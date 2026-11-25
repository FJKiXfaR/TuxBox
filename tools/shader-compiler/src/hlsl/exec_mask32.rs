//! Exact EXEC mask model for a host wave of 32 lanes.
//!
//! Model. One HLSL invocation is still one guest lane (see [`super::hlsl`] module docs), but
//! instead of keeping only *this* lane's EXEC bit the emitter keeps the whole 32-bit mask of
//! the wave in `exec_lo`, identical in every invocation, exactly as a real SGPR pair is
//! replicated per lane in the hardware wave. `exec_hi` is not maintained (a 32-lane host wave
//! has no second mask word), so every 64-bit form that reaches `exec_hi` is refused.
//!
//!   * a lane's own activity is `(exec_lo & tb_lane_bit) != 0u`, with
//!     `tb_lane_bit = 1u << WaveGetLaneIndex()` (`output.rs`), so per-lane predication of
//!     ordinary vector writes is unchanged;
//!   * a scalar (SOP) operation sees the whole mask as an integer, which is exactly the ISA
//!     semantics: `S_AND_SAVEEXEC_B32` is `D = S0 & EXEC; EXEC = D` (rdna2-isa.txt, SOP1
//!     `S_AND_SAVEEXEC_B32`), `S_FF1_I32_B64` is a bit scan of the 64-bit value
//!     (rdna2-isa.txt:6709), `S_BCNT1_I32_B*` is a population count;
//!   * instructions that ask the wave about itself are lowered to whole-wave intrinsics on
//!     that mask: `v_cmpx_*` and a `v_cmp_*` whose destination is a mask register become
//!     `WaveActiveBallot(pred).x`, so `EXEC = (EXEC & comparison)` matches "V_CMPX writes
//!     EXEC" (rdna2-isa.txt ~2699, ~2941: "write exec"); `V_MBCNT_LO/HI_U32_B32` counts the
//!     mask bits *below the lane* (`countbits(mask & ((1u << lane) - 1u))`), and
//!     `S_CBRANCH_EXECZ/VCCZ` tests the whole word, since a scalar branch is wave-uniform.
//!
//! Soundness preconditions (each one is checked by [`gate`]):
//!   1. The workgroup is one full wave — 32 threads — so the host wave *is* the guest wave
//!      on a 32-lane host. `v0` is then `WaveGetLaneIndex()`, a bijection onto the lanes, so
//!      ballot bits, `v0`-derived lane numbers and mask values agree with each other.
//!   2. No group-shared addressing and no barrier (`DS*`, `S_BARRIER`): those would observe
//!      the threadID→lane permutation instead of only the set of lanes.
//!   3. No 64-bit EXEC/VCC form and no lane read/write of a *register* (`V_READLANE_B32`,
//!      `V_WRITELANE_B32`, `V_READLANE_REGRD_B32`, ...): the lane array / spill-slot model is
//!      a different exactness domain.
//!   4. Every scalar write is executed by every lane of the wave. The emitter never puts a
//!      scalar destination (SGPR, `scc`, `exec`, `vcc`, `m0`) under a per-lane
//!      `if (exec_lo & tb_lane_bit)` guard, and [`check`] re-verifies that on the generated
//!      text; so all copies of a scalar register stay equal and every scalar branch predicate
//!      is wave-uniform.
//!
//! Residual assumption, identical to the narrower resolve model it generalizes
//! (`ray_resolve.rs`): DXIL leaves the mapping of thread IDs to lanes unspecified, so the
//! *permutation* is only closed under the bijection above. A mask value (or `v0`) that is
//! written to memory and consumed by another wave would observe the permutation; the archive
//! uses the model only for in-wave elections, ballots and reductions, which is what the
//! preconditions above cover.
use super::{compute_group, ray_resolve, Enc, Insn, Opnd, ShaderExtract, Stage};
use std::cell::Cell;

thread_local!(static ON: Cell<bool> = const { Cell::new(false) });

/// The mask model is a *retry*: a program is translated with the ordinary per-lane model
/// first, and only a failure switches to the exact wave mask. Programs that already
/// translate keep their output byte for byte.
pub(super) fn on() -> bool { ON.with(Cell::get) }

pub(super) fn retry<T>(run: impl FnOnce() -> T) -> T {
    ON.with(|c| c.set(true));
    let result = run();
    ON.with(|c| c.set(false));
    result
}

/// One full wave per workgroup, on a 32-lane host. Compute only: a pixel wave is the
/// rasteriser's, and a 64-lane guest wave needs the two-wave split instead.
pub(super) fn eligible(ex: &ShaderExtract) -> bool {
    ex.stage == Stage::Compute && compute_group(ex) == [32, 1, 1]
}

/// Whether the program actually observes EXEC as a mask. Otherwise the ordinary model is
/// left to report its own (unrelated) failure.
pub(super) fn needs_mask(insns: &[Insn]) -> bool {
    insns.iter().any(|op| {
        let m = op.mnemonic.as_str();
        m.starts_with("v_cmpx_")
            || m.contains("_saveexec_")
            || m.contains("_wrexec_")
            || (m.starts_with("v_cmp_") && matches!(op.dst[..], [Opnd::Sgpr(n)] if n & 1 != 0))
            || op.src.iter().chain(&op.dst).any(|r| {
                matches!(r, Opnd::ExecLo | Opnd::ExecHi | Opnd::VccLo | Opnd::VccHi)
            })
    })
}

/// Instructions the mask model cannot express: the lane-array forms (`V_READLANE`/
/// `V_WRITELANE`), whose spill-slot model is a separate exactness domain, group-shared
/// addressing, barriers, indirect PC and the 64-bit mask words.
pub(super) fn forbidden(ins: &Insn) -> bool {
    if ins.enc == Enc::Ds && ins.mnemonic != "ds_swizzle_b32" {
        return true;
    }
    let bad = matches!(
        ins.mnemonic.as_str(),
        "s_barrier" | "s_setpc_b64" | "s_swappc_b64"
            | "v_readlane_regrd_b32" | "v_writelane_regwr_b32"
            | "v_readlane_b32" | "v_writelane_b32"
            | "s_movreld_b32" | "s_movreld_b64" | "s_movrels_b32" | "s_movrels_b64"
    );
    bad || ins.src.iter().chain(&ins.dst).any(|r| {
        matches!(r, Opnd::VccHi | Opnd::ExecHi)
            || (matches!(r, Opnd::VccLo | Opnd::ExecLo)
                && (ins.mnemonic.ends_with("_b64") || ins.mnemonic.ends_with("_i64") && ins.enc == Enc::Sop1))
    })
}

/// Decide whether this program uses the exact mask model. `Ok(false)` means "not applicable,
/// keep the ordinary model"; `Err` is a precise refusal.
pub(super) fn select(ex: &ShaderExtract, insns: &[Insn]) -> Result<bool, String> {
    if ray_resolve::applies(ex) {
        return ray_resolve::wave32(ex, insns);
    }
    if !on() || !eligible(ex) || !needs_mask(insns) {
        return Ok(false);
    }
    if let Some(op) = insns.iter().find(|op| forbidden(op)) {
        return Err(format!(
            "EXEC mask model cannot lower {} at {:#x}: needs the 64-bit mask, a lane array or group-shared addressing",
            op.mnemonic, op.offset
        ));
    }
    Ok(true)
}

/// Post-emit assertion: nothing under a per-lane guard may write wave-wide state.
pub(super) fn check(hlsl: &str) -> Result<(), String> {
    let at = hlsl.find("void main(").or_else(|| hlsl.find(" main(")).unwrap_or(0);
    for line in hlsl[at..].lines() {
        if !line.contains("tb_lane_bit") || !line.contains("if") {
            continue;
        }
        let per_lane_guard = line.contains("exec_lo & tb_lane_bit") || line.contains("& tb_lane_bit)");
        if !per_lane_guard {
            continue;
        }
        let bytes = line.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let boundary = i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
            if boundary && bytes[i] == b's' {
                let mut j = i + 1;
                while j < bytes.len() && bytes[j].is_ascii_digit() {
                    j += 1;
                }
                if j > i + 1 && is_assignment(&line[j..]) {
                    return Err(format!(
                        "EXEC mask model: scalar {} written under a per-lane guard: {}",
                        &line[i..j],
                        line.trim()
                    ));
                }
                i = j.max(i + 1);
                continue;
            }
            i += 1;
        }
        for word in ["scc =", "exec_lo =", "vcc_lo =", "m0 ="] {
            if line.contains(word) {
                return Err(format!(
                    "EXEC mask model: wave-wide state written under a per-lane guard: {}",
                    line.trim()
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn is_assignment(rest: &str) -> bool {
    let rest = rest.trim_start();
    let bytes = rest.as_bytes();
    if bytes.first() == Some(&b'=') && bytes.get(1) != Some(&b'=') {
        return true;
    }
    if matches!(bytes.first(), Some(b'+' | b'-' | b'&' | b'|' | b'^' | b'*')) {
        return bytes.get(1) == Some(&b'=');
    }
    false
}
