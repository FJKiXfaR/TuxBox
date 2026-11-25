use super::*;
use crate::hlsl::exec_mask32::{check, forbidden, needs_mask};

fn insn(m: &str, enc: Enc, dst: Vec<Opnd>, src: Vec<Opnd>) -> Insn {
    Insn {
        offset: 0,
        dwords: 1,
        enc,
        op: 0,
        raw: vec![0],
        mnemonic: m.to_string(),
        dst,
        src,
        aux: crate::rdna2::InsnAux { imm: 0, mask: 0, imm2: 0, sdwa: None },
    }
}

#[test]
fn a_lane_mask_consumer_asks_for_the_exact_mask() {
    // `v_cmpx_lt_f32` + `s_and_saveexec_b32` + a mask-count: EXEC is read as a value, which
    // the one-bit model cannot express.
    let program = [
        insn("v_cmpx_lt_f32", Enc::Vopc, vec![Opnd::ExecLo], vec![Opnd::Vgpr(0), Opnd::Vgpr(1)]),
        insn("s_and_saveexec_b32", Enc::Sop1, vec![Opnd::Sgpr(4)], vec![Opnd::Sgpr(8)]),
        insn("s_bcnt1_i32_b32", Enc::Sop1, vec![Opnd::Sgpr(6)], vec![Opnd::ExecLo]),
    ];
    assert!(needs_mask(&program));
    assert!(program.iter().all(|i| !forbidden(i)));
    // A comparison into an odd SGPR is the other shape that keeps a whole mask.
    let ballot = [insn("v_cmp_eq_u32", Enc::Vopc, vec![Opnd::Sgpr(3)], vec![Opnd::Vgpr(0), Opnd::Vgpr(1)])];
    assert!(needs_mask(&ballot));
    // Nothing observes the wave: leave the ordinary model to report its own failure.
    let plain = [insn("v_add_f32", Enc::Vop2, vec![Opnd::Vgpr(2)], vec![Opnd::Vgpr(0), Opnd::Vgpr(1)])];
    assert!(!needs_mask(&plain));
}

#[test]
fn the_mask_model_refuses_shapes_it_cannot_express() {
    // The second mask word is not maintained on a 32-lane host.
    let wide = insn("s_and_b64", Enc::Sop2, vec![Opnd::ExecLo], vec![Opnd::Sgpr(0), Opnd::Sgpr(2)]);
    assert!(forbidden(&wide));
    let read_hi = insn("s_mov_b32", Enc::Sop1, vec![Opnd::Sgpr(4)], vec![Opnd::ExecHi]);
    assert!(forbidden(&read_hi));
    // The lane array/spill-slot model is a different exactness domain.
    assert!(forbidden(&insn("v_readlane_b32", Enc::Vop3, vec![Opnd::Vgpr(0)], vec![Opnd::Vgpr(1), Opnd::ConstInt(3)])));
    // Group-shared addressing would observe the threadID->lane permutation.
    assert!(forbidden(&insn("ds_read_b32", Enc::Ds, vec![Opnd::Vgpr(0)], vec![Opnd::Vgpr(1)])));
    assert!(forbidden(&insn("s_barrier", Enc::Sopp, vec![], vec![])));
    // `V_MBCNT_LO_U32_B32` on the mask is exactly `countbits(EXEC & lanemask_lt)`.
    assert!(!forbidden(&insn(
        "v_mbcnt_lo_u32_b32",
        Enc::Vop2,
        vec![Opnd::Vgpr(0)],
        vec![Opnd::ExecLo, Opnd::ConstInt(0)]
    )));
}

#[test]
fn a_per_lane_guard_may_not_write_wave_state() {
    let ok = "void main() {\n    uint exec_lo = 1u, tb_lane_bit = 1u;\n    if (exec_lo & tb_lane_bit) v3 = 1u;\n}\n";
    assert!(check(ok).is_ok());
    let scalar = "void main() {\n    uint exec_lo = 1u, tb_lane_bit = 1u;\n    if (exec_lo & tb_lane_bit) s3 = 1u;\n}\n";
    assert!(check(scalar).is_err());
    let scc = "void main() {\n    uint exec_lo = 1u, tb_lane_bit = 1u;\n    if ((exec_lo & tb_lane_bit) != 0u) scc = 1u;\n}\n";
    assert!(check(scc).is_err());
}
