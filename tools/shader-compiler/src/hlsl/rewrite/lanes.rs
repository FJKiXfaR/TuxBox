//! Lane reads, lane spills, local jumps and wave allocations.
use super::*;

/// Translator temporaries from this index up are lane-spill slots (declared up front).
pub(in crate::hlsl) const LANE_TEMP_BASE: u16 = 10_000;

/// SGPR spills through vector lanes.
///
/// When a shader runs out of scalar registers the console compiler parks scalars (observed:
/// texture/sampler descriptor dwords) in individual lanes of an otherwise unused VGPR with
/// `v_writelane_b32 vN, sX, <lane>` and brings them back with `v_readlane_b32 sY, vN, <lane>`
/// (VOP3 0x361 / 0x360; fixture `fortnite_lobby_desc.dxbc`). The translator models one lane
/// per VGPR, so each (VGPR, constant lane) slot is lowered to a function-scope translator
/// temporary and the two instructions become plain `s_mov_b32`. The descriptor dataflow
/// tracks those temporaries like registers (keys 1000+), so descriptors survive a spill.
///
/// Refused (explicit error): a non-constant lane index, or a spill VGPR that is also used as
/// an ordinary vector.
///
/// A constant read of a lane no `v_writelane_b32` writes is not a spill at all (see
/// [`plain_lane_read_max`]): it broadcasts another lane's ordinary VGPR value, which the
/// pipeline lowers to [`GROUP_READ`] when the launch covers that lane. A *non-constant*
/// selector that can address spilled lanes stays refused: a slot is a translator temporary
/// keyed by a constant lane, so an indexed read would have to select among slots
/// (`sel == 12 ? tb_s0 : ...`), and the ranges that remain in this cache are loop-carried
/// lane arrays — the console compiler parks a table of scalars in lanes 0..7, 12..14 of one
/// VGPR and reads it with a selector bounded only by `v_and 7` / `v_min3_u32`
/// (`50d4da5b:0x838`), so which slot a selector names depends on an earlier iteration.
pub(in crate::hlsl) const GROUP_READ: &str = "tb_group_readlane";

/// Store one scalar into one lane of a lane-spill area held in group memory
/// (`v_writelane_b32 vN, sX, <lane>`); the lane may be an SGPR.
pub(in crate::hlsl) const LANE_STORE: &str = "tb_lane_area_store";

/// Read one lane of a lane-spill area from group memory (`v_readlane_b32 sX, vN, <lane>`);
/// the lane may be an SGPR. A read takes part in the exchange rendezvous, a store does not.
pub(in crate::hlsl) const LANE_READ: &str = "tb_lane_area_read";

/// A read of one lane of the caller's 64-lane block, exchanged like `ds_swizzle_b32`.
/// A lane-area read publishes the whole per-lane area before it, so it needs the same
/// barrier hoisting and control-flow validation as a swizzle. A lane-area store is a
/// plain per-thread group-memory write and deliberately is not an exchange.
pub(in crate::hlsl) fn exchanges(i: &Insn) -> bool {
    i.mnemonic == "ds_swizzle_b32" || i.mnemonic == GROUP_READ || i.mnemonic == LANE_READ
}

/// The highest lane any `v_readlane_b32` takes from a VGPR that no
/// `v_writelane_b32` writes.
///
/// Such a read is not a spill slot: it broadcasts the value the source lane's own
/// VGPR holds (observed: the upper half of a wave64 add carry chain, and per-lane
/// tables the console compiler reads back as scalars). No slot names it, so it
/// needs a whole-wave exchange; the caller decides whether the program can
/// express one. Non-constant selectors already take the dynamic group read and
/// are ignored here.
pub(in crate::hlsl) fn plain_lane_read_max(insns: &[Insn]) -> Option<i64> {
    let written: BTreeSet<u8> = insns.iter()
        .filter(|i| i.enc == Enc::Vop3 && i.op == 0x361)
        .filter_map(|i| match i.dst.first() {
            Some(Opnd::Vgpr(v)) => Some(*v),
            _ => None,
        })
        .collect();
    let mut max = None;
    for i in insns {
        if i.enc != Enc::Vop3 || i.op != 0x360 { continue; }
        let Some(Opnd::Vgpr(v)) = i.src.first() else { continue };
        if written.contains(v) { continue; }
        if let Some(Opnd::ConstInt(lane)) = i.src.get(1) {
            if (0..64).contains(lane) {
                max = Some(max.map_or(*lane, |m: i64| m.max(*lane)));
            }
        }
    }
    max
}

/// A pixel is a wave of its own, so the first lane of a mask that covers it is its own.
/// `s_ff1` names that lane for `v_readlane`; anything else that uses the lane is left alone.
pub(in crate::hlsl) fn lower_first_lane_reads(insns: &mut [Insn]) {
    for i in 0..insns.len() {
        if insns[i].mnemonic != "v_readlane_b32" { continue; }
        let Some(&Opnd::Sgpr(lane)) = insns[i].src.get(1) else { continue };
        let Some(first) = (i.saturating_sub(64)..i).rev()
            .find(|&j| insns[j].dst.contains(&Opnd::Sgpr(lane)) || insns[j].enc == Enc::Sopp && insns[j].mnemonic.contains("branch"))
        else { continue };
        if !insns[first].mnemonic.starts_with("s_ff1_i32_b") { continue; }
        // The destination field names an SGPR (decoded generically as a VGPR).
        let Some(&(Opnd::Vgpr(result) | Opnd::Sgpr(result))) = insns[i].dst.first() else { continue };
        let reads = |j: usize| j != i && insns[j].src.contains(&Opnd::Sgpr(lane));
        if (first + 1..i).any(reads) { continue; }
        if result != lane {
            let end = (i + 1..insns.len()).find(|&j| insns[j].dst.contains(&Opnd::Sgpr(lane)) || insns[j].mnemonic.contains("branch"));
            if end.is_none_or(|end| (i + 1..=end).any(reads) || insns[end].mnemonic.contains("branch")) { continue; }
        }
        insns[first].mnemonic = "s_mov_b32".into();
        insns[first].op = 3;
        insns[first].src = vec![Opnd::ConstInt(0)];
        let ins = &mut insns[i];
        ins.enc = Enc::Vop1;
        ins.op = 2;
        ins.mnemonic = "v_readfirstlane_b32".into();
        ins.dst = vec![Opnd::Sgpr(result)];
        ins.src.truncate(1);
    }
}

/// Lower a lane-spill area read with a run-time selector to a per-(VGPR, lane) area in
/// group memory.
///
/// The console compiler reuses one VGPR as a table of scalars parked in lanes 0..7 and
/// 12..14 and rotates a selector (`v_and 7`, `v_min3_u32`) over it inside a loop, so a
/// constant-lane temporary cannot name the lane a selector picks. When the whole program
/// is one complete 32-multiple guest wave (the pipeline's `group_reads` proof covers this),
/// each scrutable lane op of a spill VGPR becomes a group-memory access:
///
/// * `v_writelane vN, sX, L` stores `sX` at `tb_lane_area[ord*stride + (base|L)]`,
/// * `v_readlane sD, vN, L` reads that slot back after the exchange rendezvous
///   (the [`LANE_READ`] pseudo-instruction is an exchange, so the emitter hoists the
///   barrier out of forward branches exactly as [`GROUP_READ`] does),
/// * an ordinary vector write to `vN` publishes its own lane's value to `tb_lane_area[..+tb_lane]`
///   (done by the emitter, which sees every VALU write).
///
/// Stores are ordinary shared-memory writes: their sources are wave-uniform scalars, so a
/// store from any active lane writes the value the guest's wave-level VGPR file gets. They do
/// not need a rendezvous, so a store may live inside a guest branch; only the read's barrier
/// is hoisted. The pipeline's existing exchange proof (complete blocks of 32, no exchange
/// after an early exit, and the single-guest-wave scalar-loop rule) then applies unchanged,
/// and the group-shared budget grows by one word per (VGPR, 64-lane block).
pub(in crate::hlsl) fn lower_dynamic_spill_areas(
    insns: &mut [Insn],
    group_reads: bool,
) -> Result<Option<BTreeMap<u8, u32>>, String> {
    const READLANE: u32 = 0x360;
    const WRITELANE: u32 = 0x361;
    if !group_reads {
        return Ok(None);
    }
    let written: BTreeSet<u8> = insns.iter()
        .filter(|i| i.enc == Enc::Vop3 && i.op == WRITELANE)
        .filter_map(|i| match i.dst.first() {
            Some(Opnd::Vgpr(v)) => Some(*v),
            _ => None,
        })
        .collect();
    if written.is_empty() { return Ok(None); }
    let dynamic = insns.iter().any(|i| {
        i.enc == Enc::Vop3
            && i.op == READLANE
            && matches!(i.src.first(), Some(Opnd::Vgpr(v)) if written.contains(v))
            && !matches!(i.src.get(1), Some(Opnd::ConstInt(_)))
    });
    if !dynamic { return Ok(None); }
    let mut area: BTreeMap<u8, u32> = BTreeMap::new();
    for ins in insns.iter() {
        if ins.enc != Enc::Vop3 || ins.op != WRITELANE { continue; }
        let Some(Opnd::Vgpr(v)) = ins.dst.first() else { continue };
        if !matches!(ins.src.first(), Some(Opnd::Sgpr(_) | Opnd::ConstInt(_) | Opnd::Literal(_))) {
            return Err(format!(
                "v_writelane at {:#x}: a lane-spill area needs a wave-uniform source",
                ins.offset
            ));
        }
        let next = area.len() as u32;
        area.entry(*v).or_insert(next);
    }
    for ins in insns.iter_mut() {
        if ins.enc != Enc::Vop3 { continue; }
        match ins.op {
            WRITELANE => {
                let Some(Opnd::Vgpr(v)) = ins.dst.first().cloned() else { continue };
                if !area.contains_key(&v) { continue; }
                ins.enc = Enc::Sop1;
                ins.mnemonic = LANE_STORE.into();
                ins.src.truncate(2);
                ins.op = 3;
            }
            READLANE => {
                let Some(Opnd::Vgpr(v)) = ins.src.first().cloned() else { continue };
                if !area.contains_key(&v) { continue; }
                let d = match ins.dst.first() {
                    Some(Opnd::Vgpr(d) | Opnd::Sgpr(d)) => *d,
                    _ => return Err(format!("v_readlane at {:#x}: unexpected destination", ins.offset)),
                };
                ins.mnemonic = LANE_READ.into();
                ins.enc = Enc::Sop1;
                ins.dst = vec![Opnd::Sgpr(d)];
                ins.src.truncate(2);
                ins.op = 3;
            }
            _ => {}
        }
    }
    Ok(Some(area))
}

pub(in crate::hlsl) fn lower_lane_spills(insns: &mut [Insn], wave32: bool, group_reads: bool) -> Result<(), String> {
    const READLANE: u32 = 0x360;
    const WRITELANE: u32 = 0x361;
    let is_lane = |i: &Insn| i.enc == Enc::Vop3 && (i.op == READLANE || i.op == WRITELANE);
    if !insns.iter().any(is_lane) {
        return Ok(());
    }
    let mut slots: BTreeMap<(u8, i64), Opnd> = BTreeMap::new();
    let mut spill_vregs: BTreeSet<u8> = BTreeSet::new();
    let written_vregs: BTreeSet<_> = insns.iter()
        .filter(|ins| wave32 && ins.enc == Enc::Vop3 && ins.op == WRITELANE)
        .filter_map(|ins| match ins.dst.first() {
            Some(Opnd::Vgpr(v)) => Some(*v),
            _ => None,
        })
        .collect();
    let any_written: BTreeSet<_> = insns.iter()
        .filter(|ins| ins.enc == Enc::Vop3 && ins.op == WRITELANE)
        .filter_map(|ins| match ins.dst.first() {
            Some(Opnd::Vgpr(v)) => Some(*v),
            _ => None,
        })
        .collect();
    for ins in insns.iter_mut() {
        if !is_lane(ins) {
            continue;
        }
        let at = ins.offset;
        if wave32 && ins.op == READLANE {
            if let ([Opnd::Vgpr(d) | Opnd::Sgpr(d)], [Opnd::Vgpr(v), lane, ..]) =
                (ins.dst.as_slice(), ins.src.as_slice())
            {
                if !written_vregs.contains(v) {
                    ins.dst = vec![Opnd::Sgpr(*d)];
                    ins.src = vec![Opnd::Vgpr(*v), *lane];
                    ins.enc = Enc::Sop1;
                    ins.mnemonic = ray_resolve::READ_LANE.into();
                    continue;
                }
            }
        }
        // A uniform Sgpr lane operand is only expressible through the whole-wave
        // group read; anything else stays refused. Spill areas (vregs written
        // by v_writelane) are still slotted per constant lane, so they remain
        // refused for dynamic lanes.
        if !wave32 && group_reads && ins.op == READLANE {
            let converted = match (ins.src.first(), ins.src.get(1), ins.dst.first()) {
                (Some(Opnd::Vgpr(v)), Some(Opnd::Sgpr(_)),
                 Some(Opnd::Vgpr(_)) | Some(Opnd::Sgpr(_)))
                    if !any_written.contains(v) =>
                {
                    let v = *v;
                    let d = match ins.dst.first() {
                        Some(Opnd::Vgpr(n)) | Some(Opnd::Sgpr(n)) => *n,
                        _ => unreachable!(),
                    };
                    ins.dst = vec![Opnd::Sgpr(d)];
                    ins.src = vec![Opnd::Vgpr(v), ins.src[1].clone()];
                    ins.enc = Enc::Sop1;
                    ins.mnemonic = GROUP_READ.into();
                    true
                }
                _ => false,
            };
            if converted {
                continue;
            }
        }
        let lane = match ins.src.get(1) {
            Some(Opnd::ConstInt(l)) if (0..64).contains(l) => *l,
            other => return Err(format!("lane op at {at:#x}: non-constant lane operand {other:?}")),
        };
        if ins.op == WRITELANE {
            let Some(Opnd::Vgpr(v)) = ins.dst.first().cloned() else {
                return Err(format!("v_writelane at {at:#x}: destination is not a VGPR"));
            };
            let next = slots.len();
            let slot = match slots.get(&(v, lane)) {
                Some(s) => *s,
                None => {
                    let s = Opnd::Temp(LANE_TEMP_BASE + next as u16);
                    slots.insert((v, lane), s);
                    s
                }
            };
            spill_vregs.insert(v);
            let value = ins.src.first().cloned().ok_or_else(|| format!("v_writelane at {at:#x}: no source"))?;
            ins.dst = vec![slot];
            ins.src = vec![value];
        } else {
            // The destination field of v_readlane names an SGPR (decoded generically as a VGPR).
            let d = match ins.dst.first() {
                Some(Opnd::Vgpr(n)) | Some(Opnd::Sgpr(n)) => *n,
                other => return Err(format!("v_readlane at {at:#x}: unexpected destination {other:?}")),
            };
            let Some(Opnd::Vgpr(v)) = ins.src.first().cloned() else {
                return Err(format!("v_readlane at {at:#x}: source is not a VGPR"));
            };
            if group_reads && !any_written.contains(&v) {
                ins.dst = vec![Opnd::Sgpr(d)];
                ins.src = vec![Opnd::Vgpr(v), Opnd::ConstInt(lane)];
                ins.enc = Enc::Sop1;
                ins.mnemonic = GROUP_READ.into();
                continue;
            }
            let Some(&slot) = slots.get(&(v, lane)) else {
                return Err(format!("v_readlane at {at:#x}: v{v} lane {lane} read before any v_writelane"));
            };
            ins.dst = vec![Opnd::Sgpr(d)];
            ins.src = vec![slot];
        }
        ins.enc = Enc::Sop1;
        ins.op = 0x03;
        ins.mnemonic = "s_mov_b32".into();
    }
    for ins in insns.iter() {
        if ins.mnemonic == "v_nop" {
            continue; // the two header words at the start of every program decode as v_nop
        }
        for o in ins.dst.iter().chain(ins.src.iter()) {
            if let Opnd::Vgpr(v) = o {
                if spill_vregs.contains(v) {
                    return Err(format!("v{v} is used both as a lane spill area and as a vector at {:#x}", ins.offset));
                }
            }
        }
    }
    Ok(())
}

/// Resolve local PC transfers when GETPC and a constant add form one basic block.
pub(in crate::hlsl) fn lower_local_pc_transfers(insns: &mut [Insn]) -> Result<(), String> {
    if !insns
        .iter()
        .any(|i| matches!(i.mnemonic.as_str(), "s_setpc_b64" | "s_swappc_b64"))
    {
        return Ok(());
    }
    let mut entries: BTreeSet<i64> = insns
        .iter()
        .filter(|i| i.enc == Enc::Sopp && Emitter::branch_cond(i.op).is_some())
        .map(|i| i.offset as i64 + 4 + 4 * i.aux.imm)
        .collect();
    let mut local = Vec::new();
    for (i, seq) in insns.windows(3).enumerate() {
        let [pc, add, jump] = seq else { unreachable!() };
        if !matches!(jump.mnemonic.as_str(), "s_setpc_b64" | "s_swappc_b64")
            || pc.mnemonic != "s_getpc_b64"
            || add.mnemonic != "s_add_u32"
            || add.dst.first() != pc.dst.first()
            || add.src.first() != pc.dst.first()
            || jump.src.first() != pc.dst.first()
        {
            continue;
        }
        let offset = match add.src.get(1) {
            Some(Opnd::ConstInt(k)) => *k,
            Some(Opnd::Literal(k)) => i64::from(*k as i32),
            _ => continue,
        };
        let target = pc.offset as i64 + 4 + offset;
        if !insns.iter().any(|x| x.offset as i64 == target) {
            return Err(format!(
                "PC transfer at {:#x} targets outside the decoded instructions",
                jump.offset
            ));
        }
        entries.insert(target);
        local.push((i, target));
    }
    for (i, target) in local {
        if entries.contains(&(insns[i + 1].offset as i64)) || entries.contains(&(insns[i + 2].offset as i64)) {
            return Err("branch enters a local PC-transfer sequence after GETPC".into());
        }
        let jump = &mut insns[i + 2];
        jump.enc = Enc::Sopp;
        jump.op = 2;
        jump.aux.imm = (target - jump.offset as i64 - 4) / 4;
        if jump.mnemonic == "s_setpc_b64" {
            jump.dst.clear();
        }
    }
    Ok(())
}

// These instruction sequences elect the lowest active lane to reserve a contiguous
// range, restore EXEC, broadcast the atomic result and add a per-lane prefix.
// The two register allocations differ. Match the complete regions before changing
// a saved EXEC mask from its usual per-invocation predicate into wave-uniform data.
pub(in crate::hlsl) fn mark_wave_allocations(insns: &mut [Insn], wave32_mask: bool, election: Option<ElectionLowering>) -> Result<bool, String> {
    const REGIONS: [&[u32]; 2] = [
        &[
        0xd7650007, 0x000100c1, 0xbe8e147e, 0xbe8c047e, 0xd7660007, 0x00020ec1,
        0xd482000e, 0x00020e0e, 0x7e0e0280, 0xbe8e240e, 0xbf880005, 0xbf800000,
        0xbe8c100c, 0x7e0e020c, 0xe0c84000, 0x80070700, 0xbefe040e, 0xd7650008,
        0x0001007e, 0xbf8c3f70, 0x7e180507, 0xd7660007, 0x0002107f, 0x4a0e0e0c
        ],
        &[
        0xd7650007, 0x000100c1, 0xbe83147e, 0xbe8e047e, 0xd7660007, 0x00020ec1,
        0xd4820012, 0x00020e03, 0x7e0e0280, 0xbe922412, 0xbf880005, 0xbf800000,
        0xbe83100e, 0x7e0e0203, 0xe0c84000, 0x80070700, 0xbefe0412, 0xd7650008,
        0x0001007e, 0xbf8c3f70, 0x7e060507, 0xd7660007, 0x0002107f, 0x4a0e0e03
        ],
    ];
    const COUNT: usize = 18;
    let snapshots = election.is_some() && predicate_snapshots(insns);
    let mut found = false;
    for i in 0..insns.len() {
        // Wave32 programs keep the previous lowering unless the election verifies.
        // Modes that model EXEC as a whole-wave mask already lower it directly.
        if !wave32_mask && insns[i].mnemonic == "s_ff1_i32_b32" && insns[i].src == [Opnd::ExecLo] {
            let mut trial = insns.to_vec();
            match verify_lane_election(&mut trial, i, election.is_some()) {
                Ok(()) => {
                    trial[i].mnemonic = "tb_first_active_lane".into();
                    insns.clone_from_slice(&trial);
                }
                Err(e) if std::env::var_os("TB_DFDEBUG").is_some() => eprintln!("{e}"),
                Err(_) => {}
            }
            continue;
        }
        if insns[i].mnemonic != "s_ff1_i32_b64" || insns[i].src != [Opnd::ExecLo] { continue; }
        let region = i.checked_sub(1).and_then(|first| insns.get(first..first + COUNT))
            .filter(|region| REGIONS.iter().any(|words| region.iter().flat_map(|op| op.raw.iter()).eq(words.iter())));
        let Some(region) = region else {
            if !snapshots {
                verify_lane_election(insns, i, election.is_some())?;
            }
            if election == Some(ElectionLowering::Shared64) { insns[i].mnemonic = "tb_first_exec_lane".into(); }
            continue;
        };
        let first = region[0].offset;
        let end = region[COUNT - 1].offset + 4;
        for branch in insns.iter().filter(|b| b.enc == Enc::Sopp && Emitter::branch_cond(b.op).is_some()) {
            let target = branch.offset as i64 + 4 + 4 * branch.aux.imm;
            if !(first..end).contains(&branch.offset) && target > first as i64 && target < end as i64 {
                return Err("external branch enters wave allocation".into());
            }
        }
        insns[i + 1].mnemonic = "tb_capture_allocation_mask".into();
        insns[i + 12].mnemonic = "tb_allocation_prefix_lo".into();
        insns[i + 15].mnemonic = "tb_allocation_prefix_hi".into();
        found = true;
    }
    Ok(found)
}
