//! Values exchanged between lanes through group memory.
use super::*;

impl<'a> Emitter<'a> {
    // READLANE ignores EXEC. Publish each lane's current VGPR even when
    // that lane skipped the guest branch, then guard only the scalar write.
    pub(in crate::hlsl) fn emit_indexed_read(&mut self, idx: usize, ins: &Insn) -> Result<(), String> {
        self.line(&format!("tb_swizzle[tb_lane] = {};", r(&ins.src[0])?));
        self.line("GroupMemoryBarrierWithGroupSync();");
        self.line(&format!("uint readlane{idx} = tb_swizzle[(tb_lane & ~63u) | ({} & 63u)];", r(&ins.src[1])?));
        self.line("GroupMemoryBarrierWithGroupSync();");
        Ok(())
    }

    /// Words per lane-spill area ordinal: one 64-lane block per launched row of the
    /// group, rounded up so every guest lane of every block has a slot.
    pub(in crate::hlsl) fn lane_area_stride(&self) -> u32 {
        let count = compute_group(self.ex).into_iter().fold(1u32, u32::saturating_mul);
        count.div_ceil(64).max(1) * 64
    }

    /// Index of one lane of one area ordinal inside `tb_lane_area`.
    pub(in crate::hlsl) fn lane_area_index(&self, ord: u32, lane: &str) -> String {
        format!("{}u + ((tb_lane & ~63u) | (({lane}) & 63u))", ord * self.lane_area_stride())
    }

    /// Index of the calling lane's own slot, for publishing an ordinary VALU write.
    pub(in crate::hlsl) fn lane_area_own(&self, ord: u32) -> String {
        format!("{}u + tb_lane", ord * self.lane_area_stride())
    }

    pub(in crate::hlsl) fn emit_compute_swizzle(&mut self, idx: usize, ins: &Insn, active: &[usize]) -> Result<(), String> {
        if ins.mnemonic == LANE_READ {
            // A run-time selector indexes the per-(VGPR, 64-lane block) area of the
            // guest's lane-spill VGPR. Publishing lanes wrote their own slot (or a
            // `v_writelane` stored a parked scalar), so the barrier makes the whole
            // guest wave's lane file visible before the indexed read. Like a lane
            // read it ignores EXEC, so the barrier is hoisted out of guest branches.
            let Some(area) = self.lane_area.as_ref() else {
                return Err("lane-area read without a lane-spill area".into());
            };
            let Some(Opnd::Vgpr(v)) = ins.src.first() else {
                return Err("lane-area read without a source VGPR".into());
            };
            let Some(ord) = area.get(v).copied() else {
                return Err(format!("v{v} is not a lane-spill area"));
            };
            let index = self.lane_area_index(ord, &r(&ins.src[1])?);
            self.line("GroupMemoryBarrierWithGroupSync();");
            self.line(&format!("uint swizzle{idx} = tb_lane_area[{index}];"));
            self.line("GroupMemoryBarrierWithGroupSync();");
            return Ok(());
        }
        if ins.mnemonic == GROUP_READ {
            // A lane read ignores EXEC.
            let guard: Vec<_> = active.iter().map(|label| format!("!j{label}")).collect();
            let value = r(&ins.src[0])?;
            let value = if guard.is_empty() { value } else { format!("({}) ? {value} : 0u", guard.join(" && ")) };
            self.line(&format!("tb_swizzle[tb_lane] = {value};"));
            self.line("GroupMemoryBarrierWithGroupSync();");
            self.line(&format!("uint swizzle{idx} = tb_swizzle[(tb_lane & ~63u) | {}];", r(&ins.src[1])?));
            self.line("GroupMemoryBarrierWithGroupSync();");
            return Ok(());
        }
        let mode = ins.aux.imm as u32;
        // RDNA2 ISA DS_SWIZZLE_B32: offset[15] selects QDMode, which permutes
        // within each quad by offset[7:0]. Other high offset bits and GDS stay refused.
        let quad = mode & 0x8000 != 0;
        if ins.raw[0] & 0x3_0000 != 0 || (quad && mode & 0xff00 != 0x8000) {
            return Err("unsupported compute DS swizzle encoding".into());
        }
        // Skipped and EXEC-masked lanes contribute zero. Rendezvous before
        // reapplying the guest branch guard to the destination write.
        let mut predicate = String::from("(exec_lo & 1u) != 0u");
        for label in active { write!(predicate, " && !j{label}").unwrap(); }
        let value = r(&ins.src[0])?;
        self.line(&format!("tb_swizzle[tb_lane] = ({predicate}) ? {value} : 0u;"));
        self.line("GroupMemoryBarrierWithGroupSync();");
        if quad {
            self.line(&format!("uint swizzle{idx} = tb_swizzle[(tb_lane & ~3u) | (({:#x}u >> (2u * (tb_lane & 3u))) & 3u)];",
                mode & 255));
        } else {
            self.line(&format!("uint swizzle{idx} = tb_swizzle[(tb_lane & ~31u) | (((tb_lane & {}u) | {}u) ^ {}u)];",
                mode & 31, (mode >> 5) & 31, (mode >> 10) & 31));
        }
        self.line("GroupMemoryBarrierWithGroupSync();");
        Ok(())
    }
}
