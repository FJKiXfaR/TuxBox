//! One instruction: dispatch by encoding, and the scalar instructions.
use super::*;

impl<'a> Emitter<'a> {
    pub(in crate::hlsl) fn emit_insn(&mut self, idx: usize, ins: &Insn, is_ps: bool, br: Option<usize>) -> Result<(), String> {
        self.emit_insn_body(idx, ins, is_ps, br)?;
        if let Some(vote) = self.lockstep.as_ref().and_then(|votes| votes.get(&idx)).cloned() {
            let mask = r(&vote.mask)?;
            let not = if vote.none { "!" } else { "" };
            self.line(&format!("scc = uint({not}WaveActiveAnyTrue(({mask} & 1u) != 0u));"));
        }
        if self.vote_defs.contains(&idx) {
            let mask = r(&ins.dst[0])?;
            self.line(&format!("tb_votes{idx} = WaveActiveCountBits(({mask} & 1u) != 0u);"));
        }
        Ok(())
    }

    pub(in crate::hlsl) fn emit_insn_body(&mut self, idx: usize, ins: &Insn, is_ps: bool, br: Option<usize>) -> Result<(), String> {
        let m = ins.mnemonic.as_str();
        let s = &ins.src;
        let d = ins.dst.first().cloned();
        self.line(&format!("// {:#06x} {m}", ins.offset));
        self.cur_idx = idx;
        if let Some(&site) = self.exec_snapshots.sites.get(&idx) {
            if site.emit(self, idx, ins)? { return Ok(()); }
        }
        if m == "tb_reflection_far_query" {
            reflection_far::emit(self, ins);
            return Ok(());
        }
        if m == "tb_radiance_query" {
            radiance_probe::emit(self, ins);
            return Ok(());
        }
        if let Some(reg) = self.packed.as_ref().and_then(|p| p.ballot_nonempty.get(&idx)) {
            self.line(&format!("s{reg} &= exec_lo; scc = uint(WaveActiveAnyTrue((s{reg} & 1u) != 0u));"));
            return Ok(());
        }
        if ins.enc == Enc::Ds && m != "ds_swizzle_b32" && !self.hoist_wave_exchanges && wave_cull::lds_words(self).is_some() {
            // Publish writes and finish reads before the next guest LDS access.
            self.line("GroupMemoryBarrierWithGroupSync();");
        }
        if let Some(width) = self.fspx_wave {
            if fspx_wave::emit(self, idx, ins, width)? { return Ok(()); }
        }
        if self.wave_cull {
            wave_cull::prepare_insn(self, idx, ins)?;
            if wave_cull::emit(self, idx, ins, br)? { return Ok(()); }
        }
        if self.page_waves && page_waves::emit(self, idx, ins, br)? { return Ok(()); }
        match m {
            "tb_light_grid_reserve" => {
                self.line(light_grid::RESERVE);
                return Ok(());
            }
            "tb_first_exec_lane" => return self.ws(&d.ok_or("missing election result")?, &format!("election{idx}")),
            "tb_indexed_readlane" => {
                if self.hoisted_swizzle != Some(idx) { self.emit_indexed_read(idx, ins)?; }
                self.ws(&d.ok_or("missing lane-read destination")?, &format!("readlane{idx}"))?;
                return Ok(());
            }
            "tb_view_occluded" => {
                self.line("hidden = (exec_lo & 1u) != 0u; hidden_id = v0; hidden_view = cull_view;");
                return Ok(());
            }
            "tb_view_candidate" => {
                self.line("cull_candidate = (exec_lo & 1u) != 0u;");
                self.line(&format!("cull_record = uint3({}, {}, {});", r(&ins.src[0])?, r(&ins.src[1])?, r(&ins.src[2])?));
                return Ok(());
            }
            "s_ff1_i32_b64" | "s_ff1_i32_b32" | "tb_first_active_lane" if lone_lane::on() && ins.src == [Opnd::ExecLo] => {
                return self.ws(&d.ok_or("missing elected lane register")?, "((exec_lo & 1u) ? 0u : 0xffffffffu)");
            }
            "s_ff1_i32_b64" if ins.src == [Opnd::ExecLo] => {
                // WaveActiveMin returns the same lane ID to all participating invocations.
                let dst = d.ok_or("missing elected lane register")?;
                return self.ws(&dst, "WaveActiveMin((exec_lo & 1u) ? WaveGetLaneIndex() : 0xffffffffu)");
            }
            "tb_subsurface_group_sync" => {
                self.line("GroupMemoryBarrierWithGroupSync();");
                return Ok(());
            }
            "tb_first_active_lane" => {
                let dst = d.ok_or("missing elected lane register")?;
                return self.ws(&dst, "WaveActiveMin((exec_lo & 1u) ? WaveGetLaneIndex() : 0xffffffffu)");
            }
            "tb_capture_allocation_mask32" => {
                let dst = d.ok_or("missing mask register")?;
                return self.ws(&dst, "WaveActiveBallot((exec_lo & 1u) != 0u).x");
            }
            "tb_capture_allocation_mask" => {
                let dst = d.ok_or("missing mask register")?;
                let upper = hi(&dst);
                self.touch(&upper);
                self.line(&format!("uint4 alloc_mask{idx} = WaveActiveBallot((exec_lo & 1u) != 0u);"));
                self.ws(&dst, &format!("alloc_mask{idx}.x"))?;
                return self.ws(&upper, &format!("alloc_mask{idx}.y"));
            }
            "tb_allocation_prefix_lo" | "tb_allocation_prefix_hi" => {
                let high = m.ends_with("_hi");
                let lane = if high { "max(32u, WaveGetLaneIndex()) - 32u" } else { "WaveGetLaneIndex()" };
                let word = if high { "y" } else { "x" };
                let prefix = format!("countbits(WaveActiveBallot((exec_lo & 1u) != 0u).{word} & {})", low_mask(lane));
                return self.wv(&d.ok_or("missing prefix destination")?, &format!("({} + {prefix})", r(&s[1])?));
            }
            _ => {}
        }
        if wave_alloc::emit(self, ins)? { return Ok(()); }
        if hair_cluster::emit(self, ins) { return Ok(()); }
        if hair_cluster_export::emit(self, ins) { return Ok(()); }
        if m == "tb_hidden_instance" {
            self.line("hidden = (exec_lo & 1u) != 0u; hidden_id = v0; if (hidden) v42 = v28;");
            return Ok(());
        }
        if instance_cull::emit(self, ins)? { return Ok(()); }
        if ps09_classify::emit(self, ins) { return Ok(()); }
        if let Some(tile) = self.lds_tile {
            if tile.emit(self, idx)? { return Ok(()); }
        }
        if let Some(&id) = self.epilog_pack.get(&idx) {
            // Export epilogue: keep the unpacked operands (see `find_export_epilogs`).
            for (k, sfx) in ["a", "b"].iter().enumerate() {
                let o = *s.get(k).ok_or("pack without two operands")?;
                self.touch(&o);
                self.regs.insert(format!("tb_pk{id}{sfx}"));
                self.line(&format!("tb_pk{id}{sfx} = {};", r(&o)?));
            }
        }
        // Run-time descriptor selectors (only for shaders with a "one of N" image op).
        if let Some((reg, cands)) = self.bsel.remove(&idx) {
            // One guarded copy of the op per candidate buffer.
            self.regs.insert(format!("tb_dsel{reg}"));
            let mut first = true;
            for (id, res) in &cands {
                self.line(&format!("{}if (tb_dsel{reg} == {id:#x}u) {{", if first { "" } else { "else " }));
                first = false;
                self.res.insert(idx, vec![*res]);
                self.emit_insn(idx, ins, is_ps, br)?;
                self.line("}");
                self.drop_heap_handles();
            }
            self.bsel.insert(idx, (reg, cands));
            return Ok(());
        }
        if !self.msel.is_empty() || !self.bsel.is_empty() {
            if let Some(&(d, table, reg, imm)) = self.dsel_runtime.get(&idx) {
                let offset = Opnd::Sgpr(reg);
                self.touch(&offset);
                self.regs.insert(format!("tb_dsel{d}"));
                self.line(&format!("tb_dsel{d} = {:#x}u | ({} + {imm:#x}u);",
                    u32::from(table) << 24, r(&offset)?));
            }
            if let Some(&(d, id)) = self.dsel_loads.get(&idx) {
                self.regs.insert(format!("tb_dsel{d}"));
                self.line(&format!("tb_dsel{d} = {id:#x}u;"));
            }
            if let Some(&(d, s)) = self.dsel_moves.get(&idx) {
                self.regs.insert(format!("tb_dsel{d}"));
                self.regs.insert(format!("tb_dsel{s}"));
                self.line(&format!("tb_dsel{d} = tb_dsel{s};"));
            }
        }
        if ins.enc == Enc::Mimg {
            if let Some(sel) = self.msel.get(&idx).cloned() {
                // One guarded copy of the op per (texture, sampler) candidate pair.
                self.regs.insert(format!("tb_dsel{}", sel.tex_reg));
                self.regs.insert(format!("tb_dsel{}", sel.samp_reg));
                let samps: Vec<Option<(u32, Res)>> =
                    if sel.samp.is_empty() { vec![None] } else { sel.samp.iter().copied().map(Some).collect() };
                let mut first = true;
                for (tid, tres) in &sel.tex {
                    for s in &samps {
                        let mut cond = vec![];
                        if sel.tex.len() > 1 {
                            cond.push(format!("tb_dsel{} == {tid:#x}u", sel.tex_reg));
                        }
                        let mut v = vec![*tres];
                        if let Some((sid, sres)) = s {
                            if sel.samp.len() > 1 {
                                cond.push(format!("tb_dsel{} == {sid:#x}u", sel.samp_reg));
                            }
                            v.push(*sres);
                        }
                        self.line(&format!("{}if ({}) {{", if first { "" } else { "else " }, cond.join(" && ")));
                        first = false;
                        self.res.insert(idx, v);
                        self.emit_mimg(idx, ins)?;
                        self.line("}");
                        self.drop_heap_handles();
                    }
                }
                return Ok(());
            }
        }
        if scalar_bits::emit(self, idx, ins)? { return Ok(()); }
        match ins.enc {
            Enc::Sopp if matches!(ins.op, 0x24 | 0x25) => {
                // MODE[7:0] exposes the guest round and denorm settings to S_GETREG.
                // V_CVT_F16_F32 also consumes these settings.
                let shift = if ins.op == 0x24 { 0 } else { 4 };
                self.regs.insert("tb_fp_mode".into());
                self.line(&format!("tb_fp_mode = (tb_fp_mode & ~{:#x}u) | {:#x}u;", 15 << shift, (ins.raw[0] & 15) << shift));
                Ok(())
            }
            Enc::Sopp if ins.op == 0x10 => self.emit_sendmsg(ins),
            Enc::Sopp if ins.op == 0x0a && self.is_cs => {
                if self.packed.as_ref().is_some_and(|p| p.device_sync_barriers.contains(&idx)) {
                    self.line("AllMemoryBarrierWithGroupSync();");
                } else {
                    self.line("GroupMemoryBarrierWithGroupSync();");
                }
                Ok(())
            }
            Enc::Sopp => {
                if m == "s_swappc_b64" {
                    let link = d.ok_or("missing local PC link register")?;
                    self.ws(&link, &format!("{:#x}u", ins.offset + 4))?;
                    self.ws(&hi(&link), "0u")?;
                }
                let cond = match ins.op {
                    2 => Some("true".to_string()),
                    4 => Some("scc == 0u".into()),
                    5 => Some("scc != 0u".into()),
                    6 => Some("(vcc_lo & 1u) == 0u".into()),
                    7 => Some("(vcc_lo & 1u) != 0u".into()),
                    // The recorded guard reaches every invocation before the
                    // full-mask reduction, including guest-inactive lanes.
                    8 if self.reduction_execz_guards.contains(&idx) =>
                        Some("!WaveActiveAnyTrue((exec_lo & 1u) != 0u)".into()),
                    8 => Some("(exec_lo & 1u) == 0u".into()),
                    9 => Some("(exec_lo & 1u) != 0u".into()),
                    // A non-final s_endpgm ends the program on this path: jump to the end label.
                    1 if br.is_some() => Some("true".to_string()),
                    0 | 1 | 0x0a | 0x0c | 0x1f | 0x20 | 0x21 | 0x23 | 0x24 | 0x25 => None,
                    _ => return Err("unsupported SOPP".into()),
                };
                if let (Some(c), Some(t)) = (cond, br) {
                    let c = if self.resolve_wave32 || self.lockstep.is_some() {
                        wave_cull::branch_condition(self, ins, &c)
                    } else { c };
                    self.line(&format!("if ({c}) j{t} = true;"));
                }
                Ok(())
            }
            Enc::Vop2 | Enc::Vop1 | Enc::Vop3 | Enc::Vopc => self.emit_vec(ins),
            Enc::Sop2 => {
                let d = d.ok_or("no dst")?;
                let (a, b) = (r(&s[0])?, r(&s[1])?);
                if matches!(m, "s_lshr_b64" | "s_ashr_i64" | "s_bfe_u64" | "s_bfe_i64") {
                    self.touch(&hi(&s[0]));
                    let upper = r(&hi(&s[0]))?;
                    let t = self.tmp;
                    self.tmp += 1;
                    let signed = m.ends_with("i64");
                    let call = if m.starts_with("s_bfe") { "tb_field64" } else { "tb_shr64" };
                    self.line(&format!("uint2 t{t} = {call}({a}, {upper}, {b}, {signed}); scc = any(t{t} != 0u);"));
                    self.ws(&d, &format!("t{t}.x"))?;
                    return self.ws(&hi(&d), &format!("t{t}.y"));
                }
                if m == "s_bfm_b64" || m == "s_cselect_b64" || m == "s_lshl_b64" {
                    let t = self.tmp;
                    self.tmp += 1;
                    let (lo, hi_e) = if m == "s_lshl_b64" {
                        self.touch(&hi(&s[0]));
                        let ah = r(&hi(&s[0]))?;
                        (
                            format!("((({b} & 63u) >= 32u) ? 0u : ({a} << ({b} & 31u)))"),
                            format!(
                                "((({b} & 63u) == 0u) ? {ah} : ((({b} & 63u) >= 32u) ? ({a} << (({b} & 63u) - 32u)) : (({ah} << ({b} & 31u)) | ({a} >> (32u - ({b} & 31u))))))"
                            ),
                        )
                    } else if m == "s_bfm_b64" {
                        // Intersect [start, end) with each 32-bit half of the destination.
                        self.line(&format!("uint t{t}start = {b} & 63u, t{t}end = t{t}start + ({a} & 63u);"));
                        let word_mask = |base: u32| {
                            let start = low_mask(&format!("max(t{t}start, {base}u) - {base}u"));
                            let end = low_mask(&format!("max(t{t}end, {base}u) - {base}u"));
                            format!("({end} & ~{start})")
                        };
                        (word_mask(0), word_mask(32))
                    } else {
                        self.touch(&hi(&s[0]));
                        self.touch(&hi(&s[1]));
                        (
                            format!("((scc != 0u) ? {a} : {b})"),
                            format!("((scc != 0u) ? {} : {})", r(&hi(&s[0]))?, r(&hi(&s[1]))?),
                        )
                    };
                    self.line(&format!("uint t{t}l = {lo}; uint t{t}h = {hi_e};"));
                    if m == "s_lshl_b64" {
                        self.line(&format!("scc = ((t{t}l | t{t}h) != 0u) ? 1u : 0u;"));
                    }
                    self.ws(&d, &format!("t{t}l"))?;
                    let dh = hi(&d);
                    self.touch(&dh);
                    return self.ws(&dh, &format!("t{t}h"));
                }
                let is64 = m.ends_with("_b64");
                let bit = |op: &str, a: &str, b: &str| -> Option<String> {
                    Some(match op {
                        "and" => format!("({a} & {b})"),
                        "or" => format!("({a} | {b})"),
                        "xor" => format!("({a} ^ {b})"),
                        "andn2" => format!("({a} & ~{b})"),
                        "orn2" => format!("({a} | ~{b})"),
                        "nand" => format!("(~({a} & {b}))"),
                        "nor" => format!("(~({a} | {b}))"),
                        "xnor" => format!("(~({a} ^ {b}))"),
                        _ => return None,
                    })
                };
                let opname = m.trim_start_matches("s_").trim_end_matches("_b32").trim_end_matches("_b64");
                if let Some(lo) = bit(opname, &a, &b) {
                    if is64 {
                        self.touch(&hi(&s[0]));
                        self.touch(&hi(&s[1]));
                        let (ah, bh) = (r(&hi(&s[0]))?, r(&hi(&s[1]))?);
                        let hi_e = bit(opname, &ah, &bh).unwrap();
                        let t = self.tmp;
                        self.tmp += 1;
                        self.line(&format!("uint t{t}l = {lo}; uint t{t}h = {hi_e};"));
                        self.ws(&d, &format!("t{t}l"))?;
                        let dh = hi(&d);
                        self.touch(&dh);
                        self.ws(&dh, &format!("t{t}h"))?;
                        self.line(&format!("scc = ((t{t}l | t{t}h) != 0u) ? 1u : 0u;"));
                        if is_ps && m == "s_andn2_b64" && pixel_quad::guards_expansion(&self.insns, idx) {
                            self.line("#if __SHADER_TARGET_MAJOR >= 6");
                            self.line(&pixel_quad::vote(t, &format!("t{t}l")));
                            self.line(&format!("scc = q{t} | (uint)WaveActiveAnyTrue(q{t} != 0u);"));
                            self.line("#endif");
                        }
                    } else {
                        self.ws(&d, &lo)?;
                        self.line(&format!("scc = ({} != 0u) ? 1u : 0u;", name(&d)?));
                    }
                    return Ok(());
                }
                let e = match m {
                    "s_add_u32" | "s_add_i32" => format!("({a} + {b})"),
                    "s_sub_u32" | "s_sub_i32" => format!("({a} - {b})"),
                    "s_addc_u32" => format!("({a} + {b} + (scc & 1u))"),
                    "s_subb_u32" => format!("({a} - {b} - (scc & 1u))"),
                    "s_mul_i32" => format!("({a} * {b})"),
                    // The subtraction wraps before taking the absolute value, including INT_MIN.
                    "s_absdiff_i32" => format!("(asint({a} - {b}) < 0 ? ({b} - {a}) : ({a} - {b}))"),
                    "s_mul_hi_u32" => format!("tb_mulhi_u32({a}, {b})"),
                    "s_mul_hi_i32" => format!("tb_mulhi_i32({a}, {b})"),
                    "s_lshl_b32" => format!("({a} << ({b} & 31u))"),
                    "s_lshr_b32" => format!("({a} >> ({b} & 31u))"),
                    "s_ashr_i32" => format!("asuint(asint({a}) >> ({b} & 31u))"),
                    "s_min_u32" => format!("min({a}, {b})"),
                    "s_max_u32" => format!("max({a}, {b})"),
                    "s_min_i32" => format!("asuint(min(asint({a}), asint({b})))"),
                    "s_max_i32" => format!("asuint(max(asint({a}), asint({b})))"),
                    "s_cselect_b32" => format!("((scc != 0u) ? {a} : {b})"),
                    "s_bfe_u32" => format!("(({a} >> ({b} & 31u)) & {})", low_mask(&format!("({b} >> 16) & 127u"))),
                    "s_bfe_i32" => format!("tb_bfe_i32({a}, {b})"),
                    "s_bfm_b32" => format!("(((1u << ({a} & 31u)) - 1u) << ({b} & 31u))"),
                    "s_pack_ll_b32_b16" => format!("(({a} & 65535u) | ({b} << 16))"),
                    "s_pack_lh_b32_b16" => format!("(({a} & 65535u) | ({b} & 0xffff0000u))"),
                    "s_pack_hh_b32_b16" => format!("(({a} >> 16) | ({b} & 0xffff0000u))"),
                    "s_lshl1_add_u32" => format!("(({a} << 1) + {b})"),
                    "s_lshl2_add_u32" => format!("(({a} << 2) + {b})"),
                    "s_lshl3_add_u32" => format!("(({a} << 3) + {b})"),
                    "s_lshl4_add_u32" => format!("(({a} << 4) + {b})"),
                    _ => return Err("unsupported SOP2".into()),
                };
                let t = self.tmp;
                self.tmp += 1;
                let result = format!("t{t}");
                self.line(&format!("uint {result} = {e};"));
                // SCC may depend on both original sources even when SDST aliases one of them.
                let scc = match m {
                    "s_add_i32" => Some(signed_add_overflow(&a, &b, &result)),
                    "s_sub_i32" => Some(format!("((({a} ^ {b}) & ({a} ^ {result})) >> 31)")),
                    // The carry is of the full sum, so bits the shift moves past
                    // bit 31 set it as well as a wrapped addition.
                    "s_lshl1_add_u32" => Some(shift_add_carry(&a, &b, &result, 1)),
                    "s_lshl2_add_u32" => Some(shift_add_carry(&a, &b, &result, 2)),
                    "s_lshl3_add_u32" => Some(shift_add_carry(&a, &b, &result, 3)),
                    "s_lshl4_add_u32" => Some(shift_add_carry(&a, &b, &result, 4)),
                    "s_add_u32" => Some(format!("(({a} + {b}) < {a}) ? 1u : 0u")),
                    "s_addc_u32" => Some(format!("((({a} + {b}) < {a}) || (({a} + {b} + (scc & 1u)) < ({a} + {b}))) ? 1u : 0u")),
                    "s_subb_u32" => Some(format!("(({b} > {a}) || ((scc & 1u) > ({a} - {b}))) ? 1u : 0u")),
                    "s_sub_u32" => Some(format!("({b} > {a}) ? 1u : 0u")),
                    "s_min_i32" => Some(format!("(asint({a}) < asint({b})) ? 1u : 0u")),
                    "s_max_i32" => Some(format!("(asint({a}) > asint({b})) ? 1u : 0u")),
                    "s_min_u32" => Some(format!("({a} < {b}) ? 1u : 0u")),
                    "s_max_u32" => Some(format!("({a} > {b}) ? 1u : 0u")),
                    "s_lshl_b32" | "s_lshr_b32" | "s_ashr_i32" | "s_bfe_u32" | "s_bfe_i32" | "s_absdiff_i32" => Some(format!("({result} != 0u) ? 1u : 0u")),
                    _ => None,
                };
                if let Some(flag) = scc {
                    self.line(&format!("scc = {flag};"));
                }
                self.ws(&d, &result)
            }
            Enc::Sop1 => {
                let d = d.ok_or("no dst")?;
                let a = r(&s[0])?;
                match m {
                    "s_mov_b32" => self.ws(&d, &a),
                    GROUP_READ => {
                        if self.hoisted_swizzle != Some(idx) { self.emit_compute_swizzle(idx, ins, &[])?; }
                        self.ws(&d, &format!("swizzle{idx}"))
                    }
                    ray_resolve::READ_LANE => {
                        let lane = r(&s[1])?;
                        self.ws(&d, &format!("WaveReadLaneAt({a}, ({lane}) & 31u)"))
                    }
                    // D[S0[4:0]] = 0 / 1
                    "s_bitset0_b32" => {
                        self.touch(&d);
                        let cur = r(&d)?;
                        self.ws(&d, &format!("({cur} & ~(1u << (({a}) & 31u)))"))
                    }
                    "s_bitset1_b32" => {
                        self.touch(&d);
                        let cur = r(&d)?;
                        self.ws(&d, &format!("({cur} | (1u << (({a}) & 31u)))"))
                    }
                    "s_wqm_b32" | "s_wqm_b64" => {
                        // For an ordinary scalar mask, replicate each nonempty nibble.
                        // EXEC instead holds one guest bit per invocation.
                        let t = self.tmp;
                        self.tmp += 1;
                        let wide = m.ends_with("_b64");
                        let lane_mask = is_ps && pixel_quad::saved_exec(&self.insns, idx);
                        if s[0] == Opnd::ExecLo && lone_lane::on() {
                            self.line(&format!("uint t{t}l = exec_lo & 1u, t{t}h = 0u;"));
                        } else if s[0] == Opnd::ExecLo && !self.resolve_wave32 {
                            self.line("#if __SHADER_TARGET_MAJOR >= 6");
                            self.line(&format!("uint4 t{t}ballot = WaveActiveBallot((exec_lo & 1u) != 0u);"));
                            self.line(&format!("uint t{t}lane = WaveGetLaneIndex(); uint t{t}l = ((t{t}ballot[t{t}lane >> 5] >> (t{t}lane & 28u)) & 15u) != 0u;"));
                            self.line("#else");
                            self.line(&format!("uint4 t{t}ballot = uint4(exec_lo,0,0,0); uint t{t}l = exec_lo & 1u;"));
                            self.line("#endif");
                            self.line(&format!("uint t{t}h = 0u;"));
                        } else if lane_mask {
                            self.line("#if __SHADER_TARGET_MAJOR >= 6");
                            self.line(&pixel_quad::vote(t, &a));
                            self.line(&format!("uint t{t}l = q{t}, t{t}h = 0u;"));
                            self.line(&format!("scc = q{t} | (uint)WaveActiveAnyTrue(q{t} != 0u);"));
                            self.line("#else");
                            self.line(&format!("uint t{t}l = {a} & 1u, t{t}h = 0u; scc = t{t}l;"));
                            self.line("#endif");
                        } else {
                            let expand = |word: &str| format!("((({word} | ({word} >> 1) | ({word} >> 2) | ({word} >> 3)) & 0x11111111u) * 15u)");
                            self.line(&format!("uint t{t}l = {};", expand(&a)));
                            let upper = if wide {
                                self.touch(&hi(&s[0]));
                                expand(&r(&hi(&s[0]))?)
                            } else { "0u".into() };
                            self.line(&format!("uint t{t}h = {upper};"));
                        }
                        if s[0] == Opnd::ExecLo && !self.resolve_wave32 && !lone_lane::on() {
                            self.line(&format!("scc = any(t{t}ballot != 0u);"));
                        } else if !lane_mask {
                            self.line(&format!("scc = ((t{t}l | t{t}h) != 0u);"));
                        }
                        self.ws(&d, &format!("t{t}l"))?;
                        if wide { self.ws(&hi(&d), &format!("t{t}h"))?; }
                        Ok(())
                    }
                    "s_mov_b64" | "s_cmov_b64" | "s_not_b64" => {
                        self.touch(&hi(&s[0]));
                        let ah = r(&hi(&s[0]))?;
                        let t = self.tmp;
                        self.tmp += 1;
                        let (lo, upper) = match m {
                            "s_not_b64" => (format!("~{a}"), format!("~{ah}")),
                            "s_cmov_b64" => (format!("scc ? {a} : {}", r(&d)?), format!("scc ? {ah} : {}", r(&hi(&d))?)),
                            _ => (a, ah),
                        };
                        self.line(&format!("uint t{t}l = {lo}; uint t{t}h = {upper};"));
                        if m == "s_not_b64" { self.line(&format!("scc = ((t{t}l | t{t}h) != 0u);")); }
                        self.ws(&d, &format!("t{t}l"))?;
                        let dh = hi(&d);
                        self.touch(&dh);
                        self.ws(&dh, &format!("t{t}h"))
                    }
                    "s_not_b32" => {
                        self.ws(&d, &format!("(~{a})"))?;
                        self.line(&format!("scc = ({} != 0u) ? 1u : 0u;", name(&d)?));
                        Ok(())
                    }
                    "s_cmov_b32" => self.ws(&d, &format!("((scc != 0u) ? {a} : {})", name(&d)?)),
                    "s_brev_b32" => self.ws(&d, &format!("reversebits({a})")),
                    COUNT_LANES => {
                        let def = *self.vote_uses.get(&idx).ok_or("lane count without its predicate")?;
                        self.ws(&d, &format!("tb_votes{def}"))?;
                        self.line(&format!("scc = ({} != 0u) ? 1u : 0u;", name(&d)?));
                        Ok(())
                    }
                    "s_bcnt1_i32_b64" => {
                        self.touch(&hi(&s[0]));
                        // Population count of a 64-bit value — typically the number of active
                        // lanes (wave-level allocation). One invocation = one lane, so a lane
                        // mask counts as 0 or 1.
                        let ah = r(&hi(&s[0]))?;
                        self.ws(&d, &format!("(countbits({a}) + countbits({ah}))"))?;
                        self.line(&format!("scc = ({} != 0u) ? 1u : 0u;", name(&d)?));
                        Ok(())
                    }
                    "s_bcnt1_i32_b32" => {
                        self.ws(&d, &format!("countbits({a})"))?;
                        self.line(&format!("scc = ({} != 0u) ? 1u : 0u;", name(&d)?));
                        Ok(())
                    }
                    "s_flbit_i32_b32" => self.ws(&d, &format!("({a} == 0u ? ~0u : 31u - firstbithigh({a}))")),
                    "s_flbit_i32_b64" => {
                        self.touch(&hi(&s[0]));
                        let ah = r(&hi(&s[0]))?;
                        self.ws(&d, &flbit_b64_expr(&a, &ah))
                    }
                    "s_ff0_i32_b32" => self.ws(&d, &ff0_b32_expr(&a)),
                    "s_ff0_i32_b64" => {
                        self.touch(&hi(&s[0]));
                        let ah = r(&hi(&s[0]))?;
                        self.ws(&d, &ff0_b64_expr(&a, &ah))
                    }
                    "s_ff1_i32_b32" => self.ws(&d, &format!("firstbitlow({a})")),
                    // s_ff1_i32_b64 is firstbitlow over the 64-bit value: the low word
                    // decides when it is nonzero, otherwise the high word.
                    "s_ff1_i32_b64" => {
                        self.touch(&hi(&s[0]));
                        let ah = r(&hi(&s[0]))?;
                        self.ws(&d, &ff1_b64_expr(&a, &ah))
                    }
                    "s_sext_i32_i8" => self.ws(&d, &format!("asuint(asint({a} << 24) >> 24)")),
                    "s_sext_i32_i16" => self.ws(&d, &format!("asuint(asint({a} << 16) >> 16)")),
                    "s_abs_i32" => {
                        self.ws(&d, &format!("asuint(abs(asint({a})))"))?;
                        self.line(&format!("scc = ({} != 0u) ? 1u : 0u;", name(&d)?));
                        Ok(())
                    }
                    "s_getpc_b64" => {
                        // Address of the next instruction, relative to the program start.
                        self.ws(&d, &format!("{:#x}u", ins.offset as u32 + 4))?;
                        let dh = hi(&d);
                        self.touch(&dh);
                        self.ws(&dh, "0u")
                    }
                    "s_swappc_b64" if self.ex.stage == Stage::Vertex => {
                        // Vertex-fetch shader call: inputs are materialised from the D3D input layout.
                        self.line("// (fetch-shader call: vertex attributes come from the input layout)");
                        Ok(())
                    }
                    _ if m.contains("_saveexec_") || m.contains("_wrexec_") => {
                        let wide = m.ends_with("_b64");
                        let combine = |src: &str, exec: &str| -> Result<String, String> {
                            Ok(match m.split('_').nth(1) {
                                Some("and") => format!("{src} & {exec}"),
                                Some("or") => format!("{src} | {exec}"),
                                Some("xor") => format!("{src} ^ {exec}"),
                                Some("nand") => format!("~({src} & {exec})"),
                                Some("andn2") => format!("{src} & ~{exec}"),
                                Some("orn2") => format!("{src} | ~{exec}"),
                                Some("andn1") => format!("~{src} & {exec}"),
                                _ => return Err("unsupported saveexec".into()),
                            })
                        };
                        let t = self.tmp;
                        self.tmp += 1;
                        self.line(&format!("uint t{t}l = {};", combine(&a, "exec_lo")?));
                        if wide {
                            self.touch(&hi(&s[0]));
                            self.line(&format!("uint t{t}h = {};", combine(&r(&hi(&s[0]))?, "exec_hi")?));
                        }
                        let (lo, high) = if m.contains("_wrexec_") {
                            (format!("t{t}l"), format!("t{t}h"))
                        } else {
                            ("exec_lo".into(), "exec_hi".into())
                        };
                        self.ws(&d, &lo)?;
                        if wide {
                            self.ws(&hi(&d), &high)?;
                            self.line(&format!("exec_hi = t{t}h;"));
                        }
                        let mask = if wide { "(exec_lo | exec_hi)" } else { "exec_lo" };
                        self.line(&format!("exec_lo = t{t}l; scc = ({mask} != 0u);"));
                        Ok(())
                    }
                    // Return from a vertex shader chained in front of a geometry shader: the
                    // program ends here (it is followed directly by s_endpgm).
                    "s_setpc_b64" if self.ex.stage == Stage::Vertex && idx + 2 >= self.insns.len() => {
                        self.line("// (return to the chained stage)");
                        Ok(())
                    }
                    _ => Err("unsupported SOP1".into()),
                }
            }
            Enc::Sopk => {
                let d = d.ok_or("no dst")?;
                let imm = format!("{:#x}u", ins.aux.imm as i32 as u32);
                let mask = modeled_hwreg_mask(ins.raw[0]);
                match m {
                    "s_version" => Ok(()),
                    "s_movk_i32" => self.ws(&d, &imm),
                    "s_cmovk_i32" => self.ws(&d, &format!("((scc != 0u) ? {imm} : {})", name(&d)?)),
                    "s_setreg_imm32_b32" | "s_setreg_b32" | "s_getreg_b32"
                        if mask != 0 => {
                        let id = ins.raw[0] & 63;
                        let shift = (ins.raw[0] >> 6) & 31;
                        let width = ((ins.raw[0] >> 11) & 31) + 1;
                        if shift + width > 32 { return Err("hardware register field exceeds word".into()); }
                        // Reads must be known in full. Writes need only update the
                        // intersection with the modeled register.
                        if m == "s_getreg_b32" && mask != u32::MAX >> (32 - width) {
                            return Err("unmodeled MODE read field".into());
                        }
                        let reg = if id == 1 { "tb_fp_mode".to_string() } else { format!("tb_flat_scratch{}", id & 1) };
                        self.regs.insert(reg.clone());
                        if m == "s_getreg_b32" {
                            return self.ws(&d, &format!("(({reg} >> {shift}) & {mask:#x}u)"));
                        }
                        let value = if m == "s_setreg_imm32_b32" {
                            format!("{:#x}u", ins.raw.get(1).ok_or("missing hardware-register literal")?)
                        } else {
                            self.touch(&d);
                            r(&d)?
                        };
                        self.line(&format!("{reg} = ({reg} & ~{:#x}u) | (({value} & {mask:#x}u) << {shift});", mask << shift));
                        Ok(())
                    }
                    // Other hardware-register writes and counter waits do not change the modeled state.
                    "s_setreg_imm32_b32" | "s_setreg_b32" | "s_waitcnt_vscnt" | "s_waitcnt_vmcnt"
                    | "s_waitcnt_expcnt" | "s_waitcnt_lgkmcnt" => Ok(()),
                    "s_addk_i32" => {
                        let a = name(&d)?;
                        let sum = format!("t{}", self.tmp);
                        self.tmp += 1;
                        self.line(&format!("uint {sum} = {a} + {imm};"));
                        self.line(&format!("scc = {};", signed_add_overflow(&a, &imm, &sum)));
                        self.ws(&d, &sum)
                    }
                    "s_mulk_i32" => self.ws(&d, &format!("({} * {imm})", name(&d)?)),
                    _ if m.starts_with("s_cmpk_") => {
                        let fake = m.replace("s_cmpk_", "s_cmp_");
                        let c = self.int_compare(&fake, &d, &Opnd::ConstInt(ins.aux.imm))?;
                        self.line(&format!("scc = {c} ? 1u : 0u;"));
                        Ok(())
                    }
                    _ => Err("unsupported SOPK".into()),
                }
            }
            Enc::Sopc => {
                let c = match m {
                    "s_bitcmp0_b32" => format!("((({} >> ({} & 31u)) & 1u) == 0u)", r(&s[0])?, r(&s[1])?),
                    "s_bitcmp1_b32" => format!("((({} >> ({} & 31u)) & 1u) != 0u)", r(&s[0])?, r(&s[1])?),
                    "s_cmp_eq_u64" => format!(
                        "({} == {} && {} == {})",
                        r(&s[0])?,
                        r(&s[1])?,
                        r(&hi(&s[0]))?,
                        r(&hi(&s[1]))?
                    ),
                    "s_cmp_lg_u64" => format!(
                        "({} != {} || {} != {})",
                        r(&s[0])?,
                        r(&s[1])?,
                        r(&hi(&s[0]))?,
                        r(&hi(&s[1]))?
                    ),
                    _ => self.int_compare(m, &s[0], &s[1])?,
                };
                self.line(&format!("scc = {c} ? 1u : 0u;"));
                Ok(())
            }
            Enc::Smem if self.packed.is_some() => self.emit_packed_load(idx, ins),
            Enc::Smem => {
                if ins.op <= 4 {
                    if let Some(Res::HeapLoad(offset)) = self.res.get(&idx).and_then(|v| v.first()).copied() {
                        let value = ssrc_name(u32::from(offset))?;
                        self.regs.insert(value.clone());
                        let dst = d.ok_or("missing direct descriptor destination")?;
                        self.touch(&dst);
                        let byte_offset = if ins.aux.imm == 0 { value }
                            else { format!("({value} + {:#x}u)", ins.aux.imm as u32) };
                        self.ws(&dst, &format!("tb_heap_index({byte_offset} >> 5)"))?;
                        return Ok(());
                    }
                    if let Some(Res::HeapSamplerLoad(offset)) = self.res.get(&idx).and_then(|v| v.first()).copied() {
                        let value = ssrc_name(u32::from(offset))?;
                        self.regs.insert(value.clone());
                        let dst = d.ok_or("missing sampler descriptor destination")?;
                        self.touch(&dst);
                        // The guest sampler-heap pointer is the heap start (as for the static
                        // path), so the byte offset over 16 is the descriptor index.
                        self.ws(&dst, &format!("({value} >> 4)"))?;
                        return Ok(());
                    }
                    let chosen = !self.chosen_samplers.is_empty() && ins.op == 2
                        && matches!(ins.src.first(), Some(Opnd::Sgpr(base)) if self.sampler_heap_base == Some(*base));
                    if let Some(Opnd::Sgpr(dst)) = d.as_ref().filter(|_| chosen) {
                        for (k, value) in [ins.aux.imm as u32 / 16, 0, 0, 0].into_iter().enumerate() {
                            let dst = sreg_opnd(dst + k as u8);
                            self.touch(&dst);
                            self.ws(&dst, &format!("{value}u"))?;
                        }
                        return Ok(());
                    }
                    if let Some(Res::CodeTbl) = self.res.get(&idx).and_then(|v| v.first()) {
                        // Constant table embedded in the code section: byte address = pointer
                        // (relative to the start of the program) + immediate.
                        let n = 1u32 << ins.op;
                        let base = sreg_num(&d.ok_or("no dst")?).ok_or("bad dst")?;
                        let ndw = ((self.ex.isa_data.len().saturating_sub(self.ex.isa.len())) / 4).max(1);
                        let t = self.tmp;
                        self.tmp += 1;
                        if let Some(offset) = self.static_smem_offset(ins) {
                            self.line(&format!("uint t{t} = ({} + {offset:#x}u) >> 2;", r(&s[0])?));
                            for k in 0..n {
                                self.line(&format!("uint t{t}_{k} = tb_code[(t{t} + {k}u - {}u) % {ndw}u];", self.ex.isa.len() / 4));
                            }
                        } else {
                            let reg = self.dynamic_code_offset(idx).ok_or("unverified dynamic code-table offset")?;
                            let offset = Opnd::Sgpr(reg);
                            self.touch(&offset);
                            self.line(&format!("uint t{t} = ({} + {} + {:#x}u - {}u) >> 2;", r(&s[0])?, r(&offset)?, ins.aux.imm, self.ex.isa.len()));
                            for k in 0..n {
                                self.line(&format!("uint t{t}_{k} = tb_code[t{t} + {k}u];"));
                            }
                        }
                        for k in 0..n {
                            let dst = sreg_opnd(base + k as u8);
                            self.touch(&dst);
                            self.ws(&dst, &format!("t{t}_{k}"))?;
                        }
                        return Ok(());
                    }
                    self.line("// (descriptor fetch from root table — resolved statically)");
                    return Ok(());
                }
                let n = 1u32 << (ins.op - 8);
                let base = sreg_num(&d.ok_or("no dst")?).ok_or("bad dst")?;
                let soff = ins.raw.get(1).copied().unwrap_or(0) >> 25;
                // SOFFSET is not an operand of the decoded instruction. VCC and the other
                // special registers are always declared.
                if soff < 106 {
                    self.regs.insert(ssrc_name(soff)?);
                }
                let slot = match self.res.get(&idx).and_then(|v| v.first()).copied() {
                    Some(Res::HeapBuffer(src)) if self.compute_heap => {
                        // A compute heap descriptor names a raw SRV (geometry and instance
                        // data read by both scalar and vector loads), not a CBV.
                        let t = self.tmp;
                        self.tmp += 1;
                        let offset = if soff != 0x7d { format!(" + {}", ssrc_name(soff)?) } else { String::new() };
                        let buf = self.bind_heap(HeapView::Raw, src);
                        self.line(&format!("uint tb_o{t} = ({}u{offset}) & ~3u;", ins.aux.imm as u32));
                        for k in 0..n {
                            self.line(&format!("uint t{t}_{k} = {buf}.Load(tb_o{t} + {}u);", 4 * k));
                        }
                        for k in 0..n {
                            let dst = sreg_opnd(base + k as u8);
                            self.touch(&dst);
                            self.ws(&dst, &format!("t{t}_{k}"))?;
                        }
                        return Ok(());
                    }
                    Some(Res::HeapBuffer(src)) => {
                        let t = self.tmp;
                        self.tmp += 1;
                        let offset = if soff != 0x7d { format!(" + {}", ssrc_name(soff)?) } else { String::new() };
                        let buf = self.bind_heap(HeapView::Constants, src);
                        self.line(&format!("uint tb_o{t} = ({}u{offset}) >> 2;", ins.aux.imm as u32));
                        for k in 0..n {
                            self.line(&format!("uint t{t}_{k} = {buf}.data[(tb_o{t}+{k}u) >> 2][(tb_o{t}+{k}u)&3u];"));
                        }
                        for k in 0..n {
                            let dst = sreg_opnd(base + k as u8);
                            self.touch(&dst);
                            self.ws(&dst, &format!("t{t}_{k}"))?;
                        }
                        return Ok(());
                    }
                    Some(Res::Cb(s)) => s,
                    Some(res @ (Res::Buf(slot, is_uav) | Res::Raw(slot, is_uav))) => {
                        // Snapshot all words before replacing any overlapping destination SGPR.
                        let raw = matches!(res, Res::Raw(..));
                        let nm = if is_uav { "ubuf" } else { "sbuf" };
                        let dynamic = if soff != 0x7d { format!(" + {}", ssrc_name(soff)?) } else { String::new() };
                        let t = self.tmp;
                        self.tmp += 1;
                        self.line(&format!("uint t{t} = {:#x}u{dynamic};", ins.aux.imm as u32));
                        for k in 0..n {
                            let load = if raw {
                                format!("raw_{nm}{slot}.Load(t{t} + {}u)", 4 * k)
                            } else {
                                format!("tb_sbw_{nm}{slot}(t{t} + {}u)", 4 * k)
                            };
                            self.line(&format!("uint t{t}_{k} = {load};"));
                        }
                        for k in 0..n {
                            let dst = sreg_opnd(base + k as u8);
                            self.touch(&dst);
                            self.ws(&dst, &format!("t{t}_{k}"))?;
                        }
                        return Ok(());
                    }
                    _ => return Err("unresolved constant buffer".into()),
                };
                if soff != 0x7d {
                    // Dynamic byte offset = imm + SGPR[soffset].
                    let so = ssrc_name(soff)?;
                    let t = self.tmp;
                    self.tmp += 1;
                    self.line(&format!("uint t{t} = ({:#x}u + {so}) >> 2;", ins.aux.imm as u32));
                    // Read all words before writing (the offset register may be a destination).
                    for k in 0..n {
                        self.line(&format!("uint t{t}_{k} = cb{slot}[((t{t} + {k}u) >> 2) & 4095u][(t{t} + {k}u) & 3u];"));
                    }
                    for k in 0..n {
                        let dst = sreg_opnd(base + k as u8);
                        self.touch(&dst);
                        self.ws(&dst, &format!("t{t}_{k}"))?;
                    }
                    return Ok(());
                }
                let w0 = ins.aux.imm as u32 / 4;
                for k in 0..n {
                    let w = w0 + k;
                    let dst = sreg_opnd(base + k as u8);
                    self.touch(&dst);
                    self.ws(&dst, &format!("cb{slot}[{}].{}", w / 4, XYZW[(w % 4) as usize]))?;
                }
                Ok(())
            }
            Enc::Mimg => self.emit_mimg(idx, ins),
            Enc::Mtbuf => self.emit_mtbuf(idx, ins),
            Enc::Mubuf if ins.op == 0x71 || ins.op == 0x72 => {
                // D3D12 exposes UAV memory visibility, not individual cache levels.
                // A device fence has no thread rendezvous; guest S_BARRIER is separate.
                if self.is_cs {
                    self.line("DeviceMemoryBarrier();");
                }
                Ok(())
            }
            Enc::Mubuf if self.packed.is_some() => self.emit_packed_buffer(idx, ins),
            Enc::Mubuf => self.emit_mubuf(idx, ins),
            Enc::Flat if m == "flat_load_ushort" => {
                code_table::halfword(&self.insns, idx, self.ex.isa.len(), self.ex.isa_data.len())
                    .ok_or("halfword load does not address a proven PC-relative table")?;
                self.uses_code = true;
                let t = self.tmp;
                self.tmp += 1;
                let addr = r(&s[0])?;
                self.line(&format!("uint t{t} = {addr} - {}u;", self.ex.isa.len()));
                self.wv(&d.ok_or("bad halfword destination")?, &format!("(tb_code[t{t} >> 2] >> ((t{t} & 2u) * 8u)) & 0xffffu"))
            }
            Enc::Flat if m.starts_with("flat_load_dword") => {
                // Per-lane load through a 64-bit address. Shaders use this for lookups in
                // constant tables embedded in their own code section (address derived from
                // `s_getpc_b64`), so it is only accepted when the program takes its PC.
                if ins.aux.mask == 1 {
                    return Err("scratch-memory load unsupported".into());
                }
                if !self.insns.iter().any(|i| i.mnemonic == "s_getpc_b64") {
                    return Err("global/flat load in a shader that never takes its PC".into());
                }
                self.uses_code = true;
                let n: u8 = match m {
                    "flat_load_dword" => 1,
                    "flat_load_dwordx2" => 2,
                    "flat_load_dwordx3" => 3,
                    _ => 4,
                };
                let extra = match &s[2] {
                    Opnd::Null => String::new(),
                    o => format!(" + {}", r(o)?),
                };
                let ndw = ((self.ex.isa_data.len().saturating_sub(self.ex.isa.len())) / 4).max(1);
                let vd = match d {
                    Some(Opnd::Vgpr(v)) => v,
                    _ => return Err("bad vdst".into()),
                };
                let t = self.tmp;
                self.tmp += 1;
                self.line(&format!("uint t{t} = ({} + {:#x}u{extra}) >> 2;", r(&s[0])?, ins.aux.imm as i32 as u32));
                for k in 0..n {
                    self.line(&format!("uint t{t}_{k} = tb_code[(t{t} + {k}u - {}u) % {ndw}u];", self.ex.isa.len() / 4));
                }
                for k in 0..n {
                    self.wv(&Opnd::Vgpr(vd + k), &format!("t{t}_{k}"))?;
                }
                Ok(())
            }
            Enc::Ds
                if !self.is_gs
                    && !self.is_cs
                    && self.ex.stage == Stage::Vertex
                    && matches!(m, "ds_write_b32" | "ds_write2_b32" | "ds_write_b64" | "ds_write2_b64") =>
            {
                // A vertex shader chained in front of a geometry shader hands its outputs over
                // through LDS instead of parameter exports: the vertex record is the output
                // registers back to back, 4 dwords each (observed in a55400ccab602283: dwords
                // 0..3 = TEXCOORD6.xyzw, 4..6 = TEXCOORD8.xyz, matching the output signature).
                // The address register is this vertex's record base and is not needed.
                let off = ins.aux.imm as u32;
                let vg = |o: &Opnd, k: u8| match o {
                    Opnd::Vgpr(n) => Ok(Opnd::Vgpr(n + k)),
                    _ => Err("LDS output write with a non-VGPR source".to_string()),
                };
                // (byte offset, source register) per written dword.
                let writes: Vec<(u32, Opnd)> = match m {
                    "ds_write_b32" => vec![(off & 0xffff, s[1])],
                    "ds_write2_b32" => vec![((off & 0xff) * 4, s[1]), (((off >> 8) & 0xff) * 4, s[2])],
                    "ds_write_b64" => vec![(off & 0xffff, s[1]), ((off & 0xffff) + 4, vg(&s[1], 1)?)],
                    _ => vec![
                        ((off & 0xff) * 8, s[1]),
                        ((off & 0xff) * 8 + 4, vg(&s[1], 1)?),
                        (((off >> 8) & 0xff) * 8, s[2]),
                        (((off >> 8) & 0xff) * 8 + 4, vg(&s[2], 1)?),
                    ],
                };
                let osig = self.ex.output_sig.clone();
                for (byte, src) in writes {
                    let dw = (byte / 4) as usize;
                    // The record is indexed by output *register* (4 dwords each), system
                    // values included: in 008bccdcf9dcecf9 register 1 is SV_Position and is
                    // written at dwords 4..7.
                    let reg = (dw / 4) as u32;
                    let comp = dw % 4;
                    self.touch(&src);
                    let mut hit = false;
                    for (ei, e) in osig.iter().enumerate() {
                        if e.register == reg && (e.mask >> comp) & 1 != 0 {
                            let idx = (0..comp).filter(|c| (e.mask >> c) & 1 != 0).count();
                            let (ty, n) = sig_type(e);
                            let field = if n == 1 { format!("o.o{ei}") } else { format!("o.o{ei}.{}", XYZW[idx]) };
                            let val = if ty.starts_with("float") { f(&src)? } else if ty.starts_with("int") { format!("asint({})", r(&src)?) } else { r(&src)? };
                            self.line(&format!("if (exec_lo & 1u) {field} = {val};"));
                            hit = true;
                        }
                    }
                    if !hit {
                        self.line(&format!("// (LDS output dword {dw} has no output-signature component: ignored)"));
                    }
                }
                Ok(())
            }
            Enc::Ds if self.is_gs && matches!(m, "ds_read_b32" | "ds_read2_b32" | "ds_write_b32" | "ds_write2_b32") => {
                self.emit_gs_lds(ins)
            }
            Enc::Ds if self.is_cs && matches!(m, "ds_read_b64" | "ds_read2_b64" | "ds_read2st64_b64" | "ds_write_b128" | "ds_write_b64" | "ds_write2_b64") => {
                if ins.raw[0] & (1 << 17) != 0 { return Err("GDS is unsupported".into()); }
                self.uses_lds = true;
                let t = self.tmp;
                self.tmp += 1;
                // Snapshot the address before any destination word can overwrite its VGPR.
                self.line(&format!("uint t{t}addr = {};", r(&s[0])?));
                let offset = ins.aux.imm as u32;
                let read = m.starts_with("ds_read");
                let paired = matches!(m, "ds_read2_b64" | "ds_read2st64_b64" | "ds_write2_b64");
                let stride = if m == "ds_read2st64_b64" { 512 } else { 8 };
                let count = if matches!(m, "ds_read_b64" | "ds_write_b64") { 2 } else { 4 };
                for word in 0..count {
                    let (base, k) = if read {
                        (ins.dst[0], word)
                    } else if paired {
                        (s[1 + (word / 2) as usize], word % 2)
                    } else {
                        (s[1], word)
                    };
                    let Opnd::Vgpr(v) = base else { return Err("expected consecutive LDS VGPRs".into()) };
                    let reg = Opnd::Vgpr(v.checked_add(k).ok_or("LDS register span exceeds v255")?);
                    let byte = if paired {
                        ((offset >> (8 * (word / 2))) & 255) * stride + u32::from(word % 2) * 4
                    } else {
                        offset + u32::from(word) * 4
                    };
                    let location = format!("tb_lds[((t{t}addr + {byte}u) >> 2) & 8191u]");
                    if read {
                        self.wv(&reg, &location)?;
                    } else {
                        self.touch(&reg);
                        self.line(&format!("if (exec_lo & 1u) {location} = {};", r(&reg)?));
                    }
                }
                Ok(())
            }
            Enc::Ds
                if self.is_cs
                    && matches!(
                        m,
                        "ds_read_b32" | "ds_read2_b32" | "ds_read2st64_b32" | "ds_write_b32" | "ds_write2_b32" | "ds_write2st64_b32"
                    ) =>
            {
                // Compute: LDS is the thread group's shared memory → an HLSL `groupshared`
                // array addressed by byte address / 4 (32 KiB, the D3D groupshared limit).
                self.uses_lds = true;
                let off = ins.aux.imm as u32;
                // Two-address forms scale their 8-bit offsets by 4 bytes ("2") or by
                // 64 dwords = 256 bytes ("2st64"); express both in units of 4 bytes.
                let scale = if m.contains("st64") { 64 } else { 1 };
                let (off0, off1) = ((off & 0xff) * scale, ((off >> 8) & 0xff) * scale);
                let addr = r(&s[0])?;
                let at = |o: u32| format!("tb_lds[(({addr} + {o:#x}u) >> 2) & 8191u]");
                let vd = match d {
                    Some(Opnd::Vgpr(v)) => v,
                    _ => return Err("bad vdst".into()),
                };
                let t = self.tmp;
                self.tmp += 1;
                match m {
                    "ds_read_b32" => {
                        if self.wave_cull && matches!(self.ex.entry.as_deref(), Some("CullPerPageDrawCommandsCs" | "VirtualSmInvalidateInstancePagesCS" | "GenerateReflectionRaysCS")) {
                            self.line(&format!("uint t{t} = 0u; if (exec_lo & 1u) t{t} = {};", at(off & 0xffff)));
                        } else {
                            self.line(&format!("uint t{t} = {};", at(off & 0xffff)));
                        }
                        self.wv(&Opnd::Vgpr(vd), &format!("t{t}"))
                    }
                    "ds_read2_b32" | "ds_read2st64_b32" => {
                        if self.wave_cull && matches!(self.ex.entry.as_deref(), Some("CullPerPageDrawCommandsCs" | "VirtualSmInvalidateInstancePagesCS" | "GenerateReflectionRaysCS")) {
                            self.line(&format!("uint t{t}a = 0u, t{t}b = 0u; if (exec_lo & 1u) {{ t{t}a = {}; t{t}b = {}; }}", at(off0 * 4), at(off1 * 4)));
                        } else {
                            self.line(&format!("uint t{t}a = {}; uint t{t}b = {};", at(off0 * 4), at(off1 * 4)));
                        }
                        self.wv(&Opnd::Vgpr(vd), &format!("t{t}a"))?;
                        self.wv(&Opnd::Vgpr(vd + 1), &format!("t{t}b"))
                    }
                    "ds_write_b32" => {
                        self.line(&format!("if (exec_lo & 1u) {} = {};", at(off & 0xffff), r(&s[1])?));
                        Ok(())
                    }
                    _ => {
                        self.line(&format!(
                            "if (exec_lo & 1u) {{ {} = {}; {} = {}; }}",
                            at(off0 * 4),
                            r(&s[1])?,
                            at(off1 * 4),
                            r(&s[2])?
                        ));
                        Ok(())
                    }
                }
            }
            Enc::Ds if self.is_cs && matches!(m, "ds_write_b16" | "ds_write_b16_d16_hi" | "ds_write_b8" | "ds_read_u16" | "ds_read_i16" | "ds_read_u8" | "ds_read_i8" | "ds_read_u16_d16" | "ds_read_u16_d16_hi") => {
                // Sub-word accesses to group-shared memory: tb_lds is addressed in dwords, so
                // read-modify-write the containing dword.
                self.uses_lds = true;
                let off = ins.aux.imm as u32 & 0xffff;
                let bits = if m.contains("16") { 16u32 } else { 8 };
                let mask = (1u32 << bits) - 1;
                let t = self.tmp;
                self.tmp += 1;
                self.line(&format!("uint t{t}a = ({}) + {off:#x}u; uint t{t}s = (t{t}a & 3u) * 8u;", r(&s[0])?));
                let word = format!("tb_lds[(t{t}a >> 2) & 8191u]");
                if m.starts_with("ds_write") {
                    let mut data = r(&s[1])?;
                    if m == "ds_write_b16_d16_hi" {
                        if ins.raw[0] & (1 << 17) != 0 {
                            return Err("unsupported global data store halfword write".into());
                        }
                        data = format!("({data} >> 16)");
                    }
                    // Adjacent lanes can target different halves of the same word.
                    // A plain read-modify-write would lose one lane's update.
                    self.line(&format!("if (exec_lo & 1u) {{ uint t{t}old; InterlockedOr({word}, 0u, t{t}old);"));
                    self.line(&format!("uint t{t}seen; [allow_uav_condition] do {{ t{t}seen = t{t}old;"));
                    self.line(&format!("uint t{t}new = (t{t}seen & ~({mask:#x}u << t{t}s)) | ((({data}) & {mask:#x}u) << t{t}s);"));
                    self.line(&format!("InterlockedCompareExchange({word}, t{t}seen, t{t}new, t{t}old); }} while (t{t}old != t{t}seen); }}"));
                    return Ok(());
                }
                let v = format!("(({word} >> t{t}s) & {mask:#x}u)");
                let dst = d.ok_or("no dst")?;
                let e = match m {
                    "ds_read_u16_d16" => format!("(({} & 0xffff0000u) | {v})", r(&dst)?),
                    "ds_read_u16_d16_hi" => format!("(({} & 0xffffu) | ({v} << 16))", r(&dst)?),
                    _ if m.contains("_i") => format!("asuint(asint({v} << {}) >> {})", 32 - bits, 32 - bits),
                    _ => v,
                };
                return self.wv(&dst, &e);
            }
            Enc::Ds
                if self.is_cs
                    && (m == "ds_wrxchg_rtn_b32" || matches!(
                        m.replace("_rtn", "").as_str(),
                        "ds_add_u32" | "ds_min_u32" | "ds_max_u32" | "ds_and_b32" | "ds_or_b32" | "ds_xor_b32"
                    )) =>
            {
                // Atomic read-modify-write on group-shared memory (the `_rtn` forms also
                // return the old value) → the matching HLSL Interlocked* intrinsic.
                self.uses_lds = true;
                if m == "ds_wrxchg_rtn_b32" && ins.raw[0] & (1 << 17) != 0 {
                    return Err("global data store exchange unsupported".into());
                }
                if m == "ds_wrxchg_rtn_b32" && self.insns.iter().any(|op|
                    op.enc == Enc::Sopp && Self::branch_cond(op.op).is_some()) {
                    return Err("LDS exchange with unverified branch convergence".into());
                }
                let func = match m.replace("_rtn", "").as_str() {
                    "ds_add_u32" => "InterlockedAdd",
                    "ds_min_u32" => "InterlockedMin",
                    "ds_max_u32" => "InterlockedMax",
                    "ds_and_b32" => "InterlockedAnd",
                    "ds_or_b32" => "InterlockedOr",
                    "ds_wrxchg_b32" => "InterlockedExchange",
                    _ => "InterlockedXor",
                };
                let off = ins.aux.imm as u32 & 0xffff;
                let addr = r(&s[0])?;
                let t = self.tmp;
                self.tmp += 1;
                self.line(&format!(
                    "uint t{t} = 0u; if (exec_lo & 1u) {{ {func}(tb_lds[(({addr} + {off:#x}u) >> 2) & 8191u], {}, t{t}); }}",
                    r(&s[1])?
                ));
                if m.contains("_rtn") {
                    self.wv(&d.ok_or("no dst")?, &format!("t{t}"))?;
                }
                Ok(())
            }
            Enc::Ds if m == "ds_bpermute_b32" => {
                if self.required_wave_size != Some(32) || ins.raw[0] & 0x3ff00 != 0 {
                    return Err("backward permute requires a verified wave32 compute layout".into());
                }
                let t = self.tmp;
                self.tmp += 1;
                let (addr, data) = (r(&s[0])?, r(&s[1])?);
                self.line(&format!("uint t{t} = WaveReadLaneAt(((exec_lo & 1u) != 0u) ? {data} : 0u, (({addr} + {}u) >> 2) & 31u);", ins.aux.imm));
                self.wv(&d.ok_or("no permute destination")?, &format!("t{t}"))
            }
            Enc::Ds if m == "ds_swizzle_b32" => {
                if self.resolve_wave32 {
                    let off = ins.aux.imm as u32;
                    if ins.raw[0] & (1 << 17) != 0 || (off & 0x8000 != 0 && off & 0xff00 != 0x8000) {
                        return Err("unsupported resolve wave swizzle mode".into());
                    }
                    let lane = "WaveGetLaneIndex()";
                    let partner = if off & 0x8000 != 0 {
                        format!("({lane} & ~3u) | (({:#x}u >> (2u * ({lane} & 3u))) & 3u)", off & 255)
                    } else {
                        format!("(({lane} & {}u) | {}u) ^ {}u", off & 31, (off >> 5) & 31, (off >> 10) & 31)
                    };
                    let source = r(&s[0])?;
                    let t = self.tmp;
                    self.tmp += 1;
                    self.line(&format!("uint t{t} = WaveReadLaneAt((exec_lo & tb_lane_bit) != 0u ? {source} : 0u, {partner});"));
                    return self.wv(&d.ok_or("no swizzle destination")?, &format!("t{t}"));
                }
                if self.is_cs && self.packed.as_ref().is_some_and(|p| p.scan_width.is_some()) {
                    let off = ins.aux.imm as u32;
                    if ins.raw[0] & (1 << 17) != 0 || off & 0x8000 != 0 { return Err("unsupported compute swizzle mode".into()); }
                    let t = self.tmp;
                    self.tmp += 1;
                    // Bitmask mode rearranges lanes within each independent group of 32.
                    self.line(&format!("uint t{t}lane = WaveGetLaneIndex();"));
                    self.line(&format!("uint t{t}src = (t{t}lane & ~31u) | (((t{t}lane & {}u) | {}u) ^ {}u);",
                        off & 31, (off >> 5) & 31, (off >> 10) & 31));
                    self.line(&format!("uint t{t}value = WaveReadLaneAt({}, t{t}src);", r(&s[0])?));
                    return self.wv(&d.ok_or("no swizzle destination")?, &format!("t{t}value"));
                }
                if self.compute_swizzle {
                    if self.hoisted_swizzle != Some(idx) { self.emit_compute_swizzle(idx, ins, &[])?; }
                    return self.wv(&d.ok_or("no swizzle destination")?, &format!("swizzle{idx}"));
                }
                if !is_ps { return Err("ds_swizzle outside a pixel shader".into()); }
                let off = ins.aux.imm as u32;
                if off & 0x8000 == 0 { return Err("ds_swizzle without QDMode unsupported".into()); }
                self.uses_quad = true;
                let src = r(&s[0])?;
                let t = self.tmp;
                self.tmp += 1;
                let dst = d.ok_or("no dst")?;
                let pattern = off & 255;
                let lane = pattern & 3;
                self.line("#if __SHADER_TARGET_MAJOR >= 6");
                // Keep reads outside the emulated EXEC predicate so helper lanes participate.
                if pattern == lane * 0x55 {
                    self.line(&format!("uint t{t}quad = QuadReadLaneAt({src}, {lane});"));
                    self.wv(&dst, &format!("t{t}quad"))?;
                } else {
                    self.line(&format!("uint4 t{t}quad = uint4(QuadReadLaneAt({src}, 0), QuadReadLaneAt({src}, 1), QuadReadLaneAt({src}, 2), QuadReadLaneAt({src}, 3));"));
                    self.wv(&dst, &format!("t{t}quad[({pattern:#x}u >> (tb_quad * 2u)) & 3u]"))?;
                }
                self.line("#else");
                self.line(&format!("float t{t} = asfloat({src}); uint t{t}s = ({:#x}u >> (tb_quad * 2u)) & 3u;", off & 255));
                self.wv(&dst, &format!("asuint(t{t} + ((float)(t{t}s & 1u) - (float)(tb_quad & 1u)) * ddx_fine(t{t}) + ((float)(t{t}s >> 1) - (float)(tb_quad >> 1)) * ddy_fine(t{t}))"))?;
                self.line("#endif");
                Ok(())
            }
            Enc::Vinterp => {
                if !is_ps {
                    return Err("v_interp outside a pixel shader".into());
                }
                let attr = ins.aux.imm as usize;
                let chan = ins.aux.imm2 as usize;
                let Param::Full(reg) = *self.ps_attrs.get(attr).ok_or_else(|| format!("PS attr{attr} has no signature register"))? else {
                    return Err(format!("32-bit read of half-float PS attr{attr}"));
                };
                let val = if let Some((ei, k)) = sig_lookup(&self.ex.input_sig, reg, chan) {
                    let acc = format!("i.i{ei}.{}", XYZW[k]);
                    if matches!(self.ex.input_sig[ei].comp_type, 3 | 6) { format!("asuint({acc})") } else { format!("(uint)({acc})") }
                } else if chan == 3 {
                    "0x3f800000u".to_string()
                } else {
                    "0u".to_string()
                };
                if self.pulled_attrs.contains(&attr) {
                    if let Some((ei, k)) = sig_lookup(&self.ex.input_sig, reg, chan) {
                        let (_, n) = sig_type(&self.ex.input_sig[ei]);
                        let field = if n == 1 { format!("i.i{ei}") } else { format!("i.i{ei}.{}", XYZW[k]) };
                        let vertex = |index| {
                            let bits = format!("GetAttributeAtVertex({field}, {index})");
                            if matches!(self.ex.input_sig[ei].comp_type, 3 | 6) { bits } else { format!("asfloat({bits})") }
                        };
                        let p0 = vertex(0);
                        let delta = |v| format!("({} - {p0})", vertex(v));
                        let dst = d.ok_or("no interpolation destination")?;
                        let expr = match ins.op {
                            0 => format!("{p0} + {} * {}", f(&s[0])?, delta(1)),
                            1 => format!("{} + {} * {}", f(&dst)?, f(&s[0])?, delta(2)),
                            2 => match ins.raw[0] & 255 {
                                0 => delta(1), 1 => delta(2), 2 => p0, _ => return Err("invalid interpolation parameter".into()),
                            },
                            _ => return Err("unsupported interpolation opcode".into()),
                        };
                        self.line("#if __SHADER_TARGET_MAJOR >= 6");
                        self.wv(&dst, &format!("asuint({expr})"))?;
                        self.line("#else");
                        self.wv(&dst, &val)?;
                        self.line("#endif");
                        return Ok(());
                    }
                    // A missing channel is constant at all three vertices.
                    let dst = d.ok_or("no interpolation destination")?;
                    let constant = match ins.op {
                        1 => return Ok(()), // the second delta is zero
                        2 if matches!(ins.raw[0] & 255, 0 | 1) => "0u",
                        _ => &val,
                    };
                    return self.wv(&dst, constant);
                }
                // p1 computes a partial term in hardware; the full interpolated value is
                // already available per-pixel here, so both steps assign it.
                self.wv(&d.ok_or("no dst")?, &val)
            }
            Enc::Exp => self.emit_exp(ins, is_ps),
            _ => Err("unsupported encoding".into()),
        }
    }
}
