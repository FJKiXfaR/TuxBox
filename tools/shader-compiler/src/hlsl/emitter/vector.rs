//! Vector arithmetic.
use super::*;

impl<'a> Emitter<'a> {
    /// Decode the packed SRC0 attribute descriptor of a Vop3 f16-interpolation
    /// instruction: (attr, channel, high).
    pub(in crate::hlsl) fn f16_interp_descriptor(second: u32) -> (usize, usize, bool) {
        ((second & 0x3f) as usize, ((second >> 6) & 3) as usize, (second >> 8) & 1 != 0)
    }

    /// Emit a Vop3-encoded f16-interpolation instruction.
    ///
    /// SRC0 in the second word packs the attribute descriptor: attr (6 bits),
    /// channel (2 bits), and the `high` bit (1 bit). SRC1 is the barycentric
    /// weight; for p2_f16, SRC2 is the p1ll's f32 partial register. An attribute
    /// packed in a 32-bit word holds two 16-bit halves, so each vertex's raw
    /// word is decoded per vertex and the halves are interpolated (mirroring
    /// the pulled-attribute path), which is also what Mesa ACO does.
    pub(in crate::hlsl) fn emit_f16_interp(&mut self, ins: &Insn) -> Result<(), String> {
        if self.ex.stage != Stage::Pixel {
            return Err("v_interp outside a pixel shader".into());
        }
        if ins.enc != Enc::Vop3 || ins.raw.len() < 2 {
            return Err("invalid f16 interpolation encoding".into());
        }
        let m = ins.mnemonic.as_str();
        let is_p2 = m == "v_interp_p2_f16";
        let (attr, chan, high) = Self::f16_interp_descriptor(ins.raw[1]);
        let clamp = (ins.raw[0] >> 15) & 1 != 0;
        let param = self.ps_attrs.get(attr).ok_or_else(|| format!("PS attr{attr} has no signature register"))?;
        // A packed half-float component arrives as the float it stands for.
        let packed = param.component(chan, high);
        let (reg, chan) = match (param, packed) {
            (Param::Full(reg), _) => (*reg, chan),
            (_, Some(found)) => found,
            _ => return Err(format!("PS attr{attr} has no signature register")),
        };
        let (ei, k) = sig_lookup(&self.ex.input_sig, reg, chan)
            .ok_or_else(|| format!("PS attr{attr} has no signature register"))?;
        if matches!(self.ex.input_sig[ei].comp_type, 1 | 2) {
            return Err("v_interp f16 with a typed signature element unsupported".into());
        }
        let (_, n) = sig_type(&self.ex.input_sig[ei]);
        let field = if n == 1 { format!("i.i{ei}") } else { format!("i.i{ei}.{}", XYZW[k]) };
        let raw = |index: usize| -> String {
            format!("asuint(GetAttributeAtVertex({field}, {index}))")
        };
        let half = |index: usize| -> String {
            let h = raw(index);
            if high { format!("(({h} >> 16u) & 0xffffu)") } else { format!("({h} & 0xffffu)") }
        };
        let vertex = |index: usize| {
            if packed.is_some() { format!("GetAttributeAtVertex({field}, {index})") } else { format!("f16tof32({})", half(index)) }
        };
        let (p0, p1, p2v) = (vertex(0), vertex(1), vertex(2));
        if packed.is_some() && !self.pulled_attrs.contains(&attr) {
            let dst = ins.dst.first().cloned().ok_or("no dst")?;
            if !is_p2 {
                return self.wv(&dst, &format!("asuint({field})"));
            }
            let value = if clamp { format!("saturate({field})") } else { field.clone() };
            let keep = r(&dst)?;
            self.touch(&dst);
            return self.wv(&dst, &format!("(({keep} & 0xffff0000u) | tb_f16_rne({value}))"));
        }
        let dst = ins.dst.first().cloned().ok_or("no dst")?;
        let name_d = name(&dst)?;
        // SM5 fallback: the raw smooth read (best-effort; the per-vertex path is SM6+).
        let fallback = format!("asuint({field})");
        if !is_p2 {
            // p1ll/p1lv: the f32 partial P0 + I*(P1-P0), consumed by p2_f16.
            let i = f(&ins.src[1])?;
            let expr = format!("{p0} + {i} * ({p1} - {p0})");
            self.line("#if __SHADER_TARGET_MAJOR >= 6");
            self.wv(&dst, &format!("asuint({expr})"))?;
            self.line("#else");
            self.wv(&dst, &fallback)?;
            self.line("#endif");
            Ok(())
        } else {
            if ins.src.len() < 3 {
                return Err("missing p2_f16 partial register".into());
            }
            let j = f(&ins.src[1])?;
            let partial = r(&ins.src[2])?;
            let full = format!("(asfloat({partial}) + {j} * ({p2v} - {p0}))");
            let full = if clamp { format!("saturate({full})") } else { full };
            let bits = format!("tb_f16_rne({full})");
            self.line("#if __SHADER_TARGET_MAJOR >= 6");
            self.wv(&dst, &format!("(({name_d} & 0xffff0000u) | {bits})"))?;
            self.line("#else");
            self.wv(&dst, &fallback)?;
            self.line("#endif");
            Ok(())
        }
    }

    pub(in crate::hlsl) fn emit_vec(&mut self, ins: &Insn) -> Result<(), String> {
        let mut m = ins.mnemonic.clone();
        let s = &ins.src;
        let d = ins.dst.first().cloned().ok_or("no dst")?;
        if self.required_wave_size == Some(64) && self.packed.as_ref().is_some_and(|p| p.lane_masks.contains(&self.cur_idx)) {
            self.touch(&hi(&s[2]));
        }
        if m == "tb_reflection_prefix32" {
            return self.wv(&d, &format!("WavePrefixSum({})", r(&s[0])?));
        }
        if m == "v_nop" {
            return Ok(());
        }
        if matches!(m.as_str(), "v_interp_p1ll_f16" | "v_interp_p1lv_f16" | "v_interp_p2_f16") {
            return self.emit_f16_interp(ins);
        }
        if ins.enc == Enc::Vop2 && ins.op == wave_sum::DPP8_EXCHANGE {
            if !self.portable_wave64 { return Err("DPP8 exchange without convergent group".into()); }
            let a = r(&s[0])?;
            let b = r(&s[1])?;
            let selector = ins.aux.imm as u32;
            let t = self.tmp;
            self.tmp += 1;
            self.line(&format!("uint t{t} = tb_reduction_read(gtid.x, {a}, (gtid.x & ~7u) | (({selector:#x}u >> (3u * (gtid.x & 7u))) & 7u));"));
            let value = match m.as_str() {
                "v_add_nc_u32" => format!("t{t} + {b}"),
                "v_and_b32" => format!("t{t} & {b}"),
                "v_or_b32" => format!("t{t} | {b}"),
                "v_xor_b32" => format!("t{t} ^ {b}"),
                "v_min_u32" => format!("min(t{t}, {b})"),
                "v_max_u32" => format!("max(t{t}, {b})"),
                "v_min_i32" => format!("asuint(min(asint(t{t}), asint({b})))"),
                "v_max_i32" => format!("asuint(max(asint(t{t}), asint({b})))"),
                _ => return Err("unsupported DPP8 exchange operation".into()),
            };
            return self.wv(&d, &value);
        }
        if ins.enc == Enc::Vop2 && ins.op == wave_sum::DPP_REDUCE {
            // Read the partner before the EXEC-guarded write, including when VDST aliases VSRC.
            let t = self.tmp;
            self.tmp += 1;
            let acc = r(&s[0])?;
            if self.pixel_wave {
                // The other lanes hold the identity, so each step keeps this one's value.
                self.pixel_sum = m.starts_with("v_add");
                return self.wv(&d, &acc);
            }
            let id = if self.portable_wave64 { "gtid.x" } else { "WaveGetLaneIndex()" };
            let control = ins.raw[1] >> 8; // The reduction matcher verified this word.
            let lane = match (ins.aux.imm, control) {
                (0, _) => format!("{id} ^ 1u"),
                (1, 0x936012) => format!("({id} & ~3u) | (({id} ^ 2u) & 2u)"),
                (2, 0x000924) => format!("({id} & ~7u) | (({id} ^ 4u) & 4u)"),
                (1, 0xb3e21a) => format!("{id} ^ 2u"),
                (2, 0x688fac) => format!("{id} ^ 4u"),
                (1 | 2, _) => format!("({id} & ~7u) | (({control:#x}u >> (3u * ({id} & 7u))) & 7u)"),
                (3, _) => format!("{id} ^ 15u"),
                _ => return Err("invalid integer sum step".into()),
            };
            let partner = if self.portable_wave64 {
                format!("tb_reduction_read(gtid.x, {acc}, {lane})")
            } else { format!("WaveReadLaneAt({acc}, {lane})") };
            let combined = match m.as_str() {
                "v_add_nc_u32" => format!("{partner} + {acc}"),
                "v_and_b32" => format!("{partner} & {acc}"),
                "v_or_b32" => format!("{partner} | {acc}"),
                "v_xor_b32" => format!("{partner} ^ {acc}"),
                "v_min_u32" => format!("min({partner}, {acc})"),
                "v_max_u32" => format!("max({partner}, {acc})"),
                "v_min_i32" => format!("asuint(min(asint({partner}), asint({acc})))"),
                "v_max_i32" => format!("asuint(max(asint({partner}), asint({acc})))"),
                "v_add_f32" => format!("asuint(asfloat({partner}) + asfloat({acc}))"),
                "v_min_f32" => format!("asuint(min(asfloat({partner}), asfloat({acc})))"),
                "v_max_f32" => format!("asuint(max(asfloat({partner}), asfloat({acc})))"),
                _ => return Err("invalid DPP reduction operation".into()),
            };
            self.line(&format!("uint t{t} = {combined};"));
            return self.wv(&d, &format!("t{t}"));
        }
        if ins.enc == Enc::Vop3 && ins.op == wave_sum::READ_LANE {
            if self.pixel_wave {
                if self.pixel_sum && s[1] == Opnd::ConstInt(32) { return self.ws(&d, "0u"); }
                return self.ws(&d, &r(&s[0])?);
            }
            let lane = r(&s[1])?;
            let lane = if matches!(s[1], Opnd::Sgpr(_)) {
                let width = self.required_wave_size.ok_or("missing width for scan lane read")?;
                format!("({lane} & {}u)", width - 1)
            } else { lane };
            let value = r(&s[0])?;
            let read = if self.portable_wave64 {
                format!("tb_reduction_read(gtid.x, {value}, {lane})")
            } else { format!("WaveReadLaneAt({value}, {lane})") };
            return self.ws(&d, &read);
        }
        if ins.enc == Enc::Vop2 && ins.op == OP_LOCAL_EXTREMA {
            let value = self.fp32_extrema(ins, r(&s[0])?)?;
            return self.wv(&d, &value);
        }
        if is_quad_exchange(ins) {
            if self.dispatched && (!self.pixel_wave || is_row_move(ins)) {
                return Err("quad exchange in a block-dispatched program".into());
            }
            if ins.op != OP_DPP_BINARY {
                let value = format!("q{}", self.cur_idx);
                let control = ins.raw[1];
                if control >> 24 != 0xff {
                    if self.ex.stage != Stage::Compute {
                        return Err("masked DPP16 move requires a known compute wave width".into());
                    }
                    // Destination masks do not affect which lanes can supply the source.
                    let rows = control >> 28;
                    let banks = (control >> 24) & 15;
                    let row = format!("({rows:#x}u >> (WaveGetLaneIndex() >> 4u))");
                    let bank = format!("({banks:#x}u >> ((WaveGetLaneIndex() >> 2u) & 3u))");
                    return self.wv(&d, &format!("(({row} & {bank} & 1u) != 0u) ? {value} : {}", r(&d)?));
                }
                return self.wv(&d, &value);
            }
            let peer = format!("q{}", self.cur_idx);
            let value = match ins.mnemonic.as_str() {
                op @ ("v_add_f32" | "v_sub_f32" | "v_subrev_f32" | "v_mul_f32" | "v_mul_legacy_f32") => {
                    let (a, b) = dpp_fp32_sources(ins, peer.clone())?;
                    match op {
                        "v_add_f32" => format!("asuint(asfloat({a}) + asfloat({b}))"),
                        "v_sub_f32" => format!("asuint(asfloat({a}) - asfloat({b}))"),
                        "v_subrev_f32" => format!("asuint(asfloat({b}) - asfloat({a}))"),
                        "v_mul_f32" => format!("asuint(asfloat({a}) * asfloat({b}))"),
                        _ => format!("asuint(tb_mull(asfloat({a}), asfloat({b})))"),
                    }
                }
                "v_min_f32" | "v_max_f32" => self.fp32_extrema(ins, peer.clone())?,
                "v_cvt_pkrtz_f16_f32" => {
                    let (lo, hi) = dpp_fp32_sources(ins, peer.clone())?;
                    format!("(tb_f16_rtz(asfloat({lo})) | (tb_f16_rtz(asfloat({hi})) << 16))")
                }
                "v_or_b32" => format!("{peer} | {}", r(&s[1])?),
                "v_and_b32" => format!("{peer} & {}", r(&s[1])?),
                "v_xor_b32" => format!("{peer} ^ {}", r(&s[1])?),
                "v_xnor_b32" => format!("~({peer} ^ {})", r(&s[1])?),
                "v_cndmask_b32" => format!("(vcc_lo & 1u) != 0u ? {} : {peer}", r(&s[1])?),
                "v_add_nc_u32" => format!("{peer} + {}", r(&s[1])?),
                "v_sub_nc_u32" => format!("{peer} - {}", r(&s[1])?),
                "v_subrev_nc_u32" => format!("{} - {peer}", r(&s[1])?),
                "v_lshlrev_b32" => format!("{} << ({peer} & 31u)", r(&s[1])?),
                "v_lshrrev_b32" => format!("{} >> ({peer} & 31u)", r(&s[1])?),
                "v_ashrrev_i32" => format!("asuint(asint({}) >> ({peer} & 31u))", r(&s[1])?),
                "v_min_i32" => format!("asuint(min(asint({peer}), asint({})))", r(&s[1])?),
                "v_max_i32" => format!("asuint(max(asint({peer}), asint({})))", r(&s[1])?),
                "v_min_u32" => format!("min({peer}, {})", r(&s[1])?),
                "v_max_u32" => format!("max({peer}, {})", r(&s[1])?),
                op => mul24_expr(op, &peer, &r(&s[1])?).ok_or("unknown quad binary operation")?,
            };
            // The source exchange precedes both the destination mask and EXEC.
            let mut guards = Vec::new();
            if let Some(mask) = group_extrema::destination_mask(ins) {
                if self.ex.stage != Stage::Compute {
                    return Err("DPP16 destination masks require a logical compute lane index".into());
                }
                let rows = mask >> 4;
                let banks = mask & 15;
                if rows != 15 {
                    let row_bits = group_extrema::row_index_mask(self, rows, banks)?;
                    guards.push(format!("(({rows:#x}u >> ((tb_lane >> 4u) & {row_bits}u)) & 1u) != 0u"));
                }
                if banks != 15 { guards.push(format!("(({banks:#x}u >> ((tb_lane >> 2u) & 3u)) & 1u) != 0u")); }
            }
            if !group_extrema::is_dpp8(ins) && ins.raw[1] & (3 << 18) == 0 {
                guards.push(format!("{peer}active"));
            }
            if guards.is_empty() { return self.wv(&d, &value); }
            return self.wv(&d, &format!("{} ? ({value}) : {}", guards.join(" && "), r(&d)?));
        }
        if ins.op == OP_WAVE_REDUCE && ins.enc == Enc::Vop3 {
            // See `lower_wave_reductions`. Every invocation computes the same value; lanes
            // that were inactive contribute the identity.
            let f = if m == "tb_wave_min_f32" { "WaveActiveMin" } else { "WaveActiveMax" };
            let e = format!(
                "asuint({f}((({} & 1u) != 0u) ? asfloat({}) : asfloat({:#x}u)))",
                r(&s[1])?,
                r(&s[0])?,
                ins.aux.imm as u32
            );
            return self.ws(&d, &e);
        }
        // ---- half-float arithmetic: operands are f16 in the low 16 bits (inline float
        // constants denote the half-float of that value); the result is written as f16.
        // VOP3 forms carry per-source abs/neg modifiers and a result clamp/output modifier,
        // exactly as for f32 (see `fm` / `fres`).
        let h16 = |o: &Opnd| -> Result<String, String> {
            let mut e = match o {
                // Inline selector 248 has distinct binary16 and binary32 values.
                Opnd::ConstF32(value) => f(&Opnd::ConstF32(inline_half_value(*value)))?,
                _ => format!("f16tof32(({}) & 0xffffu)", r(o)?),
            };
            if ins.enc == Enc::Vop3 {
                if let Some(i) = s.iter().position(|x| std::ptr::eq(x, o)) {
                    if (ins.aux.mask >> i) & 1 != 0 {
                        e = format!("abs({e})");
                    }
                    if (ins.aux.mask >> (3 + i)) & 1 != 0 {
                        e = format!("(-{e})");
                    }
                }
            }
            Ok(e)
        };
        let hres = |e| format!("tb_f16_rne({})", Self::float_result(ins, e));
        let half_binary = matches!(m.as_str(), "v_add_f16" | "v_sub_f16" | "v_subrev_f16" | "v_mul_f16" | "v_max_f16" | "v_min_f16");
        if ins.aux.sdwa.is_none() && half_binary {
            let (a, b) = (h16(&s[0])?, h16(&s[1])?);
            let e = match m.as_str() {
                "v_add_f16" => format!("{a} + {b}"),
                "v_sub_f16" => format!("{a} - {b}"),
                "v_subrev_f16" => format!("{b} - {a}"),
                "v_mul_f16" => format!("{a} * {b}"),
                "v_max_f16" => format!("max({a}, {b})"),
                _ => format!("min({a}, {b})"),
            };
            return self.wv(&d, &hres(e));
        }
        if ins.aux.sdwa.is_none() && matches!(m.as_str(), "v_sin_f16" | "v_cos_f16") {
            if ins.enc == Enc::Vop3 && (ins.aux.mask != 0 || ins.aux.imm != 0 || ins.aux.imm2 != 0) {
                return Err("half trigonometric instruction modifiers unsupported".into());
            }
            let bits = if matches!(s[0], Opnd::ConstF32(_)) { hres(f(&s[0])?) } else { r(&s[0])? };
            let lo = format!("tb_trig_f16({bits}, {})", m == "v_cos_f16");
            return self.wv(&d, &format!("(({} & 0xffff0000u) | {lo})", r(&d)?));
        }
        let unary_op = half_unary_op(&m);
        let modified_half_bits = || -> Result<String, String> {
            let mut bits = half_operand_bits(&s[0])?;
            if ins.enc == Enc::Vop3 {
                if ins.aux.mask & 1 != 0 { bits = format!("(({bits}) & 0x7fffu)"); }
                if ins.aux.mask & 8 != 0 { bits = format!("(({bits}) ^ 0x8000u)"); }
            }
            Ok(bits)
        };
        let half_int_src = integer_to_half(&m);
        if ins.aux.sdwa.is_none() && (unary_op.is_some() || half_int_src) {
            let has_mode = self.regs.contains("tb_fp_mode");
            let mode = if has_mode { "tb_fp_mode" } else { "0xc0u" };
            let narrow = |value| if has_mode {
                format!("tb_mode_half({value}, {mode})")
            } else {
                format!("tb_f16_rne({value})")
            };
            let mut bits = if let Some(op) = unary_op {
                if let Opnd::ConstF32(value) = s[0] {
                    // Inline floats are finite normal halves. Only 1/(2*pi)
                    // has a distinct rounded value in the half-precision ROM.
                    let mut value = inline_half_value(value);
                    if ins.enc == Enc::Vop3 {
                        if ins.aux.mask & 1 != 0 { value = value.abs(); }
                        if ins.aux.mask & 8 != 0 { value = -value; }
                    }
                    let x = f(&Opnd::ConstF32(value))?;
                    format!("tb_finite_unary_f16({x}, {mode}, {op}u)")
                } else {
                    format!("tb_unary_f16({}, {mode}, {op}u)", modified_half_bits()?)
                }
            } else {
                let raw = half_operand_bits(&s[0])?;
                let integer = if m == "v_cvt_f16_i16" {
                    format!("(asint(({raw}) << 16) >> 16)")
                } else {
                    format!("(({raw}) & 0xffffu)")
                };
                narrow(format!("(float){integer}"))
            };
            if ins.enc == Enc::Vop3 && (ins.aux.imm2 != 0 || ins.aux.imm != 0) {
                let scale = ["1.0", "2.0", "4.0", "0.5"][ins.aux.imm2 as usize];
                let clamp = ins.aux.imm != 0;
                bits = format!("tb_half_output({bits}, {mode}, {scale}, {clamp})");
            }
            return self.wv(&d, &format!("(({} & 0xffff0000u) | {bits})", r(&d)?));
        }
        if ins.aux.sdwa.is_none() && matches!(m.as_str(), "v_rcp_f16" | "v_sqrt_f16" | "v_rsq_f16" | "v_exp_f16") {
            let a = h16(&s[0])?;
            let e = match m.as_str() {
                "v_rcp_f16" => format!("1.0 / {a}"),
                "v_sqrt_f16" => format!("sqrt({a})"),
                "v_rsq_f16" => format!("rsqrt({a})"),
                _ => format!("exp2({a})"),
            };
            // The compact reciprocal updates the low half. The FSPX shader first
            // writes the high reciprocal via SDWA, then fills the low half here.
            if m == "v_rcp_f16" && ins.enc == Enc::Vop1 && matches!(d, Opnd::Vgpr(_)) {
                return self.wv(&d, &format!("(({} & 0xffff0000u) | {})", r(&d)?, hres(e)));
            }
            return self.wv(&d, &hres(e));
        }
        let half_int_result = half_to_integer(&m);
        if normalized_half_to_integer(&m) && ins.aux.sdwa.is_none() {
            let bits = modified_half_bits()?;
            let mode = if self.regs.contains("tb_fp_mode") { "tb_fp_mode" } else { "0xc0u" };
            let signed = m == "v_cvt_norm_i16_f16";
            return self.wv(&d, &format!("(({} & 0xffff0000u) | tb_norm_f16({bits}, {mode}, {signed}))", r(&d)?));
        }
        if half_int_result && ins.aux.sdwa.is_none() {
            let x = h16(&s[0])?;
            // Truncate after saturating. CLAMP controls the inexact exception,
            // independently of the integer value produced by the conversion.
            let mut value = if m == "v_cvt_u16_f16" {
                format!("(uint)clamp({x}, 0.0, 65535.0)")
            } else {
                format!("(asuint((int)clamp({x}, -32768.0, 32767.0)) & 0xffffu)")
            };
            if !matches!(s[0], Opnd::ConstF32(_)) {
                // Test NaN on its stored bits before DXC can assume a finite float.
                value = format!("((({}) & 0x7fffu) > 0x7c00u ? 0u : {value})", r(&s[0])?);
            }
            return self.wv(&d, &format!("(({} & 0xffff0000u) | {value})", r(&d)?));
        }
        if matches!(m.as_str(), "v_fmaak_f16" | "v_fmamk_f16") {
            // The trailing literal dword carries the half-float constant K in its low bits.
            let k = format!("f16tof32({:#x}u)", ins.raw.last().copied().unwrap_or(0) & 0xffff);
            let (a, b) = (h16(&s[0])?, h16(&s[1])?);
            let e = if m == "v_fmaak_f16" { format!("tb_fma_half({a}, {b}, {k})") } else { format!("tb_fma_half({a}, {k}, {b})") };
            return self.wv(&d, &hres(e));
        }
        if m == "v_fmac_f16" {
            self.touch(&d);
            let e = format!("{} * {} + {}", h16(&s[0])?, h16(&s[1])?, h16(&d)?);
            return self.wv(&d, &hres(e));
        }
        if m == "v_fma_f16" {
            let sel = (ins.raw[0] >> 11) & 15;
            let src = |i: usize| -> Result<String, String> {
                if sel & (1 << i) == 0 { return h16(&s[i]); }
                // Inline FP constants have a binary16 payload and a zero upper half.
                let value = if matches!(s[i], Opnd::ConstF32(_)) {
                    "0.0".to_string()
                } else {
                    format!("f16tof32(({}) >> 16)", r(&s[i])?)
                };
                Ok(Self::float_source(value, ins.aux.mask, i))
            };
            let result = hres(format!("tb_fma_half({}, {}, {})", src(0)?, src(1)?, src(2)?));
            self.touch(&d);
            let dest = r(&d)?;
            return self.wv(&d, &if sel & 8 == 0 {
                format!("(({dest} & 0xffff0000u) | {result})")
            } else {
                format!("(({dest} & 0xffffu) | ({result} << 16))")
            });
        }
        if matches!(m.as_str(), "v_dot4_i32_i8" | "v_dot4c_i32_i8") {
            let compact = ins.enc == Enc::Vop2 && ins.aux.sdwa.is_none();
            let mut rhs = r(&s[1])?;
            let acc = if compact {
                &d
            } else if ins.op == 0x416 {
                // OP_SEL and OP_SEL_HI choose source 1's half for each dot
                // pair. These three combinations were observed in FSPX.
                rhs = match (ins.raw[0] & 0xff00, ins.raw[1] >> 27) {
                    (0x4000, 3) => rhs,
                    (0x5000, 1) => format!("(({rhs} >> 16u) | ({rhs} << 16u))"),
                    (0x4000, 1) => format!("(({rhs} & 65535u) * 0x10001u)"),
                    _ => return Err("unsupported signed byte dot product modifiers".into()),
                };
                &s[2]
            } else {
                return Err("unsupported signed byte dot product encoding".into());
            };
            let lhs = r(&s[0])?;
            let acc = r(acc)?;
            let value = if self.fspx_wave.is_some() {
                format!("asuint(dot4add_i8packed({lhs}, {rhs}, asint({acc})))")
            } else {
                format!("tb_dot4_i8({lhs}, {rhs}, {acc})")
            };
            return self.wv(&d, &value);
        }
        if int16::new_half(ins) && self.ex.stage != Stage::Compute {
            return Err("unverified ordinary 16-bit integer shader stage".into());
        }
        if let Some(result) = int16::expression(ins)? {
            return self.wv(&d, &result);
        }
        if self.lower_f64(ins)? {
            return Ok(());
        }
        if int32::handles(ins) && self.ex.stage != Stage::Compute {
            return Err("unverified ordinary 32-bit integer shader stage".into());
        }
        if let Some(result) = int32::expression(ins)? {
            return self.wv(&d, &result);
        }
        if m == "v_pack_b32_f16" {
            let (w0, w1) = (ins.raw[0], ins.raw[1]);
            if w0 & 0x8700 != 0 || w1 >> 27 != 0 {
                return Err("unsupported packed 16-bit modifiers".into());
            }
            let half = |operand: &Opnd, select| {
                if let Opnd::ConstF32(value) = operand {
                    if select != 0 { return Err("high-half selection on inline float".into()); }
                    return Ok(format!("f32tof16({value:?}f)"));
                }
                int16::half(operand, select)
            };
            let lo = half(&s[0], (w0 >> 11) & 1)?;
            let high = half(&s[1], (w0 >> 12) & 1)?;
            return self.wv(&d, &format!("({lo} | ({high} << 16))"));
        }
        if m == "v_pk_fmac_f16" {
            if ins.enc != Enc::Vop2 || ins.aux.sdwa.is_some() || matches!(s[0], Opnd::ConstF32(_)) {
                return Err("unsupported packed FMAC encoding or inline float".into());
            }
            let (a, b, acc) = (r(&s[0])?, r(&s[1])?, r(&d)?);
            return self.wv(&d, &format!("tb_fmac_half({a}, {b}, {acc}) | (tb_fmac_half({a} >> 16, {b} >> 16, {acc} >> 16) << 16)"));
        }
        if m == "v_cvt_pknorm_u16_f32" {
            if ins.raw[0] & 0xfc00 != 0 || ins.raw[1] & 0x98000000 != 0 {
                return Err("unsupported UNORM16 pack modifiers".into());
            }
            if self.ex.stage == Stage::Pixel {
                // The direct two-channel export is used for R16G16_UNORM. Until
                // additional export formats have provenance, reject a skipped or
                // indirect pack instead of reinterpreting the bits as FP16.
                let e = self.insns.get(self.cur_idx + 1).ok_or("UNORM16 pack without export")?;
                if e.enc != Enc::Exp || e.aux.imm != 0 || e.aux.mask != 3 || e.aux.imm2 & 1 == 0
                    || e.src.first() != Some(&d) || self.insns.iter().any(|i| i.enc == Enc::Sopp
                        && Self::branch_cond(i.op).is_some() && i.offset as i64 + 4 + 4*i.aux.imm == e.offset as i64)
                {
                    return Err("unresolved packed UNORM16 pixel export".into());
                }
            }
            let lo = self.fm(ins, 0)?;
            let hi = self.fm(ins, 1)?;
            return self.wv(&d, &format!("((uint)round(saturate({lo}) * 65535.0) | ((uint)round(saturate({hi}) * 65535.0) << 16))"));
        }
        if m == "v_cvt_pk_u8_f32" {
            if ins.aux.mask & !9 != 0 || ins.raw[0] & 0xf800 != 0 || ins.aux.imm2 != 0 {
                return Err("unsupported byte-pack modifiers".into());
            }
            let x = self.fm(ins, 0)?;
            let t = self.tmp;
            self.tmp += 1;
            self.line(&format!("uint t{t}shift = ({} & 3u) * 8u;", r(&s[1])?));
            let mode = if self.regs.contains("tb_fp_mode") { "tb_fp_mode" } else { "0u" };
            self.line(&format!("uint t{t}byte = tb_round_u8({x}, {mode});"));
            return self.wv(&d, &format!("({} & ~(255u << t{t}shift)) | (t{t}byte << t{t}shift)", r(&s[2])?));
        }
        if matches!(m.as_str(), "v_pk_fma_f16" | "v_pk_add_f16" | "v_pk_mul_f16" | "v_pk_min_f16" | "v_pk_max_f16") {
            // Packed half-float ops (VOP3P): two f16 lanes per register. For the low result,
            // OPSEL bit i (dword0[13:11]) picks the low/high half of source i and NEG bit i
            // (dword1[31:29]) negates it; for the high result OPSEL_HI (dword1[28:27],
            // dword0[14]) and NEG_HI (dword0[10:8]) do the same.
            let (w0, w1) = (ins.raw[0], ins.raw.get(1).copied().unwrap_or(0));
            let nsrc = if m == "v_pk_fma_f16" { 3 } else { 2 };
            let sel = [(w0 >> 11) & 7, ((w1 >> 27) & 3) | (((w0 >> 14) & 1) << 2)];
            let neg = [(w1 >> 29) & 7, (w0 >> 8) & 7];
            let mut halves = Vec::new();
            for h in 0..2 {
                let mut v = Vec::new();
                for i in 0..nsrc {
                    let x = match &s[i] {
                        o @ Opnd::ConstF32(_) => format!("tb_f16_rne({})", f(o)?),
                        o => r(o)?,
                    };
                    let mut e = if (sel[h] >> i) & 1 != 0 { format!("f16tof32(({x}) >> 16)") } else { format!("f16tof32(({x}) & 0xffffu)") };
                    if (neg[h] >> i) & 1 != 0 {
                        e = format!("(-{e})");
                    }
                    v.push(e);
                }
                halves.push(match m.as_str() {
                    "v_pk_fma_f16" => format!("{} * {} + {}", v[0], v[1], v[2]),
                    "v_pk_add_f16" => format!("{} + {}", v[0], v[1]),
                    "v_pk_mul_f16" => format!("{} * {}", v[0], v[1]),
                    "v_pk_min_f16" => format!("min({}, {})", v[0], v[1]),
                    _ => format!("max({}, {})", v[0], v[1]),
                });
            }
            if w0 & (1 << 15) != 0 {
                for value in &mut halves {
                    *value = format!("saturate({value})");
                }
            }
            let pack = if self.single_wave32 && matches!(m.as_str(), "v_pk_min_f16" | "v_pk_max_f16") {
                "tb_repack_half"
            } else {
                "tb_f16_rne"
            };
            return self.wv(&d, &format!("({pack}({}) | ({pack}({}) << 16))", halves[0], halves[1]));
        }
        if matches!(m.as_str(), "v_min3_f16" | "v_max3_f16" | "v_med3_f16") {
            let (a, b, c) = (h16(&s[0])?, h16(&s[1])?, h16(&s[2])?);
            let e = match m.as_str() {
                "v_min3_f16" => format!("min(min({a}, {b}), {c})"),
                "v_max3_f16" => format!("max(max({a}, {b}), {c})"),
                _ => format!("tb_med3({a}, {b}, {c})"),
            };
            return self.wv(&d, &hres(e));
        }
        if m == "v_bcnt_u32_b32" {
            return self.wv(&d, &format!("(countbits({}) + {})", r(&s[0])?, r(&s[1])?));
        }
        if m == "v_bfm_b32" {
            return self.wv(&d, &format!("(((1u << (({}) & 31u)) - 1u) << (({}) & 31u))", r(&s[0])?, r(&s[1])?));
        }
        if m == "v_perm_b32" {
            return self.wv(&d, &format!("tb_perm({}, {}, {})", r(&s[0])?, r(&s[1])?, r(&s[2])?));
        }
        if m == "v_bfe_i32" {
            // Sign-extended bit field: offset = S1[4:0], width = S2[4:0].
            // Width 0 gives 0; a field reaching past bit 31 (offset + width >= 32) is the
            // arithmetic shift by the offset (LLVM AMDGPU's folding of V_BFE_I32; the ISA
            // guide's pseudocode omits the sign extension).
            let (a, o, w) = (r(&s[0])?, r(&s[1])?, r(&s[2])?);
            let t = self.tmp;
            self.tmp += 1;
            self.line(&format!("uint t{t}o = ({o}) & 31u, t{t}w = ({w}) & 31u;"));
            return self.wv(&d, &format!(
                "(t{t}w == 0u ? 0u : ((t{t}o + t{t}w >= 32u) ? asuint(asint({a}) >> t{t}o) : asuint(asint(({a}) << (32u - t{t}o - t{t}w)) >> (32u - t{t}w))))"));
        }
        if matches!(m.as_str(), "v_movrels_b32" | "v_movreld_b32" | "v_movrelsd_b32") {
            // Register-relative moves: the source and/or destination VGPR number is offset by
            // M0 at run time (local arrays kept in registers). Lowered to a chain over every
            // VGPR the program uses.
            let (Opnd::Vgpr(dn), Opnd::Vgpr(sn)) = (d, s[0]) else {
                return Err("register-relative move with a non-VGPR operand".into());
            };
            if !self.vreg_array {
                return Err("register-relative move outside VGPR-array mode".into());
            }
            let t = self.tmp;
            self.tmp += 1;
            let src = if m == "v_movreld_b32" { format!("tb_v[{sn}]") } else { format!("tb_v[({sn}u + m0) & 255u]") };
            self.line(&format!("uint t{t} = {src};"));
            let dst = if m == "v_movrels_b32" { format!("tb_v[{dn}]") } else { format!("tb_v[({dn}u + m0) & 255u]") };
            self.line(&format!("if (exec_lo & 1u) {dst} = t{t};"));
            return Ok(());
        }
        if matches!(m.as_str(), "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16") {
            // VOP3P: D = S0 * S1 + S2 where each source is either a full float or a half
            // float taken from the low/high 16 bits of its register and widened:
            //   OPSEL_HI bit i (dword1[28:27], dword0[14] for i = 2) = 0 -> 32-bit float,
            //                                                         1 -> 16-bit float;
            //   OPSEL bit i (dword0[13:11]) picks the high half when the source is 16-bit;
            //   NEG bit i (dword1[31:29]) negates, NEG_HI bit i (dword0[10:8]) takes |x|.
            let (w0, w1) = (ins.raw[0], ins.raw.get(1).copied().unwrap_or(0));
            let opsel = (w0 >> 11) & 7;
            let opsel_hi = ((w1 >> 27) & 3) | (((w0 >> 14) & 1) << 2);
            let (neg, abs) = ((w1 >> 29) & 7, (w0 >> 8) & 7);
            let mut v = Vec::new();
            for i in 0..3 {
                let x = r(&s[i])?;
                let mut e = if (opsel_hi >> i) & 1 != 0 {
                    if (opsel >> i) & 1 != 0 { format!("f16tof32(({x}) >> 16)") } else { format!("f16tof32(({x}) & 0xffffu)") }
                } else {
                    format!("asfloat({x})")
                };
                if (abs >> i) & 1 != 0 {
                    e = format!("abs({e})");
                }
                if (neg >> i) & 1 != 0 {
                    e = format!("(-{e})");
                }
                v.push(e);
            }
            let e = format!("{} * {} + {}", v[0], v[1], v[2]);
            return match m.as_str() {
                "v_fma_mix_f32" => self.wv(&d, &format!("asuint({e})")),
                // The f16 result replaces one half of the destination; the other half is kept.
                "v_fma_mixlo_f16" => {
                    self.touch(&d);
                    self.wv(&d, &format!("(({}) & 0xffff0000u) | tb_f16_rne({e})", r(&d)?))
                }
                _ => {
                    self.touch(&d);
                    self.wv(&d, &format!("(({}) & 0xffffu) | (tb_f16_rne({e}) << 16)", r(&d)?))
                }
            };
        }
        if m == "v_mad_u64_u32" || m == "v_mad_i64_i32" {
            let signed = m == "v_mad_i64_i32";
            let Opnd::Vgpr(low) = d else { return Err("expected VGPR pair".into()); };
            let high = Opnd::Vgpr(low.checked_add(1).ok_or("VGPR pair overflows")?);
            // LLVM writes the carry to the null register when it is unused.
            let cout = match (ins.raw[0] >> 8) & 127 {
                n @ 0..=104 if !signed => Opnd::Sgpr(n as u8),
                106 if !signed => Opnd::VccLo,
                125 => Opnd::Null,
                _ => return Err("unsupported multiply-add carry destination".into()),
            };
            if signed && (self.ex.stage != Stage::Compute
                || !ins.src[..3].iter().all(|s| matches!(s, Opnd::Vgpr(_)))) {
                return Err("unverified signed 64-bit multiply-add source or stage".into());
            }
            if ins.raw[0] & 0x8000 != 0 || ins.raw[1] >> 27 != 0 {
                return Err("unsupported multiply-add modifiers".into());
            }
            let (a, b, c, ch) = (r(&s[0])?, r(&s[1])?, r(&s[2])?, self.pair_high(&s[2])?);
            let t = self.tmp;
            self.tmp += 1;
            // The low product word is the same for signed and unsigned inputs;
            // the signed high word comes from the signed 32x32 product.
            let mulhi = if signed { "tb_mulhi_i32" } else { "tb_mulhi_u32" };
            self.line(&format!("uint t{t}lo = {a} * {b} + {c}; uint t{t}sum = {mulhi}({a}, {b}) + {ch};"));
            self.line(&format!("uint t{t}hi = t{t}sum + uint(t{t}lo < {c}); uint t{t}co = uint(t{t}sum < {ch} || t{t}hi < t{t}sum);"));
            self.wv(&d, &format!("t{t}lo"))?;
            self.touch(&high);
            self.wv(&high, &format!("t{t}hi"))?;
            if cout == Opnd::Null {
                return Ok(());
            }
            if self.resolve_wave32 {
                return self.ws(&cout, &format!("WaveActiveBallot((exec_lo & tb_lane_bit) != 0u && t{t}co != 0u).x"));
            }
            self.ws(&cout, &format!("(exec_lo & 1u) * t{t}co"))?;
            if self.one_word_vector_mask() { return Ok(()); }
            return self.ws(&hi(&cout), "0u");
        }
        if matches!(
            m.as_str(),
            "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" | "v_add_co_ci_u32" | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32"
        ) {
            // Add/subtract with carry. VOP3B: carry-out register in bits [14:8] of dword0 and
            // carry-in is src2; VOP2: both are VCC. Lane model: carry = this lane's bit.
            let vop3f = ins.enc == Enc::Vop3;
            let cout = if vop3f {
                match (ins.raw[0] >> 8) & 0x7f {
                    c @ 0..=105 => Opnd::Sgpr(c as u8),
                    106 => Opnd::VccLo,
                    125 if ins.raw[1] >> 27 == 0 => Opnd::Null,
                    c => return Err(format!("carry-out register code {c} unsupported")),
                }
            } else {
                Opnd::VccLo
            };
            let cin = if !m.contains("_ci_") {
                "0u".to_string()
            } else {
                let mask = if vop3f { s[2] } else { Opnd::VccLo };
                if self.persistent_queue {
                    self.touch(&hi(&mask));
                    format!("uint(tb_bit(uint2({}, {}), tb_lane))", r(&mask)?, r(&hi(&mask))?)
                } else if self.resolve_wave32 {
                    format!("uint(({} & tb_lane_bit) != 0u)", r(&mask)?)
                } else { r(&mask)? }
            };
            let (mut a, mut b) = (r(&s[0])?, r(&s[1])?);
            if m.starts_with("v_subrev") {
                core::mem::swap(&mut a, &mut b);
            }
            let t = self.tmp;
            self.tmp += 1;
            let add = m.starts_with("v_add");
            if add {
                self.line(&format!(
                    "uint t{t}c = ({cin}) & 1u; uint t{t}a = {a} + {b}; uint t{t} = t{t}a + t{t}c; uint t{t}o = ((t{t}a < {a}) || (t{t} < t{t}a)) ? 1u : 0u;"
                ));
            } else {
                self.line(&format!(
                    "uint t{t}c = ({cin}) & 1u; uint t{t}a = {a} - {b}; uint t{t} = t{t}a - t{t}c; uint t{t}o = (({b} > {a}) || (t{t}c > t{t}a)) ? 1u : 0u;"
                ));
            }
            if vop3f && ins.raw[0] & 0x8000 != 0 {
                // Clamp the vector result only. SDST reports the original carry or borrow.
                let limit = if add { "0xffffffffu" } else { "0u" };
                self.line(&format!("if (t{t}o != 0u) t{t} = {limit};"));
            }
            self.wv(&d, &format!("t{t}"))?;
            // A null SDST discards the outgoing flag after the incoming mask has been read.
            if cout == Opnd::Null { return Ok(()); }
            self.touch(&cout);
            if self.persistent_queue {
                self.line(&format!("uint2 carry{t} = tb_ballot((exec_lo & t{t}o) != 0u);"));
                self.ws(&cout, &format!("carry{t}.x"))?;
                return self.ws(&hi(&cout), &format!("carry{t}.y"));
            }
            if self.resolve_wave32 {
                return self.ws(&cout, &format!("WaveActiveBallot((exec_lo & tb_lane_bit) != 0u && t{t}o != 0u).x"));
            }
            self.ws(&cout, &format!("((exec_lo & 1u) != 0u) ? t{t}o : 0u"))?;
            if self.one_word_vector_mask() { return Ok(()); }
            let ch = hi(&cout);
            self.touch(&ch);
            self.ws(&ch, "0u")?;
            return Ok(());
        }
        if let Some(sd) = ins.aux.sdwa {
            // SDWA: operate on a selected byte/word of each source. Materialise the selected
            // values into temporaries, then translate the instruction as usual.
            let is_cmp = ins.enc == Enc::Vopc;
            let dst_sel = if is_cmp { 6 } else { (sd >> 8) & 7 };
            if dst_sel == 7 {
                return Err("SDWA destination select 7 is reserved".into());
            }
            let mut c = ins.clone();
            c.aux.sdwa = None;
            if clamped_product24(ins) {
                if sd & 0xc000 != 0 || (sd >> 11) & 3 == 3 {
                    return Err("unsupported SDWA destination control for 24-bit multiply".into());
                }
                c.aux.imm = 1; // Saturate the full result before packing the destination.
            }
            // Reuse VOP3 floating modifiers after source selection and before destination packing.
            let half_alu = half_binary || half_int_result || unary_op.is_some() || matches!(m.as_str(), "v_rcp_f16" | "v_exp_f16" | "v_cvt_f32_f16");
            let float_alu = half_alu || matches!(m.as_str(),
                "v_add_f32" | "v_sub_f32" | "v_subrev_f32" | "v_mul_f32" | "v_mul_legacy_f32"
                | "v_min_f32" | "v_max_f32" | "v_fract_f32" | "v_trunc_f32" | "v_ceil_f32"
                | "v_rndne_f32" | "v_floor_f32");
            // Integer conversions accept floating output modifiers but keep integer source rules.
            let int_to_float = matches!(m.as_str(),
                "v_cvt_f32_i32" | "v_cvt_f32_u32" | "v_cvt_off_f32_i4" | "v_cvt_f32_ubyte0"
                | "v_cvt_f32_ubyte1" | "v_cvt_f32_ubyte2" | "v_cvt_f32_ubyte3");
            let float_source_modifiers = float_alu || m == "v_cvt_f16_f32";
            if float_source_modifiers || int_to_float || half_int_src {
                c.enc = Enc::Vop3;
                c.aux.imm = i64::from((sd >> 13) & 1);
                c.aux.imm2 = i64::from((sd >> 14) & 3);
            }
            if sd & 0x3030_0000 != 0 {
                if !(float_source_modifiers || (is_cmp && (m.ends_with("_f16") || m.ends_with("_f32"))))
                    || (m.contains("_class_") && sd & 0x3000_0000 != 0)
                {
                    return Err("unsupported SDWA neg/abs for this operand type".into());
                }
                // SDWAB Table 89: convert the independent source fields into the
                // same ABS[1:0], NEG[4:3] mask consumed by the VOP3 comparator.
                for (i, shift) in [(0, 16), (1, 24)] {
                    c.aux.mask |= (((sd >> (shift + 5)) & 1) as u8) << i;
                    c.aux.mask |= (((sd >> (shift + 4)) & 1) as u8) << (i + 3);
                }
            }
            for (i, shift) in [(0usize, 16u32), (1usize, 24u32)] {
                if i >= c.src.len() {
                    continue;
                }
                let sel = (sd >> shift) & 7;
                let sext = (sd >> (shift + 3)) & 1 != 0;
                if sel == 6 {
                    continue;
                }
                let mut x = r(&c.src[i])?;
                let half_constant = half_int_src || ((is_cmp || half_alu) && m.ends_with("_f16")
                    && (i == 0 || !m.contains("_class_")));
                if half_constant && matches!(c.src[i], Opnd::ConstF32(_))
                {
                    // The inline constant ROM supplies binary16 before byte/word selection.
                    x = format!("tb_f16_rne(asfloat({x}))");
                }
                let (bits, lsb) = match sel {
                    0..=3 => (8u32, sel * 8),
                    4 | 5 => (16, (sel - 4) * 16),
                    _ => return Err(format!("SDWA select {sel} is reserved")),
                };
                let e = if sext {
                    format!("asuint(asint({x} << {}) >> {})", 32 - bits - lsb, 32 - bits)
                } else {
                    format!("(({x} >> {lsb}) & {:#x}u)", (1u32 << bits) - 1)
                };
                let t = self.tmp;
                self.tmp += 1;
                self.line(&format!("uint tb_s{t} = {e};"));
                c.src[i] = Opnd::Temp(t as u16);
            }
            if dst_sel == 6 && !half_round_or_conversion(&m) && m != "v_cvt_f16_f32" {
                return self.emit_vec(&c);
            }
            // Partial destination: compute into a temporary, then place the selected
            // byte/word into the register. DST_UNUSED (bits 12:11): 0 = pad with zeros,
            // 1 = sign-extend upwards, 2 = keep the other bits of the destination.
            let t = self.tmp;
            self.tmp += 1;
            self.line(&format!("uint tb_s{t} = 0u;"));
            c.dst = vec![Opnd::Temp(t as u16)];
            self.emit_vec(&c)?;
            if dst_sel == 6 {
                // SDWA DWORD writes the entire zero-extended 16-bit result.
                return self.wv(&d, &format!("tb_s{t}"));
            }
            let (bits, lsb) = if dst_sel <= 3 { (8u32, dst_sel * 8) } else { (16, (dst_sel - 4) * 16) };
            let mask = (1u32 << bits) - 1;
            let val = format!("(tb_s{t} & {mask:#x}u)");
            let e = match (sd >> 11) & 3 {
                0 => format!("({val} << {lsb})"),
                1 => format!("asuint(asint({val} << {}) >> {})", 32 - bits, 32 - bits - lsb),
                2 => {
                    self.touch(&d);
                    format!("((({}) & {:#x}u) | ({val} << {lsb}))", r(&d)?, !(mask << lsb))
                }
                _ => return Err("SDWA DST_UNUSED 3 is reserved".into()),
            };
            return self.wv(&d, &e);
        }
        // V_CMPX_*: the comparison result replaces EXEC (inactive lanes give 0). On GFX10 the
        // CMPX forms write only EXEC, not VCC/SDST (LLVM's GFX10 encodings have no SDST for
        // them; the ISA reference says "write exec"). Lane model: this lane's EXEC bit.
        if m.starts_with("v_cmp_") || m.starts_with("v_cmpx_") {
            let mask = ins.aux.mask;
            let c = if m.ends_with("_i64") || m.ends_with("_u64") {
                self.compare_int64(ins, mask)?
            } else if m.ends_with("_f64") || m.ends_with("_f32") || m.ends_with("_f16") {
                self.compare_float(ins, mask)?
            } else {
                self.int_compare(&m, &s[0], &s[1])?
            };
            if self.resolve_wave32 {
                let vote = format!("WaveActiveBallot((exec_lo & tb_lane_bit) != 0u && {c}).x");
                return self.ws(if m.starts_with("v_cmpx_") { &Opnd::ExecLo } else { &d }, &vote);
            }
            if m.starts_with("v_cmpx_") {
                self.line(&format!("exec_lo = (((exec_lo & 1u) != 0u) && {c}) ? 1u : 0u; exec_hi = 0u;"));
                return Ok(());
            }
            if self.packed.as_ref().is_some_and(|p| p.ballots.contains(&self.cur_idx)) {
                let t = self.tmp;
                self.tmp += 1;
                self.line(&format!("uint4 t{t}ballot = WaveActiveBallot(((exec_lo & 1u) != 0u) && {c});"));
                self.ws(&d, &format!("t{t}ballot.x"))?;
                return self.ws(&hi(&d), &format!("t{t}ballot.y"));
            }
            self.ws(&d, &format!("(((exec_lo & 1u) != 0u) && {c}) ? 1u : 0u"))?;
            if self.one_word_vector_mask() { return Ok(()); } // The next SGPR can hold an independent mask.
            let dh = hi(&d);
            if !matches!(dh, Opnd::Null | Opnd::ConstInt(_)) {
                self.touch(&dh);
                self.ws(&dh, "0u")?;
            }
            return Ok(());
        }
        if matches!(m.as_str(), "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64") {
            // 64-bit vector shifts: value and result are register pairs (lo, hi).
            let pair_hi = |o: &Opnd| match o {
                Opnd::Vgpr(n) => Opnd::Vgpr(n + 1),
                other => hi(other),
            };
            let n = format!("({} & 63u)", r(&s[0])?);
            self.touch(&pair_hi(&s[1]));
            let (lo, hi_e) = (r(&s[1])?, r(&pair_hi(&s[1]))?);
            let t = self.tmp;
            self.tmp += 1;
            let (rl, rh) = match m.as_str() {
                "v_lshlrev_b64" => (
                    format!("(({n} >= 32u) ? 0u : ({lo} << ({n} & 31u)))"),
                    format!("(({n} == 0u) ? {hi_e} : (({n} >= 32u) ? ({lo} << ({n} - 32u)) : (({hi_e} << ({n} & 31u)) | ({lo} >> (32u - ({n} & 31u))))))"),
                ),
                "v_lshrrev_b64" => (
                    format!("(({n} == 0u) ? {lo} : (({n} >= 32u) ? ({hi_e} >> ({n} - 32u)) : (({lo} >> ({n} & 31u)) | ({hi_e} << (32u - ({n} & 31u))))))"),
                    format!("(({n} >= 32u) ? 0u : ({hi_e} >> ({n} & 31u)))"),
                ),
                _ => (
                    format!("(({n} == 0u) ? {lo} : (({n} >= 32u) ? asuint(asint({hi_e}) >> ({n} - 32u)) : (({lo} >> ({n} & 31u)) | ({hi_e} << (32u - ({n} & 31u))))))"),
                    format!("(({n} >= 32u) ? asuint(asint({hi_e}) >> 31) : asuint(asint({hi_e}) >> ({n} & 31u)))"),
                ),
            };
            self.line(&format!("uint t{t}l = {rl}; uint t{t}h = {rh};"));
            self.wv(&d, &format!("t{t}l"))?;
            let dh = pair_hi(&d);
            self.touch(&dh);
            return self.wv(&dh, &format!("t{t}h"));
        }
        if m == "v_swap_b32" {
            let (x, y) = (name(&d)?, name(&s[0])?);
            let t = self.tmp;
            self.tmp += 1;
            self.line(&format!("if (exec_lo & 1u) {{ uint t{t} = {x}; {x} = {y}; {y} = t{t}; }}"));
            return Ok(());
        }
        if m == "v_permlanex16_b32" {
            let flags = (ins.raw[0] >> 11) & 0xf;
            if flags != 0 || ins.aux.mask != 0 || ins.aux.imm != 0 || ins.aux.imm2 != 0 {
                return Err("permlanex16 modifiers unsupported".into());
            }
            let (value, lo, hi) = (r(&s[0])?, r(&s[1])?, r(&s[2])?);
            if self.pixel_wave { return self.wv(&d, if self.pixel_sum { "0u" } else { &value }); }
            if self.portable_wave64 {
                // portable_single_wave permits only the recognized cross-row
                // exchange inside a full-EXEC reduction with selectors zero.
                if s[1..] != [Opnd::ConstInt(0), Opnd::ConstInt(0)] {
                    return Err("unverified portable wave64 cross-row selectors".into());
                }
                let t = self.tmp;
                self.tmp += 1;
                self.line(&format!("uint t{t} = tb_reduction_read(gtid.x, {value}, (gtid.x & ~15u) ^ 16u);"));
                return self.wv(&d, &format!("t{t}"));
            }
            let t = self.tmp;
            self.tmp += 1;
            self.line(&format!("uint t{t}lane = WaveGetLaneIndex(); uint t{t}sel = (t{t}lane & 8u) ? {hi} : {lo};"));
            self.line(&format!("uint t{t}src = ((t{t}lane & ~15u) ^ 16u) | ((t{t}sel >> (4u * (t{t}lane & 7u))) & 15u);"));
            self.line(&format!("uint t{t}value = WaveReadLaneAt({value}, t{t}src);"));
            // FI=0, BOUND_CTRL=0: an inactive source preserves the destination.
            self.line(&format!("uint4 t{t}active = WaveActiveBallot((exec_lo & 1u) != 0u);"));
            self.line(&format!("if (t{t}active[t{t}src >> 5] & (1u << (t{t}src & 31u))) {{"));
            self.wv(&d, &format!("t{t}value"))?;
            self.line("}");
            return Ok(());
        }
        if m == "v_readfirstlane_b32" {
            let value = r(&s[0])?;
            if self.resolve_wave32 {
                return self.ws(&d, &format!("WaveReadLaneAt({value}, exec_lo == 0u ? 0u : uint(firstbitlow(exec_lo)))"));
            }
            if self.ex.stage == Stage::Mesh || (self.is_cs && compute_group(self.ex) == [1, 1, 1]) {
                return self.ws(&d, &value);
            }
            // A pixel wave holds helper lanes, which host wave intrinsics skip. Give
            // each pixel a wave of its own: it elects itself, so a waterfall over
            // heap indices runs once with that pixel's index.
            if self.pixel_heap || lone_lane::on() {
                return self.ws(&d, &value);
            }
            // Scalar writes reach every invocation. Select the first lane whose guest
            // EXEC bit is set, even when the executing host wave also has masked lanes.
            let t = self.tmp;
            self.tmp += 1;
            self.line("#if __SHADER_TARGET_MAJOR >= 6");
            self.line(&format!("uint t{t} = WaveActiveMin((exec_lo & 1u) ? WaveGetLaneIndex() : 0xffffffffu);"));
            self.ws(&d, &format!("WaveReadLaneAt({value}, t{t} == 0xffffffffu ? 0u : t{t})"))?;
            self.line("#else");
            self.ws(&d, &value)?;
            self.line("#endif");
            return Ok(());
        }
        let vop3 = ins.enc == Enc::Vop3;
        let k = format!("asfloat({:#x}u)", ins.raw.last().copied().unwrap_or(0));
        let dn = name(&d)?;
        let fa = || self.fm(ins, 0);
        let fb = || self.fm(ins, 1);
        let fc = || self.fm(ins, 2);
        let a = self.numeric_input(&s[0])?;
        let b = s.get(1).map(|operand| self.numeric_input(operand)).transpose()?.unwrap_or_default();
        let c = s.get(2).map(|o| r(o)).transpose().unwrap_or(None).unwrap_or_default();
        if m == "v_ldexp_f32" {
            let mode = if self.regs.contains("tb_fp_mode") { "tb_fp_mode" } else { "0u" };
            let bits = Self::select_source(ins, a, 0);
            let scale = [0, 1, 2, -1][ins.aux.imm2 as usize];
            let clamp = ins.aux.imm != 0;
            return self.wv(&d, &format!("tb_ldexp_f32({bits}, asint({b}), {mode}, {scale}, {clamp})"));
        }
        if clamped_product24(ins) {
            if vop3 && (ins.raw[0] & 0x7800 != 0 || ins.aux.mask != 0 || ins.aux.imm2 != 0) {
                return Err("unsupported modifiers for clamped 24-bit multiply".into());
            }
            let acc = if m.starts_with("v_mad_") { c.as_str() } else { "0u" };
            let signed = m.ends_with("_i24");
            return self.wv(&d, &format!("tb_mad24_sat({a}, {b}, {acc}, {signed})"));
        }
        // `v_mul_legacy_f32` / `v_mad_legacy_f32`: DX9 multiply rules — the product is 0 when
        // either operand is 0, even against infinity or NaN (`tb_mull`). A plain multiply
        // only matched under fast-math; with exact arithmetic it produced NaN positions.
        let legacy = m.ends_with("_legacy_f32");
        if m.ends_with("_legacy_f32") {
            m = m.replace("_legacy", "");
        }
        let fexpr: Option<String> = match m.as_str() {
            "v_add_f32" => Some(format!("{} + {}", fa()?, fb()?)),
            "v_sub_f32" => Some(format!("{} - {}", fa()?, fb()?)),
            "v_subrev_f32" => Some(format!("{} - {}", fb()?, fa()?)),
            // Lighting multiply saturates to the most negative finite value when S2
            // is unordered/nonpositive or S1 is NaN, -INF or -FLT_MAX.
            "v_mullit_f32" => Some(format!("((asuint({c}) == 0u || asuint({c}) > 0x7f800000u || (asuint({b}) & 0x7fffffffu) > 0x7f800000u || asuint({b}) >= 0xff7fffffu) ? asfloat(0xff7fffffu) : tb_mull({a}, {b}))",
                a=fa()?, b=fb()?, c=fc()?)),
            "v_mul_f32" if legacy => Some(format!("tb_mull({}, {})", fa()?, fb()?)),
            "v_mul_f32" => Some(format!("{} * {}", fa()?, fb()?)),
            "v_min_f32" => Some(format!("min({}, {})", fa()?, fb()?)),
            "v_max_f32" => Some(format!("max({}, {})", fa()?, fb()?)),
            "v_mac_f32" | "v_fmac_f32" => Some(format!("{} * {} + asfloat({dn})", fa()?, fb()?)),
            "v_madmk_f32" | "v_fmamk_f32" => Some(format!("{} * {k} + {}", fa()?, fb()?)),
            "v_madak_f32" | "v_fmaak_f32" => Some(format!("{} * {} + {k}", fa()?, fb()?)),
            "v_mad_f32" if legacy => Some(format!("tb_mull({}, {}) + {}", fa()?, fb()?, fc()?)),
            "v_mad_f32" | "v_fma_f32" => Some(format!("{} * {} + {}", fa()?, fb()?, fc()?)),
            "v_min3_f32" => Some(format!("min(min({}, {}), {})", fa()?, fb()?, fc()?)),
            "v_max3_f32" => Some(format!("max(max({}, {}), {})", fa()?, fb()?, fc()?)),
            // ISA: any NaN operand -> V_MIN3_F32 of the three; otherwise the median.
            "v_med3_f32" => Some(format!("tb_med3({}, {}, {})", fa()?, fb()?, fc()?)),
            "v_cvt_f32_i32" | "v_cvt_f32_u32" if self.regs.contains("tb_fp_mode") => {
                let signed = m == "v_cvt_f32_i32";
                Some(format!("tb_integer_to_f32({a}, {signed}, tb_fp_mode)"))
            }
            "v_cvt_f32_i32" => Some(format!("(float)asint({a})")),
            "v_cvt_f32_u32" => Some(format!("(float)({a})")),
            "v_cvt_off_f32_i4" => Some(format!("((float)(asint({a} << 28) >> 28) * 0.0625f)")),
            "v_cvt_f32_ubyte0" => Some(format!("(float)({a} & 0xffu)")),
            "v_cvt_f32_ubyte1" => Some(format!("(float)(({a} >> 8) & 0xffu)")),
            "v_cvt_f32_ubyte2" => Some(format!("(float)(({a} >> 16) & 0xffu)")),
            "v_cvt_f32_ubyte3" => Some(format!("(float)(({a} >> 24) & 0xffu)")),
            "v_cvt_f32_f16" => Some(h16(&s[0])?),
            "v_fract_f32" => {
                let mode = if self.regs.contains("tb_fp_mode") { "tb_fp_mode" } else { "0u" };
                Some(format!("tb_fract_f32({}, {mode})", fa()?))
            }
            "v_trunc_f32" => Some(format!("trunc({})", fa()?)),
            "v_ceil_f32" => Some(format!("ceil({})", fa()?)),
            "v_rndne_f32" => Some(format!("round({})", fa()?)),
            "v_floor_f32" => Some(format!("floor({})", fa()?)),
            "v_exp_f32" => Some(format!("exp2({})", fa()?)),
            "v_log_f32" => Some(format!("log2({})", fa()?)),
            "v_rcp_f32" | "v_rcp_iflag_f32" => Some(format!("1.0 / {}", fa()?)),
            "v_rsq_f32" => Some(format!("rsqrt({})", fa()?)),
            "v_sqrt_f32" => Some(format!("sqrt({})", fa()?)),
            "v_sin_f32" => Some(format!("sin({} * 6.28318530718)", fa()?)),
            "v_cubema_f32" => Some(format!("tb_cubema({}, {}, {})", fa()?, fb()?, fc()?)),
            "v_cubesc_f32" => Some(format!("tb_cubesc({}, {}, {})", fa()?, fb()?, fc()?)),
            "v_cubetc_f32" => Some(format!("tb_cubetc({}, {}, {})", fa()?, fb()?, fc()?)),
            "v_cubeid_f32" => Some(format!("tb_cubeid({}, {}, {})", fa()?, fb()?, fc()?)),
            "v_cos_f32" => Some(format!("cos({} * 6.28318530718)", fa()?)),
            _ => None,
        };
        if let Some(e) = fexpr {
            let e = self.fres(ins, e);
            return self.wv(&d, &e);
        }
        let mask = if matches!(self.exec_snapshots.sites.get(&self.cur_idx), Some(exec_snapshot::Site::LaneMask(..))) {
            format!("select_mask{}", self.cur_idx)
        } else if self.packed.as_ref().is_some_and(|p| p.lane_masks.contains(&self.cur_idx)) {
            let upper = if self.required_wave_size == Some(64) {
                r(&hi(&s[2]))?
            } else { "0u".into() };
            format!("(((WaveGetLaneIndex() < 32u) ? {c} : {upper}) >> (WaveGetLaneIndex() & 31u))")
        } else if vop3 { c.clone() } else { "vcc_lo".to_string() };
        let e = match m.as_str() {
            "v_mov_b32" => a,
            "v_cndmask_b32" => {
                // E64 permits float sign modifiers on both data sources. Apply them
                // to the bits so signed zero and NaN payloads survive the selection.
                let bit = if self.resolve_wave32 { "tb_lane_bit" } else { "1u" };
                format!("(({mask} & {bit}) != 0u) ? {} : {}", Self::select_source(ins, b, 1), Self::select_source(ins, a, 0))
            }
            "v_and_b32" => format!("({a} & {b})"),
            "v_or_b32" => format!("({a} | {b})"),
            "v_xor_b32" => format!("({a} ^ {b})"),
            "v_xnor_b32" => format!("(~({a} ^ {b}))"),
            "v_not_b32" => format!("(~{a})"),
            "v_bfrev_b32" => format!("reversebits({a})"),
            // Find-first-bit: -1 when no bit is set (HLSL intrinsics agree for zero input).
            "v_ffbl_b32" => format!("firstbitlow({a})"),
            "v_ffbh_u32" => format!("(({a}) == 0u ? 0xffffffffu : (31u - firstbithigh({a})))"),
            "v_lshrrev_b32" => format!("({b} >> ({a} & 31u))"),
            "v_lshlrev_b32" => format!("({b} << ({a} & 31u))"),
            "v_ashrrev_i32" => format!("asuint(asint({b}) >> ({a} & 31u))"),
            // Unsigned VOP3 clamp limits the operand before the operation can wrap.
            "v_add_nc_u32" if vop3 && ins.aux.imm != 0 => format!("(min({a}, ~{b}) + {b})"),
            "v_sub_nc_u32" if vop3 && ins.aux.imm != 0 => format!("(max({a}, {b}) - {b})"),
            "v_subrev_nc_u32" if vop3 && ins.aux.imm != 0 => format!("(max({b}, {a}) - {a})"),
            "v_add_nc_u32" => format!("({a} + {b})"),
            "v_sub_nc_u32" => format!("({a} - {b})"),
            "v_subrev_nc_u32" => format!("({b} - {a})"),
            "v_min_u32" => format!("min({a}, {b})"),
            "v_max_u32" => format!("max({a}, {b})"),
            "v_min_i32" => format!("asuint(min(asint({a}), asint({b})))"),
            "v_max_i32" => format!("asuint(max(asint({a}), asint({b})))"),
            "v_mul_lo_u32" => format!("({a} * {b})"),
            // Count the mask bits below this lane in the selected 32-bit word.
            "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
                if lone_lane::on() { return self.wv(&d, &b); }
                if self.single_wave32 && m == "v_mbcnt_lo_u32_b32" && s[0] == Opnd::ConstInt(-1) {
                    return self.wv(&d, &format!("({b} + tb_lane)"));
                }
                if self.indexed_lane {
                    let lane = if m == "v_mbcnt_lo_u32_b32" { "tb_lane & 63u" } else { "max(tb_lane & 63u, 32u) - 32u" };
                    return self.wv(&d, &format!("({b} + countbits({a} & {}))", low_mask(lane)));
                }
                if self.ex.stage == Stage::Mesh {
                    // These forms count the low/high 32 lanes preceding this logical wave64 lane.
                    if s[0] != Opnd::ConstInt(-1) { return Err("mesh mbcnt with a dynamic mask".into()); }
                    let count = if m == "v_mbcnt_lo_u32_b32" { "min(lane & 63u, 32u)" } else { "max(lane & 63u, 32u) - 32u" };
                    return self.wv(&d, &format!("{} + ({count})", r(&s[1])?));
                }
                let lane = if m == "v_mbcnt_lo_u32_b32" { "WaveGetLaneIndex()" } else { "max(32u, WaveGetLaneIndex()) - 32u" };
                let t = self.tmp;
                self.tmp += 1;
                self.line("#if __SHADER_TARGET_MAJOR >= 6");
                let a = if self.packed.as_ref().is_some_and(|p| p.scan_width == Some(64))
                    && matches!(s[0], Opnd::VccLo | Opnd::VccHi) {
                    format!("WaveActiveBallot((vcc_lo & 1u) != 0u).{}", if s[0] == Opnd::VccLo { "x" } else { "y" })
                } else if matches!(s[0], Opnd::ExecLo | Opnd::ExecHi) && !self.resolve_wave32 {
                    // Each invocation holds only its own EXEC bit; the guest reads the wave mask.
                    format!("WaveActiveBallot((exec_lo & 1u) != 0u).{}", if s[0] == Opnd::ExecLo { "x" } else { "y" })
                } else { a };
                self.line(&format!("uint t{t} = countbits({a} & {});", low_mask(lane)));
                self.line("#else");
                self.line(&format!("uint t{t} = 0u;"));
                self.line("#endif");
                format!("({b} + t{t})")
            }
            "v_mul_hi_u32" => format!("tb_mulhi_u32({a}, {b})"),
            // Leading bits equal to the sign bit, from the MSB; -1 when all bits are equal.
            "v_ffbh_i32" => format!("(((asint({a}) < 0) ? ~({a}) : ({a})) == 0u ? 0xffffffffu : (31u - firstbithigh((asint({a}) < 0) ? ~({a}) : ({a}))))"),
            // {S0:S1} >> (8 * S2[1:0]), low 32 bits.
            "v_alignbyte_b32" => format!(
                "((({c}) & 3u) == 0u ? {b} : (({b} >> ((({c}) & 3u) * 8u)) | ({a} << (32u - (({c}) & 3u) * 8u))))"
            ),
            "v_mul_hi_i32" => format!("tb_mulhi_i32({a}, {b})"),
            "v_mad_u32_u24" => format!("(({a} & 0xffffffu) * ({b} & 0xffffffu) + {c})"),
            "v_mad_i32_i24" => format!("asuint((asint({a} << 8) >> 8) * (asint({b} << 8) >> 8) + asint({c}))"),
            "v_bfe_u32" => format!("(({a} >> ({b} & 31u)) & ((1u << ({c} & 31u)) - 1u))"),
            "v_bfi_b32" => format!("(({a} & {b}) | (~{a} & {c}))"),
            "v_lerp_u8" => {
                let t = self.tmp;
                self.tmp += 1;
                self.line(&format!("uint diff{t} = {a} ^ {b};"));
                // Mask each byte before shifting so bits cannot spill into its neighbour.
                // Only an odd sum rounds up when the corresponding low bit of src2 is set.
                let floor_avg = format!("(({a} & {b}) + ((diff{t} & 0xfefefefeu) >> 1))");
                format!("({floor_avg} + (diff{t} & {c} & 0x01010101u))")
            }
            "v_lshl_add_u32" => format!("(({a} << ({b} & 31u)) + {c})"),
            "v_add_lshl_u32" => format!("(({a} + {b}) << ({c} & 31u))"),
            "v_add3_u32" => format!("({a} + {b} + {c})"),
            "v_lshl_or_b32" => format!("(({a} << ({b} & 31u)) | {c})"),
            "v_and_or_b32" => format!("(({a} & {b}) | {c})"),
            "v_or3_b32" => format!("({a} | {b} | {c})"),
            "v_xor3_b32" => format!("({a} ^ {b} ^ {c})"),
            "v_xad_u32" => format!("(({a} ^ {b}) + {c})"),
            "v_min3_u32" => format!("min(min({a}, {b}), {c})"),
            "v_max3_u32" => format!("max(max({a}, {b}), {c})"),
            "v_med3_u32" => format!("max(min({a}, {b}), min(max({a}, {b}), {c}))"),
            "v_min3_i32" => format!("asuint(min(min(asint({a}), asint({b})), asint({c})))"),
            "v_max3_i32" => format!("asuint(max(max(asint({a}), asint({b})), asint({c})))"),
            "v_med3_i32" => format!(
                "asuint(max(min(asint({a}), asint({b})), min(max(asint({a}), asint({b})), asint({c}))))"
            ),
            // {S0:S1} >> S2[4:0], low 32 bits (S0 is the high word).
            "v_alignbit_b32" => format!(
                "((({c}) & 31u) == 0u ? {b} : (({b} >> (({c}) & 31u)) | ({a} << (32u - (({c}) & 31u)))))"
            ),
            "v_cvt_u32_f32" => format!("tb_cvt_u32_f32({})", fa()?),
            "v_cvt_i32_f32" => format!("tb_cvt_i32_f32({})", fa()?),
            "v_cvt_flr_i32_f32" => format!("asuint((int)floor({}))", fa()?),
            "v_cvt_rpi_i32_f32" => {
                let x = fa()?;
                let t = self.tmp;
                self.tmp += 1;
                // Adding 0.5 first can round an already integral f32 to the next integer.
                // Start with nearest-even and raise only the ties that rounded down.
                self.line(&format!("float t{t} = round({x});"));
                format!("(asuint((int)t{t}) + (uint)(({x} - t{t}) == 0.5))")
            }
            "v_cvt_f16_f32" => {
                let source = Self::select_source(ins, a, 0);
                let modified = vop3 && (ins.aux.imm != 0 || ins.aux.imm2 != 0);
                let mode = if self.regs.contains("tb_fp_mode") { "tb_fp_mode" } else { "0xc0u" };
                let mut bits = if modified || self.regs.contains("tb_fp_mode") {
                    format!("tb_mode_half(asfloat({source}), {mode})")
                } else {
                    format!("tb_f16_rne(asfloat({source}))")
                };
                if modified {
                    let scale = ["1.0", "2.0", "4.0", "0.5"][ins.aux.imm2 as usize];
                    bits = format!("tb_half_output({bits}, {mode}, {scale}, {})", ins.aux.imm != 0);
                }
                format!("(({dn} & 0xffff0000u) | {bits})")
            }
            // Round toward zero (ISA: "ignore the current rounding mode"): finite values never
            // become infinity (|x| >= 65504 gives 65504).
            "v_cvt_pkrtz_f16_f32" => format!("(tb_f16_rtz({}) | (tb_f16_rtz({}) << 16))", fa()?, fb()?),
            // Pack two signed 16-bit ints from the source into two unsigned 8-bit bytes in
            // the low 16 bits of the destination (high 16 bits cleared). ISA: V_SAT_PK_U8_I16.
            "v_sat_pk_u8_i16" => format!(
                "((uint)clamp(asint({a} << 16) >> 16, 0, 255) | ((uint)clamp(asint({a}) >> 16, 0, 255) << 8))"
            ),
            op => mul24_expr(op, &a, &b).ok_or("unsupported vector op")?,
        };
        self.wv(&d, &e)
    }
}
