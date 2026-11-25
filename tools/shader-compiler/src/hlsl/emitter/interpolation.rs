//! Pixel attribute interpolation.
use super::*;

impl<'a> Emitter<'a> {
    /// Read an interpolated attribute.
    pub(in crate::hlsl) fn emit_vinterp(&mut self, ins: &Insn, is_ps: bool) -> Result<(), String> {
        let s = &ins.src;
        let d = ins.dst.first().cloned();
        if !is_ps {
            return Err("v_interp outside a pixel shader".into());
        }
        let attr = ins.aux.imm as usize;
        let chan = ins.aux.imm2 as usize;
        let param = self
            .ps_attrs
            .get(attr)
            .cloned()
            .ok_or_else(|| format!("PS attr{attr} has no signature register"))?;
        // A half-float parameter holds two 16-bit components per 32-bit channel (ATTRCHAN
        // selects the channel, not the half). A 32-bit interpolation of such a parameter
        // therefore works on the channel's low component: the vertex program exports each
        // half as the float it stands for, so the pixel input field already holds that
        // value and the arithmetic below is plain float work. The upper half of the channel
        // is read only by the f16 interpolation opcodes, which have their own emitter.
        let (reg, chan) = match param {
            Param::Full(reg) => (reg, chan),
            Param::Half(_) => param
                .component(chan, false)
                .ok_or_else(|| format!("PS attr{attr} has no signature register"))?,
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
                        0 => delta(1), 1 => delta(2), 2 => p0, other => return Err(format!("invalid interpolation parameter {other}")),
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
        //
        // v_interp_mov_f32 selects one of P10, P20 and P0 by VSRC[1:0] (0, 1 and 2). A
        // smooth attribute read that way belongs to `pulled_attrs` above; a P0 read of a
        // flat attribute is the provoking vertex value, which is what a `nointerpolation`
        // input already holds, so the pixel value is the answer for every form.
        self.wv(&d.ok_or("no dst")?, &val)
    }
}
