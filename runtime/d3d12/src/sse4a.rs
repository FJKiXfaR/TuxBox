//! Fallback for the SSE4a bit-field instructions (`EXTRQ`/`INSERTQ`).
//!
//! Xbox Series X|S (Zen 2) implements AMD's SSE4a, which Intel CPUs do not:
//! executing `EXTRQ`/`INSERTQ` there raises `EXCEPTION_ILLEGAL_INSTRUCTION`.
//! The title uses the immediate forms to move bit fields between halves of an
//! XMM register, so the handler performs the move and resumes after the
//! instruction.
//!
//! Encodings handled:
//! - `EXTRQ xmm, imm8(index), imm8(length)` — `F2 0F 78 /0 ib ib`
//! - `INSERTQ xmm, xmm, imm8(index), imm8(length)` — `F2 0F 79 /r ib ib`
use std::sync::Mutex;
use windows::Win32::System::Diagnostics::Debug::{CONTEXT, M128A};

/// Longest instruction this module can decode (prefixes + opcode + ModRM + imms).
const MAX_LEN: usize = 8;

/// XMM register file inside the saved x64 CONTEXT.
fn xmm_base(context: &mut CONTEXT) -> *mut M128A {
    // SAFETY: the x64 CONTEXT stores Xmm0 through Xmm15 contiguously in the
    // floating-point save area.
    unsafe { &mut context.Anonymous.Anonymous.Xmm0 as *mut M128A }
}

/// Reads and writes register `index`, keeping the indices bounded.
unsafe fn xmm(base: *mut M128A, index: usize) -> Option<&'static mut M128A> {
    if index >= 16 {
        return None;
    }
    // SAFETY: `base` points at the register file and `index` is below 16.
    Some(unsafe { &mut *base.add(index) })
}

fn field_mask(length: u32) -> Option<u64> {
    if length == 0 || length > 64 {
        return None;
    }
    Some(if length == 64 { u64::MAX } else { (1u64 << length) - 1 })
}

pub(crate) fn handle_illegal_instruction(context: &mut CONTEXT, start: usize, end: usize) -> bool {
    let rip = context.Rip as usize;
    if rip < start || rip + MAX_LEN > end {
        return false;
    }
    // SAFETY: the title image stays mapped and readable around `rip`.
    let code = unsafe { core::slice::from_raw_parts(rip as *const u8, MAX_LEN) };
    let mut at = 0usize;
    let mut mandatory_f2 = false;
    loop {
        match code[at] {
            0x66 | 0x67 | 0x2e | 0x36 | 0x3e | 0x26 | 0x64 | 0x65 => at += 1,
            0xf2 => {
                mandatory_f2 = true;
                at += 1;
            }
            0xf3 => at += 1,
            _ => break,
        }
        if at >= MAX_LEN {
            return false;
        }
    }
    let mut rex = 0u8;
    if code[at] & 0xf0 == 0x40 {
        rex = code[at];
        at += 1;
    }
    if at + 3 > MAX_LEN || code[at] != 0x0f {
        return false;
    }
    let opcode = code[at + 1];
    let modrm = code[at + 2];
    if modrm >> 6 != 3 {
        return false;
    }
    let mut length = 0usize;
    let mut inserted = false;
    match (mandatory_f2, opcode) {
        // EXTRQ xmm(rm), imm8 index, imm8 length.
        (true, 0x78) if (modrm >> 3) & 7 == 0 => {
            let register = ((modrm & 7) | if rex & 1 != 0 { 8 } else { 0 }) as usize;
            let index = code[at + 3] as u32;
            let count = code[at + 4] as u32;
            let Some(mask) = field_mask(count) else { return false };
            let base = xmm_base(context);
            let Some(value) = (unsafe { xmm(base, register) }) else { return false };
            let source = value.Low as u64;
            value.Low = if index < 64 && index + count <= 64 { (source >> index) & mask } else { 0 };
            value.High = 0;
            length = at + 5;
            inserted = true;
        }
        // INSERTQ xmm(reg), xmm(rm), imm8 index, imm8 length.
        (true, 0x79) => {
            let destination = ((modrm >> 3) & 7) | if rex & 4 != 0 { 8 } else { 0 };
            let source = (modrm & 7) | if rex & 1 != 0 { 8 } else { 0 };
            let index = code[at + 3] as u32;
            let count = code[at + 4] as u32;
            let Some(mask) = field_mask(count) else { return false };
            let base = xmm_base(context);
            let Some(destination) = (unsafe { xmm(base, destination as usize) }) else { return false };
            let Some(source) = (unsafe { xmm(base, source as usize) }) else { return false };
            let (into, from) = (destination.Low as u64, source.Low as u64);
            if index < 64 && index + count <= 64 {
                destination.Low = (into & !(mask << index)) | ((from & mask) << index);
            }
            length = at + 6;
            inserted = true;
        }
        _ => {}
    }
    if !inserted {
        return false;
    }
    {
        static SITES: Mutex<Vec<usize>> = Mutex::new(Vec::new());
        let mut sites = SITES.lock().unwrap_or_else(|error| error.into_inner());
        if sites.len() < 32 && !sites.contains(&rip) {
            sites.push(rip);
            crate::diagnostics::log(&format!(
                "emulated SSE4a {} at RVA {:#x}",
                if opcode == 0x78 { "EXTRQ" } else { "INSERTQ" },
                rip - start
            ));
        }
    }
    context.Rip += length as u64;
    true
}
