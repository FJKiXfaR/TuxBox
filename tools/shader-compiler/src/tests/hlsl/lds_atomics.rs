//! LDS/DS atomic read-modify-write lowering.
//!
//! Semantics are from the RDNA 2 ISA, §12.13 "LDS & GDS Instructions" (DS
//! opcode table, pp. 188-195): every form reads `MEM[ADDR]` before the update,
//! the `_RTN` forms return it, and `DS_MIN/MAX_I32` compare signed while
//! `DS_MIN/MAX_U32` compare unsigned.
use super::*;
use crate::hlsl::float64::tests::translate;
use crate::rdna2::disassemble;
use crate::Stage;

/// Encode a DS instruction: opcode in word0[25:18], 16-bit offset in
/// word0[15:0]; VDST in word1[31:24], DATA1 in [23:16], DATA0 in [15:8],
/// ADDR in [7:0] (RDNA 2 ISA §12.13 instruction format).
fn ds(opcode: u32, offset: u32, vdst: u32, data0: u32, data1: u32, addr: u32) -> [u32; 2] {
    [
        0xd800_0000 | (opcode << 18) | (offset & 0xffff),
        (vdst << 24) | (data1 << 16) | (data0 << 8) | addr,
    ]
}

fn decode(words: &[u32; 2]) -> crate::rdna2::Insn {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    disassemble(&bytes).remove(0)
}

#[test]
fn ds_encodings_decode_to_the_isa_opcode_names() {
    // Opcode numbers from the RDNA 2 ISA §12.13 table.
    for (opcode, name) in [
        (0, "ds_add_u32"),
        (1, "ds_sub_u32"),
        (2, "ds_rsub_u32"),
        (3, "ds_inc_u32"),
        (4, "ds_dec_u32"),
        (5, "ds_min_i32"),
        (6, "ds_max_i32"),
        (7, "ds_min_u32"),
        (8, "ds_max_u32"),
        (12, "ds_mskor_b32"),
        (16, "ds_cmpst_b32"),
        (32, "ds_add_rtn_u32"),
        (37, "ds_min_rtn_i32"),
        (38, "ds_max_rtn_i32"),
        (44, "ds_mskor_rtn_b32"),
        (48, "ds_cmpst_rtn_b32"),
        (52, "ds_wrap_rtn_b32"),
    ] {
        let ins = decode(&ds(opcode, 0x10, 3, 2, 1, 1));
        assert_eq!(ins.mnemonic, name);
        assert_eq!(ins.aux.imm, 0x10);
        assert_eq!(ins.dst, vec![Opnd::Vgpr(3)]);
        assert_eq!(ins.src, vec![Opnd::Vgpr(1), Opnd::Vgpr(2), Opnd::Vgpr(1)]);
    }
}

#[test]
fn signed_extrema_compare_as_signed_and_keep_the_unsigned_word() {
    for (name, compare) in [(5, "<"), (6, ">")] {
        let hlsl = translate(Stage::Compute, &ds(name, 0x10, 3, 2, 0, 1)).unwrap();
        assert!(
            hlsl.contains(&format!("(asint(t0data) {compare} asint(t0seen)) ? t0data : t0seen")),
            "{hlsl}"
        );
        assert!(hlsl.contains("tb_lds[((v1 + 0x10u) >> 2) & 8191u]"), "{hlsl}");
        // A non-RTN form leaves VDST alone.
        assert!(!hlsl.contains("v3 = t"), "{hlsl}");
        // The result is the old memory contents.
        assert!(hlsl.contains("t0 = t0old;"), "{hlsl}");
        let rtn = translate(Stage::Compute, &ds(name + 32, 0x10, 3, 2, 0, 1)).unwrap();
        assert!(rtn.contains("v3 = t0;"), "{rtn}");
    }
    // The unsigned forms use the hardware's unsigned min/max intrinsics.
    for (name, func) in [(7, "InterlockedMin"), (8, "InterlockedMax")] {
        let hlsl = translate(Stage::Compute, &ds(name, 0, 3, 2, 0, 1)).unwrap();
        assert!(hlsl.contains(&format!("{func}(tb_lds[((v1 + 0x0u) >> 2) & 8191u], v2, t0)")), "{hlsl}");
    }
}

#[test]
fn wrapping_and_saturating_forms_use_an_atomic_compare_exchange_loop() {
    // DS_INC_U32: MEM = (tmp >= DATA) ? 0 : tmp + 1.
    let hlsl = translate(Stage::Compute, &ds(3, 0, 2, 1, 0, 1)).unwrap();
    assert!(hlsl.contains("(t0seen >= t0data) ? 0u : t0seen + 1u"), "{hlsl}");
    // DS_DEC_U32: MEM = (tmp == 0 || tmp > DATA) ? DATA : tmp - 1.
    let hlsl = translate(Stage::Compute, &ds(4, 0, 2, 1, 0, 1)).unwrap();
    assert!(hlsl.contains("(t0seen == 0u || t0seen > t0data) ? t0data : t0seen - 1u"), "{hlsl}");
    // DS_RSUB_U32: MEM = DATA - MEM.
    let hlsl = translate(Stage::Compute, &ds(2, 0, 2, 1, 0, 1)).unwrap();
    assert!(hlsl.contains("t0data - t0seen"), "{hlsl}");
    // DS_WRAP_RTN_B32: MEM = (tmp >= DATA0) ? tmp - DATA0 : tmp + DATA1.
    let hlsl = translate(Stage::Compute, &ds(52, 0, 2, 1, 3, 2)).unwrap();
    assert!(hlsl.contains("(t0seen >= t0data) ? t0seen - t0data : t0seen + t0next"), "{hlsl}");
    assert!(hlsl.contains("v2 = t0;"), "{hlsl}");
    // DS_DEC_RTN_U32 returns the pre-operation value from the loop.
    let rtn = translate(Stage::Compute, &ds(36, 0, 2, 1, 0, 1)).unwrap();
    assert!(rtn.contains("v2 = t0;"), "{rtn}");
}

#[test]
fn masked_or_and_compare_store_use_the_documented_source_order() {
    // DS_MSKOR_B32: MEM = (MEM & ~DATA0) | DATA1.
    let hlsl = translate(Stage::Compute, &ds(12, 0, 2, 1, 3, 2)).unwrap();
    assert!(hlsl.contains("uint t0mask = v1, t0src = v3;"), "{hlsl}");
    assert!(hlsl.contains("(t0seen & ~t0mask) | t0src"), "{hlsl}");
    // DS_CMPST_B32: DATA0 is the compare value, DATA1 the replacement, and the
    // order is the opposite of BUFFER_ATOMIC_CMPSWAP's.
    let hlsl = translate(Stage::Compute, &ds(16, 0, 2, 1, 3, 2)).unwrap();
    assert!(
        hlsl.contains("InterlockedCompareExchange(tb_lds[((v2 + 0x0u) >> 2) & 8191u], v1, v3, t0)"),
        "{hlsl}"
    );
    let hlsl = translate(Stage::Compute, &ds(48, 0, 2, 1, 3, 2)).unwrap();
    assert!(hlsl.contains("v2 = t0;"), "{hlsl}");
}

#[test]
fn simple_forms_use_the_matching_interlocked_intrinsic() {
    for (opcode, call) in [
        (0, "InterlockedAdd(tb_lds"),
        (9, "InterlockedAnd(tb_lds"),
        (10, "InterlockedOr(tb_lds"),
        (11, "InterlockedXor(tb_lds"),
        (45, "InterlockedExchange(tb_lds"),
    ] {
        let hlsl = translate(Stage::Compute, &ds(opcode, 0, 3, 2, 0, 1)).unwrap();
        assert!(hlsl.contains(call), "{opcode}: {hlsl}");
    }
    // DS_SUB_U32 is a wrapping add of the negated source, with the same old
    // value and stored bit pattern.
    let hlsl = translate(Stage::Compute, &ds(1, 0, 3, 2, 0, 1)).unwrap();
    assert!(hlsl.contains("InterlockedAdd(tb_lds[((v1 + 0x0u) >> 2) & 8191u], (0u - v2), t0)"), "{hlsl}");
}

#[test]
fn forms_without_a_definition_still_refuse() {
    // Floating point and 64-bit LDS atomics, and the GDS-only opcodes, have no
    // verified lowering yet.
    for words in [ds(18, 0, 3, 2, 0, 1), ds(64, 0, 3, 2, 0, 1), ds(61, 0, 3, 2, 0, 1)] {
        assert!(translate(Stage::Compute, &words).is_err());
    }
    // A GDS exchange keeps its refusal.
    let mut words = ds(45, 0, 3, 2, 0, 1);
    words[0] |= 1 << 17;
    assert!(translate(Stage::Compute, &words).is_err());
}
