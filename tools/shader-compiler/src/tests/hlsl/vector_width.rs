//! Vector data widths of buffer, typed-buffer and flat memory transfers.
//!
//! "RDNA 2" ISA 12.14 (MUBUF, table 99): BUFFER_LOAD_DWORDX2/X4/X3 are opcodes 13/14/15 and
//! BUFFER_STORE_DWORDX2/X4/X3 are 29/30/31. Each moves 2, 4 or 3 dwords through consecutive
//! VGPRs starting at the VDATA field - so x3 moves three dwords, not four, and the register
//! after the last one is not part of the access. The VGPRs of one access form a quad, so no
//! gap is allowed between them. ISA 9.7 gives FLAT the same "zero to four consecutive Dwords"
//! with the first source VGPR in the DATA field, so these tests also fix the decoded widths of
//! the FLAT store opcodes that the emitter does not translate yet.
//!
//! Encodings below were produced for gfx1030: word 0 carries OFFSET/OFFEN/IDXEN/GLC/OP and
//! word 1 carries VADDR, VDATA, SRSRC and SOFFSET.
use super::{Emitter, Res};
use crate::rdna2::disassemble;
use crate::{ShaderExtract, Stage};

fn compute_extract(words: &[u32]) -> ShaderExtract {
    let isa: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    ShaderExtract {
        stage: Stage::Compute,
        shader_model: (6, 0),
        required_features: None,
        bitcode_stripped: true,
        fspx_config: None,
        ps09_pixel_inputs: None,
        fssm: Vec::new(),
        entry: Some("main".into()),
        isa_data: isa.clone(),
        isa,
        isa_offset_in_fssm: 0,
        psv_threads: None,
        flat_inputs: Vec::new(),
        flat_outputs: Vec::new(),
        isa_preamble: [0, 64, 1, 1],
        has_input_sig: false,
        has_output_sig: false,
        has_psv0: false,
        has_root_sig: false,
        input_sig: Vec::new(),
        output_sig: Vec::new(),
        resources: Vec::new(),
        gs_info: None,
        mesh_info: None,
        primitive_sig: Vec::new(),
        root_sig: None,
    }
}

/// Raw transfers stay one dword per HLSL `Load`/`Store` (see `raw_width`), but the number of
/// transfers is the encoded width: x2 -> 2, x3 -> 3, x4 -> 4. An x3 transfer must not read or
/// write the fourth VGPR of the quad.
#[test]
fn raw_vector_transfers_move_the_encoded_number_of_dwords() {
    // OFFEN address in v2, data v[4..], srsrc s[8:11], soffset 0.
    let cases = [
        (0xe034_1000u32, "buffer_load_dwordx2", "Load", 2u32),
        (0xe074_1000, "buffer_store_dwordx2", "Store", 2),
        (0xe03c_1000, "buffer_load_dwordx3", "Load", 3),
        (0xe07c_1000, "buffer_store_dwordx3", "Store", 3),
        (0xe038_1000, "buffer_load_dwordx4", "Load", 4),
        (0xe078_1000, "buffer_store_dwordx4", "Store", 4),
    ];
    for (word, mnemonic, op, width) in cases {
        let ex = compute_extract(&[word, 0x8002_0402]);
        let insns = disassemble(&ex.isa);
        assert_eq!(insns[0].mnemonic, mnemonic);
        let mut em = Emitter::new(&ex, insns.clone());
        em.raw_transfer_on(&insns[0], "raw_ubuf0".into(), true).unwrap();
        assert_eq!(
            em.body.matches(&format!("raw_ubuf0.{op}(")).count(),
            width as usize,
            "{mnemonic} must move {width} dwords:\n{}",
            em.body
        );
        // A merged transfer (Load2/Store4/...) reintroduces the wide-access fault documented
        // in `raw_width`, and none of them exists on a ByteAddressBuffer anyway.
        for wide in ["2(", "3(", "4("] {
            assert!(!em.body.contains(&format!("{op}{wide}")), "{mnemonic} merged:\n{}", em.body);
        }
        let past = format!("v{}", 4 + width);
        assert!(!em.body.contains(&past), "{mnemonic} read past its width:\n{}", em.body);
        for k in 0..width as u8 {
            assert!(em.body.contains(&format!("v{}", 4 + k)), "{mnemonic} skipped dword {k}:\n{}", em.body);
        }
    }
}

/// A UAV structured buffer's element is declared in dwords; an x3 store writes three of those
/// words (not four) and leaves the rest of the element alone.
#[test]
fn structured_uav_vector_store_writes_one_dword_per_encoded_word() {
    // BUFFER_STORE_DWORDX3 with no OFFEN/IDXEN: DATA v4 -> v4, v5, v6; srsrc s[8:11].
    let ex = compute_extract(&[0xe07c_0000, 0x8002_0402]);
    let ops = disassemble(&ex.isa);
    assert_eq!(ops[0].mnemonic, "buffer_store_dwordx3");
    assert_eq!(ops[0].aux.mask, 0);
    let mut em = Emitter::new(&ex, ops.clone());
    em.res.insert(0, vec![Res::Buf(0, true)]);
    em.sbuf.insert((true, 0), 4); // 16-byte element
    em.emit_mubuf(0, &ops[0]).unwrap();
    assert_eq!(em.body.matches("ubuf0[").count(), 3, "{}", em.body);
    for word in 0..3 {
        assert!(em.body.contains(&format!("ubuf0[0u].w[{word}]")), "{}", em.body);
    }
    assert!(!em.body.contains(".w[3]"), "x3 must not write the fourth word:\n{}", em.body);
    assert!(em.body.contains("v6") && !em.body.contains("v7"), "{}", em.body);
}

/// An x3 access that starts in the last word of an element continues in the next element; it
/// does not assume a 4-dword (x4) element stride.
#[test]
fn structured_uav_x3_store_continues_into_the_next_element() {
    // BUFFER_STORE_DWORDX3, immediate 28 = word 7 of an 8-dword element.
    let ex = compute_extract(&[0xe07c_001c, 0x8002_0402]);
    let ops = disassemble(&ex.isa);
    assert_eq!(ops[0].aux.imm, 28);
    let mut em = Emitter::new(&ex, ops.clone());
    em.res.insert(0, vec![Res::Buf(0, true)]);
    em.sbuf.insert((true, 0), 8); // 32-byte element
    em.emit_mubuf(0, &ops[0]).unwrap();
    assert_eq!(em.body.matches("ubuf0[").count(), 3, "{}", em.body);
    assert!(em.body.contains("ubuf0[0u].w[7]"), "{}", em.body);
    assert!(em.body.contains("ubuf0[0u + 1u].w[0]"), "{}", em.body);
    assert!(em.body.contains("ubuf0[0u + 1u].w[1]"), "{}", em.body);
}

/// The FLAT/global store opcodes decode to their own widths (ISA 9.7). The emitter has no
/// bound view for a per-lane address, so each one is refused with the access it describes
/// rather than as an unknown encoding.
#[test]
fn flat_store_widths_are_reported_for_each_opcode() {
    // VADDR v10, DATA v0, SADDR null, SEG=global (2).
    let cases = [
        (0xdc70_8000u32, "flat_store_dword", 1u32),
        (0xdc74_8000, "flat_store_dwordx2", 2),
        (0xdc7c_8000, "flat_store_dwordx3", 3),
        (0xdc78_8000, "flat_store_dwordx4", 4),
        (0xdc68_8000, "flat_store_short", 1),
        (0xdc60_8000, "flat_store_byte", 1),
    ];
    for (word, mnemonic, width) in cases {
        let ex = compute_extract(&[word, 0x007d_000a]);
        let ops = disassemble(&ex.isa);
        assert_eq!(ops[0].mnemonic, mnemonic);
        assert_eq!(ops[0].aux.mask, 2, "{mnemonic} is a global store");
        let mut em = Emitter::new(&ex, ops.clone());
        let error = em.emit_flat(0, &ops[0]).unwrap_err();
        assert!(error.contains("global store"), "{mnemonic}: {error}");
        let ngpr = if width == 1 { "1 dword".to_string() } else { format!("{width} dwords") };
        assert!(error.contains(&ngpr), "{mnemonic}: {error}");
        assert!(error.contains(mnemonic), "{mnemonic}: {error}");
    }
}
