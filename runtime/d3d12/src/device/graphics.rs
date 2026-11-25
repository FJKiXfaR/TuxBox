use super::{Device, Out, compute, identity, rectangle, wrap_pageable};
use crate::proxy::check_output;
use core::{ffi::c_void, mem::ManuallyDrop};
use windows::Win32::Foundation::{E_INVALIDARG, E_NOTIMPL};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{GUID, HRESULT, Interface};

// Frosty's device slot 60 takes a graphics descriptor and an extension array
// before the IID/output pair (retail call at RVA 0x5759a60).
pub(super) unsafe extern "system" fn create_pipeline(
    this: &Device,
    desc: *const D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    extension_count: u32,
    _extensions: *const c_void,
    iid: *const GUID,
    out: Out,
) -> HRESULT {
    let checked = unsafe { check_output::<ID3D12PipelineState>(iid, out) };
    if checked.is_err() {
        return checked;
    }
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if extension_count != 0 || desc.CachedPSO.CachedBlobSizeInBytes != 0 {
        return E_NOTIMPL;
    }
    if desc.VS.pShaderBytecode.is_null() || desc.VS.BytecodeLength == 0 {
        return E_INVALIDARG;
    }
    let guest_root = desc
        .pRootSignature
        .as_ref()
        .map_or(core::ptr::null_mut(), Interface::as_raw);
    let root = match unsafe { compute::root_signature(guest_root, identity(this).as_raw()) } {
        Ok(root) => root,
        Err(error) => return error,
    };
    // The descriptor borrows its shader blobs, layout arrays and root signature
    // for this synchronous call. Replace its guest pointer without taking a
    // reference on the guest object. Release only the new native reference.
    let mut native_desc = unsafe { core::ptr::read(desc) };
    native_desc.pRootSignature = ManuallyDrop::new(root.cloned());
    super::format::host_targets(&mut native_desc.RTVFormats, &mut native_desc.DSVFormat);
    // Never hand console-native bitcode to the host compiler: it faults instead
    // of returning an error (see shader_cache::host_compatible). The rectangle
    // path supplies its own desktop shaders and is exempt.
    if native_desc.PrimitiveTopologyType.0 != rectangle::PIPELINE_TYPE {
        for (stage, code) in [
            ("VS", native_desc.VS),
            ("PS", native_desc.PS),
            ("DS", native_desc.DS),
            ("HS", native_desc.HS),
            ("GS", native_desc.GS),
        ] {
            if code.BytecodeLength == 0 || code.pShaderBytecode.is_null() {
                continue;
            }
            // SAFETY: the guest keeps each shader blob alive for this call.
            let bytes = unsafe {
                core::slice::from_raw_parts(code.pShaderBytecode.cast::<u8>(), code.BytecodeLength)
            };
            if !super::shader_cache::host_compatible(bytes) {
                crate::diagnostics::log_once(&std::sync::Once::new(), || {
                    format!("refusing console-native {stage} graphics stage; not submitted to the host compiler")
                });
                return E_NOTIMPL;
            }
        }
    }
    let label = graphics_label(desc);
    let native = if native_desc.PrimitiveTopologyType.0 == rectangle::PIPELINE_TYPE {
        crate::big_stack::run_logged(&label, || unsafe { rectangle::create_pipeline(&this.native, &mut native_desc) })
    } else {
        crate::big_stack::run_logged(&label, || unsafe {
            this.native
                .CreateGraphicsPipelineState::<ID3D12PipelineState>(&native_desc)
        })
    };
    unsafe { ManuallyDrop::drop(&mut native_desc.pRootSignature) };
    unsafe {
        wrap_pageable(
            native,
            &compute::PIPELINE,
            this,
            compute::PipelineData::default(),
            out,
        )
    }
}

fn graphics_label(desc: &D3D12_GRAPHICS_PIPELINE_STATE_DESC) -> String {
    let mut label = String::from("graphics");
    let Some(vertex) = shader_prefix(&desc.VS) else { return label };
    label = format!("VS {vertex}");
    if let Some(pixel) = shader_prefix(&desc.PS) {
        label = format!("VS {vertex} PS {pixel}");
    }
    label
}

fn shader_prefix(code: &D3D12_SHADER_BYTECODE) -> Option<String> {
    if code.pShaderBytecode.is_null() || code.BytecodeLength == 0 {
        return None;
    }
    // SAFETY: the guest retains this bytecode until CreatePipelineState returns.
    let bytes = unsafe { core::slice::from_raw_parts(code.pShaderBytecode.cast::<u8>(), code.BytecodeLength) };
    super::shader_cache::source_digest(bytes).map(|hash| hash[..8].to_string())
}
