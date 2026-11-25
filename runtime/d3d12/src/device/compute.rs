//! Wrapped root signatures and the Xbox extended compute pipeline factory.
use super::{Device, Out, identity, wrap_pageable};
use crate::proxy::{Proxy, Slots, allocate, check_output, child_slots, unwrap};
use core::{ffi::c_void, mem::ManuallyDrop, slice};
use windows::Win32::Foundation::{E_INVALIDARG, E_NOTIMPL, E_POINTER};
use windows::Win32::Graphics::Direct3D12::*;

// The host driver's shader compiler faults when many compute pipelines compile
// at once. Compile at most this many concurrently; anything beyond falls back to
// the null pipeline instead of waiting (waiting deadlocks on nested creation).
static COMPILING: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
const MAX_CONCURRENT_COMPILES: usize = 4;

struct CompilePermit;
impl CompilePermit {
    fn acquire() -> Option<Self> {
        use core::sync::atomic::Ordering;
        let mut current = COMPILING.load(Ordering::Acquire);
        loop {
            if current >= MAX_CONCURRENT_COMPILES {
                return None;
            }
            match COMPILING.compare_exchange(current, current + 1, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Some(Self),
                Err(seen) => current = seen,
            }
        }
    }
}
impl Drop for CompilePermit {
    fn drop(&mut self) {
        COMPILING.fetch_sub(1, core::sync::atomic::Ordering::AcqRel);
    }
}
use windows::core::{GUID, HRESULT, Interface};

pub(crate) use super::compute_table::ComputeTable;
pub(crate) use super::compute_scene::{RootSource, SceneRoot, SceneSource};

static ROOT: Slots<8> = child_slots::<ID3D12RootSignature, RootSource, 8>();
pub(crate) type Root = Proxy<ID3D12RootSignature, RootSource>;
pub(crate) type Pipeline = Proxy<ID3D12PipelineState, PipelineData>;
// Inline console SetPipelineState copies 32 bytes starting at +0x10 from
// this object to the guest command list. Desktop command lists use slot 25
// with an empty packet buffer, so no Xbox command packets are advertised.
#[repr(C)]
#[derive(Default)]
pub(crate) struct PipelineData {
    reserved: usize,
    packets: [u8; 32],
    pub(crate) stencil_slot: Option<u32>,
    pub(crate) draw_trace: Option<std::sync::Arc<crate::diagnostics::RasterDraw>>,
    pub(crate) scene_root: Option<SceneRoot>,
    pub(crate) reflected: Option<std::sync::Arc<ComputeTable>>,
    pub(crate) raster_table: Option<std::sync::Arc<ComputeTable>>,
}

const _: () = assert!(
    core::mem::offset_of!(Proxy<ID3D12PipelineState, PipelineData>, state)
        + core::mem::offset_of!(PipelineData, packets)
        == 0x10
);
pub(super) static PIPELINE: Slots<12> = {
    let mut slots = child_slots::<ID3D12PipelineState, PipelineData, 12>();
    slots[11] = crate::slot!(scratch_info);
    slots
};

// FidelityFX queries a blob for the console pipeline's scratch size. Desktop
// pipelines manage their own scratch. The caller skips the blob on failure.
unsafe extern "system" fn scratch_info(
    _this: &Proxy<ID3D12PipelineState, PipelineData>,
    _arg1: usize,
    _arg2: usize,
    blob: Out,
) -> HRESULT {
    if blob.is_null() {
        return E_POINTER;
    }
    // SAFETY: the caller provided storage for one COM interface pointer.
    unsafe { blob.write(core::ptr::null_mut()) };
    E_NOTIMPL
}

// A null root signature is valid for a shader with an embedded signature.
// The caller keeps every non-null guest object alive for the returned borrow.
pub(crate) unsafe fn root_proxy<'a>(
    guest: *mut c_void,
    device: *mut c_void,
) -> Result<Option<&'a Root>, HRESULT> {
    if guest.is_null() {
        return Ok(None);
    }
    // SAFETY: unwrap checks both the guest vtable and device identity.
    unsafe { unwrap::<ID3D12RootSignature, RootSource>(guest, device, &ROOT) }
        .map(Some)
        .ok_or(E_INVALIDARG)
}

pub(crate) unsafe fn root_signature<'a>(
    guest: *mut c_void,
    device: *mut c_void,
) -> Result<Option<&'a ID3D12RootSignature>, HRESULT> {
    unsafe { root_proxy(guest, device) }.map(|root| root.map(|p| &p.native))
}

pub(crate) unsafe fn pipeline<'a>(
    guest: *mut c_void,
    device: *mut c_void,
) -> Result<Option<&'a Proxy<ID3D12PipelineState, PipelineData>>, HRESULT> {
    if guest.is_null() {
        return Ok(None);
    }
    // SAFETY: as with root_signature, the caller owns the live guest reference.
    unsafe { unwrap::<ID3D12PipelineState, PipelineData>(guest, device, &PIPELINE) }
        .map(Some)
        .ok_or(E_INVALIDARG)
}

pub(super) unsafe extern "system" fn create_root_signature(
    this: &Device,
    node: u32,
    blob: *const u8,
    length: usize,
    iid: *const GUID,
    out: Out,
) -> HRESULT {
    // SAFETY: the guest supplies a readable IID and writable output slot.
    let hr = unsafe { check_output::<ID3D12RootSignature>(iid, out) };
    if hr.is_err() {
        return hr;
    }
    if blob.is_null() || length == 0 || length > isize::MAX as usize {
        return E_INVALIDARG;
    }
    // SAFETY: serialized signature bytes are borrowed only for this call.
    let bytes = unsafe { slice::from_raw_parts(blob, length) };
    match unsafe {
        this.native
            .CreateRootSignature::<ID3D12RootSignature>(node, bytes)
    } {
        // SAFETY: the returned guest child owns both its native interface and device.
        Ok(native) => unsafe {
            allocate(
                &ROOT,
                &[ID3D12DeviceChild::IID, ID3D12RootSignature::IID],
                native,
                Some(&identity(this)),
                RootSource::new(bytes),
                out,
            )
        },
        Err(error) => error.code(),
    }
}

// Slot 67 receives a serialized console PSO. FidelityFX retries slot 62 with a
// shader descriptor if its precompiled ACO pipeline cannot be deserialized.
pub(super) unsafe extern "system" fn deserialize_pipeline_x(
    _this: &Device,
    _desc: *const c_void,
    iid: *const GUID,
    out: Out,
) -> HRESULT {
    // SAFETY: the guest supplies a readable IID and writable output slot.
    let hr = unsafe { check_output::<ID3D12PipelineState>(iid, out) };
    if hr.is_err() { hr } else { E_NOTIMPL }
}

// Slot 62 takes a compute descriptor followed by an array of Xbox extensions.
// Portable DXIL and a standard serialized root signature work with zero extensions.
pub(super) unsafe extern "system" fn create_pipeline_x(
    this: &Device,
    desc: *const D3D12_COMPUTE_PIPELINE_STATE_DESC,
    extension_count: u32,
    _extensions: *const c_void,
    iid: *const GUID,
    out: Out,
) -> HRESULT {
    // SAFETY: the guest supplies a readable IID and writable output slot.
    let hr = unsafe { check_output::<ID3D12PipelineState>(iid, out) };
    if hr.is_err() {
        return hr;
    }
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if extension_count != 0 || desc.CachedPSO.CachedBlobSizeInBytes != 0 {
        return E_NOTIMPL;
    }
    if desc.CS.pShaderBytecode.is_null()
        || desc.CS.BytecodeLength == 0
        || desc.CS.BytecodeLength > isize::MAX as usize
    {
        return E_INVALIDARG;
    }
    let guest = desc
        .pRootSignature
        .as_ref()
        .map_or(core::ptr::null_mut(), Interface::as_raw);
    // SAFETY: the descriptor borrows its root signature for pipeline creation.
    let root = match unsafe { root_proxy(guest, identity(this).as_raw()) } {
        Ok(root) => root,
        Err(error) => return error,
    };
    // SAFETY: the guest retains its shader bytes until this synchronous call returns.
    let bytes =
        unsafe { slice::from_raw_parts(desc.CS.pShaderBytecode.cast(), desc.CS.BytecodeLength) };
    if std::env::var_os("TUXBOX_NULL_ALL_COMPUTE").is_some() {
        crate::diagnostics::compute_failure(bytes, E_NOTIMPL.0);
        return unsafe { null_compute(this, root.as_ref().map(|p| &p.native), desc, out) }.unwrap_or(E_NOTIMPL);
    }
    // Ray-tracing helper programs whose translated DXIL crashed the host
    // driver's shader compiler; substitute the fallback pipeline for them.
    const DRIVER_REFUSED: [&str; 3] = [
        "a206014458c885ecd189c1bbabd1a5e1415b4fa8198d8a63dae88e3d52c6406c",
        "df3e12f81ce9222b1df78cf72a0b1e7207fb8288d80321d90051e986a52b3579",
        "e708b4435708e2fcf221ca1840f819d5287a41ac3aa963ec090dcbbd5fa25f15",
    ];
    if let Some(digest) = super::shader_cache::source_digest(bytes)
        && DRIVER_REFUSED.contains(&digest.as_str())
    {
        crate::diagnostics::compute_failure(bytes, E_NOTIMPL.0);
        return unsafe { null_compute(this, root.as_ref().map(|p| &p.native), desc, out) }.unwrap_or(E_NOTIMPL);
    }
    // Optional policy for boot experiments: console programs above a size limit
    // take the host backend minutes to translate; skip straight to the fallback
    // pipeline instead of blocking the title on them.
    if let Some(kilobytes) = std::env::var("TUXBOX_SKIP_CONSOLE_SHADER_KB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value != 0)
        && bytes.len() > kilobytes.saturating_mul(1024)
        && super::shader_cache::is_xbox_native(bytes)
    {
        crate::diagnostics::log(&format!(
            "skipping {}-byte console program under the large-shader policy",
            bytes.len()
        ));
        crate::diagnostics::compute_failure(bytes, E_NOTIMPL.0);
        return unsafe { null_compute(this, root.as_ref().map(|p| &p.native), desc, out) }.unwrap_or(E_NOTIMPL);
    }
    use super::compute_table::{Program, TableLayout};
    let layout = match root.map(|r| super::compute_table::reflected(
        &this.native, &r.state, Program::Compute(bytes), desc.NodeMask,
    )).transpose() {
        Ok(layout) => layout,
        Err(error) => return error.code(),
    };
    let (translation, reflected) = match layout {
        Some(TableLayout::Reflected(table)) => {
            // Embedded RTS0 cannot be replaced. The translator reads that original
            // root and fills only the slots the appended signature adds.
            let code = super::shader_cache::compute_with_root(bytes, table.layout.payload())
                .or_else(|| super::shader_cache::prepared(bytes));
            let Some(code) = code else {
                super::compute_table::capture_stage_root(bytes, &table.original);
                crate::diagnostics::compute_failure(bytes, E_NOTIMPL.0);
                return unsafe { null_compute(this, root.map(|p| &p.native), desc, out) }.unwrap_or(E_NOTIMPL);
            };
            (Some(code), Some(table))
        }
        Some(TableLayout::Original) => (root.and_then(|r| {
            super::shader_cache::compute_with_root(bytes, r.state.payload())
        }), None),
        _ => (super::shader_cache::prepared(bytes).or_else(|| {
            super::shader_cache::compute_with_root(bytes, root?.state.payload())
        }), None),
    };
    let scene_root = if let Some(translated) = &translation {
        match super::compute_scene::private_root(
            &this.native,
            bytes,
            translated,
            root.map(|p| &p.state),
            desc.NodeMask,
        ) {
            Ok(root) => root,
            Err(e) => return e.code(),
        }
    } else {
        None
    };
    if translation.is_none() && !super::shader_cache::host_compatible(bytes) {
        // No host translation exists for this console-native container (console
        // DXIL, or an Xbox program stripped to XDAT/NOBC): the host compiler
        // cannot consume it and faults on it, so use the fallback pipeline
        // instead of handing the original bytes over.
        crate::diagnostics::compute_failure(bytes, E_NOTIMPL.0);
        return unsafe { null_compute(this, root.as_ref().map(|p| &p.native), desc, out) }.unwrap_or(E_NOTIMPL);
    }
    let code = translation.as_deref().unwrap_or(bytes);
    // Very large translated kernels take the host driver's backend minutes and
    // can fault it outright; keep the title running with the fallback pipeline.
    // Very large translated programs are usable once the translator emits them
    // compactly; the limit only guards the host backend against absurd sizes.
    let max_host_code = std::env::var("TUXBOX_MAX_HOST_CODE_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1024 * 1024);
    if max_host_code != 0 && code.len() > max_host_code {
        crate::diagnostics::log(&format!(
            "substituted oversized compute program ({} bytes of DXIL) to keep the host compiler alive",
            code.len()
        ));
        crate::diagnostics::compute_failure(bytes, E_NOTIMPL.0);
        return unsafe { null_compute(this, root.as_ref().map(|p| &p.native), desc, out) }.unwrap_or(E_NOTIMPL);
    }
    // Translated programs whose DXIL faults the host driver's shader compiler;
    // the pipeline is replaced by the fallback one instead of being submitted.
    const DRIVER_REFUSED_DXIL: [&str; 7] = [
        "1a4cc49811c9ba8a9a629dd89b66cebc1354b0474379db7410019032ba44da06",
        "e951f7018607ad7f5c3abac8bc625c21acac087010ecd07bc70be1f5638c280e",
        "c93880131432c6b01b3b43a49fefeeb38f96542feaf3c371c6c98127c32803db",
        "3e0a42e6f112262c31133d592cbc2e8e00f82bf2ba423d77c207c11048459c04",
        "4553864955f532c9cb654aeece3170b8f3c62e6e33da3e72c50b8fb9ff1f6561",
        "6577a0c9c6cd7cc2857c96855661fa790950b5aea62d5ad0e2a7ef9e626093f6",
        // Variant discovered by cptest: the host backend faults on this translation.
        "5cce092978870dd9b0c5715f90b1c5e6cd46bde85aef464a9bb7d4b76102a50e",
    ];
    if let Some(digest) = super::shader_cache::source_digest(code)
        && DRIVER_REFUSED_DXIL.contains(&digest.as_str())
    {
        crate::diagnostics::compute_failure(bytes, E_NOTIMPL.0);
        return unsafe { null_compute(this, root.as_ref().map(|p| &p.native), desc, out) }.unwrap_or(E_NOTIMPL);
    }
    let label = super::shader_cache::source_digest(bytes)
        .map(|hash| format!("CS {}", &hash[..8]))
        .unwrap_or_else(|| "CS".to_string());
    let mut host = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(
            scene_root
                .as_ref()
                .map(|s| &s.native)
                .or_else(|| reflected.as_ref().map(|t| &t.native))
                .or(root.map(|p| &p.native))
                .cloned(),
        ),
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: code.as_ptr().cast(),
            BytecodeLength: code.len(),
        },
        NodeMask: desc.NodeMask,
        Flags: desc.Flags,
        ..Default::default()
    };
    // SAFETY: native creation consumes the descriptor synchronously. The owned
    // clone in the C layout must be released explicitly on success and failure.
    let create = |descriptor: &D3D12_COMPUTE_PIPELINE_STATE_DESC| crate::big_stack::run_logged(&label, || unsafe {
        this.native
            .CreateComputePipelineState::<ID3D12PipelineState>(descriptor)
    });
    let Some(_permit) = CompilePermit::acquire() else {
        // Too many concurrent compiles: keep the title running with a substitute.
        crate::diagnostics::compute_failure(bytes, E_NOTIMPL.0);
        return unsafe { null_compute(this, root.as_ref().map(|p| &p.native), desc, out) }.unwrap_or(E_NOTIMPL);
    };
    let mut native = create(&host);
    if translation.is_some()
        && reflected.is_none()
        && scene_root.is_none()
        && native.is_err()
        && super::shader_cache::host_compatible(bytes)
    {
        // Structural checks cannot validate DXIL instructions or root bindings.
        // Let the adapter handle the original guest shader if it rejects the
        // cache, but only when the guest bytecode is host-parseable: retrying a
        // console-native container faults inside the host compiler.
        host.CS = desc.CS;
        native = create(&host);
    }
    unsafe { ManuallyDrop::drop(&mut host.pRootSignature) };
    if let Err(error) = &native {
        if let Some(root) = root { super::compute_table::capture_stage_root(bytes, &root.state); }
        if let Ok(dump) = std::env::var("TUXBOX_SHADER_DUMP") {
            let _ = std::fs::write(format!("{dump}.dxil"), code);
            let _ = std::fs::write(format!("{dump}.src"), bytes);
        }
        crate::diagnostics::compute_failure(bytes, error.code().0);
        if let Some(status) = unsafe { null_compute(this, root.map(|p| &p.native), desc, out) } {
            return status;
        }
    } else if let Some(table) = &reflected {
        static FIRST: std::sync::Once = std::sync::Once::new();
        crate::diagnostics::log_once(&FIRST, || format!(
            "created compute pipeline with {} reflected FSSM descriptor slots", table.lengths.iter().map(|&v| u64::from(v)).sum::<u64>()
        ));
    }
    unsafe {
        wrap_pageable(
            native,
            &PIPELINE,
            this,
            PipelineData {
                scene_root,
                reflected,
                ..Default::default()
            },
            out,
        )
    }
}

/// Create a resource-free compute pipeline on the guest root (see `fallback`).
unsafe fn null_compute(
    this: &Device, root: Option<&ID3D12RootSignature>, desc: &D3D12_COMPUTE_PIPELINE_STATE_DESC, out: Out,
) -> Option<HRESULT> {
    use super::fallback::{Stage, bytecode, enabled, shader, substituted};
    if !enabled() { return None; }
    let mut host = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(root.cloned()),
        CS: bytecode(shader(Stage::Compute)?),
        NodeMask: desc.NodeMask,
        Flags: desc.Flags,
        ..Default::default()
    };
    let native = crate::big_stack::run_logged("null CS", || unsafe { this.native.CreateComputePipelineState::<ID3D12PipelineState>(&host) });
    unsafe { ManuallyDrop::drop(&mut host.pRootSignature) };
    if native.is_err() { return None; }
    if let Ok(pipeline) = &native {
        let code = unsafe { std::slice::from_raw_parts(desc.CS.pShaderBytecode.cast::<u8>(), desc.CS.BytecodeLength) };
        if crate::diagnostics::display::enabled() && let Some(digest) = super::shader_cache::source_digest(code) {
            crate::diagnostics::display::mark_null_compute(pipeline.as_raw(), digest);
        }
    }
    substituted("compute");
    Some(unsafe { wrap_pageable(native, &PIPELINE, this, PipelineData::default(), out) })
}
