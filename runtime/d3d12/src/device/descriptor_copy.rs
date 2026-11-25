//! CPU descriptor handles returned by our heaps are native on this device.
use super::Device;
use core::slice;
use windows::Win32::Graphics::Direct3D12::*;

struct Ranges<'a> {
    starts: &'a [D3D12_CPU_DESCRIPTOR_HANDLE],
    sizes: Option<&'a [u32]>,
}

impl<'a> Ranges<'a> {
    // SAFETY: the guest owns count readable entries in each supplied array
    // for the duration of the synchronous device call.
    unsafe fn read(
        count: u32,
        starts: *const D3D12_CPU_DESCRIPTOR_HANDLE,
        sizes: *const u32,
    ) -> Option<Self> {
        if count == 0 {
            return Some(Self {
                starts: &[],
                sizes: None,
            });
        }
        if starts.is_null() {
            return None;
        }
        let count = count as usize;
        Some(Self {
            starts: unsafe { slice::from_raw_parts(starts, count) },
            sizes: (!sizes.is_null()).then(|| unsafe { slice::from_raw_parts(sizes, count) }),
        })
    }

    fn spans(&self) -> impl Iterator<Item = (usize, u32)> + '_ {
        self.starts
            .iter()
            .enumerate()
            .map(|(i, handle)| (handle.ptr, self.sizes.map_or(1, |sizes| sizes[i])))
    }

    fn count(&self, stride: usize) -> Option<u64> {
        self.spans()
            .filter(|(_, size)| *size > 0)
            .try_fold(0u64, |total, (start, size)| {
                if start == 0 {
                    return None;
                }
                start.checked_add(size as usize * stride)?;
                Some(total + u64::from(size))
            })
    }

    fn handles(&self, stride: usize) -> impl Iterator<Item = D3D12_CPU_DESCRIPTOR_HANDLE> + '_ {
        self.spans().flat_map(move |(start, size)| {
            (0..size).map(move |i| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start + i as usize * stride,
            })
        })
    }
}

/// Strips marker bits only the title sets above the descriptor address.
fn ptr_norm(handle: D3D12_CPU_DESCRIPTOR_HANDLE) -> D3D12_CPU_DESCRIPTOR_HANDLE {
    D3D12_CPU_DESCRIPTOR_HANDLE {
        ptr: handle.ptr & 0x00ff_ffff_ffff_ffff,
    }
}

fn plausible_handle(handle: usize) -> bool {
    (0x1_0000..0x0000_8000_0000_0000).contains(&handle)
}

pub(super) unsafe extern "system" fn ranges(
    device: &Device,
    dst_count: u32,
    dst: *const D3D12_CPU_DESCRIPTOR_HANDLE,
    dst_sizes: *const u32,
    src_count: u32,
    src: *const D3D12_CPU_DESCRIPTOR_HANDLE,
    src_sizes: *const u32,
    kind: D3D12_DESCRIPTOR_HEAP_TYPE,
) {
    // The console's type 4 heap is a CBV/SRV/UAV heap on the host.
    let kind = super::host_heap_type(kind);
    // SetUAVs copies N independent sources to one contiguous destination.
    // A size array can be omitted independently on either side (one per range).
    if (0..D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES.0).contains(&kind.0)
        && let (Some(to), Some(from)) =
            (unsafe { Ranges::read(dst_count, dst, dst_sizes) }, unsafe {
                Ranges::read(src_count, src, src_sizes)
            })
    {
        let stride = unsafe { device.native.GetDescriptorHandleIncrementSize(kind) } as usize;
        if stride != 0
            && let Some(count) = to.count(stride)
            && Some(count) == from.count(stride)
        {
            if kind == D3D12_DESCRIPTOR_HEAP_TYPE_DSV {
                device
                    .state
                    .depth_views
                    .copy(stride, to.spans(), from.spans());
            }
            for (destination, source) in to.handles(stride).zip(from.handles(stride)) {
                unsafe { copy_host_range(device, 1, destination, source, kind) };
            }
            return;
        }
    }
    // Translate the range starts before any native forwarding. Arena-backed
    // heaps start at a guest address, so forwarding it raw makes vkd3d decode a
    // bogus heap pointer and fault. Untranslatable ranges are dropped.
    let registry = &device.state.descriptor_arenas;
    // The title may mark handles in bits above the address width.
    let normalize = |handle: D3D12_CPU_DESCRIPTOR_HANDLE| -> D3D12_CPU_DESCRIPTOR_HANDLE {
        ptr_norm(handle)
    };
    let mut dst_starts: Vec<D3D12_CPU_DESCRIPTOR_HANDLE> = Vec::new();
    let mut src_starts: Vec<D3D12_CPU_DESCRIPTOR_HANDLE> = Vec::new();
    let mut usable = true;
    if let Some(to) = unsafe { Ranges::read(dst_count, dst, dst_sizes) } {
        for start in to.starts {
            let handle = registry.copy_source(normalize(*start));
            usable &= plausible_handle(handle.ptr) || registry.find(handle.ptr).is_some();
            dst_starts.push(handle);
        }
    }
    if let Some(from) = unsafe { Ranges::read(src_count, src, src_sizes) } {
        for start in from.starts {
            let handle = registry.copy_source(normalize(*start));
            usable &= plausible_handle(handle.ptr) || registry.find(handle.ptr).is_some();
            src_starts.push(handle);
        }
    }
    if usable && dst_starts.len() == dst_count as usize && src_starts.len() == src_count as usize {
        unsafe {
            device.native.CopyDescriptors(
                dst_count,
                dst_starts.as_ptr(),
                Some(dst_sizes),
                src_count,
                src_starts.as_ptr(),
                Some(src_sizes),
                kind,
            );
        }
        return;
    }
    static SKIPS: std::sync::LazyLock<std::sync::Mutex<u32>> =
        std::sync::LazyLock::new(Default::default);
    let mut skips = SKIPS.lock().unwrap_or_else(|error| error.into_inner());
    *skips += 1;
    if skips.is_power_of_two() {
        crate::diagnostics::log(&format!(
            "descriptor range copy skipped: kind={} dst_count={dst_count} src_count={src_count}",
            kind.0
        ));
    }
}

pub(super) unsafe extern "system" fn simple(
    device: &Device,
    count: u32,
    dst: D3D12_CPU_DESCRIPTOR_HANDLE,
    src: D3D12_CPU_DESCRIPTOR_HANDLE,
    kind: D3D12_DESCRIPTOR_HEAP_TYPE,
) {
    let kind = super::host_heap_type(kind);
    if kind == D3D12_DESCRIPTOR_HEAP_TYPE_DSV {
        let stride = unsafe { device.native.GetDescriptorHandleIncrementSize(kind) } as usize;
        device.state.depth_views.copy(
            stride,
            std::iter::once((dst.ptr, count)),
            std::iter::once((src.ptr, count)),
        );
    }
    unsafe { copy_host_range(device, count, dst, src, kind); }
}

unsafe fn copy_host_range(device: &Device, count: u32, dst: D3D12_CPU_DESCRIPTOR_HANDLE, src: D3D12_CPU_DESCRIPTOR_HANDLE, kind: D3D12_DESCRIPTOR_HEAP_TYPE) {
    let registry = &device.state.descriptor_arenas;
    let arena = registry.find(dst.ptr);
    let translated = arena.as_ref().map_or(dst, |a| a.cpu(dst.ptr));
    if count > 1 && kind == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV && registry.find(src.ptr).is_some() {
        let stride = unsafe { device.native.GetDescriptorHandleIncrementSize(kind) } as usize;
        for i in 0..count as usize {
            let offset = i * stride;
            unsafe { copy_host_range(device, 1, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: dst.ptr + offset },
                D3D12_CPU_DESCRIPTOR_HANDLE { ptr: src.ptr + offset }, kind) };
        }
        return;
    }
    let from = registry.copy_source(src);
    if kind == D3D12_DESCRIPTOR_HEAP_TYPE_RTV {
        let stride = unsafe { device.native.GetDescriptorHandleIncrementSize(kind) } as usize;
        device.state.placements.targets.copy(stride, std::iter::once((dst.ptr, count)), std::iter::once((src.ptr, count)));
    }
    if kind == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV {
        let stride = unsafe { device.native.GetDescriptorHandleIncrementSize(kind) } as usize;
        let to = std::iter::once((translated.ptr, count));
        let source = std::iter::once((from.ptr, count));
        device.state.imported_views.copy(stride, to.clone(), source.clone());
        device.state.blas.views.copy(stride, to.clone(), source.clone());
        crate::diagnostics::display::copy_texture_views(translated.ptr, from.ptr, count, stride);
        device.state.placements.views.copy(stride, to, source);
    }
    // A handle the host cannot decode makes vkd3d compute a bogus descriptor
    // address and fault. Guest (console) handles are small values; host
    // descriptor addresses are ordinary pointers.
    let plausible = |handle: usize| (0x1_0000..0x0000_8000_0000_0000).contains(&handle);
    if !plausible(translated.ptr) || !plausible(from.ptr) {
        static SEEN: std::sync::LazyLock<std::sync::Mutex<std::collections::HashSet<String>>> =
            std::sync::LazyLock::new(Default::default);
        let key = format!("{}-{}-{}", kind.0, count > 1, arena.is_some());
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if seen.len() < 32 && seen.insert(key) {
            drop(seen);
            crate::diagnostics::log(&format!(
                "descriptor copy skipped: kind={} count={count} dst={:#x} src={:#x} arena={}",
                kind.0, translated.ptr, from.ptr, arena.is_some()
            ));
        }
        return;
    }
    unsafe { device.native.CopyDescriptorsSimple(count, translated, from, kind) };
    if let Some(arena) = arena {
        arena.author(dst.ptr, count);
        arena.publish(dst.ptr, count);
    }
}
