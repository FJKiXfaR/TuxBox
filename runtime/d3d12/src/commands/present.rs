//! Presenting the title's frames.
use super::*;

pub(super) unsafe extern "system" fn present(
    queue: &Queue,
    count: u32,
    planes: *const PresentPlane,
    params: *const PresentParameters,
) -> HRESULT {
    let result = unsafe { present_planes(queue, count, planes, params) };
    if result.is_err() {
        static FIRST: std::sync::Once = std::sync::Once::new();
        crate::diagnostics::log_once(&FIRST, || format!("PresentX failed planes={count} HRESULT={:#010x}", result.0));
        if result == windows::Win32::Graphics::Dxgi::DXGI_ERROR_ACCESS_LOST {
            // The host window association was lost (the title recreated or hid
            // its window, the display mode changed, or a desktop switch). A
            // console title never closes a desktop window: drop the host window
            // and swap chain so the next present builds a fresh one, and keep
            // the title running; it may still be in its loading phase.
            static LOST: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let lost = LOST.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            crate::diagnostics::log(&format!(
                "presentation access lost; recreating host window/swap chain (#{})",
                lost + 1
            ));
            queue.state.presentation.reset();
        }
        // A rejected present still ends this frame for the title's pacing.
        if count >= 1 && let Some(plane) = unsafe { planes.as_ref() } {
            unsafe { device::frame_events::display_flip(parent(&queue.parent), plane.token) };
        }
    }
    result
}

/// Say every ten seconds how many frames reached the screen.
pub(super) fn frame_heartbeat(token: u64) {
    static FRAMES: Mutex<Option<(std::time::Instant, u64, u64)>> = Mutex::new(None);
    let mut state = FRAMES.lock().unwrap_or_else(|e| e.into_inner());
    let (since, recent, total) = state.get_or_insert_with(|| (std::time::Instant::now(), 0, 0));
    *recent += 1;
    *total += 1;
    let elapsed = since.elapsed().as_secs_f32();
    if elapsed >= 10.0 {
        crate::diagnostics::log(&format!(
            "presented {recent} frames in {elapsed:.0}s, {total} in total, token={token}"
        ));
        *since = std::time::Instant::now();
        *recent = 0;
    }
}

pub(super) unsafe fn present_planes(
    queue: &Queue, count: u32, planes: *const PresentPlane, params: *const PresentParameters,
) -> HRESULT {
    let mut submitted = queue
        .state
        .submission
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    *submitted = true;
    let hr = HRESULT(queue.state.placement.error.load(Ordering::Acquire));
    if hr.is_err() {
        return hr;
    }
    if queue.state.kind != D3D12_COMMAND_LIST_TYPE_DIRECT {
        return E_NOTIMPL;
    }
    if count == 0 || planes.is_null() {
        return E_INVALIDARG;
    }
    if count != 1 {
        return E_NOTIMPL;
    }
    let plane = unsafe { &*planes };
    if plane.resource_count == 0 || plane.resources.is_null() {
        return E_INVALIDARG;
    }
    if plane.resource_count != 1 {
        return E_NOTIMPL;
    }
    let Some(source) = (unsafe { device::resource(*plane.resources, parent(&queue.parent)) })
    else {
        return E_INVALIDARG;
    };
    if let Some(params) = unsafe { params.as_ref() } {
        if !(0.0..=100.0).contains(&params.immediate_threshold_percent) {
            return E_INVALIDARG;
        }
        let full = unsafe { &*planes.cast::<RetailPresentPlane>() };
        let desc = unsafe { source.GetDesc() };
        let rect = unsafe { full.source_rects.as_ref() };
        let destination = unsafe { full.destination.as_ref() };
        static FIRST: std::sync::Once = std::sync::Once::new();
        crate::diagnostics::log_once(&FIRST, || format!(
            "present plane {}x{} source {rect:?} destination {destination:?} color {} parameters {:?}",
            desc.Width, desc.Height, full.color_space, params.options
        ));
        if rect.is_some_and(|r| r.left != 0 || r.top != 0 || r.right as u64 != desc.Width || r.bottom as u32 != desc.Height)
            || destination.is_some_and(|r| *r != [0.0, 0.0, 1.0, 1.0])
            || full.color_space != 0 || full.remaining != [0; 7] || params.options != [0; 7] {
            return E_NOTIMPL;
        }
    }
    // The finished frame usually lands in a placed render target while the
    // present names the committed display surface. Show the render target the
    // title actually drew into when its shape matches (see present::redirect).
    static PRESENT_CALLERS: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
    if let Some(rva) = crate::survey::title_rva() {
        let mut seen = PRESENT_CALLERS.lock().unwrap_or_else(|e| e.into_inner());
        if seen.len() < 8 && !seen.contains(&rva) {
            seen.push(rva);
            drop(seen);
            crate::diagnostics::log(&format!("PresentX submitted from title RVA {rva:#x}"));
        }
    }
    let redirected = crate::present::redirect::source(&source);
    static SOURCE_NOTE: std::sync::Once = std::sync::Once::new();
    crate::diagnostics::log_once(&SOURCE_NOTE, || format!(
        "present names {:p}; redirect {}; drawn display {}; last render target {}",
        source.as_raw(),
        redirected.as_ref().map(|r| format!("{:p}", r.as_raw())).unwrap_or_else(|| "none".into()),
        crate::present::redirect::drawn_display()
            .map(|r| format!("{:p}", r.as_raw())).unwrap_or_else(|| "none".into()),
        crate::present::redirect::last_target()
            .map(|r| format!("{:p}", r.as_raw())).unwrap_or_else(|| "none".into()),
    ));
    if redirected.is_some() {
        static FIRST: std::sync::Once = std::sync::Once::new();
        crate::diagnostics::log_once(&FIRST, || {
            "presenting the title's full-size render target in place of the display surface".into()
        });
    }
    let result = queue
        .state
        .presentation
        .present(&queue.native, redirected.as_ref().unwrap_or(&source), Some(&source));
    if result == S_OK {
        frame_heartbeat(plane.token);
    }
    // Frame pacing must advance even if the host present failed; the title
    // otherwise waits forever for this token's flip.
    unsafe { device::frame_events::display_flip(parent(&queue.parent), plane.token) };
    if result == S_OK {
        static COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        static EARLY: [std::sync::Once;4] = [const { std::sync::Once::new() };4];
        if COUNT.load(Ordering::Relaxed) < EARLY.len() {
            let index = COUNT.fetch_add(1, Ordering::Relaxed);
            if let Some(site) = EARLY.get(index) {
                crate::diagnostics::log_once(site, || format!("completed PresentX #{} token={} source={:p}",index+1,plane.token,unsafe { *plane.resources }));
            }
        }
    }
    result
}
