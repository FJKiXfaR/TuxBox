//! Host pipeline creation on a thread with a large stack.
//!
//! Proton's vkd3d-proton compiles DXIL to SPIR-V synchronously inside
//! Create*PipelineState, on the calling thread. Its control-flow analysis
//! recurses deeply on large translated shaders. Xbox titles create pipelines
//! on worker threads with 1 MiB stacks, so those threads overflowed their stack
//! inside the compiler and died (ProjectScarlett stayed on a black screen).

/// Reserved, not committed: untouched pages cost only address space.
const STACK_BYTES: usize = 64 << 20;

struct AssertSend<T>(T);
// SAFETY: D3D12 devices and the descriptors they read are free-threaded, and
// the caller blocks until the helper thread has finished with them.
unsafe impl<T> Send for AssertSend<T> {}

/// Run `job` on a scoped helper thread with a 64 MiB stack and return its
/// result. Falls back to the current thread if the helper cannot be created.
///
/// `label` is reported while `CreatePipelineState` is still inside the driver.
/// Cached creates return in milliseconds and stay quiet. A compile that is
/// still going after five seconds is logged, then again every fifteen seconds,
/// because that call produces no other log line.
pub(crate) fn run_logged<R, F: FnOnce() -> R>(label: &str, job: F) -> R {
    let _watch = PipelineWatch::start(label);
    let mut slot = Some(job);
    let pointer = AssertSend(&mut slot as *mut Option<F>);
    let result = std::thread::scope(|scope| {
        let spawned = std::thread::Builder::new()
            .name("tuxbox-pipeline".into())
            .stack_size(STACK_BYTES)
            .spawn_scoped(scope, move || {
                let pointer = pointer;
                // Record the OS thread that runs the host compiler so a crash
                // inside it can be matched to the pipeline being created.
                let tid = unsafe { windows::Win32::System::Threading::GetCurrentThreadId() };
                crate::diagnostics::log(&format!("pipeline job tid={tid:x} enter"));
                // SAFETY: the spawning thread waits for this scope to end
                // before reading `slot` again.
                let job = unsafe { (*pointer.0).take() }.expect("pipeline job already taken");
                AssertSend(job())
            });
        match spawned {
            Ok(handle) => Some(handle.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic)).0),
            Err(_) => None,
        }
    });
    match result {
        Some(value) => value,
        None => (slot.take().expect("pipeline job already taken"))(),
    }
}

struct Flight {
    id: u64,
    label: String,
    started: std::time::Instant,
    /// Seconds from the start at which the next progress line is due.
    next_due: u64,
}

struct PipelineWatch {
    id: Option<u64>,
    label: String,
    started: std::time::Instant,
}

impl PipelineWatch {
    fn start(label: &str) -> Self {
        if !crate::diagnostics::logging() {
            return Self { id: None, label: String::new(), started: std::time::Instant::now() };
        }
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        static WATCH: std::sync::Once = std::sync::Once::new();
        WATCH.call_once(|| {
            let _ = std::thread::Builder::new().name("tuxbox-pso-log".into()).spawn(watch);
        });
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let started = std::time::Instant::now();
        crate::diagnostics::log(&format!("CreatePipelineState #{id} {label} started"));
        flights().push(Flight { id, label: label.to_string(), started, next_due: 5 });
        Self { id: Some(id), label: label.to_string(), started }
    }
}

impl Drop for PipelineWatch {
    fn drop(&mut self) {
        let Some(id) = self.id else { return };
        let elapsed = self.started.elapsed().as_secs();
        flights().retain(|flight| flight.id != id);
        if elapsed >= 5 {
            crate::diagnostics::log(&format!(
                "CreatePipelineState #{id} {} finished in {elapsed}s",
                self.label
            ));
        }
    }
}

fn flights() -> std::sync::MutexGuard<'static, Vec<Flight>> {
    static FLIGHTS: std::sync::Mutex<Vec<Flight>> = std::sync::Mutex::new(Vec::new());
    FLIGHTS.lock().unwrap_or_else(|error| error.into_inner())
}

fn watch() {
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
        let mut notes = Vec::new();
        {
            let mut open = flights();
            for flight in open.iter_mut() {
                let elapsed = flight.started.elapsed().as_secs();
                if elapsed < flight.next_due {
                    continue;
                }
                notes.push((flight.id, format!(
                    "CreatePipelineState #{} {} still running {elapsed}s",
                    flight.id, flight.label
                )));
                flight.next_due = elapsed.saturating_add(15);
            }
        }
        for (id, note) in notes {
            // The compile can finish between the snapshot and this write.
            if flights().iter().any(|flight| flight.id == id) {
                crate::diagnostics::log(&note);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn deep_recursion_fits_on_the_helper_stack() {
        fn depth(n: u32) -> u32 {
            let pad = std::hint::black_box([0u8; 1024]);
            if n == 0 { pad[0] as u32 } else { depth(n - 1) + std::hint::black_box(1) }
        }
        // About 8 MiB of frames: more than a 1 MiB thread stack holds.
        assert_eq!(super::run_logged("test", || depth(8000)), 8000);
    }

    #[test]
    fn returns_borrowed_results() {
        let text = String::from("pso");
        assert_eq!(super::run_logged("test", || text.len()), 3);
    }
}
