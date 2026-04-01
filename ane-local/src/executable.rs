use std::cell::UnsafeCell;

use objc2::rc::Retained;
use objc2_foundation::NSQualityOfService;

use crate::ane_client::ANEClient;
use crate::ane_in_memory_model::ANEInMemoryModel;
use crate::ane_performance_stats::ANEPerformanceStats;
use crate::request::Request;
use crate::tensor_data::TensorData;
use crate::Error;

/// A compiled, loaded ANE program ready for repeated evaluation.
///
/// Obtained from [`Graph::compile`](crate::Graph::compile).
/// Automatically unloads from ANE hardware on drop.
pub struct Executable {
    pub(crate) inner: Retained<ANEInMemoryModel>,
    pub(crate) qos: NSQualityOfService,
    /// Cached request for `run_cached`. Lazily initialized on first call.
    pub(crate) cached_request: UnsafeCell<Option<Request>>,
    /// Cached stats-enabled request for `run_cached_with_stats`.
    pub(crate) stats_request: UnsafeCell<Option<Request>>,
    /// Cached perf stats object (attached to stats_request).
    pub(crate) perf_stats: UnsafeCell<Option<Retained<ANEPerformanceStats>>>,
}

unsafe impl Send for Executable {}
unsafe impl Sync for Executable {}

impl Executable {
    /// Run the compiled program on the ANE.
    ///
    /// `inputs` and `outputs` are positional [`TensorData`] arrays matching the
    /// order of [`placeholder`](crate::Graph::placeholder) calls and output tensors
    /// in the graph.
    pub fn run(&self, inputs: &[&TensorData], outputs: &[&TensorData]) -> Result<(), Error> {
        let input_surfaces: Vec<&objc2_io_surface::IOSurface> = inputs
            .iter()
            .map(|tensor_data| tensor_data.surface())
            .collect();
        let output_surfaces: Vec<&objc2_io_surface::IOSurface> = outputs
            .iter()
            .map(|tensor_data| tensor_data.surface())
            .collect();
        let request = Request::new(&input_surfaces, &output_surfaces)?;
        self.inner
            .evaluate(self.qos, &request.inner)
            .map_err(|error| Error::Evaluate(error.localizedDescription().to_string()))
    }

    /// Run the compiled program, caching the ANE request object for reuse.
    ///
    /// On the first call, creates and caches an `_ANERequest` from the given
    /// IOSurfaces. Subsequent calls reuse the cached request, saving ~0.095ms
    /// of Objective-C object allocation per dispatch.
    ///
    /// # Safety requirement
    /// The caller must pass the **same** `inputs` and `outputs` TensorData on
    /// every call. The IOSurface backing buffers may be mutated between calls
    /// (that's the dynamic-weight pattern), but the TensorData/IOSurface
    /// objects themselves must be the same.
    pub fn run_cached(&self, inputs: &[&TensorData], outputs: &[&TensorData]) -> Result<(), Error> {
        // SAFETY: Executable is not Sync for concurrent mutation — single-threaded
        // access to cached_request is guaranteed by the borrow of &self in the
        // training loop (one dispatch at a time).
        let cached = unsafe { &mut *self.cached_request.get() };
        if cached.is_none() {
            let input_surfaces: Vec<&objc2_io_surface::IOSurface> =
                inputs.iter().map(|td| td.surface()).collect();
            let output_surfaces: Vec<&objc2_io_surface::IOSurface> =
                outputs.iter().map(|td| td.surface()).collect();
            *cached = Some(Request::new(&input_surfaces, &output_surfaces)?);
        }
        let request = cached.as_ref().unwrap();
        self.inner
            .evaluate(self.qos, &request.inner)
            .map_err(|error| Error::Evaluate(error.localizedDescription().to_string()))
    }

    /// Run the compiled program with hardware performance stats collection.
    ///
    /// Returns `hw_execution_time_ns`. The hw time is the actual
    /// nanoseconds spent executing on the ANE hardware, excluding XPC overhead.
    /// Returns 0 if the runtime doesn't populate perf stats.
    ///
    /// Uses the same cached request as `run_cached`. Sets perfStatsMask on first call.
    pub fn run_cached_with_stats(
        &self,
        inputs: &[&TensorData],
        outputs: &[&TensorData],
    ) -> Result<u64, Error> {
        let cached_req = unsafe { &mut *self.stats_request.get() };
        let cached_stats = unsafe { &mut *self.perf_stats.get() };

        if cached_req.is_none() {
            // Enable hardware timing collection (bit 0 = hw execution time)
            self.inner.set_perf_stats_mask(0x1);

            // Create a blank stats object and attach it to the request
            let stats = ANEPerformanceStats::new().ok_or(Error::Evaluate(
                "failed to create ANEPerformanceStats".into(),
            ))?;
            let input_surfaces: Vec<&objc2_io_surface::IOSurface> =
                inputs.iter().map(|td| td.surface()).collect();
            let output_surfaces: Vec<&objc2_io_surface::IOSurface> =
                outputs.iter().map(|td| td.surface()).collect();
            let req = Request::new(&input_surfaces, &output_surfaces)?;
            req.attach_perf_stats(&stats);
            *cached_req = Some(req);
            *cached_stats = Some(stats);
        }

        let request = cached_req.as_ref().unwrap();

        self.inner
            .evaluate(self.qos, &request.inner)
            .map_err(|error| Error::Evaluate(error.localizedDescription().to_string()))?;

        // Read hw time from the stats object attached to the request
        Ok(request.hw_execution_time())
    }

    /// Run the compiled program using direct evaluation (XPC bypass).
    ///
    /// Uses `_ANEClient.doEvaluateDirectWithModel:` instead of the daemon path.
    /// Same caching semantics as `run_cached`.
    pub fn run_cached_direct(
        &self,
        inputs: &[&TensorData],
        outputs: &[&TensorData],
    ) -> Result<(), Error> {
        let cached = unsafe { &mut *self.cached_request.get() };
        if cached.is_none() {
            let input_surfaces: Vec<&objc2_io_surface::IOSurface> =
                inputs.iter().map(|td| td.surface()).collect();
            let output_surfaces: Vec<&objc2_io_surface::IOSurface> =
                outputs.iter().map(|td| td.surface()).collect();
            *cached = Some(Request::new(&input_surfaces, &output_surfaces)?);
        }
        let request = cached.as_ref().unwrap();
        let client = ANEClient::shared_connection().ok_or(Error::Evaluate(
            "failed to get ANEClient shared connection".into(),
        ))?;
        client
            .evaluate_direct(&self.inner, &request.inner, self.qos.0 as u32)
            .map_err(|error| Error::Evaluate(error.localizedDescription().to_string()))
    }
}

impl Drop for Executable {
    fn drop(&mut self) {
        self.inner.unload(self.qos);
    }
}
