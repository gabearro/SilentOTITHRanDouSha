use objc2::rc::Retained;
use objc2::runtime::NSObject;
use objc2::{extern_class, extern_conformance, msg_send, ClassType};
use objc2_foundation::NSObjectProtocol;

extern_class!(
    #[unsafe(super(NSObject))]
    #[name = "_ANEPerformanceStats"]
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub(crate) struct ANEPerformanceStats;
);

extern_conformance!(
    unsafe impl NSObjectProtocol for ANEPerformanceStats {}
);

impl ANEPerformanceStats {
    /// Create a blank performance stats object.
    /// The ANE runtime populates it during evaluation when perfStatsMask is set.
    pub fn new() -> Option<Retained<ANEPerformanceStats>> {
        let zero: u64 = 0;
        unsafe { msg_send![Self::class(), statsWithHardwareExecutionNS: zero] }
    }

    /// Hardware execution time in nanoseconds, measured by the ANE hardware.
    /// Returns 0 if perf stats were not populated (perfStatsMask not set on model).
    pub fn hw_execution_time(&self) -> u64 {
        unsafe { msg_send![self, hwExecutionTime] }
    }
}
