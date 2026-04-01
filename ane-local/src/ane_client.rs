use objc2::rc::Retained;
use objc2::runtime::NSObject;
use objc2::{extern_class, extern_conformance, msg_send, ClassType};
use objc2_foundation::{NSDictionary, NSError, NSObjectProtocol, NSString};

use crate::ane_in_memory_model::ANEInMemoryModel;
use crate::ane_request::ANERequest;

extern_class!(
    #[unsafe(super(NSObject))]
    #[name = "_ANEClient"]
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub(crate) struct ANEClient;
);

extern_conformance!(
    unsafe impl NSObjectProtocol for ANEClient {}
);

impl ANEClient {
    /// Get the shared ANE client connection (singleton).
    pub fn shared_connection() -> Option<Retained<ANEClient>> {
        unsafe { msg_send![Self::class(), sharedConnection] }
    }

    /// Evaluate directly, bypassing the ANE daemon XPC path.
    pub fn evaluate_direct(
        &self,
        model: &ANEInMemoryModel,
        request: &ANERequest,
        qos: u32,
    ) -> Result<(), Retained<NSError>> {
        let opts: Retained<NSDictionary<NSString, NSObject>> = NSDictionary::new();
        unsafe {
            msg_send![self,
                doEvaluateDirectWithModel: model,
                options: &*opts,
                request: request,
                qos: qos,
                error: _]
        }
    }
}
