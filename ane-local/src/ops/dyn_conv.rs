/// Dynamic-weight convolution (weights from a variable tensor, not constants).
///
/// Unlike `ConvOp`, the weights are NOT constant blobs — they come from a
/// runtime tensor (slice/reshape of the input IOSurface), enabling the
/// dynamic-weight training pattern used in rustane.
///
/// Supports arbitrary kernel sizes. For 1×1 kernels the weight tensor has
/// shape `[OC, IC, 1, 1]`. For general kernels it is `[OC, IC/groups, KH, KW]`.
///
/// MIL equivalent:
///   conv(x = acts_f16, weight = dyn_weights_f16, ...)
/// where dyn_weights_f16 is a reshaped slice of the input placeholder.
#[derive(Clone)]
pub struct DynConvOp {
    pub name: String,
    /// Source (activation) tensor name.
    pub source: String,
    /// Dynamic weight tensor name: `[OC, IC/groups, KH, KW]`.
    pub weight_source: String,
    /// Output tensor name.
    pub top: String,
    pub input_channels: usize,
    pub output_channels: usize,
    pub kernel_height: usize,
    pub kernel_width: usize,
    pub groups: usize,
    pub pad_mode: crate::PadMode,
}
