/// Unary ops that map to single MIL operations with no weight blobs.
/// Each takes one input tensor and optional scalar/vector parameters.

/// gelu(x, mode) — Gaussian Error Linear Unit
#[derive(Clone)]
pub struct GeluOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    /// "EXACT", "TANH_APPROXIMATION", or "SIGMOID_APPROXIMATION"
    pub mode: String,
}

/// silu(x) — Sigmoid Linear Unit (x * sigmoid(x))
#[derive(Clone)]
pub struct SiluOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
}

/// clip(x, alpha=min, beta=max) — Clamp values to [alpha, beta]
#[derive(Clone)]
pub struct ClipOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub alpha: f32, // min
    pub beta: f32,  // max
}

/// relu6(x) — ReLU capped at 6
#[derive(Clone)]
pub struct Relu6Op {
    pub name: String,
    pub bottom: String,
    pub top: String,
}

/// clamped_relu(x, alpha, beta) — leaky relu with ceiling
#[derive(Clone)]
pub struct ClampedReluOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub alpha: f64, // negative slope
    pub beta: f64,  // ceiling
}

/// thresholded_relu(x, alpha) — zero below threshold
#[derive(Clone)]
pub struct ThresholdedReluOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub alpha: f64,
}

/// scaled_tanh(x, alpha, beta) — alpha * tanh(beta * x)
#[derive(Clone)]
pub struct ScaledTanhOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub alpha: f64,
    pub beta: f64,
}

/// prelu(x, alpha) — per-channel leaky relu
#[derive(Clone)]
pub struct PreluOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub channels: usize,
    pub alpha: crate::ops::weights::WeightBlob,
}

/// cast(x, dtype) — type conversion
#[derive(Clone)]
pub struct CastOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub dtype: String, // "fp16", "fp32", "int32", "bool"
}

/// cumsum(x, axis, exclusive, reverse)
#[derive(Clone)]
pub struct CumsumOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub axis: i32,
    pub exclusive: bool,
    pub reverse: bool,
}

/// tile(x, reps)
#[derive(Clone)]
pub struct TileOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub reps: Box<[i32]>,
}

/// expand_dims(x, axes)
#[derive(Clone)]
pub struct ExpandDimsOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub axes: Box<[i32]>,
}

/// squeeze(x, axes)
#[derive(Clone)]
pub struct SqueezeOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub axes: Option<Box<[i32]>>,
}
