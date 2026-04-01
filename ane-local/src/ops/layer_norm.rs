/// Layer normalization — normalizes across specified axes.
///
/// Unlike `InstanceNormOp` (which normalizes per-channel across spatial dims),
/// `LayerNormOp` normalizes across arbitrary axes, typically the channel/hidden dimension.
///
/// MIL equivalent:
///   layer_norm(axes = [...], epsilon = eps, x = input)
/// Optionally with gamma/beta affine parameters.
#[derive(Clone)]
pub struct LayerNormOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    /// Axes to normalize over (e.g., [-1] for last dim, [1] for channels).
    pub axes: Box<[i32]>,
    pub epsilon: f64,
    /// If true, gamma/beta constants are included. If false, identity normalization.
    pub has_affine: bool,
    /// Gamma (scale) weight blob — only used if has_affine=true.
    pub gamma: Option<crate::ops::weights::WeightBlob>,
}
