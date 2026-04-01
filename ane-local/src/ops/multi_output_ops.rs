/// Ops that produce multiple output tensors.

/// split(x, axis, num_splits/split_sizes) — split tensor into parts
#[derive(Clone)]
pub struct SplitOp {
    pub name: String,
    pub bottom: String,
    pub tops: Box<[String]>,
    pub axis: i32,
    pub num_splits: usize,
    pub split_sizes: Option<Box<[i32]>>,
}

/// topk(x, k, axis, ascending) — top-k values and indices
#[derive(Clone)]
pub struct TopkOp {
    pub name: String,
    pub bottom: String,
    pub top_values: String,
    pub top_indices: String,
    pub k: i32,
    pub axis: i32,
    pub ascending: bool,
}

/// argsort(x, axis, ascending)
#[derive(Clone)]
pub struct ArgsortOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub axis: i32,
    pub ascending: bool,
}

/// reduce_argmax(x, axis, keep_dims)
#[derive(Clone)]
pub struct ReduceArgmaxOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub axis: i32,
    pub keep_dims: bool,
}

/// reduce_argmin(x, axis, keep_dims)
#[derive(Clone)]
pub struct ReduceArgminOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub axis: i32,
    pub keep_dims: bool,
}

/// select(cond, a, b) — where/conditional select
#[derive(Clone)]
pub struct SelectOp {
    pub name: String,
    pub cond: String,
    pub a: String,
    pub b: String,
    pub top: String,
}

/// l2_norm(x, epsilon)
#[derive(Clone)]
pub struct L2NormOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub epsilon: f64,
}

/// batch_norm(x, mean, variance, gamma, beta, epsilon)
#[derive(Clone)]
pub struct BatchNormOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub channels: usize,
    pub epsilon: f64,
    pub mean: crate::ops::weights::WeightBlob,
    pub variance: crate::ops::weights::WeightBlob,
    pub gamma: Option<crate::ops::weights::WeightBlob>,
    pub beta: Option<crate::ops::weights::WeightBlob>,
}

/// reduce_prod(x, axes, keep_dims)
#[derive(Clone)]
pub struct ReduceProdOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub axes: Box<[i32]>,
    pub keep_dims: bool,
}
