use crate::ops::activation::ActivationOp;
use crate::ops::concat::ConcatOp;
use crate::ops::constant::ConstantOp;
use crate::ops::conv::ConvOp;
use crate::ops::deconv::DeconvOp;
use crate::ops::dyn_conv::DynConvOp;
use crate::ops::elementwise::ElementwiseOp;
use crate::ops::flatten::FlattenOp;
use crate::ops::inner_product::InnerProductOp;
use crate::ops::instance_norm::InstanceNormOp;
use crate::ops::layer_norm::LayerNormOp;
use crate::ops::matmul::MatmulOp;
use crate::ops::multi_output_ops::*;
use crate::ops::padding::PaddingOp;
use crate::ops::pooling::PoolingOp;
use crate::ops::reduction::ReductionOp;
use crate::ops::reshape::ReshapeOp;
use crate::ops::scalar::ScalarOp;
use crate::ops::slice::SliceBySizeOp;
use crate::ops::softmax::SoftmaxOp;
use crate::ops::transpose::TransposeOp;
use crate::ops::unary_ops::*;

#[derive(Clone)]
pub enum Op {
    Constant(ConstantOp),
    InnerProduct(InnerProductOp),
    Conv(ConvOp),
    Elementwise(ElementwiseOp),
    Activation(ActivationOp),
    Softmax(SoftmaxOp),
    Concat(ConcatOp),
    Reshape(ReshapeOp),
    InstanceNorm(InstanceNormOp),
    Pooling(PoolingOp),
    Deconv(DeconvOp),
    Padding(PaddingOp),
    Flatten(FlattenOp),
    Reduction(ReductionOp),
    Matmul(MatmulOp),
    Transpose(TransposeOp),
    SliceBySize(SliceBySizeOp),
    ScalarOp(ScalarOp),
    DynConv(DynConvOp),
    LayerNorm(LayerNormOp),
    // New native MIL ops
    Gelu(GeluOp),
    Silu(SiluOp),
    Clip(ClipOp),
    Relu6(Relu6Op),
    ClampedRelu(ClampedReluOp),
    ThresholdedRelu(ThresholdedReluOp),
    ScaledTanh(ScaledTanhOp),
    Prelu(PreluOp),
    Cast(CastOp),
    Cumsum(CumsumOp),
    Tile(TileOp),
    ExpandDims(ExpandDimsOp),
    Squeeze(SqueezeOp),
    Split(SplitOp),
    Topk(TopkOp),
    Argsort(ArgsortOp),
    ReduceArgmax(ReduceArgmaxOp),
    ReduceArgmin(ReduceArgminOp),
    Select(SelectOp),
    L2Norm(L2NormOp),
    BatchNorm(BatchNormOp),
    ReduceProd(ReduceProdOp),
}

impl Op {
    pub fn name(&self) -> &str {
        match self {
            Self::Constant(operation) => &operation.name,
            Self::InnerProduct(operation) => &operation.name,
            Self::Conv(operation) => &operation.name,
            Self::Elementwise(operation) => &operation.name,
            Self::Activation(operation) => &operation.name,
            Self::Softmax(operation) => &operation.name,
            Self::Concat(operation) => &operation.name,
            Self::Reshape(operation) => &operation.name,
            Self::InstanceNorm(operation) => &operation.name,
            Self::Pooling(operation) => &operation.name,
            Self::Deconv(operation) => &operation.name,
            Self::Padding(operation) => &operation.name,
            Self::Flatten(operation) => &operation.name,
            Self::Reduction(operation) => &operation.name,
            Self::Matmul(operation) => &operation.name,
            Self::Transpose(operation) => &operation.name,
            Self::SliceBySize(operation) => &operation.name,
            Self::ScalarOp(operation) => &operation.name,
            Self::DynConv(operation) => &operation.name,
            Self::LayerNorm(operation) => &operation.name,
            Self::Gelu(o) => &o.name,
            Self::Silu(o) => &o.name,
            Self::Clip(o) => &o.name,
            Self::Relu6(o) => &o.name,
            Self::ClampedRelu(o) => &o.name,
            Self::ThresholdedRelu(o) => &o.name,
            Self::ScaledTanh(o) => &o.name,
            Self::Prelu(o) => &o.name,
            Self::Cast(o) => &o.name,
            Self::Cumsum(o) => &o.name,
            Self::Tile(o) => &o.name,
            Self::ExpandDims(o) => &o.name,
            Self::Squeeze(o) => &o.name,
            Self::Split(o) => &o.name,
            Self::Topk(o) => &o.name,
            Self::Argsort(o) => &o.name,
            Self::ReduceArgmax(o) => &o.name,
            Self::ReduceArgmin(o) => &o.name,
            Self::Select(o) => &o.name,
            Self::L2Norm(o) => &o.name,
            Self::BatchNorm(o) => &o.name,
            Self::ReduceProd(o) => &o.name,
        }
    }

    pub fn top(&self) -> &str {
        match self {
            Self::Constant(operation) => &operation.top,
            Self::InnerProduct(operation) => &operation.top,
            Self::Conv(operation) => &operation.top,
            Self::Elementwise(operation) => &operation.top,
            Self::Activation(operation) => &operation.top,
            Self::Softmax(operation) => &operation.top,
            Self::Concat(operation) => &operation.top,
            Self::Reshape(operation) => &operation.top,
            Self::InstanceNorm(operation) => &operation.top,
            Self::Pooling(operation) => &operation.top,
            Self::Deconv(operation) => &operation.top,
            Self::Padding(operation) => &operation.top,
            Self::Flatten(operation) => &operation.top,
            Self::Reduction(operation) => &operation.top,
            Self::Matmul(operation) => &operation.top,
            Self::Transpose(operation) => &operation.top,
            Self::SliceBySize(operation) => &operation.top,
            Self::ScalarOp(operation) => &operation.top,
            Self::DynConv(operation) => &operation.top,
            Self::LayerNorm(operation) => &operation.top,
            Self::Gelu(o) => &o.top,
            Self::Silu(o) => &o.top,
            Self::Clip(o) => &o.top,
            Self::Relu6(o) => &o.top,
            Self::ClampedRelu(o) => &o.top,
            Self::ThresholdedRelu(o) => &o.top,
            Self::ScaledTanh(o) => &o.top,
            Self::Prelu(o) => &o.top,
            Self::Cast(o) => &o.top,
            Self::Cumsum(o) => &o.top,
            Self::Tile(o) => &o.top,
            Self::ExpandDims(o) => &o.top,
            Self::Squeeze(o) => &o.top,
            Self::Split(o) => o.tops.first().map(|s| s.as_str()).unwrap_or(""),
            Self::Topk(o) => &o.top_values,
            Self::Argsort(o) => &o.top,
            Self::ReduceArgmax(o) => &o.top,
            Self::ReduceArgmin(o) => &o.top,
            Self::Select(o) => &o.top,
            Self::L2Norm(o) => &o.top,
            Self::BatchNorm(o) => &o.top,
            Self::ReduceProd(o) => &o.top,
        }
    }

    pub(crate) fn bottom_names(&self) -> Vec<&str> {
        match self {
            Self::Constant(_) => vec![],
            Self::Concat(l) => l.bottoms.iter().map(|string| string.as_str()).collect(),
            Self::Elementwise(l) => l.bottoms.iter().map(|string| string.as_str()).collect(),
            Self::Matmul(l) => vec![l.bottom_x.as_str(), l.bottom_y.as_str()],
            Self::InnerProduct(l) => vec![l.bottom.as_str()],
            Self::Conv(l) => vec![l.bottom.as_str()],
            Self::Deconv(l) => vec![l.bottom.as_str()],
            Self::Activation(l) => vec![l.bottom.as_str()],
            Self::Softmax(l) => vec![l.bottom.as_str()],
            Self::Reshape(l) => vec![l.bottom.as_str()],
            Self::InstanceNorm(l) => vec![l.bottom.as_str()],
            Self::Pooling(l) => vec![l.bottom.as_str()],
            Self::Padding(l) => vec![l.bottom.as_str()],
            Self::Flatten(l) => vec![l.bottom.as_str()],
            Self::Reduction(l) => vec![l.bottom.as_str()],
            Self::Transpose(l) => vec![l.bottom.as_str()],
            Self::SliceBySize(l) => vec![l.bottom.as_str()],
            Self::ScalarOp(l) => vec![l.bottom.as_str()],
            Self::DynConv(l) => vec![l.source.as_str(), l.weight_source.as_str()],
            Self::LayerNorm(l) => vec![l.bottom.as_str()],
            Self::Gelu(o) => vec![o.bottom.as_str()],
            Self::Silu(o) => vec![o.bottom.as_str()],
            Self::Clip(o) => vec![o.bottom.as_str()],
            Self::Relu6(o) => vec![o.bottom.as_str()],
            Self::ClampedRelu(o) => vec![o.bottom.as_str()],
            Self::ThresholdedRelu(o) => vec![o.bottom.as_str()],
            Self::ScaledTanh(o) => vec![o.bottom.as_str()],
            Self::Prelu(o) => vec![o.bottom.as_str()],
            Self::Cast(o) => vec![o.bottom.as_str()],
            Self::Cumsum(o) => vec![o.bottom.as_str()],
            Self::Tile(o) => vec![o.bottom.as_str()],
            Self::ExpandDims(o) => vec![o.bottom.as_str()],
            Self::Squeeze(o) => vec![o.bottom.as_str()],
            Self::Split(o) => vec![o.bottom.as_str()],
            Self::Topk(o) => vec![o.bottom.as_str()],
            Self::Argsort(o) => vec![o.bottom.as_str()],
            Self::ReduceArgmax(o) => vec![o.bottom.as_str()],
            Self::ReduceArgmin(o) => vec![o.bottom.as_str()],
            Self::Select(o) => vec![o.cond.as_str(), o.a.as_str(), o.b.as_str()],
            Self::L2Norm(o) => vec![o.bottom.as_str()],
            Self::BatchNorm(o) => vec![o.bottom.as_str()],
            Self::ReduceProd(o) => vec![o.bottom.as_str()],
        }
    }
}
