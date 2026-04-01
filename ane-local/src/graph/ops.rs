use crate::ops::weights::WeightBlob;
use crate::{
    ActivationMode, ActivationOp, ConcatOp, ConvOp, DeconvOp, ElementwiseOp, ElementwiseOpType,
    FlattenOp, InstanceNormOp, MatmulOp, Op, PadFillMode, PadMode, PaddingOp, PoolType, PoolingOp,
    ReductionMode, ReductionOp, ReshapeOp, Shape, SliceBySizeOp, SoftmaxOp, TransposeOp,
};

use super::descriptor::{Convolution2dDescriptor, ConvolutionTranspose2dDescriptor};
use super::tensor::Tensor;
use super::{Graph, GraphOp};

impl Graph {
    /// Declare a placeholder tensor fed via IOSurface at runtime.
    pub fn placeholder(&mut self, shape: Shape) -> Tensor {
        let tensor = self.alloc(shape);
        self.inputs.push(tensor);
        tensor
    }

    /// Create a constant tensor from f32 data (converted to fp16 internally).
    pub fn constant(&mut self, data: &[f32], shape: Shape) -> Tensor {
        let tensor = self.alloc(shape);
        self.constants
            .insert(tensor.id, (WeightBlob::from_f32(data).data, shape));
        tensor
    }

    /// Create a constant tensor from pre-encoded fp16 bytes.
    pub fn constant_with_f16_bytes(&mut self, data: &[u8], shape: Shape) -> Tensor {
        let tensor = self.alloc(shape);
        self.constants.insert(tensor.id, (data.into(), shape));
        tensor
    }

    /// Create a constant tensor filled with a scalar value.
    pub fn constant_with_scalar(&mut self, scalar: f32, shape: Shape) -> Tensor {
        let count = shape.total_elements();
        let data = vec![scalar; count];
        self.constant(&data, shape)
    }

    pub(crate) fn resolve_constant(&self, tensor: Tensor) -> WeightBlob {
        let (bytes, _) = self
            .constants
            .get(&tensor.id)
            .expect("tensor is not a constant");
        WeightBlob::from_f16_bytes(bytes.clone())
    }

    pub fn convolution_2d_1x1(
        &mut self,
        source: Tensor,
        weights: Tensor,
        bias: Option<Tensor>,
    ) -> Tensor {
        self.convolution_2d(source, weights, bias, &Convolution2dDescriptor::default())
    }

    /// Dynamic-weight 1x1 convolution.
    ///
    /// Unlike `convolution_2d_1x1`, the weights come from a runtime tensor
    /// (slice/reshape of input), not from constant data. Enables the dynamic-weight
    /// training pattern where weights are packed into the input IOSurface.
    ///
    /// `source`: activation tensor [1, IC, 1, SEQ]
    /// `weights`: dynamic weight tensor [OC, IC, 1, 1]
    ///   (weights.shape.batch = OC, weights.shape.channels = IC)
    /// Output: [1, OC, 1, SEQ]
    pub fn convolution_2d_1x1_dynamic(&mut self, source: Tensor, weights: Tensor) -> Tensor {
        let out_channels = weights.shape.batch;
        let output = self.alloc(Shape {
            batch: 1,
            channels: out_channels,
            height: 1,
            width: source.shape.width,
        });
        self.ops.push(GraphOp {
            op: Op::DynConv(crate::ops::dyn_conv::DynConvOp {
                name: Self::op_name(output),
                source: Self::tensor_name(source),
                weight_source: Self::tensor_name(weights),
                top: Self::tensor_name(output),
                input_channels: source.shape.channels,
                output_channels: out_channels,
                kernel_height: 1,
                kernel_width: 1,
                groups: 1,
                pad_mode: crate::PadMode::Valid,
            }),
            output,
        });
        output
    }

    /// Dynamic-weight convolution with arbitrary kernel size.
    ///
    /// Weights come from a runtime tensor (placeholder / IOSurface), not from
    /// constant data.
    ///
    /// **ANE hardware limitation (M1–M5):** The ANE compiler only accepts
    /// dynamic-weight convolutions with 1×1 kernels. Graphs with larger
    /// kernels compile at the MIL level but fail with `ANECCompile() FAILED`.
    /// For larger kernels, use constant-weight `convolution_2d()` instead.
    ///
    /// `source`: activation tensor `[1, IC/S, 1, S]`
    /// `weights`: dynamic weight tensor `[OC, IC/S, KH, KW]`
    ///   (weights.shape.batch = OC, weights.shape.channels = IC/S)
    /// Output shape depends on pad_mode:
    ///   Valid → `[1, OC, 1, 1]` when S == KW and KH == 1
    pub fn convolution_2d_dynamic(
        &mut self,
        source: Tensor,
        weights: Tensor,
        descriptor: &Convolution2dDescriptor,
    ) -> Tensor {
        let out_channels = weights.shape.batch;
        let kh = weights.shape.height;
        let kw = weights.shape.width;
        let (out_h, out_w) = match descriptor.pad_mode {
            crate::PadMode::Valid => (
                source.shape.height.saturating_sub(kh) + 1,
                source.shape.width.saturating_sub(kw) + 1,
            ),
            crate::PadMode::Same => (source.shape.height, source.shape.width),
        };
        let output = self.alloc(Shape {
            batch: 1,
            channels: out_channels,
            height: out_h,
            width: out_w,
        });
        self.ops.push(GraphOp {
            op: Op::DynConv(crate::ops::dyn_conv::DynConvOp {
                name: Self::op_name(output),
                source: Self::tensor_name(source),
                weight_source: Self::tensor_name(weights),
                top: Self::tensor_name(output),
                input_channels: source.shape.channels,
                output_channels: out_channels,
                kernel_height: kh,
                kernel_width: kw,
                groups: descriptor.groups,
                pad_mode: descriptor.pad_mode.clone(),
            }),
            output,
        });
        output
    }

    pub fn convolution_2d(
        &mut self,
        source: Tensor,
        weights: Tensor,
        bias: Option<Tensor>,
        descriptor: &Convolution2dDescriptor,
    ) -> Tensor {
        let out_channels = weights.shape.channels;
        let kernel_h = weights.shape.height;
        let kernel_w = weights.shape.width;
        let out_h = match descriptor.pad_mode {
            PadMode::Valid => source.shape.height.saturating_sub(kernel_h) + 1,
            PadMode::Same => source.shape.height,
        };
        let out_w = match descriptor.pad_mode {
            PadMode::Valid => source.shape.width.saturating_sub(kernel_w) + 1,
            PadMode::Same => source.shape.width,
        };
        let output = self.alloc(Shape {
            channels: out_channels,
            height: out_h,
            width: out_w,
            batch: 1,
        });
        self.ops.push(GraphOp {
            op: Op::Conv(ConvOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(source),
                top: Self::tensor_name(output),
                input_channels: source.shape.channels,
                output_channels: out_channels,
                kernel_height: kernel_h,
                kernel_width: kernel_w,
                groups: descriptor.groups,
                pad_mode: descriptor.pad_mode.clone(),
                pad_top: 0,
                pad_bottom: 0,
                pad_left: 0,
                pad_right: 0,
                weights: self.resolve_constant(weights),
                bias: bias.map(|b| self.resolve_constant(b)),
                fused_relu: false,
                fused_tanh: false,
            }),
            output,
        });
        output
    }

    pub fn convolution_transpose_2d(
        &mut self,
        source: Tensor,
        weights: Tensor,
        bias: Option<Tensor>,
        descriptor: &ConvolutionTranspose2dDescriptor,
    ) -> Tensor {
        let out_channels = weights.shape.channels;
        let kernel_h = weights.shape.height;
        let kernel_w = weights.shape.width;
        let out_h = source.shape.height * descriptor.stride_height;
        let out_w = source.shape.width * descriptor.stride_width;
        let output = self.alloc(Shape {
            channels: out_channels,
            height: out_h,
            width: out_w,
            batch: 1,
        });
        self.ops.push(GraphOp {
            op: Op::Deconv(DeconvOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(source),
                top: Self::tensor_name(output),
                input_channels: source.shape.channels,
                output_channels: out_channels,
                kernel_height: kernel_h,
                kernel_width: kernel_w,
                stride_height: descriptor.stride_height,
                stride_width: descriptor.stride_width,
                groups: descriptor.groups,
                pad_mode: descriptor.pad_mode.clone(),
                pad_top: 0,
                pad_bottom: 0,
                pad_left: 0,
                pad_right: 0,
                output_padding_height: 0,
                output_padding_width: 0,
                weights: self.resolve_constant(weights),
                bias: bias.map(|b| self.resolve_constant(b)),
                fused_relu: false,
                fused_tanh: false,
            }),
            output,
        });
        output
    }

    fn activation(&mut self, input: Tensor, mode: ActivationMode) -> Tensor {
        let output = self.alloc(input.shape);
        self.ops.push(GraphOp {
            op: Op::Activation(ActivationOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                mode,
            }),
            output,
        });
        output
    }

    pub fn relu(&mut self, input: Tensor) -> Tensor {
        self.activation(input, ActivationMode::Relu)
    }

    pub fn tanh(&mut self, input: Tensor) -> Tensor {
        self.activation(input, ActivationMode::Tanh)
    }

    pub fn sigmoid(&mut self, input: Tensor) -> Tensor {
        self.activation(input, ActivationMode::Sigmoid)
    }

    pub fn leaky_relu(&mut self, input: Tensor, negative_slope: f64) -> Tensor {
        self.activation(input, ActivationMode::LeakyRelu { negative_slope })
    }

    pub fn elu(&mut self, input: Tensor, alpha: f64) -> Tensor {
        self.activation(input, ActivationMode::Elu { alpha })
    }

    pub fn hard_sigmoid(&mut self, input: Tensor, alpha: f64, beta: f64) -> Tensor {
        self.activation(input, ActivationMode::SigmoidHard { alpha, beta })
    }

    pub fn linear(&mut self, input: Tensor, alpha: f64, beta: f64) -> Tensor {
        self.activation(input, ActivationMode::Linear { alpha, beta })
    }

    pub fn softplus(&mut self, input: Tensor) -> Tensor {
        self.activation(input, ActivationMode::SoftPlus)
    }

    pub fn softsign(&mut self, input: Tensor) -> Tensor {
        self.activation(input, ActivationMode::SoftSign)
    }

    // ── Composable convenience methods ──────────────────────────────────────
    // These combine existing ops for common patterns. No new Op variants needed.

    /// SiLU (Sigmoid Linear Unit): x * sigmoid(x). Used by SwiGLU in MoE experts.
    pub fn silu(&mut self, input: Tensor) -> Tensor {
        let sig = self.sigmoid(input);
        self.multiplication(input, sig)
    }

    /// GELU (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    /// Uses the tanh approximation to avoid erf which falls back to CPU.
    pub fn gelu(&mut self, input: Tensor) -> Tensor {
        let bs = Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        };
        let half = self.constant_with_scalar(0.5, bs);
        let one = self.constant_with_scalar(1.0, bs);
        let coeff = self.constant_with_scalar(0.044715, bs);
        let sqrt_2_pi = self.constant_with_scalar(0.7978845608028654, bs);

        let x_sq = self.multiplication(input, input);
        let x_cube = self.multiplication(x_sq, input);
        let scaled_cube = self.multiplication(coeff, x_cube);
        let inner = self.addition(input, scaled_cube);
        let tanh_arg = self.multiplication(sqrt_2_pi, inner);
        let tanh_val = self.tanh(tanh_arg);
        let one_plus_tanh = self.addition(one, tanh_val);
        let half_x = self.multiplication(half, input);
        self.multiplication(half_x, one_plus_tanh)
    }

    /// Clamp / clip: max(min(x, hi), lo). Keeps values in [lo, hi] range.
    pub fn clip(&mut self, input: Tensor, lo: f32, hi: f32) -> Tensor {
        let bs = Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        };
        let lo_c = self.constant_with_scalar(lo, bs);
        let hi_c = self.constant_with_scalar(hi, bs);
        let clamped_hi = self.minimum(input, hi_c);
        self.maximum(clamped_hi, lo_c)
    }

    /// Negate: -x. Equivalent to multiplication by -1.
    pub fn neg(&mut self, input: Tensor) -> Tensor {
        let bs = Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        };
        let neg_one = self.constant_with_scalar(-1.0, bs);
        self.multiplication(input, neg_one)
    }

    /// Square: x * x.
    pub fn square(&mut self, input: Tensor) -> Tensor {
        self.multiplication(input, input)
    }

    /// Split a tensor along an axis into multiple parts by size.
    /// Returns a Vec of tensors, one per split.
    /// `sizes` must sum to the input dimension along `axis`.
    pub fn split(&mut self, input: Tensor, sizes: &[usize], axis: usize) -> Vec<Tensor> {
        let shape = input.shape;
        let dims = [shape.batch, shape.channels, shape.height, shape.width];
        let mut results = Vec::with_capacity(sizes.len());
        let mut offset = [0usize; 4];
        for &size in sizes {
            let mut sz = dims;
            sz[axis] = size;
            results.push(self.slice(input, offset, sz));
            offset[axis] += size;
        }
        results
    }

    /// Expand a dimension by repeating via multiplication with ones.
    /// Creates a tensor of ones with the target shape and multiplies.
    /// Useful for GQA head expansion (repeat KV heads to match Q heads).
    pub fn expand(&mut self, input: Tensor, target_shape: Shape) -> Tensor {
        let ones = self.constant_with_scalar(1.0, target_shape);
        self.multiplication(input, ones)
    }

    fn elementwise_binary(
        &mut self,
        left_hand_side: Tensor,
        right_hand_side: Tensor,
        op: ElementwiseOpType,
    ) -> Tensor {
        let output = self.alloc(Shape {
            batch: left_hand_side.shape.batch.max(right_hand_side.shape.batch),
            channels: left_hand_side
                .shape
                .channels
                .max(right_hand_side.shape.channels),
            height: left_hand_side
                .shape
                .height
                .max(right_hand_side.shape.height),
            width: left_hand_side.shape.width.max(right_hand_side.shape.width),
        });
        let left_hand_side_name = Self::tensor_name(left_hand_side);
        let right_hand_side_name = Self::tensor_name(right_hand_side);
        self.ops.push(GraphOp {
            op: Op::Elementwise(ElementwiseOp {
                name: Self::op_name(output),
                bottoms: vec![left_hand_side_name, right_hand_side_name].into_boxed_slice(),
                top: Self::tensor_name(output),
                operation: op,
                alpha: 1.0,
                beta: 0.0,
                fused_relu: false,
            }),
            output,
        });
        output
    }

    fn elementwise_unary(&mut self, input: Tensor, op: ElementwiseOpType) -> Tensor {
        let output = self.alloc(input.shape);
        self.ops.push(GraphOp {
            op: Op::Elementwise(ElementwiseOp {
                name: Self::op_name(output),
                bottoms: vec![Self::tensor_name(input)].into_boxed_slice(),
                top: Self::tensor_name(output),
                operation: op,
                alpha: 1.0,
                beta: 0.0,
                fused_relu: false,
            }),
            output,
        });
        output
    }

    pub fn addition(&mut self, left_hand_side: Tensor, right_hand_side: Tensor) -> Tensor {
        self.elementwise_binary(left_hand_side, right_hand_side, ElementwiseOpType::Add)
    }

    pub fn subtraction(&mut self, left_hand_side: Tensor, right_hand_side: Tensor) -> Tensor {
        self.elementwise_binary(left_hand_side, right_hand_side, ElementwiseOpType::Sub)
    }

    pub fn multiplication(&mut self, left_hand_side: Tensor, right_hand_side: Tensor) -> Tensor {
        self.elementwise_binary(left_hand_side, right_hand_side, ElementwiseOpType::Multiply)
    }

    pub fn division(&mut self, left_hand_side: Tensor, right_hand_side: Tensor) -> Tensor {
        self.elementwise_binary(left_hand_side, right_hand_side, ElementwiseOpType::Div)
    }

    pub fn power(&mut self, left_hand_side: Tensor, right_hand_side: Tensor) -> Tensor {
        self.elementwise_binary(left_hand_side, right_hand_side, ElementwiseOpType::Pow)
    }

    pub fn maximum(&mut self, left_hand_side: Tensor, right_hand_side: Tensor) -> Tensor {
        self.elementwise_binary(left_hand_side, right_hand_side, ElementwiseOpType::Max)
    }

    pub fn minimum(&mut self, left_hand_side: Tensor, right_hand_side: Tensor) -> Tensor {
        self.elementwise_binary(left_hand_side, right_hand_side, ElementwiseOpType::Min)
    }

    pub fn absolute(&mut self, input: Tensor) -> Tensor {
        self.elementwise_unary(input, ElementwiseOpType::Abs)
    }

    pub fn square_root(&mut self, input: Tensor) -> Tensor {
        self.elementwise_unary(input, ElementwiseOpType::Sqrt)
    }

    pub fn reciprocal_square_root(&mut self, input: Tensor) -> Tensor {
        self.elementwise_unary(input, ElementwiseOpType::Rsqrt)
    }

    pub fn exponent(&mut self, input: Tensor) -> Tensor {
        self.elementwise_unary(input, ElementwiseOpType::Exp)
    }

    pub fn logarithm(&mut self, input: Tensor) -> Tensor {
        self.elementwise_unary(input, ElementwiseOpType::Log)
    }

    pub fn reciprocal(&mut self, input: Tensor) -> Tensor {
        self.elementwise_unary(input, ElementwiseOpType::Inverse)
    }

    /// Softmax along the specified axis.
    pub fn soft_max(&mut self, input: Tensor, axis: i64) -> Tensor {
        let output = self.alloc(input.shape);
        self.ops.push(GraphOp {
            op: Op::Softmax(SoftmaxOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                axis,
            }),
            output,
        });
        output
    }

    /// Concatenate tensors along the specified axis.
    pub fn concat(&mut self, inputs: &[Tensor], axis: usize) -> Tensor {
        assert!(!inputs.is_empty(), "concat requires at least one input");
        let base = inputs[0].shape;
        let sum_dim: usize = inputs
            .iter()
            .map(|t| {
                let dims = [
                    t.shape.batch,
                    t.shape.channels,
                    t.shape.height,
                    t.shape.width,
                ];
                dims[axis]
            })
            .sum();
        let mut out_dims = [base.batch, base.channels, base.height, base.width];
        out_dims[axis] = sum_dim;
        let output = self.alloc(Shape {
            batch: out_dims[0],
            channels: out_dims[1],
            height: out_dims[2],
            width: out_dims[3],
        });
        let bottoms: Box<[String]> = inputs.iter().map(|t| Self::tensor_name(*t)).collect();
        self.ops.push(GraphOp {
            op: Op::Concat(ConcatOp {
                name: Self::op_name(output),
                bottoms,
                top: Self::tensor_name(output),
                axis,
            }),
            output,
        });
        output
    }

    /// Matrix multiplication: `out = x @ y` (with optional transposes on last two dims).
    pub fn matrix_multiplication(
        &mut self,
        left_hand_side: Tensor,
        right_hand_side: Tensor,
        transpose_x: bool,
        transpose_y: bool,
    ) -> Tensor {
        let out_h = if transpose_x {
            left_hand_side.shape.width
        } else {
            left_hand_side.shape.height
        };
        let out_w = if transpose_y {
            right_hand_side.shape.height
        } else {
            right_hand_side.shape.width
        };
        let output = self.alloc(Shape {
            batch: left_hand_side.shape.batch,
            channels: left_hand_side.shape.channels,
            height: out_h,
            width: out_w,
        });
        self.ops.push(GraphOp {
            op: Op::Matmul(MatmulOp {
                name: Self::op_name(output),
                bottom_x: Self::tensor_name(left_hand_side),
                bottom_y: Self::tensor_name(right_hand_side),
                top: Self::tensor_name(output),
                transpose_x,
                transpose_y,
            }),
            output,
        });
        output
    }

    /// Permute the dimensions of `x` according to `perm` (4-element NCHW permutation).
    pub fn transpose(&mut self, input: Tensor, perm: [usize; 4]) -> Tensor {
        let dimensions = [
            input.shape.batch,
            input.shape.channels,
            input.shape.height,
            input.shape.width,
        ];
        let output = self.alloc(Shape {
            batch: dimensions[perm[0]],
            channels: dimensions[perm[1]],
            height: dimensions[perm[2]],
            width: dimensions[perm[3]],
        });
        self.ops.push(GraphOp {
            op: Op::Transpose(TransposeOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                perm,
            }),
            output,
        });
        output
    }

    /// Extract a sub-tensor starting at `begin` with dimensions `size` (both in NCHW order).
    pub fn slice(&mut self, input: Tensor, begin: [usize; 4], size: [usize; 4]) -> Tensor {
        let output = self.alloc(Shape {
            batch: size[0],
            channels: size[1],
            height: size[2],
            width: size[3],
        });
        self.ops.push(GraphOp {
            op: Op::SliceBySize(SliceBySizeOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                begin,
                size,
            }),
            output,
        });
        output
    }

    pub fn reshape(&mut self, input: Tensor, target: Shape) -> Tensor {
        let output = self.alloc(target);
        self.ops.push(GraphOp {
            op: Op::Reshape(ReshapeOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                target_shape: [target.batch, target.channels, target.height, target.width],
            }),
            output,
        });
        output
    }

    /// Flatten all dimensions into a single channel axis: `[1, k*h*w, 1, 1]`.
    pub fn flatten_2d(&mut self, input: Tensor) -> Tensor {
        let flat_k = input.shape.total_elements();
        let output = self.alloc(Shape::channels(flat_k));
        self.ops.push(GraphOp {
            op: Op::Flatten(FlattenOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
            }),
            output,
        });
        output
    }

    fn pool(
        &mut self,
        input: Tensor,
        pool_type: PoolType,
        kernel_h: usize,
        kernel_w: usize,
        stride_height: usize,
        stride_width: usize,
        pad_mode: PadMode,
        global: bool,
    ) -> Tensor {
        let (out_h, out_w) = if global {
            (1, 1)
        } else {
            match pad_mode {
                PadMode::Valid => (
                    (input.shape.height.saturating_sub(kernel_h)) / stride_height + 1,
                    (input.shape.width.saturating_sub(kernel_w)) / stride_width + 1,
                ),
                PadMode::Same => (
                    (input.shape.height + stride_height - 1) / stride_height,
                    (input.shape.width + stride_width - 1) / stride_width,
                ),
            }
        };
        let output = self.alloc(Shape {
            channels: input.shape.channels,
            height: out_h,
            width: out_w,
            batch: 1,
        });
        self.ops.push(GraphOp {
            op: Op::Pooling(PoolingOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                pool_type,
                kernel_height: kernel_h,
                kernel_width: kernel_w,
                stride_height,
                stride_width,
                pad_mode,
                pad_top: 0,
                pad_bottom: 0,
                pad_left: 0,
                pad_right: 0,
                global_pooling: global,
            }),
            output,
        });
        output
    }

    pub fn max_pool(
        &mut self,
        input: Tensor,
        kernel_h: usize,
        kernel_w: usize,
        stride_height: usize,
        stride_width: usize,
        pad_mode: PadMode,
    ) -> Tensor {
        self.pool(
            input,
            PoolType::Max,
            kernel_h,
            kernel_w,
            stride_height,
            stride_width,
            pad_mode,
            false,
        )
    }

    pub fn avg_pool(
        &mut self,
        input: Tensor,
        kernel_h: usize,
        kernel_w: usize,
        stride_height: usize,
        stride_width: usize,
        pad_mode: PadMode,
    ) -> Tensor {
        self.pool(
            input,
            PoolType::Average,
            kernel_h,
            kernel_w,
            stride_height,
            stride_width,
            pad_mode,
            false,
        )
    }

    pub fn global_avg_pool(&mut self, input: Tensor) -> Tensor {
        let kh = input.shape.height;
        let kw = input.shape.width;
        self.pool(input, PoolType::Average, kh, kw, 1, 1, PadMode::Valid, true)
    }

    pub fn pad(
        &mut self,
        input: Tensor,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        mode: PadFillMode,
        value: f64,
    ) -> Tensor {
        let output = self.alloc(Shape {
            channels: input.shape.channels,
            height: input.shape.height + top + bottom,
            width: input.shape.width + left + right,
            batch: 1,
        });
        self.ops.push(GraphOp {
            op: Op::Padding(PaddingOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                pad_top: top,
                pad_bottom: bottom,
                pad_left: left,
                pad_right: right,
                pad_fill_mode: mode,
                pad_value: value,
            }),
            output,
        });
        output
    }

    fn reduce(&mut self, input: Tensor, mode: ReductionMode, axis: i64) -> Tensor {
        let mut out_shape = input.shape;
        match axis.rem_euclid(4) {
            0 => out_shape.batch = 1,
            1 => out_shape.channels = 1,
            2 => out_shape.height = 1,
            3 => out_shape.width = 1,
            _ => {}
        }
        let output = self.alloc(out_shape);
        self.ops.push(GraphOp {
            op: Op::Reduction(ReductionOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                mode,
                axis,
            }),
            output,
        });
        output
    }

    pub fn reduce_sum(&mut self, input: Tensor, axis: i64) -> Tensor {
        self.reduce(input, ReductionMode::Sum, axis)
    }

    pub fn reduce_mean(&mut self, input: Tensor, axis: i64) -> Tensor {
        self.reduce(input, ReductionMode::Mean, axis)
    }

    pub fn reduce_min(&mut self, input: Tensor, axis: i64) -> Tensor {
        self.reduce(input, ReductionMode::Min, axis)
    }

    pub fn reduce_max(&mut self, input: Tensor, axis: i64) -> Tensor {
        self.reduce(input, ReductionMode::Max, axis)
    }

    pub fn instance_norm(&mut self, source: Tensor, params: Tensor, epsilon: f64) -> Tensor {
        let output = self.alloc(source.shape);
        self.ops.push(GraphOp {
            op: Op::InstanceNorm(InstanceNormOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(source),
                top: Self::tensor_name(output),
                channels: source.shape.channels,
                epsilon,
                params: self.resolve_constant(params),
            }),
            output,
        });
        output
    }

    /// Layer normalization across specified axes (fused ANE op).
    ///
    /// Unlike manual reduce_sum(x²) which overflows fp16 for large dims,
    /// the fused layer_norm op handles variance computation internally
    /// without materializing squared values in fp16.
    ///
    /// For RMSNorm via the ANEMLL trick: concat [x, -x], layer_norm, slice.
    pub fn layer_norm(&mut self, source: Tensor, axes: &[i32], epsilon: f64) -> Tensor {
        let output = self.alloc(source.shape);
        self.ops.push(GraphOp {
            op: Op::LayerNorm(crate::ops::layer_norm::LayerNormOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(source),
                top: Self::tensor_name(output),
                axes: axes.into(),
                epsilon,
                has_affine: false,
                gamma: None,
            }),
            output,
        });
        output
    }

    /// Layer normalization with learned affine gamma parameter.
    pub fn layer_norm_affine(
        &mut self,
        source: Tensor,
        axes: &[i32],
        gamma: Tensor,
        epsilon: f64,
    ) -> Tensor {
        let output = self.alloc(source.shape);
        self.ops.push(GraphOp {
            op: Op::LayerNorm(crate::ops::layer_norm::LayerNormOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(source),
                top: Self::tensor_name(output),
                axes: axes.into(),
                epsilon,
                has_affine: true,
                gamma: Some(self.resolve_constant(gamma)),
            }),
            output,
        });
        output
    }

    // ── Native MIL ops (emitted as first-class MIL operations) ──────────

    /// GELU activation as native MIL op (may fuse better than composed version).
    /// Mode: "EXACT", "TANH_APPROXIMATION", "SIGMOID_APPROXIMATION".
    pub fn gelu_native(&mut self, input: Tensor, mode: &str) -> Tensor {
        let output = self.alloc(input.shape);
        self.ops.push(GraphOp {
            op: Op::Gelu(crate::ops::unary_ops::GeluOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                mode: mode.to_string(),
            }),
            output,
        });
        output
    }

    /// SiLU activation as native MIL op.
    pub fn silu_native(&mut self, input: Tensor) -> Tensor {
        let output = self.alloc(input.shape);
        self.ops.push(GraphOp {
            op: Op::Silu(crate::ops::unary_ops::SiluOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
            }),
            output,
        });
        output
    }

    /// Clip/clamp as native MIL op: clamp to [alpha, beta].
    pub fn clip_native(&mut self, input: Tensor, alpha: f32, beta: f32) -> Tensor {
        let output = self.alloc(input.shape);
        self.ops.push(GraphOp {
            op: Op::Clip(crate::ops::unary_ops::ClipOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                alpha,
                beta,
            }),
            output,
        });
        output
    }

    /// ReLU6 as native MIL op.
    pub fn relu6(&mut self, input: Tensor) -> Tensor {
        let output = self.alloc(input.shape);
        self.ops.push(GraphOp {
            op: Op::Relu6(crate::ops::unary_ops::Relu6Op {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
            }),
            output,
        });
        output
    }

    /// Clamped ReLU: leaky relu with ceiling.
    pub fn clamped_relu(&mut self, input: Tensor, alpha: f64, beta: f64) -> Tensor {
        let output = self.alloc(input.shape);
        self.ops.push(GraphOp {
            op: Op::ClampedRelu(crate::ops::unary_ops::ClampedReluOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                alpha,
                beta,
            }),
            output,
        });
        output
    }

    /// Thresholded ReLU: zero below threshold alpha.
    pub fn thresholded_relu(&mut self, input: Tensor, alpha: f64) -> Tensor {
        let output = self.alloc(input.shape);
        self.ops.push(GraphOp {
            op: Op::ThresholdedRelu(crate::ops::unary_ops::ThresholdedReluOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                alpha,
            }),
            output,
        });
        output
    }

    /// Scaled tanh: alpha * tanh(beta * x).
    pub fn scaled_tanh(&mut self, input: Tensor, alpha: f64, beta: f64) -> Tensor {
        let output = self.alloc(input.shape);
        self.ops.push(GraphOp {
            op: Op::ScaledTanh(crate::ops::unary_ops::ScaledTanhOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                alpha,
                beta,
            }),
            output,
        });
        output
    }

    /// PReLU: per-channel leaky relu with learned alpha.
    pub fn prelu(&mut self, input: Tensor, alpha: Tensor) -> Tensor {
        let output = self.alloc(input.shape);
        self.ops.push(GraphOp {
            op: Op::Prelu(crate::ops::unary_ops::PreluOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                channels: input.shape.channels,
                alpha: self.resolve_constant(alpha),
            }),
            output,
        });
        output
    }

    /// Type cast.
    pub fn cast(&mut self, input: Tensor, dtype: &str, output_shape: Shape) -> Tensor {
        let output = self.alloc(output_shape);
        self.ops.push(GraphOp {
            op: Op::Cast(crate::ops::unary_ops::CastOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                dtype: dtype.to_string(),
            }),
            output,
        });
        output
    }

    /// Cumulative sum along axis.
    pub fn cumsum(&mut self, input: Tensor, axis: i32, exclusive: bool, reverse: bool) -> Tensor {
        let output = self.alloc(input.shape);
        self.ops.push(GraphOp {
            op: Op::Cumsum(crate::ops::unary_ops::CumsumOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                axis,
                exclusive,
                reverse,
            }),
            output,
        });
        output
    }

    /// Tile (repeat) tensor along dimensions.
    pub fn tile(&mut self, input: Tensor, reps: &[i32], output_shape: Shape) -> Tensor {
        let output = self.alloc(output_shape);
        self.ops.push(GraphOp {
            op: Op::Tile(crate::ops::unary_ops::TileOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                reps: reps.into(),
            }),
            output,
        });
        output
    }

    /// Insert size-1 dimensions at specified axes.
    pub fn expand_dims(&mut self, input: Tensor, axes: &[i32], output_shape: Shape) -> Tensor {
        let output = self.alloc(output_shape);
        self.ops.push(GraphOp {
            op: Op::ExpandDims(crate::ops::unary_ops::ExpandDimsOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                axes: axes.into(),
            }),
            output,
        });
        output
    }

    /// Remove size-1 dimensions. If axes is None, removes all.
    pub fn squeeze(&mut self, input: Tensor, axes: Option<&[i32]>, output_shape: Shape) -> Tensor {
        let output = self.alloc(output_shape);
        self.ops.push(GraphOp {
            op: Op::Squeeze(crate::ops::unary_ops::SqueezeOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                axes: axes.map(|a| a.into()),
            }),
            output,
        });
        output
    }

    /// Top-k values and indices. Returns (values_tensor, indices_tensor).
    pub fn topk(
        &mut self,
        input: Tensor,
        k: i32,
        axis: i32,
        ascending: bool,
        val_shape: Shape,
        idx_shape: Shape,
    ) -> (Tensor, Tensor) {
        let val_out = self.alloc(val_shape);
        let idx_out = self.alloc(idx_shape);
        self.ops.push(GraphOp {
            op: Op::Topk(crate::ops::multi_output_ops::TopkOp {
                name: Self::op_name(val_out),
                bottom: Self::tensor_name(input),
                top_values: Self::tensor_name(val_out),
                top_indices: Self::tensor_name(idx_out),
                k,
                axis,
                ascending,
            }),
            output: val_out, // primary output for shape tracking
        });
        (val_out, idx_out)
    }

    /// Argsort: returns indices that would sort the tensor.
    pub fn argsort(&mut self, input: Tensor, axis: i32, ascending: bool) -> Tensor {
        let output = self.alloc(input.shape);
        self.ops.push(GraphOp {
            op: Op::Argsort(crate::ops::multi_output_ops::ArgsortOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                axis,
                ascending,
            }),
            output,
        });
        output
    }

    /// Argmax along axis.
    pub fn reduce_argmax(
        &mut self,
        input: Tensor,
        axis: i32,
        keep_dims: bool,
        output_shape: Shape,
    ) -> Tensor {
        let output = self.alloc(output_shape);
        self.ops.push(GraphOp {
            op: Op::ReduceArgmax(crate::ops::multi_output_ops::ReduceArgmaxOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                axis,
                keep_dims,
            }),
            output,
        });
        output
    }

    /// Argmin along axis.
    pub fn reduce_argmin(
        &mut self,
        input: Tensor,
        axis: i32,
        keep_dims: bool,
        output_shape: Shape,
    ) -> Tensor {
        let output = self.alloc(output_shape);
        self.ops.push(GraphOp {
            op: Op::ReduceArgmin(crate::ops::multi_output_ops::ReduceArgminOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                axis,
                keep_dims,
            }),
            output,
        });
        output
    }

    /// Conditional select: where(cond, a, b). Returns a where cond is true, b otherwise.
    pub fn select(&mut self, cond: Tensor, a: Tensor, b: Tensor) -> Tensor {
        let output = self.alloc(a.shape);
        self.ops.push(GraphOp {
            op: Op::Select(crate::ops::multi_output_ops::SelectOp {
                name: Self::op_name(output),
                cond: Self::tensor_name(cond),
                a: Self::tensor_name(a),
                b: Self::tensor_name(b),
                top: Self::tensor_name(output),
            }),
            output,
        });
        output
    }

    /// L2 normalization.
    pub fn l2_norm(&mut self, input: Tensor, epsilon: f64) -> Tensor {
        let output = self.alloc(input.shape);
        self.ops.push(GraphOp {
            op: Op::L2Norm(crate::ops::multi_output_ops::L2NormOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                epsilon,
            }),
            output,
        });
        output
    }

    /// Product reduction along axes.
    pub fn reduce_prod(
        &mut self,
        input: Tensor,
        axes: &[i32],
        keep_dims: bool,
        output_shape: Shape,
    ) -> Tensor {
        let output = self.alloc(output_shape);
        self.ops.push(GraphOp {
            op: Op::ReduceProd(crate::ops::multi_output_ops::ReduceProdOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                axes: axes.into(),
                keep_dims,
            }),
            output,
        });
        output
    }
}
