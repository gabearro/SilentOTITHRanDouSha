use std::collections::HashSet;

use super::{
    activation_mode::ActivationMode,
    elementwise::ElementwiseOpType,
    op::Op,
    pad_fill_mode::PadFillMode,
    pad_mode::PadMode,
    pool_type::PoolType,
    reduction_mode::ReductionMode,
    scalar::ScalarOpType,
    shape::Shape,
    weights::{build_mil_weight_blob, mil_blob_chunk_offset, WeightBlob},
};

const MIL_BUILD_INFO: &str = r#"[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]"#;

/// Emits the complete MIL program text and the packed weight blob.
///
/// Returns `(mil_text, weight_bytes)`. `weight_bytes` is empty when there are
/// no learnable weights.
pub(crate) fn emit_mil(ops: &[Op], shapes: &[(String, Shape)]) -> (String, Box<[u8]>) {
    let shape_map: std::collections::HashMap<&str, Shape> = shapes
        .iter()
        .map(|(name, shape)| (name.as_str(), *shape))
        .collect();

    let all_tops: HashSet<&str> = ops.iter().flat_map(|l| tops(l)).collect();
    let all_bottoms: HashSet<&str> = ops.iter().flat_map(|l| bottoms(l)).collect();

    let input_names: Vec<&str> = ops
        .iter()
        .flat_map(|l| bottoms(l))
        .filter(|b| !all_tops.contains(b))
        .collect::<Vec<_>>()
        .into_iter()
        .fold(vec![], |mut acc, n| {
            if !acc.contains(&n) {
                acc.push(n);
            }
            acc
        });

    let output_names: Vec<&str> = ops
        .iter()
        .flat_map(|l| tops(l))
        .filter(|t| !all_bottoms.contains(t))
        .collect::<Vec<_>>()
        .into_iter()
        .fold(vec![], |mut acc, n| {
            if !acc.contains(&n) {
                acc.push(n);
            }
            acc
        });

    let mut weight_blobs: Vec<&WeightBlob> = Vec::new();
    for layer in ops.iter() {
        collect_weights(layer, &mut weight_blobs);
    }

    let weight_bytes: Box<[u8]> = if weight_blobs.is_empty() {
        Box::new([])
    } else {
        build_mil_weight_blob(&weight_blobs)
    };

    let mut out = String::new();
    out.push_str("program(1.3)\n");
    out.push_str(MIL_BUILD_INFO);
    out.push_str("\n{\n");

    // fp16 I/O: inputs and outputs are fp16 IOSurfaces.
    // No cast ops needed — ANE natively operates in fp16.
    // This halves DMA bandwidth and eliminates 2-3 cast ops per kernel.
    out.push_str("    func main<ios18>(");
    let sig_parts: Vec<String> = input_names
        .iter()
        .map(|name| {
            let shape = shape_map.get(name).copied().unwrap_or(Shape::channels(1));
            format!("tensor<fp16, {}> {}", mil_shape(shape), name)
        })
        .collect();
    out.push_str(&sig_parts.join(", "));
    out.push_str(") {\n");

    // Input is already fp16 — create aliases with _f16 suffix for downstream ops
    for name in &input_names {
        let shape = shape_map.get(name).copied().unwrap_or(Shape::channels(1));
        out.push_str(&format!(
            "        tensor<fp16, {s}> {n}_f16 = identity(x = {n})[name = string(\"alias_{n}\")];\n",
            s = mil_shape(shape),
            n = name,
        ));
    }

    let mut blob_index = 0usize;
    for layer in ops.iter() {
        emit_layer(layer, &shape_map, &weight_blobs, &mut blob_index, &mut out);
    }

    // Output stays fp16 — just reference the _f16 output directly
    let ret_parts: Vec<String> = output_names.iter().map(|n| format!("{n}_f16")).collect();
    let ret = ret_parts.join(", ");
    out.push_str(&format!("    }} -> ({ret});\n"));
    out.push_str("}\n");

    (out, weight_bytes)
}

fn mil_shape(s: Shape) -> String {
    format!("[{}, {}, {}, {}]", s.batch, s.channels, s.height, s.width)
}

fn tops(layer: &Op) -> Vec<&str> {
    vec![layer.top()]
}

fn bottoms(layer: &Op) -> Vec<&str> {
    layer.bottom_names()
}

fn collect_weights<'a>(layer: &'a Op, out: &mut Vec<&'a WeightBlob>) {
    match layer {
        Op::Constant(l) => {
            out.push(&l.data);
        }
        Op::InnerProduct(l) => {
            out.push(&l.weights);
            if let Some(b) = &l.bias {
                out.push(b);
            }
        }
        Op::Conv(l) => {
            out.push(&l.weights);
            if let Some(b) = &l.bias {
                out.push(b);
            }
        }
        Op::Deconv(l) => {
            out.push(&l.weights);
            if let Some(b) = &l.bias {
                out.push(b);
            }
        }
        Op::InstanceNorm(l) => {
            out.push(&l.params);
        }
        Op::LayerNorm(l) => {
            if let Some(ref gamma) = l.gamma {
                out.push(gamma);
            }
        }
        Op::Prelu(l) => {
            out.push(&l.alpha);
        }
        Op::BatchNorm(l) => {
            out.push(&l.mean);
            out.push(&l.variance);
            if let Some(ref g) = l.gamma {
                out.push(g);
            }
            if let Some(ref b) = l.beta {
                out.push(b);
            }
        }
        Op::Matmul(_)
        | Op::Transpose(_)
        | Op::SliceBySize(_)
        | Op::ScalarOp(_)
        | Op::Elementwise(_)
        | Op::Activation(_)
        | Op::Softmax(_)
        | Op::Concat(_)
        | Op::Reshape(_)
        | Op::Pooling(_)
        | Op::Padding(_)
        | Op::Flatten(_)
        | Op::Reduction(_)
        | Op::DynConv(_)
        | Op::Gelu(_)
        | Op::Silu(_)
        | Op::Clip(_)
        | Op::Relu6(_)
        | Op::ClampedRelu(_)
        | Op::ThresholdedRelu(_)
        | Op::ScaledTanh(_)
        | Op::Cast(_)
        | Op::Cumsum(_)
        | Op::Tile(_)
        | Op::ExpandDims(_)
        | Op::Squeeze(_)
        | Op::Split(_)
        | Op::Topk(_)
        | Op::Argsort(_)
        | Op::ReduceArgmax(_)
        | Op::ReduceArgmin(_)
        | Op::Select(_)
        | Op::L2Norm(_)
        | Op::ReduceProd(_) => {}
    }
}

fn blobfile_ref(
    all_blobs: &[&WeightBlob],
    blob_index: usize,
    shape_str: &str,
    var_name: &str,
    out: &mut String,
) {
    let offset = mil_blob_chunk_offset(all_blobs, blob_index);
    out.push_str(&format!(
        "        tensor<fp16, {s}> {v} = const()[name = string(\"{v}\"), \
         val = tensor<fp16, {s}>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), \
         offset = uint64({o})))];\n",
        s = shape_str,
        v = var_name,
        o = offset,
    ));
}

fn emit_layer(
    layer: &Op,
    shape_map: &std::collections::HashMap<&str, Shape>,
    all_blobs: &[&WeightBlob],
    blob_index: &mut usize,
    out: &mut String,
) {
    match layer {
        Op::Constant(l) => {
            let shape = shape_map
                .get(l.top.as_str())
                .copied()
                .unwrap_or(Shape::channels(1));
            let sh = mil_shape(shape);
            blobfile_ref(all_blobs, *blob_index, &sh, &format!("{}_f16", l.top), out);
            *blob_index += 1;
        }

        Op::InnerProduct(l) => {
            let in_shape = shape_map
                .get(l.bottom.as_str())
                .copied()
                .unwrap_or(Shape::channels(1));
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .unwrap_or(Shape::channels(1));
            let in_ch = in_shape.channels;
            let out_ch = out_shape.channels;
            let n = &l.name;

            emit_conv_constants(n, 0, 0, 0, 0, 1, 1, 1, 1, "valid", out);

            let w_shape = format!("[{out_ch}, {in_ch}, 1, 1]");
            let w_var = format!("{n}_W");
            blobfile_ref(all_blobs, *blob_index, &w_shape, &w_var, out);
            *blob_index += 1;

            let out_sh = mil_shape(out_shape);

            let bias_param = if l.bias.is_some() {
                let b_shape = format!("[{out_ch}]");
                let b_var = format!("{n}_b");
                blobfile_ref(all_blobs, *blob_index, &b_shape, &b_var, out);
                *blob_index += 1;
                format!(", bias = {b_var}")
            } else {
                String::new()
            };

            let conv_out = if l.has_relu || l.has_tanh {
                format!("{n}_conv_out")
            } else {
                format!("{}_f16", l.top)
            };

            out.push_str(&format!(
                "        tensor<fp16, {out_sh}> {conv_out} = conv(\
                 dilations = {n}_dilations, groups = {n}_groups, pad = {n}_pad, \
                 pad_type = {n}_pad_type, strides = {n}_strides{bias_param}, weight = {w_var}, \
                 x = {bot}_f16)[name = string(\"{n}\")];\n",
                bot = l.bottom,
            ));

            if l.has_relu {
                out.push_str(&format!(
                    "        tensor<fp16, {out_sh}> {top}_f16 = relu(x = {conv_out})[name = string(\"{n}_relu\")];\n",
                    top = l.top,
                ));
            } else if l.has_tanh {
                out.push_str(&format!(
                    "        tensor<fp16, {out_sh}> {top}_f16 = tanh(x = {conv_out})[name = string(\"{n}_tanh\")];\n",
                    top = l.top,
                ));
            }
        }

        Op::Conv(l) => {
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .unwrap_or(Shape::channels(1));
            let n = &l.name;
            let in_ch = l.input_channels;
            let out_ch = l.output_channels;
            let kh = l.kernel_height;
            let kw = l.kernel_width;
            let groups = l.groups;

            let pad_type_str = match l.pad_mode {
                PadMode::Valid => "valid",
                PadMode::Same => "same_lower",
            };
            emit_conv_constants(
                n,
                l.pad_top,
                l.pad_bottom,
                l.pad_left,
                l.pad_right,
                1,
                1,
                groups,
                groups,
                pad_type_str,
                out,
            );

            let w_shape = format!(
                "[{out_ch}, {per_group}, {kh}, {kw}]",
                per_group = in_ch / groups
            );
            let w_var = format!("{n}_W");
            blobfile_ref(all_blobs, *blob_index, &w_shape, &w_var, out);
            *blob_index += 1;

            let out_sh = mil_shape(out_shape);

            let bias_param = if l.bias.is_some() {
                let b_shape = format!("[{out_ch}]");
                let b_var = format!("{n}_b");
                blobfile_ref(all_blobs, *blob_index, &b_shape, &b_var, out);
                *blob_index += 1;
                format!(", bias = {b_var}")
            } else {
                String::new()
            };

            let conv_out = if l.fused_relu || l.fused_tanh {
                format!("{n}_conv_out")
            } else {
                format!("{}_f16", l.top)
            };

            out.push_str(&format!(
                "        tensor<fp16, {out_sh}> {conv_out} = conv(\
                 dilations = {n}_dilations, groups = {n}_groups, pad = {n}_pad, \
                 pad_type = {n}_pad_type, strides = {n}_strides{bias_param}, weight = {w_var}, \
                 x = {bot}_f16)[name = string(\"{n}\")];\n",
                bot = l.bottom,
            ));

            if l.fused_relu {
                out.push_str(&format!(
                    "        tensor<fp16, {out_sh}> {top}_f16 = relu(x = {conv_out})[name = string(\"{n}_relu\")];\n",
                    top = l.top,
                ));
            } else if l.fused_tanh {
                out.push_str(&format!(
                    "        tensor<fp16, {out_sh}> {top}_f16 = tanh(x = {conv_out})[name = string(\"{n}_tanh\")];\n",
                    top = l.top,
                ));
            }
        }

        Op::Deconv(l) => {
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .unwrap_or(Shape::channels(1));
            let n = &l.name;
            let in_ch = l.input_channels;
            let out_ch = l.output_channels;
            let kh = l.kernel_height;
            let kw = l.kernel_width;
            let groups = l.groups;
            let sh = l.stride_height;
            let sw = l.stride_width;

            let pad_type_str = match l.pad_mode {
                PadMode::Valid => "valid",
                PadMode::Same => "same_lower",
            };

            out.push_str(&format!(
                "        string {n}_pad_type = const()[name = string(\"{n}_pad_type\"), val = string(\"{pad_type_str}\")];\n",
            ));
            out.push_str(&format!(
                "        tensor<int32, [2]> {n}_strides = const()[name = string(\"{n}_strides\"), val = tensor<int32, [2]>([{sh}, {sw}])];\n",
            ));
            out.push_str(&format!(
                "        tensor<int32, [4]> {n}_pad = const()[name = string(\"{n}_pad\"), val = tensor<int32, [4]>([{}, {}, {}, {}])];\n",
                l.pad_top, l.pad_bottom, l.pad_left, l.pad_right,
            ));
            out.push_str(&format!(
                "        tensor<int32, [2]> {n}_dilations = const()[name = string(\"{n}_dilations\"), val = tensor<int32, [2]>([1, 1])];\n",
            ));
            out.push_str(&format!(
                "        int32 {n}_groups = const()[name = string(\"{n}_groups\"), val = int32({groups})];\n",
            ));
            if l.output_padding_height > 0 || l.output_padding_width > 0 {
                out.push_str(&format!(
                    "        tensor<int32, [2]> {n}_out_pad = const()[name = string(\"{n}_out_pad\"), val = tensor<int32, [2]>([{}, {}])];\n",
                    l.output_padding_height, l.output_padding_width,
                ));
            }

            let w_shape = format!(
                "[{in_ch}, {per_group}, {kh}, {kw}]",
                per_group = out_ch / groups
            );
            let w_var = format!("{n}_W");
            blobfile_ref(all_blobs, *blob_index, &w_shape, &w_var, out);
            *blob_index += 1;

            let out_sh = mil_shape(out_shape);
            let deconv_out = if l.bias.is_some() || l.fused_relu || l.fused_tanh {
                format!("{n}_deconv_out")
            } else {
                format!("{}_f16", l.top)
            };

            if l.output_padding_height > 0 || l.output_padding_width > 0 {
                out.push_str(&format!(
                    "        tensor<fp16, {out_sh}> {deconv_out} = conv_transpose(\
                     dilations = {n}_dilations, groups = {n}_groups, pad = {n}_pad, \
                     pad_type = {n}_pad_type, strides = {n}_strides, weight = {w_var}, \
                     output_padding = {n}_out_pad, x = {bot}_f16)[name = string(\"{n}\")];\n",
                    bot = l.bottom,
                ));
            } else {
                out.push_str(&format!(
                    "        tensor<fp16, {out_sh}> {deconv_out} = conv_transpose(\
                     dilations = {n}_dilations, groups = {n}_groups, pad = {n}_pad, \
                     pad_type = {n}_pad_type, strides = {n}_strides, weight = {w_var}, \
                     x = {bot}_f16)[name = string(\"{n}\")];\n",
                    bot = l.bottom,
                ));
            }

            let after_bias = if l.bias.is_some() {
                let b_shape = format!("[{out_ch}]");
                let b_var = format!("{n}_b");
                blobfile_ref(all_blobs, *blob_index, &b_shape, &b_var, out);
                *blob_index += 1;
                let biased = if l.fused_relu || l.fused_tanh {
                    format!("{n}_biased")
                } else {
                    format!("{}_f16", l.top)
                };
                out.push_str(&format!(
                    "        tensor<fp16, {out_sh}> {biased} = add(x = {deconv_out}, y = {b_var})[name = string(\"{n}_bias\")];\n",
                ));
                biased
            } else {
                deconv_out
            };

            if l.fused_relu {
                out.push_str(&format!(
                    "        tensor<fp16, {out_sh}> {top}_f16 = relu(x = {after_bias})[name = string(\"{n}_relu\")];\n",
                    top = l.top,
                ));
            } else if l.fused_tanh {
                out.push_str(&format!(
                    "        tensor<fp16, {out_sh}> {top}_f16 = tanh(x = {after_bias})[name = string(\"{n}_tanh\")];\n",
                    top = l.top,
                ));
            }
        }

        Op::Activation(l) => {
            let shape = shape_map
                .get(l.top.as_str())
                .copied()
                .unwrap_or(Shape::channels(1));
            let sh = mil_shape(shape);
            let n = &l.name;
            let bot = &l.bottom;
            let top = &l.top;

            match l.mode {
                ActivationMode::Relu => {
                    out.push_str(&format!(
                        "        tensor<fp16, {sh}> {top}_f16 = relu(x = {bot}_f16)[name = string(\"{n}\")];\n",
                    ));
                }
                ActivationMode::Tanh => {
                    out.push_str(&format!(
                        "        tensor<fp16, {sh}> {top}_f16 = tanh(x = {bot}_f16)[name = string(\"{n}\")];\n",
                    ));
                }
                ActivationMode::LeakyRelu { negative_slope } => {
                    out.push_str(&format!(
                        "        fp32 {n}_alpha = const()[name = string(\"{n}_alpha\"), val = fp32({negative_slope})];\n",
                    ));
                    out.push_str(&format!(
                        "        tensor<fp16, {sh}> {top}_f16 = leaky_relu(alpha = {n}_alpha, x = {bot}_f16)[name = string(\"{n}\")];\n",
                    ));
                }
                ActivationMode::Sigmoid => {
                    out.push_str(&format!(
                        "        tensor<fp16, {sh}> {top}_f16 = sigmoid(x = {bot}_f16)[name = string(\"{n}\")];\n",
                    ));
                }
                ActivationMode::Elu { alpha } => {
                    out.push_str(&format!(
                        "        fp32 {n}_alpha = const()[name = string(\"{n}_alpha\"), val = fp32({alpha})];\n",
                    ));
                    out.push_str(&format!(
                        "        tensor<fp16, {sh}> {top}_f16 = elu(alpha = {n}_alpha, x = {bot}_f16)[name = string(\"{n}\")];\n",
                    ));
                }
                ActivationMode::Linear { alpha, beta } => {
                    out.push_str(&format!(
                        "        fp32 {n}_alpha = const()[name = string(\"{n}_alpha\"), val = fp32({alpha})];\n",
                    ));
                    out.push_str(&format!(
                        "        fp32 {n}_beta = const()[name = string(\"{n}_beta\"), val = fp32({beta})];\n",
                    ));
                    out.push_str(&format!(
                        "        tensor<fp16, {sh}> {top}_f16 = linear_activation(alpha = {n}_alpha, beta = {n}_beta, x = {bot}_f16)[name = string(\"{n}\")];\n",
                    ));
                }
                ActivationMode::SigmoidHard { alpha, beta } => {
                    out.push_str(&format!(
                        "        fp32 {n}_alpha = const()[name = string(\"{n}_alpha\"), val = fp32({alpha})];\n",
                    ));
                    out.push_str(&format!(
                        "        fp32 {n}_beta = const()[name = string(\"{n}_beta\"), val = fp32({beta})];\n",
                    ));
                    out.push_str(&format!(
                        "        fp32 {n}_clip_lo = const()[name = string(\"{n}_clip_lo\"), val = fp32(0.0)];\n",
                    ));
                    out.push_str(&format!(
                        "        fp32 {n}_clip_hi = const()[name = string(\"{n}_clip_hi\"), val = fp32(1.0)];\n",
                    ));
                    out.push_str(&format!(
                        "        tensor<fp16, {sh}> {n}_linear = linear_activation(alpha = {n}_alpha, beta = {n}_beta, x = {bot}_f16)[name = string(\"{n}_linear\")];\n",
                    ));
                    out.push_str(&format!(
                        "        tensor<fp16, {sh}> {top}_f16 = clip(alpha = {n}_clip_lo, beta = {n}_clip_hi, x = {n}_linear)[name = string(\"{n}\")];\n",
                    ));
                }
                ActivationMode::SoftPlus => {
                    out.push_str(&format!(
                        "        tensor<fp16, {sh}> {top}_f16 = softplus(x = {bot}_f16)[name = string(\"{n}\")];\n",
                    ));
                }
                ActivationMode::SoftSign => {
                    out.push_str(&format!(
                        "        tensor<fp16, {sh}> {top}_f16 = softsign(x = {bot}_f16)[name = string(\"{n}\")];\n",
                    ));
                }
            }
        }

        Op::Elementwise(l) => {
            let shape = shape_map
                .get(l.top.as_str())
                .copied()
                .unwrap_or(Shape::channels(1));
            let sh = mil_shape(shape);
            let n = &l.name;
            let top = &l.top;

            let (mil_op, is_binary) = match l.operation {
                ElementwiseOpType::Add => ("add", true),
                ElementwiseOpType::Multiply => ("mul", true),
                ElementwiseOpType::Max => ("maximum", true),
                ElementwiseOpType::Min => ("minimum", true),
                ElementwiseOpType::Sub => ("sub", true),
                ElementwiseOpType::Div => ("real_div", true),
                ElementwiseOpType::Pow => ("pow", true),
                ElementwiseOpType::Abs => ("abs", false),
                ElementwiseOpType::Sqrt => ("sqrt", false),
                ElementwiseOpType::Rsqrt => ("rsqrt", false),
                ElementwiseOpType::Inverse => ("inverse", false),
                ElementwiseOpType::Exp => ("exp", false),
                ElementwiseOpType::Log => ("log", false),
                ElementwiseOpType::Threshold => ("threshold", false),
            };

            if is_binary && l.bottoms.len() >= 2 {
                let a = &l.bottoms[0];
                let b = &l.bottoms[1];
                out.push_str(&format!(
                    "        tensor<fp16, {sh}> {top}_f16 = {mil_op}(x = {a}_f16, y = {b}_f16)[name = string(\"{n}\")];\n",
                ));
            } else {
                let bot = l.bottoms.first().map(|s| s.as_str()).unwrap_or("");
                out.push_str(&format!(
                    "        tensor<fp16, {sh}> {top}_f16 = {mil_op}(x = {bot}_f16)[name = string(\"{n}\")];\n",
                ));
            }

            if l.fused_relu {
                out.push_str(&format!(
                    "        tensor<fp16, {sh}> {top}_relu_f16 = relu(x = {top}_f16)[name = string(\"{n}_relu\")];\n",
                ));
                out.push_str(&format!(
                    "        tensor<fp16, {sh}> {top}_f16_final = identity(x = {top}_relu_f16)[name = string(\"{n}_id\")];\n",
                ));
            }
        }

        Op::Softmax(l) => {
            let shape = shape_map
                .get(l.top.as_str())
                .copied()
                .unwrap_or(Shape::channels(1));
            let sh = mil_shape(shape);
            let n = &l.name;
            let axis = l.axis;
            out.push_str(&format!(
                "        int32 {n}_axis = const()[name = string(\"{n}_axis\"), val = int32({axis})];\n",
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = softmax(axis = {n}_axis, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top,
                bot = l.bottom,
            ));
        }

        Op::Concat(l) => {
            let shape = shape_map
                .get(l.top.as_str())
                .copied()
                .unwrap_or(Shape::channels(1));
            let sh = mil_shape(shape);
            let n = &l.name;
            let axis = l.axis;
            let inputs: Vec<String> = l.bottoms.iter().map(|b| format!("{b}_f16")).collect();
            let inputs_str = inputs.join(", ");
            out.push_str(&format!(
                "        int32 {n}_axis = const()[name = string(\"{n}_axis\"), val = int32({axis})];\n",
            ));
            out.push_str(&format!(
                "        bool {n}_interleave = const()[name = string(\"{n}_interleave\"), val = bool(false)];\n",
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = concat(axis = {n}_axis, interleave = {n}_interleave, values = ({inputs_str}))[name = string(\"{n}\")];\n",
                top = l.top,
            ));
        }

        Op::Reshape(l) => {
            let shape = shape_map
                .get(l.top.as_str())
                .copied()
                .unwrap_or(Shape::channels(1));
            let sh = mil_shape(shape);
            let n = &l.name;
            let [s0, s1, s2, s3] = l.target_shape;
            out.push_str(&format!(
                "        tensor<int32, [4]> {n}_shape = const()[name = string(\"{n}_shape\"), val = tensor<int32, [4]>([{s0}, {s1}, {s2}, {s3}])];\n",
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = reshape(shape = {n}_shape, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top,
                bot = l.bottom,
            ));
        }

        Op::Flatten(l) => {
            let in_shape = shape_map
                .get(l.bottom.as_str())
                .copied()
                .unwrap_or(Shape::channels(1));
            let flat = in_shape.total_elements();
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .unwrap_or(Shape::channels(flat));
            let sh = mil_shape(out_shape);
            let n = &l.name;
            out.push_str(&format!(
                "        tensor<int32, [2]> {n}_shape = const()[name = string(\"{n}_shape\"), val = tensor<int32, [2]>([1, {flat}])];\n",
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = reshape(shape = {n}_shape, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top,
                bot = l.bottom,
            ));
        }

        Op::InstanceNorm(l) => {
            let shape = shape_map
                .get(l.top.as_str())
                .copied()
                .unwrap_or(Shape::channels(1));
            let sh = mil_shape(shape);
            let n = &l.name;
            let ch = l.channels;
            let eps = l.epsilon;

            let gamma_shape = format!("[{ch}]");
            let gamma_var = format!("{n}_gamma");
            let beta_var = format!("{n}_beta");

            blobfile_ref(all_blobs, *blob_index, &gamma_shape, &gamma_var, out);
            *blob_index += 1;

            out.push_str(&format!(
                "        tensor<fp16, [{ch}]> {beta_var} = const()[name = string(\"{beta_var}\"), val = tensor<fp16, [{ch}]>({})];\n",
                format!("[{}]", vec!["0.0"; ch as usize].join(", ")),
            ));
            out.push_str(&format!(
                "        fp32 {n}_eps = const()[name = string(\"{n}_eps\"), val = fp32({eps})];\n",
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = instance_norm(beta = {beta_var}, eps = {n}_eps, gamma = {gamma_var}, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top,
                bot = l.bottom,
            ));
        }

        Op::Pooling(l) => {
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .unwrap_or(Shape::channels(1));
            let sh = mil_shape(out_shape);
            let n = &l.name;
            let kh = l.kernel_height;
            let kw = l.kernel_width;
            let sh_s = l.stride_height;
            let sw_s = l.stride_width;

            let pad_type_str = match l.pad_mode {
                PadMode::Valid => "valid",
                PadMode::Same => "same_lower",
            };

            out.push_str(&format!(
                "        tensor<int32, [2]> {n}_kernel = const()[name = string(\"{n}_kernel\"), val = tensor<int32, [2]>([{kh}, {kw}])];\n",
            ));
            out.push_str(&format!(
                "        tensor<int32, [2]> {n}_strides = const()[name = string(\"{n}_strides\"), val = tensor<int32, [2]>([{sh_s}, {sw_s}])];\n",
            ));
            out.push_str(&format!(
                "        tensor<int32, [4]> {n}_pad = const()[name = string(\"{n}_pad\"), val = tensor<int32, [4]>([{}, {}, {}, {}])];\n",
                l.pad_top, l.pad_bottom, l.pad_left, l.pad_right,
            ));
            out.push_str(&format!(
                "        string {n}_pad_type = const()[name = string(\"{n}_pad_type\"), val = string(\"{pad_type_str}\")];\n",
            ));
            out.push_str(&format!(
                "        bool {n}_ceil = const()[name = string(\"{n}_ceil\"), val = bool(false)];\n",
            ));

            let mil_pool_op = match l.pool_type {
                PoolType::Max => "max_pool",
                PoolType::Average | PoolType::L2 => "avg_pool",
            };

            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = {mil_pool_op}(ceil_mode = {n}_ceil, \
                 kernel_sizes = {n}_kernel, pad = {n}_pad, pad_type = {n}_pad_type, strides = {n}_strides, \
                 x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top,
                bot = l.bottom,
            ));
        }

        Op::Padding(l) => {
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .unwrap_or(Shape::channels(1));
            let sh = mil_shape(out_shape);
            let n = &l.name;

            let mode_str = match l.pad_fill_mode {
                PadFillMode::Constant => "constant",
                PadFillMode::Reflect => "reflect",
                PadFillMode::Replicate => "replicate",
            };

            let (pt, pb, pl, pr) = (l.pad_top, l.pad_bottom, l.pad_left, l.pad_right);
            out.push_str(&format!(
                "        tensor<int32, [2, 2]> {n}_amounts = const()[name = string(\"{n}_amounts\"), val = tensor<int32, [2, 2]>([{pt}, {pb}, {pl}, {pr}])];\n",
            ));
            out.push_str(&format!(
                "        string {n}_mode = const()[name = string(\"{n}_mode\"), val = string(\"{mode_str}\")];\n",
            ));
            out.push_str(&format!(
                "        fp32 {n}_val = const()[name = string(\"{n}_val\"), val = fp32({})];\n",
                l.pad_value,
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = pad(constant_val = {n}_val, mode = {n}_mode, \
                 pad = {n}_amounts, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top,
                bot = l.bottom,
            ));
        }

        Op::Reduction(l) => {
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .unwrap_or(Shape::channels(1));
            let sh = mil_shape(out_shape);
            let n = &l.name;

            let mil_reduce_op = match l.mode {
                ReductionMode::Sum => "reduce_sum",
                ReductionMode::Mean => "reduce_mean",
                ReductionMode::Min => "reduce_min",
                ReductionMode::Max => "reduce_max",
            };

            out.push_str(&format!(
                "        tensor<int32, [1]> {n}_axes = const()[name = string(\"{n}_axes\"), val = tensor<int32, [1]>([{}])];\n",
                l.axis,
            ));
            out.push_str(&format!(
                "        bool {n}_keep = const()[name = string(\"{n}_keep\"), val = bool(true)];\n",
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = {mil_reduce_op}(axes = {n}_axes, keep_dims = {n}_keep, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top,
                bot = l.bottom,
            ));
        }

        Op::Matmul(l) => {
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .unwrap_or(Shape::channels(1));
            let sh = mil_shape(out_shape);
            let n = &l.name;
            let tx = if l.transpose_x { "true" } else { "false" };
            let ty = if l.transpose_y { "true" } else { "false" };
            out.push_str(&format!(
                "        bool {n}_tx = const()[name = string(\"{n}_tx\"), val = bool({tx})];\n",
            ));
            out.push_str(&format!(
                "        bool {n}_ty = const()[name = string(\"{n}_ty\"), val = bool({ty})];\n",
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = matmul(transpose_x = {n}_tx, transpose_y = {n}_ty, \
                 x = {bx}_f16, y = {by}_f16)[name = string(\"{n}\")];\n",
                top = l.top,
                bx = l.bottom_x,
                by = l.bottom_y,
            ));
        }

        Op::Transpose(l) => {
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .unwrap_or(Shape::channels(1));
            let sh = mil_shape(out_shape);
            let n = &l.name;
            let [p0, p1, p2, p3] = l.perm;
            out.push_str(&format!(
                "        tensor<int32, [4]> {n}_perm = const()[name = string(\"{n}_perm\"), val = tensor<int32, [4]>([{p0}, {p1}, {p2}, {p3}])];\n",
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = transpose(perm = {n}_perm, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top,
                bot = l.bottom,
            ));
        }

        Op::SliceBySize(l) => {
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .unwrap_or(Shape::channels(1));
            let sh = mil_shape(out_shape);
            let n = &l.name;
            let [b0, b1, b2, b3] = l.begin;
            let [s0, s1, s2, s3] = l.size;
            out.push_str(&format!(
                "        tensor<int32, [4]> {n}_begin = const()[name = string(\"{n}_begin\"), val = tensor<int32, [4]>([{b0}, {b1}, {b2}, {b3}])];\n",
            ));
            out.push_str(&format!(
                "        tensor<int32, [4]> {n}_size = const()[name = string(\"{n}_size\"), val = tensor<int32, [4]>([{s0}, {s1}, {s2}, {s3}])];\n",
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = slice_by_size(x = {bot}_f16, begin = {n}_begin, size = {n}_size)[name = string(\"{n}\")];\n",
                top = l.top,
                bot = l.bottom,
            ));
        }

        Op::ScalarOp(l) => {
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .unwrap_or(Shape::channels(1));
            let sh = mil_shape(out_shape);
            let n = &l.name;
            let s = l.scalar;
            out.push_str(&format!(
                "        fp16 {n}_s = const()[name = string(\"{n}_s\"), val = fp16({s})];\n",
            ));
            let op_str = match l.op {
                ScalarOpType::Mul => format!("mul(x = {bot}_f16, y = {n}_s)", bot = l.bottom),
                ScalarOpType::Add => format!("add(x = {bot}_f16, y = {n}_s)", bot = l.bottom),
                ScalarOpType::RSub => format!("sub(x = {n}_s, y = {bot}_f16)", bot = l.bottom),
                ScalarOpType::Pow => format!("pow(x = {bot}_f16, y = {n}_s)", bot = l.bottom),
            };
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = {op_str}[name = string(\"{n}\")];\n",
                top = l.top,
            ));
        }

        Op::DynConv(l) => {
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .unwrap_or(Shape::channels(1));
            let out_sh = mil_shape(out_shape);
            let n = &l.name;

            let pad_type_str = match l.pad_mode {
                PadMode::Valid => "valid",
                PadMode::Same => "same_lower",
            };
            emit_conv_constants(n, 0, 0, 0, 0, 1, 1, l.groups, l.groups, pad_type_str, out);

            // Reference the dynamic weight tensor variable (not a blob)
            out.push_str(&format!(
                "        tensor<fp16, {out_sh}> {top}_f16 = conv(\
                 dilations = {n}_dilations, groups = {n}_groups, pad = {n}_pad, \
                 pad_type = {n}_pad_type, strides = {n}_strides, weight = {wt}_f16, \
                 x = {src}_f16)[name = string(\"{n}\")];\n",
                top = l.top,
                src = l.source,
                wt = l.weight_source,
            ));
        }

        Op::LayerNorm(l) => {
            let shape = shape_map
                .get(l.top.as_str())
                .copied()
                .unwrap_or(Shape::channels(1));
            let sh = mil_shape(shape);
            let n = &l.name;
            let eps = l.epsilon;
            let axes_str = l
                .axes
                .iter()
                .map(|a| a.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let axes_len = l.axes.len();

            // Axes constant
            out.push_str(&format!(
                "        tensor<int32, [{axes_len}]> {n}_axes = const()[name = string(\"{n}_axes\"), val = tensor<int32, [{axes_len}]>([{axes_str}])];\n",
            ));
            // Epsilon constant
            out.push_str(&format!(
                "        fp32 {n}_eps = const()[name = string(\"{n}_eps\"), val = fp32({eps})];\n",
            ));

            if l.has_affine {
                // Gamma from weight blob
                let norm_size: usize = l
                    .axes
                    .iter()
                    .map(|&a| {
                        let dim = if a >= 0 { a as usize } else { (4 + a) as usize };
                        match dim {
                            0 => shape.batch,
                            1 => shape.channels,
                            2 => shape.height,
                            _ => shape.width,
                        }
                    })
                    .product();
                let gamma_shape = format!("[{norm_size}]");
                let gamma_var = format!("{n}_gamma");
                blobfile_ref(all_blobs, *blob_index, &gamma_shape, &gamma_var, out);
                *blob_index += 1;

                out.push_str(&format!(
                    "        tensor<fp16, {sh}> {top}_f16 = layer_norm(axes = {n}_axes, epsilon = {n}_eps, gamma = {gamma_var}, x = {bot}_f16)[name = string(\"{n}\")];\n",
                    top = l.top,
                    bot = l.bottom,
                ));
            } else {
                // No affine — just normalize
                out.push_str(&format!(
                    "        tensor<fp16, {sh}> {top}_f16 = layer_norm(axes = {n}_axes, epsilon = {n}_eps, x = {bot}_f16)[name = string(\"{n}\")];\n",
                    top = l.top,
                    bot = l.bottom,
                ));
            }
        }

        // ── New native MIL ops ────────────────────────────────────────────
        Op::Gelu(l) => {
            let sh = mil_shape(
                shape_map
                    .get(l.top.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            let n = &l.name;
            if l.mode != "EXACT" {
                out.push_str(&format!(
                    "        string {n}_mode = const()[name = string(\"{n}_mode\"), val = string(\"{}\")];\n", l.mode));
                out.push_str(&format!(
                    "        tensor<fp16, {sh}> {top}_f16 = gelu(mode = {n}_mode, x = {bot}_f16)[name = string(\"{n}\")];\n",
                    top = l.top, bot = l.bottom));
            } else {
                out.push_str(&format!(
                    "        tensor<fp16, {sh}> {top}_f16 = gelu(x = {bot}_f16)[name = string(\"{n}\")];\n",
                    top = l.top, bot = l.bottom));
            }
        }

        Op::Silu(l) => {
            let sh = mil_shape(
                shape_map
                    .get(l.top.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = silu(x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top, bot = l.bottom, n = l.name));
        }

        Op::Clip(l) => {
            let sh = mil_shape(
                shape_map
                    .get(l.top.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            let n = &l.name;
            out.push_str(&format!(
                "        fp32 {n}_alpha = const()[name = string(\"{n}_alpha\"), val = fp32({})];\n",
                l.alpha
            ));
            out.push_str(&format!(
                "        fp32 {n}_beta = const()[name = string(\"{n}_beta\"), val = fp32({})];\n",
                l.beta
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = clip(alpha = {n}_alpha, beta = {n}_beta, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top, bot = l.bottom));
        }

        Op::Relu6(l) => {
            let sh = mil_shape(
                shape_map
                    .get(l.top.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = relu6(x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top, bot = l.bottom, n = l.name));
        }

        Op::ClampedRelu(l) => {
            let sh = mil_shape(
                shape_map
                    .get(l.top.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            let n = &l.name;
            out.push_str(&format!(
                "        fp32 {n}_alpha = const()[name = string(\"{n}_alpha\"), val = fp32({})];\n",
                l.alpha
            ));
            out.push_str(&format!(
                "        fp32 {n}_beta = const()[name = string(\"{n}_beta\"), val = fp32({})];\n",
                l.beta
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = clamped_relu(alpha = {n}_alpha, beta = {n}_beta, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top, bot = l.bottom));
        }

        Op::ThresholdedRelu(l) => {
            let sh = mil_shape(
                shape_map
                    .get(l.top.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            let n = &l.name;
            out.push_str(&format!(
                "        fp32 {n}_alpha = const()[name = string(\"{n}_alpha\"), val = fp32({})];\n",
                l.alpha
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = thresholded_relu(alpha = {n}_alpha, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top, bot = l.bottom));
        }

        Op::ScaledTanh(l) => {
            let sh = mil_shape(
                shape_map
                    .get(l.top.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            let n = &l.name;
            out.push_str(&format!(
                "        fp32 {n}_alpha = const()[name = string(\"{n}_alpha\"), val = fp32({})];\n",
                l.alpha
            ));
            out.push_str(&format!(
                "        fp32 {n}_beta = const()[name = string(\"{n}_beta\"), val = fp32({})];\n",
                l.beta
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = scaled_tanh(alpha = {n}_alpha, beta = {n}_beta, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top, bot = l.bottom));
        }

        Op::Prelu(l) => {
            let sh = mil_shape(
                shape_map
                    .get(l.top.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            let n = &l.name;
            let ch = l.channels;
            let alpha_var = format!("{n}_alpha");
            blobfile_ref(all_blobs, *blob_index, &format!("[{ch}]"), &alpha_var, out);
            *blob_index += 1;
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = prelu(alpha = {alpha_var}, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top, bot = l.bottom));
        }

        Op::Cast(l) => {
            let sh = mil_shape(
                shape_map
                    .get(l.top.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            let n = &l.name;
            let dt = &l.dtype;
            out.push_str(&format!(
                "        tensor<{dt}, {sh}> {top} = cast(dtype = \"{dt}\", x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top, bot = l.bottom));
        }

        Op::Cumsum(l) => {
            let sh = mil_shape(
                shape_map
                    .get(l.top.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            let n = &l.name;
            out.push_str(&format!(
                "        int32 {n}_axis = const()[name = string(\"{n}_axis\"), val = int32({})];\n",
                l.axis
            ));
            out.push_str(&format!(
                "        bool {n}_excl = const()[name = string(\"{n}_excl\"), val = bool({})];\n",
                l.exclusive
            ));
            out.push_str(&format!(
                "        bool {n}_rev = const()[name = string(\"{n}_rev\"), val = bool({})];\n",
                l.reverse
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = cumsum(axis = {n}_axis, exclusive = {n}_excl, reverse = {n}_rev, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top, bot = l.bottom));
        }

        Op::Tile(l) => {
            let sh = mil_shape(
                shape_map
                    .get(l.top.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            let n = &l.name;
            let reps_str = l
                .reps
                .iter()
                .map(|r| r.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let reps_len = l.reps.len();
            out.push_str(&format!(
                "        tensor<int32, [{reps_len}]> {n}_reps = const()[name = string(\"{n}_reps\"), val = tensor<int32, [{reps_len}]>([{reps_str}])];\n"));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = tile(reps = {n}_reps, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top, bot = l.bottom));
        }

        Op::ExpandDims(l) => {
            let sh = mil_shape(
                shape_map
                    .get(l.top.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            let n = &l.name;
            let axes_str = l
                .axes
                .iter()
                .map(|a| a.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let axes_len = l.axes.len();
            out.push_str(&format!(
                "        tensor<int32, [{axes_len}]> {n}_axes = const()[name = string(\"{n}_axes\"), val = tensor<int32, [{axes_len}]>([{axes_str}])];\n"));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = expand_dims(axes = {n}_axes, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top, bot = l.bottom));
        }

        Op::Squeeze(l) => {
            let sh = mil_shape(
                shape_map
                    .get(l.top.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            let n = &l.name;
            if let Some(ref axes) = l.axes {
                let axes_str = axes
                    .iter()
                    .map(|a| a.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let axes_len = axes.len();
                out.push_str(&format!(
                    "        tensor<int32, [{axes_len}]> {n}_axes = const()[name = string(\"{n}_axes\"), val = tensor<int32, [{axes_len}]>([{axes_str}])];\n"));
                out.push_str(&format!(
                    "        tensor<fp16, {sh}> {top}_f16 = squeeze(axes = {n}_axes, x = {bot}_f16)[name = string(\"{n}\")];\n",
                    top = l.top, bot = l.bottom));
            } else {
                out.push_str(&format!(
                    "        tensor<fp16, {sh}> {top}_f16 = squeeze(x = {bot}_f16)[name = string(\"{n}\")];\n",
                    top = l.top, bot = l.bottom));
            }
        }

        Op::Split(l) => {
            let n = &l.name;
            out.push_str(&format!(
                "        int32 {n}_axis = const()[name = string(\"{n}_axis\"), val = int32({})];\n",
                l.axis
            ));
            out.push_str(&format!(
                "        int32 {n}_num = const()[name = string(\"{n}_num\"), val = int32({})];\n",
                l.num_splits
            ));
            if let Some(ref sizes) = l.split_sizes {
                let sizes_str = sizes
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let sizes_len = sizes.len();
                out.push_str(&format!(
                    "        tensor<int32, [{sizes_len}]> {n}_sizes = const()[name = string(\"{n}_sizes\"), val = tensor<int32, [{sizes_len}]>([{sizes_str}])];\n"));
            }
            // Multi-output LHS
            let lhs: Vec<String> = l
                .tops
                .iter()
                .enumerate()
                .map(|(i, top)| {
                    let s = shape_map
                        .get(top.as_str())
                        .copied()
                        .unwrap_or(Shape::channels(1));
                    format!("tensor<fp16, {}> {}_f16", mil_shape(s), top)
                })
                .collect();
            let lhs_str = lhs.join(", ");
            if l.split_sizes.is_some() {
                out.push_str(&format!(
                    "        {lhs_str} = split(axis = {n}_axis, num_splits = {n}_num, split_sizes = {n}_sizes, x = {bot}_f16)[name = string(\"{n}\")];\n",
                    bot = l.bottom));
            } else {
                out.push_str(&format!(
                    "        {lhs_str} = split(axis = {n}_axis, num_splits = {n}_num, x = {bot}_f16)[name = string(\"{n}\")];\n",
                    bot = l.bottom));
            }
        }

        Op::Topk(l) => {
            let val_sh = mil_shape(
                shape_map
                    .get(l.top_values.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            let idx_sh = mil_shape(
                shape_map
                    .get(l.top_indices.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            let n = &l.name;
            out.push_str(&format!(
                "        int32 {n}_k = const()[name = string(\"{n}_k\"), val = int32({})];\n",
                l.k
            ));
            out.push_str(&format!(
                "        int32 {n}_axis = const()[name = string(\"{n}_axis\"), val = int32({})];\n",
                l.axis
            ));
            out.push_str(&format!(
                "        bool {n}_asc = const()[name = string(\"{n}_asc\"), val = bool({})];\n",
                l.ascending
            ));
            out.push_str(&format!(
                "        tensor<fp16, {val_sh}> {vals}_f16, tensor<int32, {idx_sh}> {idxs} = topk(ascending = {n}_asc, axis = {n}_axis, k = {n}_k, x = {bot}_f16)[name = string(\"{n}\")];\n",
                vals = l.top_values, idxs = l.top_indices, bot = l.bottom));
        }

        Op::Argsort(l) => {
            let sh = mil_shape(
                shape_map
                    .get(l.top.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            let n = &l.name;
            out.push_str(&format!(
                "        int32 {n}_axis = const()[name = string(\"{n}_axis\"), val = int32({})];\n",
                l.axis
            ));
            out.push_str(&format!(
                "        bool {n}_asc = const()[name = string(\"{n}_asc\"), val = bool({})];\n",
                l.ascending
            ));
            out.push_str(&format!(
                "        tensor<int32, {sh}> {top} = argsort(ascending = {n}_asc, axis = {n}_axis, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top, bot = l.bottom));
        }

        Op::ReduceArgmax(l) => {
            let sh = mil_shape(
                shape_map
                    .get(l.top.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            let n = &l.name;
            out.push_str(&format!(
                "        int32 {n}_axis = const()[name = string(\"{n}_axis\"), val = int32({})];\n",
                l.axis
            ));
            out.push_str(&format!(
                "        bool {n}_keep = const()[name = string(\"{n}_keep\"), val = bool({})];\n",
                l.keep_dims
            ));
            out.push_str(&format!(
                "        tensor<int32, {sh}> {top} = reduce_argmax(axis = {n}_axis, keep_dims = {n}_keep, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top, bot = l.bottom));
        }

        Op::ReduceArgmin(l) => {
            let sh = mil_shape(
                shape_map
                    .get(l.top.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            let n = &l.name;
            out.push_str(&format!(
                "        int32 {n}_axis = const()[name = string(\"{n}_axis\"), val = int32({})];\n",
                l.axis
            ));
            out.push_str(&format!(
                "        bool {n}_keep = const()[name = string(\"{n}_keep\"), val = bool({})];\n",
                l.keep_dims
            ));
            out.push_str(&format!(
                "        tensor<int32, {sh}> {top} = reduce_argmin(axis = {n}_axis, keep_dims = {n}_keep, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top, bot = l.bottom));
        }

        Op::Select(l) => {
            let sh = mil_shape(
                shape_map
                    .get(l.top.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = select(a = {a}_f16, b = {b}_f16, cond = {cond})[name = string(\"{n}\")];\n",
                top = l.top, a = l.a, b = l.b, cond = l.cond, n = l.name));
        }

        Op::L2Norm(l) => {
            let sh = mil_shape(
                shape_map
                    .get(l.top.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            let n = &l.name;
            out.push_str(&format!(
                "        fp32 {n}_eps = const()[name = string(\"{n}_eps\"), val = fp32({})];\n",
                l.epsilon
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = l2_norm(epsilon = {n}_eps, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top, bot = l.bottom));
        }

        Op::BatchNorm(l) => {
            let sh = mil_shape(
                shape_map
                    .get(l.top.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            let n = &l.name;
            let ch = l.channels;
            // mean (required)
            let mean_var = format!("{n}_mean");
            blobfile_ref(all_blobs, *blob_index, &format!("[{ch}]"), &mean_var, out);
            *blob_index += 1;
            // variance (required)
            let var_var = format!("{n}_var");
            blobfile_ref(all_blobs, *blob_index, &format!("[{ch}]"), &var_var, out);
            *blob_index += 1;
            // gamma (optional)
            let gamma_part = if l.gamma.is_some() {
                let gv = format!("{n}_gamma");
                blobfile_ref(all_blobs, *blob_index, &format!("[{ch}]"), &gv, out);
                *blob_index += 1;
                format!(", gamma = {gv}")
            } else {
                String::new()
            };
            // beta (optional)
            let beta_part = if l.beta.is_some() {
                let bv = format!("{n}_beta");
                blobfile_ref(all_blobs, *blob_index, &format!("[{ch}]"), &bv, out);
                *blob_index += 1;
                format!(", beta = {bv}")
            } else {
                String::new()
            };
            out.push_str(&format!(
                "        fp32 {n}_eps = const()[name = string(\"{n}_eps\"), val = fp32({})];\n",
                l.epsilon
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = batch_norm(epsilon = {n}_eps{gamma_part}{beta_part}, mean = {mean_var}, variance = {var_var}, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top, bot = l.bottom));
        }

        Op::ReduceProd(l) => {
            let sh = mil_shape(
                shape_map
                    .get(l.top.as_str())
                    .copied()
                    .unwrap_or(Shape::channels(1)),
            );
            let n = &l.name;
            let axes_str = l
                .axes
                .iter()
                .map(|a| a.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let axes_len = l.axes.len();
            out.push_str(&format!(
                "        tensor<int32, [{axes_len}]> {n}_axes = const()[name = string(\"{n}_axes\"), val = tensor<int32, [{axes_len}]>([{axes_str}])];\n"));
            out.push_str(&format!(
                "        bool {n}_keep = const()[name = string(\"{n}_keep\"), val = bool({})];\n",
                l.keep_dims
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = reduce_prod(axes = {n}_axes, keep_dims = {n}_keep, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top, bot = l.bottom));
        }
    }
}

fn emit_conv_constants(
    n: &str,
    pt: usize,
    pb: usize,
    pl: usize,
    pr: usize,
    sh: usize,
    sw: usize,
    groups: usize,
    _n_parallel: usize,
    pad_type: &str,
    out: &mut String,
) {
    out.push_str(&format!(
        "        string {n}_pad_type = const()[name = string(\"{n}_pad_type\"), val = string(\"{pad_type}\")];\n",
    ));
    out.push_str(&format!(
        "        tensor<int32, [2]> {n}_strides = const()[name = string(\"{n}_strides\"), val = tensor<int32, [2]>([{sh}, {sw}])];\n",
    ));
    out.push_str(&format!(
        "        tensor<int32, [4]> {n}_pad = const()[name = string(\"{n}_pad\"), val = tensor<int32, [4]>([{pt}, {pb}, {pl}, {pr}])];\n",
    ));
    out.push_str(&format!(
        "        tensor<int32, [2]> {n}_dilations = const()[name = string(\"{n}_dilations\"), val = tensor<int32, [2]>([1, 1])];\n",
    ));
    out.push_str(&format!(
        "        int32 {n}_groups = const()[name = string(\"{n}_groups\"), val = int32({groups})];\n",
    ));
}
