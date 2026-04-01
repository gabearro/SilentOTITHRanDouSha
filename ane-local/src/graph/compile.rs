use std::collections::HashSet;

use objc2_foundation::NSQualityOfService;

use crate::ops::weights::WeightBlob;
use crate::{ConstantOp, Op, Shape};

use super::tensor::Tensor;
use super::Graph;

/// Minimum spatial width (the W dimension in NCHW) that the ANE hardware
/// can execute.  Programs compiled with smaller widths succeed at the MIL
/// level but fail at runtime with `ANEProgramProcessRequestDirect` status
/// 0x1d.  Discovered empirically on M-series (M1–M5) / macOS 25–26.
pub const MIN_SPATIAL_WIDTH: usize = 64;

impl Graph {
    pub(crate) fn to_ops_and_shapes(&self) -> (Box<[Op]>, Box<[(String, Shape)]>) {
        let mut shapes: Vec<(String, Shape)> = self
            .inputs
            .iter()
            .map(|t| (Self::tensor_name(*t), t.shape))
            .collect();

        let regular_ops: Vec<Op> = self
            .ops
            .iter()
            .map(|op| {
                shapes.push((Self::tensor_name(op.output), op.output.shape));
                op.op.clone()
            })
            .collect();

        let all_bottoms: HashSet<&str> = regular_ops
            .iter()
            .flat_map(|op| op.bottom_names())
            .collect();

        let mut all_ops: Vec<Op> = Vec::new();
        for (&id, (data, shape)) in &self.constants {
            let name = Self::tensor_name(Tensor { id, shape: *shape });
            if all_bottoms.contains(name.as_str()) {
                shapes.push((name.clone(), *shape));
                all_ops.push(Op::Constant(ConstantOp {
                    name: format!("const_{name}"),
                    top: name,
                    data: WeightBlob::from_f16_bytes(data.clone()),
                }));
            }
        }
        all_ops.extend(regular_ops);

        (all_ops.into_boxed_slice(), shapes.into_boxed_slice())
    }

    /// Collect tensor names that are used as dynamic convolution weight sources.
    /// These placeholders represent conv kernels (not activation tensors) and
    /// are exempt from the MIN_SPATIAL_WIDTH check. For 1×1 kernels the shape
    /// is `[OC, IC, 1, 1]` (width=1); for general kernels it may be
    /// `[OC, IC/groups, KH, KW]` where KW ≥ 64.
    fn dyn_conv_weight_names(&self) -> HashSet<String> {
        self.ops
            .iter()
            .filter_map(|op| match &op.op {
                Op::DynConv(dc) => Some(dc.weight_source.clone()),
                _ => None,
            })
            .collect()
    }

    /// Compile this graph to an ANE executable.
    ///
    /// Returns [`crate::Error::SpatialWidthTooSmall`] if any non-weight placeholder
    /// has a width smaller than [`MIN_SPATIAL_WIDTH`]. Placeholders used as weight
    /// sources for dynamic convolutions are exempt (they have kernel dimensions,
    /// not spatial dimensions).
    pub fn compile(
        &self,
        quality_of_service: NSQualityOfService,
    ) -> Result<crate::Executable, crate::Error> {
        let weight_names = self.dyn_conv_weight_names();
        for input in &self.inputs {
            let name = Self::tensor_name(*input);
            if weight_names.contains(&name) {
                continue; // weight placeholder for dynamic conv — exempt from spatial check
            }
            if input.shape.width < MIN_SPATIAL_WIDTH {
                return Err(crate::Error::SpatialWidthTooSmall {
                    name,
                    width: input.shape.width,
                    min: MIN_SPATIAL_WIDTH,
                });
            }
        }
        crate::client::compile_network(self, quality_of_service)
    }
}
