//! Metal GPU dispatch for MoE expert SwiGLU with fused MXFP4 dequantization.

use metal::*;

#[repr(C)]
#[derive(Clone, Copy)]
struct MoeConstants {
    hidden_size: u32,
    inter_size: u32,
    swiglu_clamp: f32,
}

/// Pre-loaded GPU buffers for one expert's MXFP4 data.
pub struct ExpertGpuBuffers {
    pub gu_blocks: Buffer,
    pub gu_scales: Buffer,
    pub down_blocks: Buffer,
    pub down_scales: Buffer,
}

/// Pre-loaded GPU buffers for all experts in all layers.
pub struct PreloadedExperts {
    /// experts[layer][expert_idx]
    pub experts: Vec<Vec<ExpertGpuBuffers>>,
    /// bias[layer] — combined gate_up_bias + down_bias for all experts
    pub biases: Vec<Vec<Buffer>>, // [layer][expert] = buffer with [2*inter + hidden] f32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LmConstants {
    vocab_size: u32,
    hidden_size: u32,
}

pub struct GpuMoeDispatcher {
    device: Device,
    queue: CommandQueue,
    pipeline_gate_up: ComputePipelineState,
    pipeline_down: ComputePipelineState,
    pipeline_lm_head: ComputePipelineState,
    input_buf: Buffer,
    gated_buf: Buffer,
    output_buf: Buffer,
    bias_buf: Buffer,
    gu_blocks_buf: Buffer,
    gu_scales_buf: Buffer,
    down_blocks_buf: Buffer,
    down_scales_buf: Buffer,
    // LM head buffers
    lm_input_buf: Buffer,
    pub lm_output_buf: Option<Buffer>,
    pub lm_weight_buf: Option<Buffer>,
    hidden_size: usize,
    inter_size: usize,
    swiglu_clamp: f32,
}

impl GpuMoeDispatcher {
    pub fn new(hidden_size: usize, inter_size: usize, swiglu_clamp: f32) -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal GPU")?;
        let queue = device.new_command_queue();

        let lib_data = include_bytes!(concat!(env!("OUT_DIR"), "/moe_swiglu.metallib"));
        let library = device
            .new_library_with_data(lib_data)
            .map_err(|e| format!("Metal MoE library load: {}", e))?;

        let fn_gate_up = library
            .get_function("moe_gate_up", None)
            .map_err(|e| format!("moe_gate_up not found: {}", e))?;
        let fn_down = library
            .get_function("moe_down_proj", None)
            .map_err(|e| format!("moe_down_proj not found: {}", e))?;
        let fn_lm = library
            .get_function("lm_head_matmul", None)
            .map_err(|e| format!("lm_head_matmul not found: {}", e))?;

        let pipeline_gate_up = device
            .new_compute_pipeline_state_with_function(&fn_gate_up)
            .map_err(|e| format!("gate_up pipeline: {}", e))?;
        let pipeline_down = device
            .new_compute_pipeline_state_with_function(&fn_down)
            .map_err(|e| format!("down pipeline: {}", e))?;
        let pipeline_lm_head = device
            .new_compute_pipeline_state_with_function(&fn_lm)
            .map_err(|e| format!("lm_head pipeline: {}", e))?;

        let input_buf = device.new_buffer(
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let gated_buf = device.new_buffer(
            (inter_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buf = device.new_buffer(
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let bias_buf = device.new_buffer(
            ((2 * inter_size + hidden_size) * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Pre-allocate reusable MXFP4 buffers (max expert size)
        let gu_elems = 2 * inter_size * hidden_size;
        let d_elems = hidden_size * inter_size;
        let gu_blocks_buf =
            device.new_buffer((gu_elems / 2) as u64, MTLResourceOptions::StorageModeShared);
        let gu_scales_buf = device.new_buffer(
            (gu_elems / 32) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let down_blocks_buf =
            device.new_buffer((d_elems / 2) as u64, MTLResourceOptions::StorageModeShared);
        let down_scales_buf =
            device.new_buffer((d_elems / 32) as u64, MTLResourceOptions::StorageModeShared);

        let lm_input_buf = device.new_buffer(
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(GpuMoeDispatcher {
            device,
            queue,
            pipeline_gate_up,
            pipeline_down,
            pipeline_lm_head,
            input_buf,
            gated_buf,
            output_buf,
            bias_buf,
            gu_blocks_buf,
            gu_scales_buf,
            down_blocks_buf,
            down_scales_buf,
            lm_input_buf,
            lm_output_buf: None,
            lm_weight_buf: None,
            hidden_size,
            inter_size,
            swiglu_clamp,
        })
    }

    /// Pre-load all expert MXFP4 data to GPU buffers at model load time.
    pub fn preload_experts(
        &self,
        layers: &[crate::gpt_oss::weights::LayerWeights],
        config: &crate::gpt_oss::config::GptOssConfig,
    ) -> PreloadedExperts {
        let ne = config.num_local_experts;
        let inter = config.intermediate_size;
        let hs = config.hidden_size;

        eprint!("  Pre-loading expert MXFP4 to GPU...");
        let experts: Vec<Vec<ExpertGpuBuffers>> = layers
            .iter()
            .map(|lw| {
                (0..ne)
                    .map(|ei| {
                        let expert = &lw.experts[ei];
                        ExpertGpuBuffers {
                            gu_blocks: self.device.new_buffer_with_data(
                                expert.gate_up_blocks.as_ptr() as *const _,
                                expert.gate_up_blocks.len() as u64,
                                MTLResourceOptions::StorageModeShared,
                            ),
                            gu_scales: self.device.new_buffer_with_data(
                                expert.gate_up_scales.as_ptr() as *const _,
                                expert.gate_up_scales.len() as u64,
                                MTLResourceOptions::StorageModeShared,
                            ),
                            down_blocks: self.device.new_buffer_with_data(
                                expert.down_blocks.as_ptr() as *const _,
                                expert.down_blocks.len() as u64,
                                MTLResourceOptions::StorageModeShared,
                            ),
                            down_scales: self.device.new_buffer_with_data(
                                expert.down_scales.as_ptr() as *const _,
                                expert.down_scales.len() as u64,
                                MTLResourceOptions::StorageModeShared,
                            ),
                        }
                    })
                    .collect()
            })
            .collect();

        let biases: Vec<Vec<Buffer>> = layers
            .iter()
            .map(|lw| {
                (0..ne)
                    .map(|ei| {
                        let expert = &lw.experts[ei];
                        let total_bias = 2 * inter + hs;
                        let mut bias_data = vec![0.0f32; total_bias];
                        let gu_len = expert.gate_up_bias.len().min(2 * inter);
                        bias_data[..gu_len].copy_from_slice(&expert.gate_up_bias[..gu_len]);
                        let db_len = expert.down_bias.len().min(hs);
                        bias_data[2 * inter..2 * inter + db_len]
                            .copy_from_slice(&expert.down_bias[..db_len]);
                        self.device.new_buffer_with_data(
                            bias_data.as_ptr() as *const _,
                            (total_bias * 4) as u64,
                            MTLResourceOptions::StorageModeShared,
                        )
                    })
                    .collect()
            })
            .collect();

        let total_mb: f64 = experts
            .iter()
            .flat_map(|l| l.iter())
            .map(|e| {
                (e.gu_blocks.length()
                    + e.gu_scales.length()
                    + e.down_blocks.length()
                    + e.down_scales.length()) as f64
            })
            .sum::<f64>()
            / 1e6;
        eprintln!(" {:.0}MB", total_mb);

        PreloadedExperts { experts, biases }
    }

    /// Dispatch expert using pre-loaded GPU buffers (no per-dispatch allocation).
    pub fn dispatch_expert_preloaded(
        &self,
        input: &[f32],
        preloaded: &PreloadedExperts,
        layer: usize,
        expert_idx: usize,
    ) -> Vec<f32> {
        let hs = self.hidden_size;
        let inter = self.inter_size;
        let bufs = &preloaded.experts[layer][expert_idx];
        let bias = &preloaded.biases[layer][expert_idx];

        // Upload only the input activation (2880 × 4B = 11.5KB)
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.input_buf.contents() as *mut f32,
                hs,
            );
        }

        let consts = MoeConstants {
            hidden_size: hs as u32,
            inter_size: inter as u32,
            swiglu_clamp: self.swiglu_clamp,
        };

        // Phase 1: gate_up
        {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline_gate_up);
            enc.set_buffer(0, Some(&bufs.gu_blocks), 0);
            enc.set_buffer(1, Some(&bufs.gu_scales), 0);
            enc.set_buffer(2, Some(bias), 0);
            enc.set_buffer(3, Some(&self.input_buf), 0);
            enc.set_buffer(4, Some(&self.gated_buf), 0);
            enc.set_bytes(
                5,
                std::mem::size_of::<MoeConstants>() as u64,
                &consts as *const MoeConstants as *const _,
            );
            let grid = MTLSize::new(inter as u64, 1, 1);
            let tg = MTLSize::new(
                self.pipeline_gate_up
                    .thread_execution_width()
                    .min(inter as u64),
                1,
                1,
            );
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        // Phase 2: down projection
        {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline_down);
            enc.set_buffer(0, Some(&bufs.down_blocks), 0);
            enc.set_buffer(1, Some(&bufs.down_scales), 0);
            enc.set_buffer(2, Some(bias), (2 * inter * 4) as u64); // down_bias offset
            enc.set_buffer(3, Some(&self.gated_buf), 0);
            enc.set_buffer(4, Some(&self.output_buf), 0);
            enc.set_bytes(
                5,
                std::mem::size_of::<MoeConstants>() as u64,
                &consts as *const MoeConstants as *const _,
            );
            let grid = MTLSize::new(hs as u64, 1, 1);
            let tg = MTLSize::new(
                self.pipeline_down.thread_execution_width().min(hs as u64),
                1,
                1,
            );
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let mut output = vec![0.0f32; hs];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.output_buf.contents() as *const f32,
                output.as_mut_ptr(),
                hs,
            );
        }
        output
    }

    /// Dispatch expert using reusable GPU buffers (memcpy instead of new_buffer_with_data).
    pub fn dispatch_expert(
        &self,
        input: &[f32],
        gate_up_blocks: &[u8],
        gate_up_scales: &[u8],
        gate_up_bias: &[f32],
        down_blocks: &[u8],
        down_scales: &[u8],
        down_bias: &[f32],
    ) -> Vec<f32> {
        let hs = self.hidden_size;
        let inter = self.inter_size;

        // Copy input activation (11.5KB — fast)
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.input_buf.contents() as *mut f32,
                hs,
            );
        }

        // Copy biases into reusable buffer
        {
            let bias_ptr = self.bias_buf.contents() as *mut f32;
            let gu_len = gate_up_bias.len().min(2 * inter);
            let db_len = down_bias.len().min(hs);
            unsafe {
                std::ptr::copy_nonoverlapping(gate_up_bias.as_ptr(), bias_ptr, gu_len);
                std::ptr::write_bytes(bias_ptr.add(gu_len), 0, 2 * inter - gu_len);
                std::ptr::copy_nonoverlapping(down_bias.as_ptr(), bias_ptr.add(2 * inter), db_len);
                std::ptr::write_bytes(bias_ptr.add(2 * inter + db_len), 0, hs - db_len);
            }
        }

        // Copy MXFP4 data into reusable buffers (memcpy into existing GPU shared memory)
        unsafe {
            let gu_len = gate_up_blocks
                .len()
                .min(self.gu_blocks_buf.length() as usize);
            std::ptr::copy_nonoverlapping(
                gate_up_blocks.as_ptr(),
                self.gu_blocks_buf.contents() as *mut u8,
                gu_len,
            );
            let gus_len = gate_up_scales
                .len()
                .min(self.gu_scales_buf.length() as usize);
            std::ptr::copy_nonoverlapping(
                gate_up_scales.as_ptr(),
                self.gu_scales_buf.contents() as *mut u8,
                gus_len,
            );
            let d_len = down_blocks
                .len()
                .min(self.down_blocks_buf.length() as usize);
            std::ptr::copy_nonoverlapping(
                down_blocks.as_ptr(),
                self.down_blocks_buf.contents() as *mut u8,
                d_len,
            );
            let ds_len = down_scales
                .len()
                .min(self.down_scales_buf.length() as usize);
            std::ptr::copy_nonoverlapping(
                down_scales.as_ptr(),
                self.down_scales_buf.contents() as *mut u8,
                ds_len,
            );
        }

        let consts = MoeConstants {
            hidden_size: hs as u32,
            inter_size: inter as u32,
            swiglu_clamp: self.swiglu_clamp,
        };

        // Single command buffer with both phases (eliminates 1 commit+wait)
        let cmd = self.queue.new_command_buffer();

        // Phase 1: gate_up → gated
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline_gate_up);
            enc.set_buffer(0, Some(&self.gu_blocks_buf), 0);
            enc.set_buffer(1, Some(&self.gu_scales_buf), 0);
            enc.set_buffer(2, Some(&self.bias_buf), 0);
            enc.set_buffer(3, Some(&self.input_buf), 0);
            enc.set_buffer(4, Some(&self.gated_buf), 0);
            enc.set_bytes(
                5,
                std::mem::size_of::<MoeConstants>() as u64,
                &consts as *const MoeConstants as *const _,
            );
            let grid = MTLSize::new(inter as u64, 1, 1);
            let tg = MTLSize::new(
                self.pipeline_gate_up
                    .thread_execution_width()
                    .min(inter as u64),
                1,
                1,
            );
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
        }

        // Phase 2: down → output (same command buffer, serialized after phase 1)
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline_down);
            enc.set_buffer(0, Some(&self.down_blocks_buf), 0);
            enc.set_buffer(1, Some(&self.down_scales_buf), 0);
            enc.set_buffer(2, Some(&self.bias_buf), (2 * inter * 4) as u64);
            enc.set_buffer(3, Some(&self.gated_buf), 0);
            enc.set_buffer(4, Some(&self.output_buf), 0);
            enc.set_bytes(
                5,
                std::mem::size_of::<MoeConstants>() as u64,
                &consts as *const MoeConstants as *const _,
            );
            let grid = MTLSize::new(hs as u64, 1, 1);
            let tg = MTLSize::new(
                self.pipeline_down.thread_execution_width().min(hs as u64),
                1,
                1,
            );
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
        }

        cmd.commit();
        cmd.wait_until_completed();

        let mut output = vec![0.0f32; hs];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.output_buf.contents() as *const f32,
                output.as_mut_ptr(),
                hs,
            );
        }
        output
    }

    /// Dispatch multiple experts in one command buffer, accumulating weighted outputs.
    /// Eliminates per-expert commit+wait overhead (3 fewer round-trips for 4 experts).
    pub fn dispatch_experts_batched(
        &self,
        input: &[f32],
        experts: &[(usize, f32, &[u8], &[u8], &[f32], &[u8], &[u8], &[f32])],
        // Each tuple: (expert_idx, weight, gu_blocks, gu_scales, gu_bias, d_blocks, d_scales, d_bias)
    ) -> Vec<f32> {
        let hs = self.hidden_size;
        let inter = self.inter_size;

        // Upload input once
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.input_buf.contents() as *mut f32,
                hs,
            );
        }

        let consts = MoeConstants {
            hidden_size: hs as u32,
            inter_size: inter as u32,
            swiglu_clamp: self.swiglu_clamp,
        };
        let mut accumulated = vec![0.0f32; hs];

        // Process each expert sequentially but with batched command buffers
        for &(_, weight, gu_blocks, gu_scales, gu_bias, d_blocks, d_scales, d_bias) in experts {
            // Copy MXFP4 data + bias into reusable buffers
            unsafe {
                let gu_len = gu_blocks.len().min(self.gu_blocks_buf.length() as usize);
                std::ptr::copy_nonoverlapping(
                    gu_blocks.as_ptr(),
                    self.gu_blocks_buf.contents() as *mut u8,
                    gu_len,
                );
                let gus_len = gu_scales.len().min(self.gu_scales_buf.length() as usize);
                std::ptr::copy_nonoverlapping(
                    gu_scales.as_ptr(),
                    self.gu_scales_buf.contents() as *mut u8,
                    gus_len,
                );
                let d_len = d_blocks.len().min(self.down_blocks_buf.length() as usize);
                std::ptr::copy_nonoverlapping(
                    d_blocks.as_ptr(),
                    self.down_blocks_buf.contents() as *mut u8,
                    d_len,
                );
                let ds_len = d_scales.len().min(self.down_scales_buf.length() as usize);
                std::ptr::copy_nonoverlapping(
                    d_scales.as_ptr(),
                    self.down_scales_buf.contents() as *mut u8,
                    ds_len,
                );
            }
            {
                let bias_ptr = self.bias_buf.contents() as *mut f32;
                let gu_len = gu_bias.len().min(2 * inter);
                let db_len = d_bias.len().min(hs);
                unsafe {
                    std::ptr::copy_nonoverlapping(gu_bias.as_ptr(), bias_ptr, gu_len);
                    std::ptr::write_bytes(bias_ptr.add(gu_len), 0, 2 * inter - gu_len);
                    std::ptr::copy_nonoverlapping(d_bias.as_ptr(), bias_ptr.add(2 * inter), db_len);
                    std::ptr::write_bytes(bias_ptr.add(2 * inter + db_len), 0, hs - db_len);
                }
            }

            // Single command buffer: gate_up + down
            let cmd = self.queue.new_command_buffer();

            {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.pipeline_gate_up);
                enc.set_buffer(0, Some(&self.gu_blocks_buf), 0);
                enc.set_buffer(1, Some(&self.gu_scales_buf), 0);
                enc.set_buffer(2, Some(&self.bias_buf), 0);
                enc.set_buffer(3, Some(&self.input_buf), 0);
                enc.set_buffer(4, Some(&self.gated_buf), 0);
                enc.set_bytes(
                    5,
                    std::mem::size_of::<MoeConstants>() as u64,
                    &consts as *const MoeConstants as *const _,
                );
                enc.dispatch_threads(
                    MTLSize::new(inter as u64, 1, 1),
                    MTLSize::new(
                        self.pipeline_gate_up
                            .thread_execution_width()
                            .min(inter as u64),
                        1,
                        1,
                    ),
                );
                enc.end_encoding();
            }
            {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.pipeline_down);
                enc.set_buffer(0, Some(&self.down_blocks_buf), 0);
                enc.set_buffer(1, Some(&self.down_scales_buf), 0);
                enc.set_buffer(2, Some(&self.bias_buf), (2 * inter * 4) as u64);
                enc.set_buffer(3, Some(&self.gated_buf), 0);
                enc.set_buffer(4, Some(&self.output_buf), 0);
                enc.set_bytes(
                    5,
                    std::mem::size_of::<MoeConstants>() as u64,
                    &consts as *const MoeConstants as *const _,
                );
                enc.dispatch_threads(
                    MTLSize::new(hs as u64, 1, 1),
                    MTLSize::new(
                        self.pipeline_down.thread_execution_width().min(hs as u64),
                        1,
                        1,
                    ),
                );
                enc.end_encoding();
            }

            cmd.commit();
            cmd.wait_until_completed();

            // Accumulate weighted output
            unsafe {
                let out = self.output_buf.contents() as *const f32;
                for i in 0..hs {
                    let v = *out.add(i);
                    if v.is_finite() {
                        accumulated[i] += weight * v;
                    }
                }
            }
        }

        accumulated
    }

    /// Initialize LM head GPU buffers with pre-loaded weights.
    /// Call once at model load time.
    pub fn init_lm_head(&mut self, weights: &[f32], vocab_size: usize) {
        self.lm_weight_buf = Some(self.device.new_buffer_with_data(
            weights.as_ptr() as *const _,
            (weights.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        self.lm_output_buf = Some(self.device.new_buffer(
            (vocab_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
    }

    /// Dispatch LM head matmul on GPU. Returns logits[vocab_size].
    pub fn dispatch_lm_head(&self, input: &[f32], vocab_size: usize) -> Vec<f32> {
        let hs = self.hidden_size;
        let weight_buf = self
            .lm_weight_buf
            .as_ref()
            .expect("call init_lm_head first");
        let output_buf = self
            .lm_output_buf
            .as_ref()
            .expect("call init_lm_head first");

        // Upload input
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.lm_input_buf.contents() as *mut f32,
                hs,
            );
        }

        let consts = LmConstants {
            vocab_size: vocab_size as u32,
            hidden_size: hs as u32,
        };

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline_lm_head);
        enc.set_buffer(0, Some(weight_buf), 0);
        enc.set_buffer(1, Some(&self.lm_input_buf), 0);
        enc.set_buffer(2, Some(output_buf), 0);
        enc.set_bytes(
            3,
            std::mem::size_of::<LmConstants>() as u64,
            &consts as *const LmConstants as *const _,
        );

        let grid = MTLSize::new(vocab_size as u64, 1, 1);
        let tg = MTLSize::new(
            self.pipeline_lm_head
                .thread_execution_width()
                .min(vocab_size as u64),
            1,
            1,
        );
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mut logits = vec![0.0f32; vocab_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_buf.contents() as *const f32,
                logits.as_mut_ptr(),
                vocab_size,
            );
        }
        logits
    }
}
