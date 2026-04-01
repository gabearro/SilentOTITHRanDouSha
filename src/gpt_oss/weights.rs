//! Weight loading for GPT-OSS-20B with memory-mapped safetensors.
//!
//! BF16 tensors (attention, embeddings, LM head) are converted to f32 on demand.
//! MXFP4 tensors (expert weights) are kept as raw bytes for GPU/ANE dequantization.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use half::bf16;
use hf_hub::api::sync::ApiBuilder;
use memmap2::Mmap;
use serde::Deserialize;

use super::config::GptOssConfig;

/// Raw MXFP4 data for one expert's weights.
/// Blocks: 4-bit values packed 2 per byte, 32 values per block → 16 bytes per block.
/// Scales: 1 E8M0 byte per block of 32 elements.
pub struct ExpertMxfp4 {
    pub gate_up_blocks: Box<[u8]>, // [5760, 90, 16] = [out, blocks, 16]
    pub gate_up_scales: Box<[u8]>, // [5760, 90]
    pub gate_up_bias: Box<[f32]>,  // [5760]
    pub down_blocks: Box<[u8]>,    // [2880, 90, 16]
    pub down_scales: Box<[u8]>,    // [2880, 90]
    pub down_bias: Box<[f32]>,     // [2880]
}

pub struct LayerWeights {
    // RMSNorm
    pub input_layernorm_weight: Box<[f32]>, // [hidden_size]
    pub post_attn_layernorm_weight: Box<[f32]>, // [hidden_size]
    // Self-attention (BF16 → f32)
    pub q_proj_weight: Box<[f32]>, // [q_dim, hidden_size]
    pub q_proj_bias: Box<[f32]>,   // [q_dim]
    pub k_proj_weight: Box<[f32]>, // [kv_dim, hidden_size]
    pub k_proj_bias: Box<[f32]>,   // [kv_dim]
    pub v_proj_weight: Box<[f32]>, // [kv_dim, hidden_size]
    pub v_proj_bias: Box<[f32]>,   // [kv_dim]
    pub o_proj_weight: Box<[f32]>, // [hidden_size, q_dim]
    pub o_proj_bias: Box<[f32]>,   // [hidden_size]
    pub sinks: Box<[f32]>,         // [num_attention_heads]
    // MoE router
    pub router_weight: Box<[f32]>, // [num_experts, hidden_size]
    pub router_bias: Box<[f32]>,   // [num_experts]
    // MoE experts (MXFP4)
    pub experts: Box<[ExpertMxfp4]>,
}

pub struct ModelWeights {
    /// Embeddings stay as raw BF16 bytes (memory-mapped). Converted to f32 on access per token.
    pub embed_tokens_bf16: Box<[u8]>,
    pub layers: Box<[LayerWeights]>,
    pub norm_weight: Box<[f32]>, // [hidden_size] — small, kept as f32
    /// LM head stays as raw BF16 bytes (memory-mapped). Converted to f32 on access per row.
    pub lm_head_bf16: Box<[u8]>,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

impl ModelWeights {
    /// Get embedding for a single token, converting BF16 → f32 on demand.
    pub fn embed_token(&self, token_id: usize, out: &mut [f32]) {
        let offset = token_id * self.hidden_size * 2; // 2 bytes per BF16
        let bytes = &self.embed_tokens_bf16[offset..offset + self.hidden_size * 2];
        for (i, chunk) in bytes.chunks_exact(2).enumerate() {
            out[i] = bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
        }
    }

    /// Compute dot product of LM head row `v` with `input`, converting BF16 → f32 on the fly.
    /// Avoids materializing the full LM head weight matrix in f32.
    pub fn lm_head_dot(&self, v: usize, input: &[f32]) -> f32 {
        let offset = v * self.hidden_size * 2;
        let bytes = &self.lm_head_bf16[offset..offset + self.hidden_size * 2];
        let mut acc = 0.0f32;
        for (i, chunk) in bytes.chunks_exact(2).enumerate() {
            let w = bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
            acc += w * input[i];
        }
        acc
    }
}

pub struct ModelFiles {
    pub config: GptOssConfig,
    pub tokenizer_path: PathBuf,
    pub shard_paths: Vec<PathBuf>,
}

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

#[derive(Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

struct ShardReader {
    mmap: Mmap,
    header: HashMap<String, TensorInfo>,
    data_offset: usize,
}

impl ShardReader {
    fn open(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let header_json = &mmap[8..8 + header_size];
        let mut header: HashMap<String, TensorInfo> = serde_json::from_slice(header_json)?;
        header.remove("__metadata__");

        Ok(ShardReader {
            mmap,
            header,
            data_offset: 8 + header_size,
        })
    }

    fn get_bytes(&self, name: &str) -> &[u8] {
        let info = self
            .header
            .get(name)
            .unwrap_or_else(|| panic!("tensor not found in shard: {name}"));
        let start = self.data_offset + info.data_offsets[0];
        let end = self.data_offset + info.data_offsets[1];
        &self.mmap[start..end]
    }

    fn has(&self, name: &str) -> bool {
        self.header.contains_key(name)
    }
}

/// Convert BF16 bytes to f32 slice.
fn bf16_to_f32(bytes: &[u8]) -> Box<[f32]> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

/// Convert BF16 bytes directly to fp16 bytes for ANE constants.
/// BF16 (1-8-7) → fp16 (1-5-10): truncation-free for values within fp16 range.
/// For values outside fp16 range, clamps to fp16 max/min.
pub fn bf16_to_fp16_bytes(bytes: &[u8]) -> Box<[u8]> {
    use half::f16;
    bytes
        .chunks_exact(2)
        .flat_map(|c| {
            let val = bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
            let fp16 = f16::from_f32(val);
            fp16.to_le_bytes()
        })
        .collect()
}

/// Copy raw U8 bytes.
fn copy_u8(bytes: &[u8]) -> Box<[u8]> {
    bytes.into()
}

pub fn download_model(repo_id: &str) -> Result<ModelFiles, Box<dyn std::error::Error>> {
    let api = ApiBuilder::new().with_progress(true).build()?;
    let repo = api.model(repo_id.to_string());

    eprint!("  Downloading config.json...");
    let config_path = repo.get("config.json")?;
    let config: GptOssConfig = serde_json::from_reader(fs::File::open(&config_path)?)?;
    eprintln!(
        " done ({} layers, {}d, {} experts)",
        config.num_hidden_layers, config.hidden_size, config.num_local_experts
    );

    eprint!("  Downloading tokenizer.json...");
    let tokenizer_path = repo.get("tokenizer.json")?;
    eprintln!(" done");

    eprint!("  Downloading safetensors index...");
    let index_path = repo.get("model.safetensors.index.json")?;
    let index: SafetensorsIndex = serde_json::from_reader(fs::File::open(&index_path)?)?;
    eprintln!(" done");

    // Determine unique shard filenames
    let mut shard_names: Vec<String> = index.weight_map.values().cloned().collect();
    shard_names.sort();
    shard_names.dedup();

    let mut shard_paths = Vec::new();
    for shard_name in &shard_names {
        eprint!("  Downloading {shard_name}...");
        let path = repo.get(shard_name)?;
        eprintln!(" done");
        shard_paths.push(path);
    }

    Ok(ModelFiles {
        config,
        tokenizer_path,
        shard_paths,
    })
}

/// Find which shard reader contains a tensor and return its bytes.
fn get_tensor_bytes<'a>(shards: &'a [ShardReader], name: &str) -> &'a [u8] {
    for shard in shards {
        if shard.has(name) {
            return shard.get_bytes(name);
        }
    }
    panic!("tensor {name} not found in any shard");
}

fn load_bf16(shards: &[ShardReader], name: &str) -> Box<[f32]> {
    bf16_to_f32(get_tensor_bytes(shards, name))
}

fn load_u8(shards: &[ShardReader], name: &str) -> Box<[u8]> {
    copy_u8(get_tensor_bytes(shards, name))
}

pub fn load_weights(files: &ModelFiles) -> Result<ModelWeights, Box<dyn std::error::Error>> {
    let config = &files.config;
    let ne = config.num_local_experts;
    let inter = config.intermediate_size;
    let hs = config.hidden_size;

    eprintln!(
        "  Memory-mapping {} safetensor shards...",
        files.shard_paths.len()
    );
    let shards: Vec<ShardReader> = files
        .shard_paths
        .iter()
        .map(|p| ShardReader::open(p).unwrap())
        .collect();

    // Keep embeddings and LM head as raw BF16 bytes (saves ~4.6GB RAM).
    // Converted to f32 on demand per token/row.
    eprint!("  Loading embeddings (BF16 mmap)...");
    let embed_tokens_bf16 = load_u8(&shards, "model.embed_tokens.weight");
    eprintln!(" done ({:.0}MB BF16)", embed_tokens_bf16.len() as f64 / 1e6);

    eprint!("  Loading LM head (BF16 mmap)...");
    let lm_head_bf16 = load_u8(&shards, "lm_head.weight");
    eprintln!(" done ({:.0}MB BF16)", lm_head_bf16.len() as f64 / 1e6);

    let norm_weight = load_bf16(&shards, "model.norm.weight");

    let layers: Box<[LayerWeights]> = (0..config.num_hidden_layers)
        .map(|li| {
            eprint!(
                "\r  Loading layer {}/{}...",
                li + 1,
                config.num_hidden_layers
            );
            let lp = format!("model.layers.{li}");

            let input_layernorm_weight =
                load_bf16(&shards, &format!("{lp}.input_layernorm.weight"));
            let post_attn_layernorm_weight =
                load_bf16(&shards, &format!("{lp}.post_attention_layernorm.weight"));

            let q_proj_weight = load_bf16(&shards, &format!("{lp}.self_attn.q_proj.weight"));
            let q_proj_bias = load_bf16(&shards, &format!("{lp}.self_attn.q_proj.bias"));
            let k_proj_weight = load_bf16(&shards, &format!("{lp}.self_attn.k_proj.weight"));
            let k_proj_bias = load_bf16(&shards, &format!("{lp}.self_attn.k_proj.bias"));
            let v_proj_weight = load_bf16(&shards, &format!("{lp}.self_attn.v_proj.weight"));
            let v_proj_bias = load_bf16(&shards, &format!("{lp}.self_attn.v_proj.bias"));
            let o_proj_weight = load_bf16(&shards, &format!("{lp}.self_attn.o_proj.weight"));
            let o_proj_bias = load_bf16(&shards, &format!("{lp}.self_attn.o_proj.bias"));
            let sinks = load_bf16(&shards, &format!("{lp}.self_attn.sinks"));

            let router_weight = load_bf16(&shards, &format!("{lp}.mlp.router.weight"));
            let router_bias = load_bf16(&shards, &format!("{lp}.mlp.router.bias"));

            // MXFP4 expert weights — load all experts from packed tensors
            let gu_blocks_all = load_u8(&shards, &format!("{lp}.mlp.experts.gate_up_proj_blocks"));
            let gu_scales_all = load_u8(&shards, &format!("{lp}.mlp.experts.gate_up_proj_scales"));
            let gu_bias_all = load_bf16(&shards, &format!("{lp}.mlp.experts.gate_up_proj_bias"));
            let d_blocks_all = load_u8(&shards, &format!("{lp}.mlp.experts.down_proj_blocks"));
            let d_scales_all = load_u8(&shards, &format!("{lp}.mlp.experts.down_proj_scales"));
            let d_bias_all = load_bf16(&shards, &format!("{lp}.mlp.experts.down_proj_bias"));

            // Split packed expert tensors into per-expert slices
            let num_blocks = hs / 32; // 2880/32 = 90
            let gu_out = 2 * inter; // 5760
            let gu_blocks_per_expert = gu_out * num_blocks * 16;
            let gu_scales_per_expert = gu_out * num_blocks;
            let d_blocks_per_expert = hs * num_blocks * 16;
            let d_scales_per_expert = hs * num_blocks;

            let experts: Box<[ExpertMxfp4]> = (0..ne)
                .map(|ei| {
                    let gb_start = ei * gu_blocks_per_expert;
                    let gs_start = ei * gu_scales_per_expert;
                    let db_start = ei * d_blocks_per_expert;
                    let ds_start = ei * d_scales_per_expert;
                    ExpertMxfp4 {
                        gate_up_blocks: gu_blocks_all[gb_start..gb_start + gu_blocks_per_expert]
                            .into(),
                        gate_up_scales: gu_scales_all[gs_start..gs_start + gu_scales_per_expert]
                            .into(),
                        gate_up_bias: gu_bias_all[ei * gu_out..(ei + 1) * gu_out].into(),
                        down_blocks: d_blocks_all[db_start..db_start + d_blocks_per_expert].into(),
                        down_scales: d_scales_all[ds_start..ds_start + d_scales_per_expert].into(),
                        down_bias: d_bias_all[ei * hs..(ei + 1) * hs].into(),
                    }
                })
                .collect();

            LayerWeights {
                input_layernorm_weight,
                post_attn_layernorm_weight,
                q_proj_weight,
                q_proj_bias,
                k_proj_weight,
                k_proj_bias,
                v_proj_weight,
                v_proj_bias,
                o_proj_weight,
                o_proj_bias,
                sinks,
                router_weight,
                router_bias,
                experts,
            }
        })
        .collect();
    eprintln!(" done");

    Ok(ModelWeights {
        embed_tokens_bf16,
        layers,
        norm_weight,
        lm_head_bf16,
        hidden_size: hs,
        vocab_size: config.vocab_size,
    })
}

/// Dequantize a single MXFP4 row: out_dim values from blocks + scales.
/// MXFP4: 4-bit E2M1 values with shared E8M0 scale per block of 32.
pub fn dequant_mxfp4_row(blocks: &[u8], scales: &[u8], out: &mut [f32]) {
    // FP4 E2M1 lookup table: maps 4-bit value to f32
    const FP4_LUT: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];

    let num_blocks = scales.len();
    for bi in 0..num_blocks {
        // E8M0 scale: value = 2^(scale_byte - 127)
        let scale = f32::from_bits((scales[bi] as u32) << 23);
        let block_bytes = &blocks[bi * 16..(bi + 1) * 16];

        for j in 0..16 {
            let byte = block_bytes[j];
            let lo = (byte & 0x0F) as usize;
            let hi = (byte >> 4) as usize;
            let idx = bi * 32 + j * 2;
            out[idx] = FP4_LUT[lo] * scale;
            out[idx + 1] = FP4_LUT[hi] * scale;
        }
    }
}

/// Dequantize an entire expert's gate_up matrix: [out_dim, in_dim] from MXFP4.
pub fn dequant_expert_gate_up(expert: &ExpertMxfp4, out_dim: usize, in_dim: usize) -> Box<[f32]> {
    let mut result = vec![0.0f32; out_dim * in_dim];
    dequant_expert_gate_up_into(expert, out_dim, in_dim, &mut result);
    result.into_boxed_slice()
}

/// Dequantize gate_up into a pre-allocated buffer.
pub fn dequant_expert_gate_up_into(
    expert: &ExpertMxfp4,
    out_dim: usize,
    in_dim: usize,
    result: &mut [f32],
) {
    let num_blocks = in_dim / 32;
    for row in 0..out_dim {
        let blocks = &expert.gate_up_blocks[row * num_blocks * 16..(row + 1) * num_blocks * 16];
        let scales = &expert.gate_up_scales[row * num_blocks..(row + 1) * num_blocks];
        dequant_mxfp4_row(
            blocks,
            scales,
            &mut result[row * in_dim..(row + 1) * in_dim],
        );
    }
}

/// Dequantize an entire expert's down matrix: [out_dim, in_dim] from MXFP4.
pub fn dequant_expert_down(expert: &ExpertMxfp4, out_dim: usize, in_dim: usize) -> Box<[f32]> {
    let mut result = vec![0.0f32; out_dim * in_dim];
    dequant_expert_down_into(expert, out_dim, in_dim, &mut result);
    result.into_boxed_slice()
}

/// Dequantize down into a pre-allocated buffer.
pub fn dequant_expert_down_into(
    expert: &ExpertMxfp4,
    out_dim: usize,
    in_dim: usize,
    result: &mut [f32],
) {
    let num_blocks = in_dim / 32;
    for row in 0..out_dim {
        let blocks = &expert.down_blocks[row * num_blocks * 16..(row + 1) * num_blocks * 16];
        let scales = &expert.down_scales[row * num_blocks..(row + 1) * num_blocks];
        dequant_mxfp4_row(
            blocks,
            scales,
            &mut result[row * in_dim..(row + 1) * in_dim],
        );
    }
}

/// Dequantize gate_up with reordering: interleaved [g0,u0,g1,u1,...] → contiguous [g0,g1,...,u0,u1,...].
/// Output: [2*inter, in_dim] with first `inter` rows being gate, next `inter` rows being up.
/// This matches the ANE expert FFN graph which uses contiguous slice for gate/up split.
pub fn dequant_expert_gate_up_reordered(
    expert: &ExpertMxfp4,
    inter: usize,
    in_dim: usize,
    result: &mut [f32],
) {
    let num_blocks = in_dim / 32;
    for i in 0..inter {
        let src_row = 2 * i;
        let blocks =
            &expert.gate_up_blocks[src_row * num_blocks * 16..(src_row + 1) * num_blocks * 16];
        let scales = &expert.gate_up_scales[src_row * num_blocks..(src_row + 1) * num_blocks];
        dequant_mxfp4_row(blocks, scales, &mut result[i * in_dim..(i + 1) * in_dim]);

        let src_row = 2 * i + 1;
        let blocks =
            &expert.gate_up_blocks[src_row * num_blocks * 16..(src_row + 1) * num_blocks * 16];
        let scales = &expert.gate_up_scales[src_row * num_blocks..(src_row + 1) * num_blocks];
        dequant_mxfp4_row(
            blocks,
            scales,
            &mut result[(inter + i) * in_dim..(inter + i + 1) * in_dim],
        );
    }
}

/// Reorder gate_up bias: interleaved [g0,u0,g1,u1,...] → contiguous [g0,g1,...,u0,u1,...].
pub fn reorder_gate_up_bias(bias: &[f32], inter: usize, result: &mut [f32]) {
    for i in 0..inter {
        result[i] = bias[2 * i];
        result[inter + i] = bias[2 * i + 1];
    }
}

// ── MXFP4 → fp16 direct dequantization ──────────────────────────────────────

/// FP4 E2M1 lookup table as fp16 bits.
const FP4_LUT_F16: [u16; 16] = {
    use half::f16;
    [
        f16::from_f32_const(0.0).to_bits(),
        f16::from_f32_const(0.5).to_bits(),
        f16::from_f32_const(1.0).to_bits(),
        f16::from_f32_const(1.5).to_bits(),
        f16::from_f32_const(2.0).to_bits(),
        f16::from_f32_const(3.0).to_bits(),
        f16::from_f32_const(4.0).to_bits(),
        f16::from_f32_const(6.0).to_bits(),
        f16::from_f32_const(-0.0).to_bits(),
        f16::from_f32_const(-0.5).to_bits(),
        f16::from_f32_const(-1.0).to_bits(),
        f16::from_f32_const(-1.5).to_bits(),
        f16::from_f32_const(-2.0).to_bits(),
        f16::from_f32_const(-3.0).to_bits(),
        f16::from_f32_const(-4.0).to_bits(),
        f16::from_f32_const(-6.0).to_bits(),
    ]
};

/// Dequantize a single MXFP4 row directly to fp16 using NEON SIMD.
///
/// Processes 32 values (16 packed bytes) per iteration using:
/// - vqtbl1q_u8: 4-bit → fp16 byte lookup (decomposed into lo/hi byte tables)
/// - vzip: recombine bytes into fp16 values
/// - vmulq_f16: scale by E8M0 power-of-2
///
/// ~5 cycles per 32 values on Apple Silicon (vs ~160 cycles scalar).
#[cfg(target_arch = "aarch64")]
fn dequant_mxfp4_row_f16(blocks: &[u8], scales: &[u8], out: &mut [u16]) {
    use std::arch::aarch64::*;

    // Decompose the 16-entry fp16 LUT into separate low-byte and high-byte tables.
    // Each table has 16 entries of 1 byte, fitting in one NEON register.
    // The 4-bit index selects from both tables; ZIP recombines into fp16.
    let lut_lo_bytes: [u8; 16] = {
        let mut t = [0u8; 16];
        for i in 0..16 {
            t[i] = (FP4_LUT_F16[i] & 0xFF) as u8;
        }
        t
    };
    let lut_hi_bytes: [u8; 16] = {
        let mut t = [0u8; 16];
        for i in 0..16 {
            t[i] = (FP4_LUT_F16[i] >> 8) as u8;
        }
        t
    };

    unsafe {
        let tbl_lo = vld1q_u8(lut_lo_bytes.as_ptr());
        let tbl_hi = vld1q_u8(lut_hi_bytes.as_ptr());
        let mask_0f = vdupq_n_u8(0x0F);

        let num_blocks = scales.len();
        for bi in 0..num_blocks {
            // E8M0 scale → fp16: 2^(byte - 127) as fp16 bits
            let scale_exp = scales[bi] as i16 - 112;
            let scale_f16 = if scale_exp <= 0 {
                0u16
            } else if scale_exp >= 31 {
                0x7C00u16
            } else {
                (scale_exp as u16) << 10
            };
            let scale_vec = vreinterpretq_f16_u16(vdupq_n_u16(scale_f16));

            let block_ptr = blocks.as_ptr().add(bi * 16);
            let packed = vld1q_u8(block_ptr);

            // Split nibbles: lo = packed & 0x0F, hi = packed >> 4
            let lo_nib = vandq_u8(packed, mask_0f);
            let hi_nib = vshrq_n_u8::<4>(packed);

            // LUT lookup: 4-bit index → fp16 low byte and high byte separately
            let lo_lo = vqtbl1q_u8(tbl_lo, lo_nib); // fp16 low bytes for lo nibbles
            let lo_hi = vqtbl1q_u8(tbl_hi, lo_nib); // fp16 high bytes for lo nibbles
            let hi_lo = vqtbl1q_u8(tbl_lo, hi_nib); // fp16 low bytes for hi nibbles
            let hi_hi = vqtbl1q_u8(tbl_hi, hi_nib); // fp16 high bytes for hi nibbles

            // Interleave lo/hi bytes to form fp16 values.
            // lo nibbles produce values at even positions, hi nibbles at odd positions.
            // But the output layout is: [lo0, hi0, lo1, hi1, ...] (interleaved by input byte).
            // ZIP1 combines first halves, ZIP2 combines second halves of 16-byte vectors.

            // First 16 fp16 values (from input bytes 0-7):
            let fp16_a_bytes = vzip1q_u8(lo_lo, lo_hi); // [lo_byte, hi_byte] pairs for lo nibbles 0-7
            let fp16_b_bytes = vzip1q_u8(hi_lo, hi_hi); // [lo_byte, hi_byte] pairs for hi nibbles 0-7

            // We need interleaved: [lo_nib_0, hi_nib_0, lo_nib_1, hi_nib_1, ...]
            // That's 8 fp16 from lo nibbles (bytes 0-7) interleaved with 8 fp16 from hi nibbles (bytes 0-7)
            // Each pair: (lo_nibble_value, hi_nibble_value) for input byte j

            // Reinterpret as u16 for interleaving at fp16 granularity
            let lo_vals_0 = vreinterpretq_u16_u8(fp16_a_bytes); // 8 fp16 values from lo nibbles of bytes 0-7
            let hi_vals_0 = vreinterpretq_u16_u8(fp16_b_bytes); // 8 fp16 values from hi nibbles of bytes 0-7

            // Second 16 fp16 values (from input bytes 8-15):
            let fp16_c_bytes = vzip2q_u8(lo_lo, lo_hi);
            let fp16_d_bytes = vzip2q_u8(hi_lo, hi_hi);
            let lo_vals_1 = vreinterpretq_u16_u8(fp16_c_bytes);
            let hi_vals_1 = vreinterpretq_u16_u8(fp16_d_bytes);

            // Interleave lo/hi at u16 level: [lo0, hi0, lo1, hi1, ...] per group of 8
            let interleaved_0 = vzip1q_u16(lo_vals_0, hi_vals_0); // 8 values: lo0,hi0,lo1,hi1,...,lo3,hi3
            let interleaved_1 = vzip2q_u16(lo_vals_0, hi_vals_0); // 8 values: lo4,hi4,...,lo7,hi7
            let interleaved_2 = vzip1q_u16(lo_vals_1, hi_vals_1); // 8 values: lo8,hi8,...,lo11,hi11
            let interleaved_3 = vzip2q_u16(lo_vals_1, hi_vals_1); // 8 values: lo12,hi12,...,lo15,hi15

            // Multiply by E8M0 scale (native fp16 multiply)
            let r0 = vmulq_f16(vreinterpretq_f16_u16(interleaved_0), scale_vec);
            let r1 = vmulq_f16(vreinterpretq_f16_u16(interleaved_1), scale_vec);
            let r2 = vmulq_f16(vreinterpretq_f16_u16(interleaved_2), scale_vec);
            let r3 = vmulq_f16(vreinterpretq_f16_u16(interleaved_3), scale_vec);

            // Store 32 fp16 values
            let out_ptr = out.as_mut_ptr().add(bi * 32);
            vst1q_u16(out_ptr, vreinterpretq_u16_f16(r0));
            vst1q_u16(out_ptr.add(8), vreinterpretq_u16_f16(r1));
            vst1q_u16(out_ptr.add(16), vreinterpretq_u16_f16(r2));
            vst1q_u16(out_ptr.add(24), vreinterpretq_u16_f16(r3));
        }
    }
}

/// Scalar fallback for non-aarch64 targets.
#[cfg(not(target_arch = "aarch64"))]
fn dequant_mxfp4_row_f16(blocks: &[u8], scales: &[u8], out: &mut [u16]) {
    let num_blocks = scales.len();
    for bi in 0..num_blocks {
        let scale_exp = scales[bi] as i16 - 112;
        let scale_f16 = if scale_exp <= 0 {
            0u16
        } else if scale_exp >= 31 {
            0x7C00u16
        } else {
            (scale_exp as u16) << 10
        };
        let block_bytes = &blocks[bi * 16..(bi + 1) * 16];
        for j in 0..16 {
            let byte = block_bytes[j];
            let lo = (byte & 0x0F) as usize;
            let hi = (byte >> 4) as usize;
            let idx = bi * 32 + j * 2;
            out[idx] = f16_mul_scalar(FP4_LUT_F16[lo], scale_f16);
            out[idx + 1] = f16_mul_scalar(FP4_LUT_F16[hi], scale_f16);
        }
    }
}

/// Scalar fp16 multiply for fallback path.
#[inline]
fn f16_mul_scalar(val_bits: u16, scale_bits: u16) -> u16 {
    if val_bits == 0 || val_bits == 0x8000 {
        return val_bits;
    }
    if scale_bits == 0 {
        return val_bits & 0x8000;
    }
    let sign = val_bits & 0x8000;
    let val_exp = ((val_bits >> 10) & 0x1F) as i16;
    let val_mant = val_bits & 0x03FF;
    let scale_exp = ((scale_bits >> 10) & 0x1F) as i16;
    let new_exp = val_exp + scale_exp - 15;
    if new_exp <= 0 {
        return sign;
    }
    if new_exp >= 31 {
        return sign | 0x7C00;
    }
    sign | ((new_exp as u16) << 10) | val_mant
}

/// Dequantize gate_up reordered directly to fp16 buffer.
pub fn dequant_expert_gate_up_reordered_f16(
    expert: &ExpertMxfp4,
    inter: usize,
    in_dim: usize,
    result: &mut [u16],
) {
    let num_blocks = in_dim / 32;
    for i in 0..inter {
        let src_row = 2 * i;
        let blocks =
            &expert.gate_up_blocks[src_row * num_blocks * 16..(src_row + 1) * num_blocks * 16];
        let scales = &expert.gate_up_scales[src_row * num_blocks..(src_row + 1) * num_blocks];
        dequant_mxfp4_row_f16(blocks, scales, &mut result[i * in_dim..(i + 1) * in_dim]);

        let src_row = 2 * i + 1;
        let blocks =
            &expert.gate_up_blocks[src_row * num_blocks * 16..(src_row + 1) * num_blocks * 16];
        let scales = &expert.gate_up_scales[src_row * num_blocks..(src_row + 1) * num_blocks];
        dequant_mxfp4_row_f16(
            blocks,
            scales,
            &mut result[(inter + i) * in_dim..(inter + i + 1) * in_dim],
        );
    }
}

/// Dequantize down directly to fp16 buffer.
pub fn dequant_expert_down_f16(
    expert: &ExpertMxfp4,
    out_dim: usize,
    in_dim: usize,
    result: &mut [u16],
) {
    let num_blocks = in_dim / 32;
    for row in 0..out_dim {
        let blocks = &expert.down_blocks[row * num_blocks * 16..(row + 1) * num_blocks * 16];
        let scales = &expert.down_scales[row * num_blocks..(row + 1) * num_blocks];
        dequant_mxfp4_row_f16(
            blocks,
            scales,
            &mut result[row * in_dim..(row + 1) * in_dim],
        );
    }
}

/// Reorder gate_up bias to fp16: interleaved → contiguous.
pub fn reorder_gate_up_bias_f16(bias: &[f32], inter: usize, result: &mut [u16]) {
    use half::f16;
    for i in 0..inter {
        result[i] = f16::from_f32(bias[2 * i]).to_bits();
        result[inter + i] = f16::from_f32(bias[2 * i + 1]).to_bits();
    }
}

/// Convert f32 bias slice to fp16.
pub fn f32_to_f16_slice(src: &[f32], dst: &mut [u16]) {
    use half::f16;
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = f16::from_f32(*s).to_bits();
    }
}
