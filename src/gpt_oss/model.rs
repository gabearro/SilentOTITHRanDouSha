//! GPT-OSS-20B model: compiled ANE executables + weights.

use super::config::GptOssConfig;
use super::executables::{self, CompiledExecutables, LayerExecutables};
use super::weights::{self, ModelFiles, ModelWeights};

const DEFAULT_ANE_COMPILE_RETRIES: usize = 5;

fn parse_lm_head_chunk_size_env() -> Option<usize> {
    let raw = std::env::var("GPT_OSS_LM_HEAD_CHUNK_SIZE").ok()?;
    let parsed = raw.trim().parse::<usize>().ok()?;
    if parsed == 0 {
        return None;
    }
    Some(parsed)
}

fn parse_ane_compile_retries_env() -> usize {
    std::env::var("GPT_OSS_ANE_COMPILE_RETRIES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_ANE_COMPILE_RETRIES)
}

fn cleanup_ane_tmp_dirs() {
    if let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) {
        for entry in entries.flatten() {
            if entry.file_name().to_string_lossy().len() == 32 {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }
}

fn compile_with_retries<T, E, F>(
    label: &str,
    max_attempts: usize,
    mut compile_once: F,
) -> Result<T, String>
where
    E: std::fmt::Display,
    F: FnMut() -> Result<T, E>,
{
    for attempt in 1..=max_attempts {
        match compile_once() {
            Ok(v) => return Ok(v),
            Err(e) => {
                if attempt == max_attempts {
                    return Err(e.to_string());
                }
                let delay_ms = 500u64.saturating_mul(1u64 << (attempt.saturating_sub(1) as u32));
                eprintln!(
                    "\n  {label} compile failed (attempt {attempt}/{max_attempts}): {e}; retrying in {delay_ms}ms..."
                );
                cleanup_ane_tmp_dirs();
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
            }
        }
    }
    unreachable!("max_attempts >= 1")
}

pub struct OssModel {
    pub config: GptOssConfig,
    pub weights: ModelWeights,
    pub executables: CompiledExecutables,
    pub max_sequence_length: usize,
}

impl OssModel {
    pub fn load(
        files: ModelFiles,
        max_sequence_length: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let config = files.config.clone();
        let weights = weights::load_weights(&files)?;
        let compile_retries = parse_ane_compile_retries_env();

        eprintln!("  Compiling ANE executables (RMSNorm on CPU, projections on ANE)...");

        let mut layers_vec: Vec<LayerExecutables> = Vec::with_capacity(config.num_hidden_layers);
        for (li, lw) in weights.layers.iter().enumerate() {
            eprint!(
                "\r  Compiling layer {}/{}...",
                li + 1,
                config.num_hidden_layers
            );

            let qkv = compile_with_retries(&format!("qkv layer {li}"), compile_retries, || {
                executables::build_decode_qkv(lw, &config)
            })
            .unwrap_or_else(|e| panic!("qkv layer {li}: {e}"));
            let output_proj_router = compile_with_retries(
                &format!("output_proj_router layer {li}"),
                compile_retries,
                || executables::build_decode_output_proj_router(lw, &config),
            )
            .unwrap_or_else(|e| panic!("output_proj_router layer {li}: {e}"));

            layers_vec.push(LayerExecutables {
                qkv,
                output_proj_router,
            });

            // Clean up temp files from ANE compiler to prevent disk exhaustion.
            cleanup_ane_tmp_dirs();
        }
        let layers = layers_vec.into_boxed_slice();
        eprintln!(" done");

        eprint!("  Compiling shared expert FFN (dynamic weights)...");
        let expert_ffn = compile_with_retries("expert_ffn", compile_retries, || {
            executables::build_expert_ffn_dynamic(&config)
        })
        .unwrap_or_else(|e| panic!("expert_ffn: {e}"));
        eprintln!(" done");

        eprint!("  Compiling final norm + residual add + LM head chunk...");
        let final_norm = compile_with_retries("final_norm", compile_retries, || {
            executables::build_final_norm(&weights.norm_weight, &config)
        })
        .unwrap_or_else(|e| panic!("final_norm: {e}"));
        let residual_add = compile_with_retries("residual_add", compile_retries, || {
            executables::build_residual_add(&config)
        })
        .unwrap_or_else(|e| panic!("residual_add: {e}"));
        let lm_chunk_candidates: Vec<usize> = if let Some(v) = parse_lm_head_chunk_size_env() {
            vec![v]
        } else {
            vec![
                executables::LM_HEAD_CHUNK_SIZE_PREFERRED,
                executables::LM_HEAD_CHUNK_SIZE_FALLBACK,
            ]
        };
        let mut last_err: Option<String> = None;
        let mut selected_chunk: Option<usize> = None;
        let mut lm_head_chunk_exec: Option<ane::Executable> = None;
        for chunk in lm_chunk_candidates.iter().copied() {
            match compile_with_retries(
                &format!("lm_head_chunk (chunk={chunk})"),
                compile_retries,
                || executables::build_lm_head_chunk(&config, chunk),
            ) {
                Ok(exec) => {
                    selected_chunk = Some(chunk);
                    lm_head_chunk_exec = Some(exec);
                    break;
                }
                Err(e) => {
                    last_err = Some(format!("chunk={chunk}: {e}"));
                }
            }
        }
        let lm_head_chunk_size = selected_chunk.unwrap_or_else(|| {
            panic!(
                "lm_head_chunk compile failed for candidates {:?}: {}",
                lm_chunk_candidates,
                last_err.unwrap_or_else(|| "unknown error".to_string())
            )
        });
        let lm_head_chunk = lm_head_chunk_exec.expect("lm_head chunk executable missing");
        eprintln!(" done (chunk={lm_head_chunk_size})");

        Ok(OssModel {
            config,
            weights,
            executables: CompiledExecutables {
                layers,
                expert_ffn,
                final_norm,
                residual_add,
                lm_head_chunk,
                lm_head_chunk_size,
            },
            max_sequence_length,
        })
    }
}
