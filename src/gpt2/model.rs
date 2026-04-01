use safetensors::SafeTensors;

use super::config::Gpt2Config;
use super::executables::{self, CompiledExecutables, DecodeLayer, PrefillLayer};
use super::weights::{self, ModelWeights};

pub struct CompiledModel {
    pub config: Gpt2Config,
    pub weights: ModelWeights,
    pub executables: CompiledExecutables,
    pub max_sequence_length: usize,
}

impl CompiledModel {
    pub fn from_safetensors(
        config: Gpt2Config,
        safetensors: &SafeTensors,
        padded_prompt_length: usize,
        max_sequence_length: usize,
    ) -> Result<Self, ane::Error> {
        eprintln!("  Loading weights...");
        let model_weights = weights::load_weights(safetensors, &config);
        let nl = config.n_layer;

        let prefill: Box<[PrefillLayer]> = model_weights
            .layers
            .iter()
            .enumerate()
            .map(|(i, lw)| {
                eprint!("\r  Compiling prefill layer {}/{}...", i + 1, nl);
                Ok(PrefillLayer {
                    attention: executables::build_prefill_attention(
                        lw,
                        &config,
                        padded_prompt_length,
                    )?,
                    feed_forward: executables::build_prefill_feed_forward(
                        lw,
                        &config,
                        padded_prompt_length,
                    )?,
                })
            })
            .collect::<Result<_, ane::Error>>()?;
        eprintln!(" done");

        let decode: Box<[DecodeLayer]> = model_weights
            .layers
            .iter()
            .enumerate()
            .map(|(i, lw)| {
                eprint!("\r  Compiling decode layer {}/{}...", i + 1, nl);
                Ok(DecodeLayer {
                    attention: executables::build_decode_attention(
                        lw,
                        &config,
                        max_sequence_length,
                    )?,
                    feed_forward: executables::build_decode_feed_forward(lw, &config)?,
                })
            })
            .collect::<Result<_, ane::Error>>()?;
        eprintln!(" done");

        eprint!("  Compiling LM head...");
        let lm_head = executables::build_lm_head(
            &model_weights.ln_f_weight,
            &model_weights.ln_f_bias,
            &model_weights.wte,
            &config,
        )?;
        eprintln!(" done");

        Ok(Self {
            config,
            weights: model_weights,
            executables: CompiledExecutables {
                prefill,
                decode,
                lm_head,
            },
            max_sequence_length,
        })
    }

    /// Compile only what the private session needs: FFN + LM head (no attention kernels).
    /// The private session compiles its own QKV and output_proj kernels separately.
    pub fn for_private_inference(
        config: Gpt2Config,
        safetensors: &SafeTensors,
        max_sequence_length: usize,
    ) -> Result<Self, ane::Error> {
        eprintln!("  Loading weights...");
        let model_weights = weights::load_weights(safetensors, &config);
        let nl = config.n_layer;

        // No prefill kernels needed for private inference
        let prefill: Box<[PrefillLayer]> = Box::new([]);

        // Only FFN kernels — attention is handled by the private session's split kernels
        let decode: Box<[DecodeLayer]> = model_weights
            .layers
            .iter()
            .enumerate()
            .map(|(i, lw)| {
                eprint!("\r  Compiling FFN layer {}/{}...", i + 1, nl);
                Ok(DecodeLayer {
                    // Dummy: won't be used — private session uses its own QKV+output_proj
                    attention: executables::build_decode_feed_forward(lw, &config)?,
                    feed_forward: executables::build_decode_feed_forward(lw, &config)?,
                })
            })
            .collect::<Result<_, ane::Error>>()?;
        eprintln!(" done");

        eprint!("  Compiling LM head...");
        let lm_head = executables::build_lm_head(
            &model_weights.ln_f_weight,
            &model_weights.ln_f_bias,
            &model_weights.wte,
            &config,
        )?;
        eprintln!(" done");

        Ok(Self {
            config,
            weights: model_weights,
            executables: CompiledExecutables {
                prefill,
                decode,
                lm_head,
            },
            max_sequence_length,
        })
    }
}
