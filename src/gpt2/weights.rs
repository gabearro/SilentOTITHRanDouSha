use std::fs;
use std::path::PathBuf;

use half::{bf16, f16};
use hf_hub::api::sync::ApiBuilder;
use safetensors::{Dtype, SafeTensors};

use super::config::Gpt2Config;

pub struct LayerWeights {
    pub ln1_weight: Box<[f32]>,
    pub ln1_bias: Box<[f32]>,
    pub qkv_weight: Box<[f32]>,
    pub qkv_bias: Box<[f32]>,
    pub attn_proj_weight: Box<[f32]>,
    pub attn_proj_bias: Box<[f32]>,
    pub ln2_weight: Box<[f32]>,
    pub ln2_bias: Box<[f32]>,
    pub fc_weight: Box<[f32]>,
    pub fc_bias: Box<[f32]>,
    pub fc_proj_weight: Box<[f32]>,
    pub fc_proj_bias: Box<[f32]>,
}

pub struct ModelWeights {
    pub wte: Box<[f32]>,
    pub wpe: Box<[f32]>,
    pub layers: Box<[LayerWeights]>,
    pub ln_f_weight: Box<[f32]>,
    pub ln_f_bias: Box<[f32]>,
}

pub struct ModelFiles {
    pub config: Gpt2Config,
    pub tokenizer_path: PathBuf,
    pub safetensors_bytes: Vec<u8>,
}

pub fn download_model(repo_id: &str) -> Result<ModelFiles, Box<dyn std::error::Error>> {
    let api = ApiBuilder::new().with_progress(true).build()?;
    let repo = api.model(repo_id.to_string());

    eprint!("  Downloading config.json...");
    let config_path = repo.get("config.json")?;
    let config: Gpt2Config = serde_json::from_reader(fs::File::open(&config_path)?)?;
    eprintln!(" done");

    eprint!("  Downloading tokenizer.json...");
    let tokenizer_path = repo.get("tokenizer.json")?;
    eprintln!(" done");

    eprint!("  Downloading model.safetensors...");
    let safetensors_path = repo.get("model.safetensors")?;
    let safetensors_bytes = fs::read(&safetensors_path)?;
    eprintln!(" done ({:.1}MB)", safetensors_bytes.len() as f64 / 1e6);

    Ok(ModelFiles {
        config,
        tokenizer_path,
        safetensors_bytes,
    })
}

fn tensor_to_f32(safetensors: &SafeTensors, name: &str) -> Box<[f32]> {
    let tensor = safetensors
        .tensor(name)
        .unwrap_or_else(|_| panic!("tensor not found: {name}"));
    let bytes = tensor.data();
    match tensor.dtype() {
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect(),
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        other => panic!("unsupported dtype: {other:?}"),
    }
}

fn tensor_to_f32_transposed(
    safetensors: &SafeTensors,
    name: &str,
    rows: usize,
    cols: usize,
) -> Box<[f32]> {
    let raw = tensor_to_f32(safetensors, name);
    assert_eq!(raw.len(), rows * cols, "shape mismatch for {name}");
    let mut transposed = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            transposed[col * rows + row] = raw[row * cols + col];
        }
    }
    transposed.into_boxed_slice()
}

pub fn load_weights(safetensors: &SafeTensors, config: &Gpt2Config) -> ModelWeights {
    let embedding_dim = config.n_embd;

    let wte = tensor_to_f32(safetensors, "wte.weight");
    let wpe = tensor_to_f32(safetensors, "wpe.weight");

    let layers: Box<[LayerWeights]> = (0..config.n_layer)
        .map(|layer_index| {
            let lp = format!("h.{layer_index}");
            LayerWeights {
                ln1_weight: tensor_to_f32(safetensors, &format!("{lp}.ln_1.weight")),
                ln1_bias: tensor_to_f32(safetensors, &format!("{lp}.ln_1.bias")),
                qkv_weight: tensor_to_f32_transposed(
                    safetensors,
                    &format!("{lp}.attn.c_attn.weight"),
                    embedding_dim,
                    3 * embedding_dim,
                ),
                qkv_bias: tensor_to_f32(safetensors, &format!("{lp}.attn.c_attn.bias")),
                attn_proj_weight: tensor_to_f32_transposed(
                    safetensors,
                    &format!("{lp}.attn.c_proj.weight"),
                    embedding_dim,
                    embedding_dim,
                ),
                attn_proj_bias: tensor_to_f32(safetensors, &format!("{lp}.attn.c_proj.bias")),
                ln2_weight: tensor_to_f32(safetensors, &format!("{lp}.ln_2.weight")),
                ln2_bias: tensor_to_f32(safetensors, &format!("{lp}.ln_2.bias")),
                fc_weight: tensor_to_f32_transposed(
                    safetensors,
                    &format!("{lp}.mlp.c_fc.weight"),
                    embedding_dim,
                    4 * embedding_dim,
                ),
                fc_bias: tensor_to_f32(safetensors, &format!("{lp}.mlp.c_fc.bias")),
                fc_proj_weight: tensor_to_f32_transposed(
                    safetensors,
                    &format!("{lp}.mlp.c_proj.weight"),
                    4 * embedding_dim,
                    embedding_dim,
                ),
                fc_proj_bias: tensor_to_f32(safetensors, &format!("{lp}.mlp.c_proj.bias")),
            }
        })
        .collect();

    let ln_f_weight = tensor_to_f32(safetensors, "ln_f.weight");
    let ln_f_bias = tensor_to_f32(safetensors, "ln_f.bias");

    ModelWeights {
        wte,
        wpe,
        layers,
        ln_f_weight,
        ln_f_bias,
    }
}
