use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    pub rope_type: String,
    pub factor: f64,
    pub original_max_position_embeddings: usize,
    #[serde(default = "default_beta_fast")]
    pub beta_fast: f64,
    #[serde(default = "default_beta_slow")]
    pub beta_slow: f64,
    #[serde(default)]
    pub truncate: bool,
}

fn default_beta_fast() -> f64 {
    32.0
}
fn default_beta_slow() -> f64 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct GptOssConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub layer_types: Vec<LayerType>,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    #[serde(default = "default_swiglu_limit")]
    pub swiglu_limit: f32,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}
fn default_rope_theta() -> f64 {
    150000.0
}
fn default_sliding_window() -> usize {
    128
}
fn default_swiglu_limit() -> f32 {
    7.0
}
fn default_max_position_embeddings() -> usize {
    131072
}

impl GptOssConfig {
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
    pub fn gqa_ratio(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}
