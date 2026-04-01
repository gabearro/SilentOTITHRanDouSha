//! Shared private-runtime types for GPT-2 private inference modes.

use crate::field32::Fp32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrivateExecutionPolicy {
    /// Fast local prototype path.
    PrototypeFast,
    /// Prototype path with distributed Beaver openings enabled.
    PrototypeDistributed,
    /// Intended end-to-end private mode (currently incomplete in this codebase).
    StrictPrivate,
}

impl PrivateExecutionPolicy {
    pub fn from_env() -> Self {
        match std::env::var("PRIVATE_EXECUTION_POLICY")
            .unwrap_or_else(|_| "prototype_distributed".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "prototype_fast" | "fast" => PrivateExecutionPolicy::PrototypeFast,
            "strict_private" | "strict" => PrivateExecutionPolicy::StrictPrivate,
            _ => PrivateExecutionPolicy::PrototypeDistributed,
        }
    }

    #[inline]
    pub fn use_distributed_beaver(self) -> bool {
        match self {
            PrivateExecutionPolicy::PrototypeFast => false,
            PrivateExecutionPolicy::PrototypeDistributed => true,
            PrivateExecutionPolicy::StrictPrivate => true,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SharedScale {
    pub numerator: i32,
    pub denominator: i32,
}

impl SharedScale {
    pub const UNIT: SharedScale = SharedScale {
        numerator: 1,
        denominator: 1,
    };
}

#[derive(Clone, Debug)]
pub struct SharedTensor32 {
    pub n_parties: usize,
    pub rows: usize,
    pub cols: usize,
    pub scale: SharedScale,
    /// Party-major layout: `[party][rows * cols]`.
    pub data: Vec<Vec<Fp32>>,
}

impl SharedTensor32 {
    pub fn new(n_parties: usize, rows: usize, cols: usize, scale: SharedScale) -> Self {
        SharedTensor32 {
            n_parties,
            rows,
            cols,
            scale,
            data: vec![vec![Fp32::ZERO; rows * cols]; n_parties],
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

#[derive(Clone, Debug, Default)]
pub struct PrivateRuntimeStats {
    pub triples_consumed: u64,
    pub distributed_beaver_batches: u64,
    pub distributed_open_values: u64,
    pub distributed_softmax_open_values: u64,
    pub strict_mode_blocks: u64,
}
