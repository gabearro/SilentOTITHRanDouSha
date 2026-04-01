//! Private GPT-OSS-20B inference session (placeholder).
//!
//! Q·K^T via Beaver multiplication protocol.
//! Full implementation pending — cleartext session must be validated first.

use crate::gpu::BeaverTripleBatch32;
use rand::Rng;
use std::sync::mpsc::Receiver;

use super::model::OssModel;
use super::session::Session;

pub struct PrivateOssSession<'model> {
    session: Session<'model>,
    pub triples_consumed: u64,
}

impl<'model> PrivateOssSession<'model> {
    pub fn new(
        model: &'model OssModel,
        _n: usize,
        _t: usize,
        _triple_rx: Receiver<BeaverTripleBatch32>,
        max_seq: usize,
    ) -> Self {
        PrivateOssSession {
            session: Session::new(model, max_seq),
            triples_consumed: 0,
        }
    }

    /// Placeholder: delegates to cleartext session.
    /// TODO: Replace attention Q·K^T with Beaver protocol.
    pub fn decode_step_private(&mut self, token: u32, _rng: &mut impl Rng) -> Vec<f32> {
        self.session.decode_step(token).to_vec()
    }
}
