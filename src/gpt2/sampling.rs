use rand::Rng;

pub fn sample(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    generated_token_ids: &[u32],
    rng: &mut impl Rng,
) -> u32 {
    let mut penalized: Vec<f32> = logits.to_vec();
    if repetition_penalty != 1.0 {
        for &tid in generated_token_ids {
            let i = tid as usize;
            if i < penalized.len() {
                if penalized[i] > 0.0 {
                    penalized[i] /= repetition_penalty;
                } else {
                    penalized[i] *= repetition_penalty;
                }
            }
        }
    }
    for v in &mut penalized {
        if !v.is_finite() {
            *v = f32::NEG_INFINITY;
        }
    }

    if temperature <= 0.0 {
        return penalized
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(50256);
    }

    let max_logit = penalized.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max_logit.is_finite() {
        return 50256;
    }
    let mut probs: Vec<(usize, f32)> = penalized
        .iter()
        .enumerate()
        .map(|(i, &l)| {
            if l.is_finite() {
                (i, ((l - max_logit) / temperature).exp())
            } else {
                (i, 0.0)
            }
        })
        .collect();
    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let total: f32 = probs.iter().map(|(_, p)| p).sum();
    if !total.is_finite() || total <= 0.0 {
        return 50256;
    }
    let cutoff = top_p * total;
    let mut cum = 0.0f32;
    let mut candidates = Vec::new();
    for &(ti, p) in &probs {
        if p <= 0.0 || !p.is_finite() {
            continue;
        }
        cum += p;
        candidates.push((ti, p));
        if cum >= cutoff {
            break;
        }
    }

    let cand_total: f32 = candidates.iter().map(|(_, p)| p).sum();
    if !cand_total.is_finite() || cand_total <= 0.0 {
        return 50256;
    }
    let threshold = rng.gen::<f32>() * cand_total;
    let mut acc = 0.0f32;
    for &(ti, p) in &candidates {
        acc += p;
        if acc >= threshold {
            return ti as u32;
        }
    }
    candidates.last().map(|(i, _)| *i as u32).unwrap_or(50256)
}
