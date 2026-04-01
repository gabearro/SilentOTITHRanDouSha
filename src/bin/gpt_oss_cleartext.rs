//! GPT-OSS-20B inference with Harmony chat format — ANE + CPU.

use std::io::{self, BufRead, Write};
use std::time::Instant;

use tokenizers::Tokenizer;

use silent_ot_randousha::gpt2::sampling;
use silent_ot_randousha::gpt_oss::model::OssModel;
use silent_ot_randousha::gpt_oss::session::Session;
use silent_ot_randousha::gpt_oss::weights;

const REPO_ID: &str = "openai/gpt-oss-20b";
const MAX_SEQUENCE_LENGTH: usize = 1024;
const DEFAULT_MAX_NEW_TOKENS: usize = 256;
const DEFAULT_TEMPERATURE: f32 = 0.6;
const DEFAULT_TOP_P: f32 = 0.9;
const DEFAULT_REPETITION_PENALTY: f32 = 1.03;

// Harmony special tokens
const TOK_START: &str = "<|start|>";
const TOK_END: &str = "<|end|>";
const TOK_MESSAGE: &str = "<|message|>";
const TOK_CHANNEL: &str = "<|channel|>";
const TOK_RETURN: u32 = 200002;
const TOK_CALL: u32 = 200012;
const TOK_END_ID: u32 = 200007;

/// Format a user message in Harmony chat template.
fn format_harmony_prompt(user_message: &str, system_prompt: Option<&str>) -> String {
    let sys = system_prompt.unwrap_or(
        "You are a helpful assistant.\n\
Knowledge cutoff: 2024-06\n\n\
Give direct, accurate answers.",
    );
    format!(
        "{TOK_START}system{TOK_MESSAGE}{sys}{TOK_END}{TOK_START}user{TOK_MESSAGE}{user_message}{TOK_END}{TOK_START}assistant{TOK_CHANNEL}final{TOK_MESSAGE}"
    )
}

fn harmony_content_prefix(decoded: &str) -> &str {
    let mut cut = decoded.len();
    for marker in [TOK_END, TOK_START, TOK_CHANNEL, TOK_MESSAGE] {
        if let Some(i) = decoded.find(marker) {
            cut = cut.min(i);
        }
    }
    &decoded[..cut]
}

fn common_prefix_len(a: &str, b: &str) -> usize {
    let mut n = 0usize;
    for (ca, cb) in a.chars().zip(b.chars()) {
        if ca != cb {
            break;
        }
        n += ca.len_utf8();
    }
    n
}

fn parse_f32_env(name: &str, default: f32, min: f32, max: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .map(|v| v.clamp(min, max))
        .unwrap_or(default)
}

fn has_repeated_ngram_tail(tokens: &[u32], n: usize, repeats: usize) -> bool {
    if n == 0 || repeats < 2 || tokens.len() < n * repeats {
        return false;
    }
    let tail = &tokens[tokens.len() - (n * repeats)..];
    for r in 1..repeats {
        if tail[0..n] != tail[r * n..(r + 1) * n] {
            return false;
        }
    }
    true
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║  GPT-OSS-20B — ANE inference (Harmony format)                  ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════╝\n");

    let start = Instant::now();

    let model_files = weights::download_model(REPO_ID)?;
    eprintln!(
        "  {}L × {}d, {}q/{}kv GQA, {} experts (top-{})",
        model_files.config.num_hidden_layers,
        model_files.config.hidden_size,
        model_files.config.num_attention_heads,
        model_files.config.num_key_value_heads,
        model_files.config.num_local_experts,
        model_files.config.num_experts_per_tok
    );

    let tokenizer =
        Tokenizer::from_file(&model_files.tokenizer_path).map_err(|e| format!("tokenizer: {e}"))?;

    let model = OssModel::load(model_files, MAX_SEQUENCE_LENGTH)?;
    let mut session = Session::new(&model, MAX_SEQUENCE_LENGTH);
    eprintln!("  Ready ({:.1}s)\n", start.elapsed().as_secs_f64());

    let stdin = io::stdin();
    let mut rng = rand::thread_rng();
    let max_new_tokens = std::env::var("GPT_OSS_MAX_NEW_TOKENS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_MAX_NEW_TOKENS);
    let temperature = parse_f32_env("GPT_OSS_TEMPERATURE", DEFAULT_TEMPERATURE, 0.0, 5.0);
    let top_p = parse_f32_env("GPT_OSS_TOP_P", DEFAULT_TOP_P, 0.01, 1.0);
    let repetition_penalty = parse_f32_env(
        "GPT_OSS_REPETITION_PENALTY",
        DEFAULT_REPETITION_PENALTY,
        1.0,
        2.0,
    );

    loop {
        eprint!("> ");
        io::stderr().flush()?;

        let mut user_input = String::new();
        let bytes_read = stdin.lock().read_line(&mut user_input)?;
        if bytes_read == 0 {
            break;
        }
        let user_input = user_input.trim();
        if user_input.is_empty() {
            continue;
        }
        if user_input == "quit" || user_input == "exit" {
            break;
        }

        // Format with Harmony template
        let formatted = format_harmony_prompt(user_input, None);
        let encoding = tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| format!("encode: {e}"))?;
        let prompt_ids = encoding.get_ids();

        if prompt_ids.len() + max_new_tokens > MAX_SEQUENCE_LENGTH {
            eprintln!(
                "  (prompt too long, max {} tokens)",
                MAX_SEQUENCE_LENGTH - max_new_tokens
            );
            continue;
        }

        session.reset();
        let prefill_start = Instant::now();

        // Prefill
        eprintln!("  Prefilling {} tokens...", prompt_ids.len());
        let last_prompt = *prompt_ids
            .last()
            .expect("tokenizer returned empty prompt unexpectedly");
        for &tid in &prompt_ids[..prompt_ids.len().saturating_sub(1)] {
            session.decode_step_prefill(tid);
        }
        let prefill_secs = prefill_start.elapsed().as_secs_f64();
        eprintln!("  Prefill done ({prefill_secs:.1}s)");
        let gen_start = Instant::now();

        // Generate and stream only the assistant final-channel message content.
        let mut generated: Vec<u32> = prompt_ids.to_vec();
        let mut generated_count = 0usize;
        let mut prev_display = String::new();
        let mut printed_any = false;

        for tok_idx in 0..max_new_tokens {
            let logits = if tok_idx == 0 {
                session.decode_step(last_prompt)
            } else {
                let last = *generated.last().unwrap();
                session.decode_step(last)
            };

            let next = sampling::sample(
                logits,
                temperature,
                top_p,
                repetition_penalty,
                &generated,
                &mut rng,
            );
            generated.push(next);
            generated_count += 1;

            // Stop tokens
            if next == TOK_RETURN || next == TOK_CALL || next == TOK_END_ID {
                break;
            }

            // Decode full generated suffix for robust byte-fallback handling.
            let decoded = tokenizer
                .decode(&generated[prompt_ids.len()..], false)
                .map_err(|e| format!("{e}"))?;

            let display = harmony_content_prefix(&decoded);
            if display.starts_with(&prev_display) {
                let delta = &display[prev_display.len()..];
                if !delta.is_empty() {
                    printed_any = true;
                    eprint!("{delta}");
                    io::stderr().flush()?;
                }
            } else {
                let cp = common_prefix_len(&prev_display, display);
                let delta = &display[cp..];
                if !delta.is_empty() {
                    printed_any = true;
                    eprint!("{delta}");
                    io::stderr().flush()?;
                }
            }
            prev_display.clear();
            prev_display.push_str(display);

            // Stop when Harmony control markup appears after content.
            if display.len() < decoded.len() {
                break;
            }

            // Stop runaway repetition loops (common failure mode when end token is missed).
            let answer_tokens = &generated[prompt_ids.len()..];
            if (generated_count >= 40 && has_repeated_ngram_tail(answer_tokens, 8, 2))
                || (generated_count >= 48 && has_repeated_ngram_tail(answer_tokens, 4, 3))
            {
                break;
            }
        }
        eprintln!();
        if !printed_any {
            eprintln!("  (no final-channel content emitted)");
        }

        let gen_elapsed = gen_start.elapsed().as_secs_f64();
        let total_elapsed = prefill_secs + gen_elapsed;
        eprintln!(
            "  [prefill {:.1}s | {} gen tokens in {:.1}s ({:.2} tok/s) | end-to-end {:.1}s]\n",
            prefill_secs,
            generated_count,
            gen_elapsed,
            generated_count as f64 / gen_elapsed.max(1e-9),
            total_elapsed,
        );
    }

    Ok(())
}
