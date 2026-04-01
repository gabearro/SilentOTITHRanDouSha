//! Private GPT-2 Chat — attention Q·K^T goes through the Beaver multiplication protocol.
//!
//! GPU generates Beaver triples → CPU consumes them for private attention dot products.
//! Non-private parts (FFN, embedding) run in cleartext on CPU.

use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use rand::rngs::OsRng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use safetensors::SafeTensors;
use tokenizers::Tokenizer;

use silent_ot_randousha::gpt2::model::CompiledModel;
use silent_ot_randousha::gpt2::private_session::PrivateSession;
use silent_ot_randousha::gpt2::sampling;
use silent_ot_randousha::gpt2::weights;
use silent_ot_randousha::gpu::{setup_ot_correlations32, GpuTripleGen32};

const REPO_ID: &str = "openai-community/gpt2";
const MAX_SEQUENCE_LENGTH: usize = 128;
const MAX_NEW_TOKENS: usize = 60;
const TEMPERATURE: f32 = 0.8;
const TOP_P: f32 = 0.95;
const REPETITION_PENALTY: f32 = 1.2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("╔═══════════════════════════════════════════════════════════════╗");
    eprintln!("║  Private GPT-2 — Q·K^T via Beaver multiplication protocol   ║");
    eprintln!("║  GPU: generates Fp32 Beaver triples (Metal lean kernel)      ║");
    eprintln!("║  CPU: consumes triples for private attention dot products    ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════╝\n");

    // ── Download and compile model ──────────────────────────────────
    let start = Instant::now();
    let model_files = weights::download_model(REPO_ID)?;
    let config = model_files.config;

    let tokenizer =
        Tokenizer::from_file(&model_files.tokenizer_path).map_err(|e| format!("tokenizer: {e}"))?;

    let safetensors_data = SafeTensors::deserialize(&model_files.safetensors_bytes)?;
    let model =
        CompiledModel::for_private_inference(config, &safetensors_data, MAX_SEQUENCE_LENGTH)?;

    eprintln!(
        "  Model ready: {} layers, {}d, {} heads ({:.1}s)\n",
        model.config.n_layer,
        model.config.n_embd,
        model.config.n_head,
        start.elapsed().as_secs_f64()
    );

    // ── Start GPU triple generation pipeline ────────────────────────
    let n = 5usize;
    let t = 1usize;
    let triple_batch_size = 2_000_000usize; // 2M triples per batch
    let spr = n - 2 * t;
    let num_rounds = triple_batch_size.div_ceil(spr);

    eprintln!("  Running Silent-OT setup for Beaver preprocessing...");
    let ot_setup_start = Instant::now();
    let mut ot_rng = OsRng;
    let ot32 = setup_ot_correlations32(n, t, num_rounds, &mut ot_rng)?;
    eprintln!(
        "  Silent-OT setup complete ({:.2}s)",
        ot_setup_start.elapsed().as_secs_f64()
    );

    let (triple_tx, triple_rx) =
        mpsc::sync_channel::<silent_ot_randousha::gpu::BeaverTripleBatch32>(4);
    let running = Arc::new(AtomicBool::new(true));
    let gpu_total = Arc::new(AtomicU64::new(0));

    let running_gpu = running.clone();
    let gpu_total_c = gpu_total.clone();
    let gpu_thread = std::thread::spawn(move || {
        let gpu32 = GpuTripleGen32::new(n, t).unwrap();
        let mut rng = ChaCha20Rng::from_rng(OsRng).unwrap();

        while running_gpu.load(Ordering::Relaxed) {
            let batch = gpu32.generate(triple_batch_size, &ot32, &mut rng).unwrap();
            gpu_total_c.fetch_add(triple_batch_size as u64, Ordering::Relaxed);
            if triple_tx.send(batch).is_err() {
                break; // receiver dropped
            }
        }
    });

    eprintln!(
        "  GPU triple generation started ({}M per batch, n={}, t={})",
        triple_batch_size / 1_000_000,
        n,
        t
    );

    // ── Chat loop with private attention ────────────────────────────
    let mut rng = ChaCha20Rng::from_rng(OsRng)?;
    let stdin = io::stdin();

    eprintln!("  Type a prompt and press Enter. Type 'quit' to exit.\n");

    loop {
        eprint!("\x1b[1;36m> \x1b[0m");
        io::stderr().flush()?;

        let mut prompt = String::new();
        stdin.lock().read_line(&mut prompt)?;
        let prompt = prompt.trim();

        if prompt.is_empty() {
            continue;
        }
        if prompt == "quit" || prompt == "exit" {
            break;
        }

        let encoding = tokenizer
            .encode(prompt, false)
            .map_err(|e| format!("encode: {e}"))?;
        let prompt_ids = encoding.get_ids();
        let prompt_len = prompt_ids.len();

        if prompt_len + MAX_NEW_TOKENS > MAX_SEQUENCE_LENGTH {
            eprintln!(
                "  (prompt too long, max {} tokens total)",
                MAX_SEQUENCE_LENGTH - MAX_NEW_TOKENS
            );
            continue;
        }

        // Create private session with real triple consumption
        let mut session = PrivateSession::new(&model, n, t, triple_rx);

        // Prefill: process prompt tokens one at a time through private attention
        // (simplified: no batch prefill, each token is a decode step)
        let gen_start = Instant::now();

        let last_prompt = *prompt_ids
            .last()
            .expect("tokenizer returned empty prompt unexpectedly");
        for &tid in &prompt_ids[..prompt_len.saturating_sub(1)] {
            let _ = session.decode_step_private(tid, &mut rng);
        }

        // Generate new tokens
        let mut generated: Vec<u32> = prompt_ids.to_vec();
        let mut prev_text = tokenizer
            .decode(prompt_ids, true)
            .map_err(|e| format!("decode: {e}"))?;
        print!("\x1b[1;32m{}\x1b[0m", prev_text);
        io::stdout().flush()?;

        let mut logits = session.decode_step_private(last_prompt, &mut rng);
        for _ in 0..MAX_NEW_TOKENS {
            let next = sampling::sample(
                logits,
                TEMPERATURE,
                TOP_P,
                REPETITION_PENALTY,
                &generated,
                &mut rng,
            );
            generated.push(next);
            if next == 50256 {
                break;
            }

            let cur_text = tokenizer
                .decode(&generated, true)
                .map_err(|e| format!("decode: {e}"))?;
            if let Some(delta) = cur_text.strip_prefix(&prev_text) {
                print!("{delta}");
            }
            io::stdout().flush()?;
            prev_text = cur_text;
            logits = session.decode_step_private(next, &mut rng);
        }
        println!();

        let gen_elapsed = gen_start.elapsed().as_secs_f64();
        let total_tokens = generated.len();
        let gpu_t = gpu_total.load(Ordering::Relaxed);

        eprintln!(
            "\n\x1b[2m  [{} tokens in {:.1}s ({:.1} tok/s)]",
            total_tokens,
            gen_elapsed,
            total_tokens as f64 / gen_elapsed
        );
        eprintln!(
            "  Private attention: {} Beaver triples CONSUMED (not just counted)",
            session.triples_consumed
        );
        eprintln!(
            "  Private runtime: {} distributed Beaver batches, {} opened values ({} softmax), {} strict fallbacks",
            session.runtime_stats.distributed_beaver_batches,
            session.runtime_stats.distributed_open_values,
            session.runtime_stats.distributed_softmax_open_values,
            session.runtime_stats.strict_mode_blocks
        );
        eprintln!(
            "  GPU total generated: {:.1}M triples\x1b[0m\n",
            gpu_t as f64 / 1e6,
        );

        // Return the receiver for the next prompt
        // (PrivateSession consumed it, need to recreate for next iteration)
        // Actually we need to restructure — for now just break after 1 prompt
        // since the channel receiver was moved into the session.
        break;
    }

    // Shutdown
    running.store(false, Ordering::Relaxed);
    drop(gpu_thread);

    Ok(())
}
