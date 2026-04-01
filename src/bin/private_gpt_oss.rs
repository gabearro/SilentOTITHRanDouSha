//! Private GPT-OSS-20B Chat — GQA attention Q·K^T via Beaver multiplication protocol.
//!
//! ANE: QKV projection, output projection (linear ops)
//! CPU: RoPE, Beaver Q·K^T (private), softmax, attn·V, MoE (expert SwiGLU)
//! GPU: Beaver triple generation in background

use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use rand::rngs::OsRng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use tokenizers::Tokenizer;

use silent_ot_randousha::gpt2::sampling;
use silent_ot_randousha::gpt_oss::model::OssModel;
use silent_ot_randousha::gpt_oss::private_session::PrivateOssSession;
use silent_ot_randousha::gpt_oss::weights;
use silent_ot_randousha::gpu::{setup_ot_correlations32, GpuTripleGen32};

const REPO_ID: &str = "openai/gpt-oss-20b";
const MAX_SEQUENCE_LENGTH: usize = 1024;
const MAX_NEW_TOKENS: usize = 10; // Reduced: MoE FFN is CPU-heavy (~2880² × 4 experts per layer)
const TEMPERATURE: f32 = 0.8;
const TOP_P: f32 = 0.95;
const REPETITION_PENALTY: f32 = 1.2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║  Private GPT-OSS-20B — GQA Q·K^T via Beaver multiplication     ║");
    eprintln!("║  ANE: QKV+output proj | CPU: Beaver+MoE | GPU: triple gen      ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════╝\n");

    let start = Instant::now();

    // Download and load model
    let model_files = weights::download_model(REPO_ID)?;
    eprintln!(
        "  Architecture: {}L × {}d, {}q/{}kv GQA, {} MoE experts (top-{})",
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

    eprintln!("  Model ready ({:.1}s)\n", start.elapsed().as_secs_f64());

    // Start GPU triple generation
    let n = 5usize;
    let t = 1usize;
    let triple_batch = 10_000_000usize;
    let spr = n - 2 * t;
    let num_rounds = triple_batch.div_ceil(spr);

    eprintln!("  Running Silent-OT setup for Beaver preprocessing...");
    let ot_setup_start = Instant::now();
    let mut ot_rng = OsRng;
    let ot = setup_ot_correlations32(n, t, num_rounds, &mut ot_rng)?;
    eprintln!(
        "  Silent-OT setup complete ({:.2}s)",
        ot_setup_start.elapsed().as_secs_f64()
    );

    let (triple_tx, triple_rx) = mpsc::sync_channel(4);
    let running = Arc::new(AtomicBool::new(true));
    let gpu_total = Arc::new(AtomicU64::new(0));

    let running_gpu = running.clone();
    let gpu_total_c = gpu_total.clone();
    let gpu_thread = std::thread::spawn(move || {
        let gpu = GpuTripleGen32::new(n, t).unwrap();
        let mut rng = ChaCha20Rng::from_rng(OsRng).unwrap();

        while running_gpu.load(Ordering::Relaxed) {
            let batch = gpu.generate(triple_batch, &ot, &mut rng).unwrap();
            gpu_total_c.fetch_add(triple_batch as u64, Ordering::Relaxed);
            if triple_tx.send(batch).is_err() {
                break;
            }
        }
    });

    eprintln!(
        "  GPU triple generation started ({}M per batch)",
        triple_batch / 1_000_000
    );
    eprintln!("  Type a prompt and press Enter. Type 'quit' to exit.\n");

    let mut rng = ChaCha20Rng::from_rng(OsRng)?;
    let stdin = io::stdin();

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

        if prompt_ids.len() + MAX_NEW_TOKENS > MAX_SEQUENCE_LENGTH {
            eprintln!(
                "  (prompt too long, max {} tokens)",
                MAX_SEQUENCE_LENGTH - MAX_NEW_TOKENS
            );
            continue;
        }

        let mut session = PrivateOssSession::new(&model, n, t, triple_rx, MAX_SEQUENCE_LENGTH);

        let gen_start = Instant::now();

        // Process prompt tokens
        eprint!("  Prefilling {} tokens...", prompt_ids.len());
        io::stderr().flush()?;
        for &tid in prompt_ids {
            let _ = session.decode_step_private(tid, &mut rng);
        }
        eprintln!(" done ({:.1}s)", gen_start.elapsed().as_secs_f64());

        // Generate
        let mut generated: Vec<u32> = prompt_ids.to_vec();
        let mut prev_text = tokenizer
            .decode(prompt_ids, true)
            .map_err(|e| format!("{e}"))?;
        print!("\x1b[1;32m{}\x1b[0m", prev_text);
        io::stdout().flush()?;

        for _i in 0..MAX_NEW_TOKENS {
            let last = *generated.last().unwrap();
            let logits = session.decode_step_private(last, &mut rng);

            // Debug first token's logits
            if generated.len() == prompt_ids.len() {
                let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
                indexed.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                let nan_count = logits.iter().filter(|v| !v.is_finite()).count();
                eprintln!("  Top-5 logits: {:?}", &indexed[..5]);
                eprintln!("  NaN: {}/{}", nan_count, logits.len());
            }

            let next = sampling::sample(
                &logits,
                TEMPERATURE,
                TOP_P,
                REPETITION_PENALTY,
                &generated,
                &mut rng,
            );
            generated.push(next);
            eprint!("[{}]", next); // show token IDs

            let cur = tokenizer
                .decode(&generated, true)
                .map_err(|e| format!("{e}"))?;
            if let Some(delta) = cur.strip_prefix(&prev_text) {
                print!("{delta}");
            }
            io::stdout().flush()?;
            prev_text = cur;
        }
        println!();

        let elapsed = gen_start.elapsed().as_secs_f64();
        let total_tokens = prompt_ids.len() + MAX_NEW_TOKENS;
        eprintln!(
            "\n\x1b[2m  [{} tokens in {:.1}s ({:.1} tok/s)]",
            total_tokens,
            elapsed,
            total_tokens as f64 / elapsed
        );
        eprintln!(
            "  Private attention: {} Beaver triples consumed",
            session.triples_consumed
        );
        eprintln!(
            "  GPU generated: {:.1}M triples\x1b[0m\n",
            gpu_total.load(Ordering::Relaxed) as f64 / 1e6
        );

        break; // single prompt (receiver moved into session)
    }

    running.store(false, Ordering::Relaxed);
    drop(gpu_thread);

    Ok(())
}
