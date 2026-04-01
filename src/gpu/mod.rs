//! GPU-accelerated Beaver triple generation and MoE expert dispatch using Apple Metal.

pub mod moe_dispatch;

use crate::beaver::BeaverTripleBatch;
use crate::error::{ProtocolError, Result};
use crate::field::{AesCtrRng, Fp};
use crate::randousha::HyperInvertibleMatrix;
use crate::shamir::Shamir;
use crate::silent_ot::{DistributedSilentOt, ExpandedCorrelations, SilentOtParams};

use metal::*;
use rand::Rng;
use rayon;

fn ensure_embedded_metallib_nonempty(bytes: &[u8], kernel: &str) -> Result<()> {
    if bytes.is_empty() {
        return Err(ProtocolError::InvalidParams(format!(
            "{kernel} metallib is empty. This is usually a stale dummy build artifact from SILENT_OT_ALLOW_DUMMY_METAL=1. Rebuild with SILENT_OT_ALLOW_DUMMY_METAL=0 (or unset) after cleaning."
        )));
    }
    Ok(())
}

/// GPU constants struct matching the Metal shader's `Constants` layout exactly.
#[repr(C)]
#[derive(Clone, Copy)]
struct GpuConstants {
    n: u32,
    t: u32,
    count: u32,
    spr: u32,
    num_rounds: u32,
    party: u32,
    eval_raw: [u64; 8],
    him_rows: [[u64; 8]; 3],
    lag_sum: u64,
    lag_x_sum: u64,
    lag_xsq_sum: u64,
}

/// GPU-accelerated Beaver triple generator.
///
/// Precompiles the Metal pipeline once, then reuses it for multiple batches.
/// Uses `StorageModeShared` (unified memory) for zero-copy CPU↔GPU transfer.
pub struct GpuTripleGen {
    device: Device,
    queue: CommandQueue,
    pipeline_all: ComputePipelineState,
    pipeline_single: ComputePipelineState,
    constants: GpuConstants,
}

impl GpuTripleGen {
    /// Create a GPU triple generator. Compiles the Metal pipeline (~5ms one-time cost).
    pub fn new(n: usize, t: usize) -> Result<Self> {
        if n <= 2 * t {
            return Err(ProtocolError::InvalidParams(format!(
                "need n > 2t, got n={}, t={}",
                n, t
            )));
        }
        if n > 8 {
            return Err(ProtocolError::InvalidParams(format!(
                "GPU kernel supports n <= 8, got {}",
                n
            )));
        }

        let device = Device::system_default()
            .ok_or_else(|| ProtocolError::InvalidParams("no Metal GPU available".into()))?;
        let queue = device.new_command_queue();

        let lib_data = include_bytes!(concat!(env!("OUT_DIR"), "/beaver_triple.metallib"));
        ensure_embedded_metallib_nonempty(lib_data, "beaver_triple")?;
        let library = device.new_library_with_data(lib_data).map_err(|e| {
            ProtocolError::InvalidParams(format!("Metal library load failed: {}", e))
        })?;

        let fn_all = library
            .get_function("beaver_triple_gen", None)
            .map_err(|e| ProtocolError::InvalidParams(format!("kernel not found: {}", e)))?;
        let fn_single = library
            .get_function("beaver_triple_gen_single_party", None)
            .map_err(|e| ProtocolError::InvalidParams(format!("kernel not found: {}", e)))?;

        let pipeline_all = device
            .new_compute_pipeline_state_with_function(&fn_all)
            .map_err(|e| {
                ProtocolError::InvalidParams(format!("pipeline creation failed: {}", e))
            })?;
        let pipeline_single = device
            .new_compute_pipeline_state_with_function(&fn_single)
            .map_err(|e| {
                ProtocolError::InvalidParams(format!("pipeline creation failed: {}", e))
            })?;

        // Precompute constants
        let spr = n - 2 * t;
        let shamir_t = Shamir::new(n, t)?;
        let shamir_2t = Shamir::new(n, 2 * t)?;
        let him = HyperInvertibleMatrix::new(n);

        let eval_raw_vec: Vec<u64> = shamir_t.eval_points.iter().map(|fp| fp.raw()).collect();
        let lag_raw: Vec<u64> = shamir_2t
            .lagrange_coefficients()
            .iter()
            .map(|fp| fp.raw())
            .collect();

        let mut lag_sum: u64 = 0;
        let mut lag_x_sum: u64 = 0;
        let mut lag_xsq_sum: u64 = 0;
        for p in 0..n {
            lag_sum = Fp::add_raw(lag_sum, lag_raw[p]);
            lag_x_sum = Fp::add_raw(lag_x_sum, Fp::mul_raw(lag_raw[p], eval_raw_vec[p]));
            lag_xsq_sum = Fp::add_raw(
                lag_xsq_sum,
                Fp::mul_raw(lag_raw[p], Fp::mul_raw(eval_raw_vec[p], eval_raw_vec[p])),
            );
        }

        let mut eval_raw = [0u64; 8];
        eval_raw[..n].copy_from_slice(&eval_raw_vec);

        let mut him_rows = [[0u64; 8]; 3];
        for j in 0..spr.min(3) {
            for i in 0..n {
                him_rows[j][i] = him.get(j, i).raw();
            }
        }

        let constants = GpuConstants {
            n: n as u32,
            t: t as u32,
            count: 0,
            spr: spr as u32,
            num_rounds: 0,
            party: 0,
            eval_raw,
            him_rows,
            lag_sum,
            lag_x_sum,
            lag_xsq_sum,
        };

        Ok(GpuTripleGen {
            device,
            queue,
            pipeline_all,
            pipeline_single,
            constants,
        })
    }

    /// Generate `count` Beaver triples for all `n` parties on the GPU.
    /// Returns a `BeaverTripleBatch` with party-major SoA layout.
    pub fn generate<R: Rng>(
        &self,
        count: usize,
        ot_correlations: &[ExpandedCorrelations],
        rng: &mut R,
    ) -> Result<BeaverTripleBatch> {
        let n = self.constants.n as usize;
        let spr = self.constants.spr as usize;
        let num_rounds = count.div_ceil(spr);

        let t0 = std::time::Instant::now();

        // CPU prep: interleave OT + generate AES-CTR fresh buffer in parallel
        let fresh_per_round = 2 * spr;
        let fresh_total = num_rounds * fresh_per_round;
        let aes_seed: u64 = rng.gen();
        let round_seeds: Vec<u64> = (0..num_rounds).map(|_| rng.gen()).collect();

        // Parallel: OT interleave on one thread, AES-CTR on another
        let (ot_interleaved, fresh_buf) = rayon::join(
            || {
                let mut ot = vec![0u64; num_rounds * n];
                for round in 0..num_rounds {
                    for i in 0..n {
                        ot[round * n + i] =
                            unsafe { ot_correlations[i].get_random_raw_unchecked(round) };
                    }
                }
                ot
            },
            || {
                let mut buf = vec![0u64; fresh_total];
                let mut aes_rng = AesCtrRng::from_seed(aes_seed);
                aes_rng.fill_field_raw(&mut buf);
                buf
            },
        );

        let t1 = std::time::Instant::now();

        let mut consts = self.constants;
        consts.count = count as u32;
        consts.num_rounds = num_rounds as u32;

        // Use new_buffer_with_bytes_no_copy where possible to avoid memcpy.
        // For output buffers, allocate empty with StorageModeShared.
        let ot_buf = self.new_buffer_from_slice(&ot_interleaved);
        let fresh_gpu = self.new_buffer_from_slice(&fresh_buf);
        let seed_buf = self.new_buffer_from_slice(&round_seeds);

        let total = n * count;
        let out_bytes = (total * 8) as u64;
        let a_buf = self
            .device
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
        let b_buf = self
            .device
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
        let c_buf = self
            .device
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

        let t2 = std::time::Instant::now();

        self.dispatch(
            &self.pipeline_all,
            &[&ot_buf, &fresh_gpu, &a_buf, &b_buf, &c_buf],
            &consts,
            &seed_buf,
            num_rounds,
        );

        let t3 = std::time::Instant::now();

        // Zero-copy: reinterpret GPU shared memory buffer as Vec<u64> directly.
        // SAFETY: StorageModeShared guarantees CPU can read GPU output after wait_until_completed().
        // We transfer ownership of the buffer contents into a Vec.
        let a_values = self.read_buffer(&a_buf, total);
        let b_values = self.read_buffer(&b_buf, total);
        let c_values = self.read_buffer(&c_buf, total);

        let t4 = std::time::Instant::now();
        let _ = (t0, t1, t2, t3, t4);

        Ok(BeaverTripleBatch {
            n,
            count,
            a_values,
            b_values,
            c_values,
        })
    }

    /// Generate `count` triples for a single party.
    /// Returns `(a_values, b_values, c_values)` each of length `count`.
    pub fn generate_single_party<R: Rng>(
        &self,
        party: usize,
        count: usize,
        ot_correlations: &[ExpandedCorrelations],
        rng: &mut R,
    ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>)> {
        let n = self.constants.n as usize;
        let spr = self.constants.spr as usize;
        let num_rounds = count.div_ceil(spr);

        let mut ot_interleaved = vec![0u64; num_rounds * n];
        for round in 0..num_rounds {
            for i in 0..n {
                ot_interleaved[round * n + i] =
                    unsafe { ot_correlations[i].get_random_raw_unchecked(round) };
            }
        }

        let fresh_per_round = 2 * spr;
        let fresh_total = num_rounds * fresh_per_round;
        let mut fresh_buf = vec![0u64; fresh_total];
        let seed: u64 = rng.gen();
        let mut aes_rng = AesCtrRng::from_seed(seed);
        aes_rng.fill_field_raw(&mut fresh_buf);

        let round_seeds: Vec<u64> = (0..num_rounds).map(|_| rng.gen()).collect();

        let mut consts = self.constants;
        consts.count = count as u32;
        consts.num_rounds = num_rounds as u32;
        consts.party = party as u32;

        let ot_buf = self.new_buffer_from_slice(&ot_interleaved);
        let fresh_gpu = self.new_buffer_from_slice(&fresh_buf);
        let seed_buf = self.new_buffer_from_slice(&round_seeds);

        let out_bytes = (count * 8) as u64;
        let a_buf = self
            .device
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
        let b_buf = self
            .device
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
        let c_buf = self
            .device
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

        self.dispatch(
            &self.pipeline_single,
            &[&ot_buf, &fresh_gpu, &a_buf, &b_buf, &c_buf],
            &consts,
            &seed_buf,
            num_rounds,
        );

        Ok((
            self.read_buffer(&a_buf, count),
            self.read_buffer(&b_buf, count),
            self.read_buffer(&c_buf, count),
        ))
    }

    fn new_buffer_from_slice(&self, data: &[u64]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            (data.len() * 8) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn read_buffer(&self, buf: &Buffer, count: usize) -> Vec<u64> {
        // Use copy_nonoverlapping for maximum memcpy speed
        let mut out = Vec::with_capacity(count);
        unsafe {
            out.set_len(count);
            std::ptr::copy_nonoverlapping(buf.contents() as *const u64, out.as_mut_ptr(), count);
        }
        out
    }

    fn dispatch(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[&Buffer],
        consts: &GpuConstants,
        seed_buf: &Buffer,
        num_rounds: usize,
    ) {
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);

        for (i, buf) in buffers.iter().enumerate() {
            enc.set_buffer(i as u64, Some(buf), 0);
        }
        enc.set_bytes(
            5,
            std::mem::size_of::<GpuConstants>() as u64,
            consts as *const GpuConstants as *const _,
        );
        enc.set_buffer(6, Some(seed_buf), 0);

        let grid = MTLSize::new(num_rounds as u64, 1, 1);
        let tg_width = pipeline.thread_execution_width().min(num_rounds as u64);
        let tg = MTLSize::new(tg_width, 1, 1);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();
    }
}

// ── 32-bit Mersenne field GPU triple generator ──────────────────────

use crate::field32::{Fp32, PRIME32};

/// Batch of Beaver triples using the 32-bit Mersenne field (p = 2^31-1).
pub struct BeaverTripleBatch32 {
    pub n: usize,
    pub count: usize,
    pub a_values: Vec<u32>,
    pub b_values: Vec<u32>,
    pub c_values: Vec<u32>,
}

impl BeaverTripleBatch32 {
    pub fn triple_values(&self, k: usize, party: usize) -> (u32, u32, u32) {
        let idx = party * self.count + k;
        (self.a_values[idx], self.b_values[idx], self.c_values[idx])
    }
}

/// GPU constants for 32-bit field, matching Metal's Constants32 layout.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuConstants32 {
    pub n: u32,
    pub t: u32,
    pub count: u32,
    pub spr: u32,
    pub num_rounds: u32,
    pub party: u32,
    pub eval_raw: [u32; 8],
    pub him_rows: [[u32; 8]; 3],
    pub lag_sum: u32,
    pub lag_x_sum: u32,
    pub lag_xsq_sum: u32,
    pub ot_offsets: [u32; 8], // byte offset into packed OT buffer for each party
}

/// 32-bit OT correlations for GPU use.
pub struct ExpandedCorrelations32 {
    pub party_id: usize,
    pub num_ots: usize,
    pub random_values: Vec<u32>,
}

impl ExpandedCorrelations32 {
    pub fn from_expanded(source: &ExpandedCorrelations, num_ots: usize) -> Result<Self> {
        if source.num_ots() < num_ots {
            return Err(ProtocolError::InvalidParams(format!(
                "ExpandedCorrelations32::from_expanded needs {} OTs, got {}",
                num_ots,
                source.num_ots()
            )));
        }
        let random_values: Vec<u32> = (0..num_ots)
            .map(|i| (source.get_random_raw(i) % PRIME32 as u64) as u32)
            .collect();
        Ok(ExpandedCorrelations32 {
            party_id: source.party_id,
            num_ots,
            random_values,
        })
    }

    pub fn from_random<R: Rng>(party_id: usize, num_ots: usize, rng: &mut R) -> Self {
        let seed: u64 = rng.gen();
        let mut mix = crate::field::SplitMix64::new(seed);
        let random_values: Vec<u32> = (0..num_ots)
            .map(|_| {
                let v = (mix.next_u64() >> 33) as u32; // top 31 bits
                if v >= PRIME32 {
                    v - PRIME32
                } else {
                    v
                }
            })
            .collect();
        ExpandedCorrelations32 {
            party_id,
            num_ots,
            random_values,
        }
    }
}

/// Run distributed Silent-OT setup and convert expanded correlations to the
/// 32-bit field used by the GPU Beaver kernel.
///
/// This avoids demo-only `from_random` preprocessing and produces
/// protocol-derived correlations for each party.
pub fn setup_ot_correlations32<R: Rng>(
    n: usize,
    t: usize,
    num_ots: usize,
    rng: &mut R,
) -> Result<Vec<ExpandedCorrelations32>> {
    let params = SilentOtParams::with_arity(n, t, num_ots.max(16), 4)?;
    let protocol = DistributedSilentOt::new(params);
    let mut states: Vec<_> = (0..n).map(|i| protocol.init_party(i, rng)).collect();

    // Round A
    let mut round_a = vec![Vec::new(); n];
    for s in states.iter() {
        for (to, commitment, punct_idx) in DistributedSilentOt::round_a_messages(s) {
            round_a[to].push((s.party_id, commitment, punct_idx));
        }
    }
    for (i, s) in states.iter_mut().enumerate() {
        DistributedSilentOt::process_round_a(s, &round_a[i])?;
    }

    // Round B
    let mut round_b = vec![Vec::new(); n];
    for s in states.iter() {
        for (to, path, seed) in DistributedSilentOt::round_b_messages(s)? {
            round_b[to].push((s.party_id, path, seed));
        }
    }
    for (i, s) in states.iter_mut().enumerate() {
        DistributedSilentOt::process_round_b(s, &round_b[i])?;
    }

    let expanded: Vec<ExpandedCorrelations> = states
        .iter()
        .map(DistributedSilentOt::expand)
        .collect::<Result<_>>()?;

    expanded
        .iter()
        .map(|e| ExpandedCorrelations32::from_expanded(e, num_ots))
        .collect()
}

/// Pre-allocated GPU buffers for zero-alloc repeated dispatch.
pub struct PreallocatedBuffers {
    ot_buf: Buffer,
    seed_buf: Buffer,
    pub a_buf: Buffer,
    pub b_buf: Buffer,
    pub c_buf: Buffer,
    pub count: usize,
    pub num_rounds: usize,
}

/// GPU-accelerated Beaver triple generator for the 32-bit Mersenne field.
///
/// Uses native u32×u32→u64 multiply on GPU — no mulhi needed, half the register
/// pressure of the 64-bit kernel, enabling ~2x higher occupancy.
pub struct GpuTripleGen32 {
    device: Device,
    queue: CommandQueue,
    pipeline_all: ComputePipelineState,
    pipeline_single: ComputePipelineState,
    constants: GpuConstants32,
}

impl GpuTripleGen32 {
    pub fn new(n: usize, t: usize) -> Result<Self> {
        if n <= 2 * t {
            return Err(ProtocolError::InvalidParams(format!(
                "need n > 2t, got n={}, t={}",
                n, t
            )));
        }
        if n > 8 {
            return Err(ProtocolError::InvalidParams(format!(
                "GPU kernel supports n <= 8, got {}",
                n
            )));
        }

        let device = Device::system_default()
            .ok_or_else(|| ProtocolError::InvalidParams("no Metal GPU available".into()))?;
        let queue = device.new_command_queue();

        let lib_data = include_bytes!(concat!(env!("OUT_DIR"), "/beaver_triple32.metallib"));
        ensure_embedded_metallib_nonempty(lib_data, "beaver_triple32")?;
        let library = device.new_library_with_data(lib_data).map_err(|e| {
            ProtocolError::InvalidParams(format!("Metal 32-bit library load failed: {}", e))
        })?;

        let fn_all = library
            .get_function("beaver_triple_gen_32", None)
            .map_err(|e| ProtocolError::InvalidParams(format!("kernel not found: {}", e)))?;
        let fn_single = library
            .get_function("beaver_triple_gen_single_party_32", None)
            .map_err(|e| ProtocolError::InvalidParams(format!("kernel not found: {}", e)))?;

        let pipeline_all = device
            .new_compute_pipeline_state_with_function(&fn_all)
            .map_err(|e| ProtocolError::InvalidParams(format!("pipeline failed: {}", e)))?;
        let pipeline_single = device
            .new_compute_pipeline_state_with_function(&fn_single)
            .map_err(|e| ProtocolError::InvalidParams(format!("pipeline failed: {}", e)))?;

        // Precompute constants in 32-bit field
        let spr = n - 2 * t;
        let shamir_t = crate::shamir::Shamir::new(n, t)?;
        let shamir_2t = crate::shamir::Shamir::new(n, 2 * t)?;
        let him = HyperInvertibleMatrix::new(n);

        // Eval points: 1, 2, ..., n (same as 64-bit but reduced mod PRIME32)
        let eval_raw_vec: Vec<u32> = (1..=n as u32).map(|v| Fp32::new(v).raw()).collect();

        // Lagrange coefficients in 32-bit field
        // Recompute from scratch using Fp32 arithmetic
        let lag_raw: Vec<u32> = {
            let points: Vec<Fp32> = (1..=n as u32).map(Fp32::new).collect();
            let mut lags = vec![Fp32::ZERO; n];
            for i in 0..n {
                let mut num = Fp32::ONE;
                let mut den = Fp32::ONE;
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    num *= -points[j];
                    den *= points[i] - points[j];
                }
                lags[i] = num * den.inv();
            }
            lags.iter().map(|f| f.raw()).collect()
        };

        let mut lag_sum: u32 = 0;
        let mut lag_x_sum: u32 = 0;
        let mut lag_xsq_sum: u32 = 0;
        for p in 0..n {
            lag_sum = Fp32::add_raw(lag_sum, lag_raw[p]);
            lag_x_sum = Fp32::add_raw(lag_x_sum, Fp32::mul_raw(lag_raw[p], eval_raw_vec[p]));
            lag_xsq_sum = Fp32::add_raw(
                lag_xsq_sum,
                Fp32::mul_raw(lag_raw[p], Fp32::mul_raw(eval_raw_vec[p], eval_raw_vec[p])),
            );
        }

        // HIM rows in 32-bit
        let mut him_rows = [[0u32; 8]; 3];
        for j in 0..spr.min(3) {
            for i in 0..n {
                // HIM entry = alpha_{i+1}^j, computed in 32-bit field
                let alpha = Fp32::new((i + 1) as u32);
                him_rows[j][i] = alpha.pow(j as u32).raw();
            }
        }

        let mut eval_raw = [0u32; 8];
        eval_raw[..n].copy_from_slice(&eval_raw_vec);

        let constants = GpuConstants32 {
            n: n as u32,
            t: t as u32,
            count: 0,
            spr: spr as u32,
            num_rounds: 0,
            party: 0,
            eval_raw,
            him_rows,
            lag_sum,
            lag_x_sum,
            lag_xsq_sum,
            ot_offsets: [0; 8], // set per-dispatch based on actual OT layout
        };

        Ok(GpuTripleGen32 {
            device,
            queue,
            pipeline_all,
            pipeline_single,
            constants,
        })
    }

    pub fn generate<R: Rng>(
        &self,
        count: usize,
        ot_correlations: &[ExpandedCorrelations32],
        rng: &mut R,
    ) -> Result<BeaverTripleBatch32> {
        let n = self.constants.n as usize;
        let spr = self.constants.spr as usize;
        let num_rounds = count.div_ceil(spr);

        let t0 = std::time::Instant::now();

        let round_seeds: Vec<u32> = (0..num_rounds).map(|_| rng.gen::<u32>()).collect();
        let mut ot_packed = Vec::with_capacity(num_rounds * n);
        let mut ot_offsets = [0u32; 8];
        for i in 0..n {
            ot_offsets[i] = ot_packed.len() as u32;
            ot_packed.extend_from_slice(&ot_correlations[i].random_values[..num_rounds]);
        }

        let t1 = std::time::Instant::now();

        let mut consts = self.constants;
        consts.count = count as u32;
        consts.num_rounds = num_rounds as u32;
        consts.ot_offsets = ot_offsets;

        let ot_buf = self.new_buffer_from_u32_slice(&ot_packed);
        let seed_buf = self.new_buffer_from_u32_slice(&round_seeds);

        let total = n * count;
        let out_bytes = (total * 4) as u64;
        let a_buf = self
            .device
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
        let b_buf = self
            .device
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
        let c_buf = self
            .device
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

        let t2 = std::time::Instant::now();

        self.dispatch(
            &self.pipeline_all,
            &[&ot_buf, &a_buf, &b_buf, &c_buf],
            &consts,
            &seed_buf,
            num_rounds,
        );

        let t3 = std::time::Instant::now();

        let a_values = self.read_u32_buffer(&a_buf, total);
        let b_values = self.read_u32_buffer(&b_buf, total);
        let c_values = self.read_u32_buffer(&c_buf, total);

        let t4 = std::time::Instant::now();
        let _ = (t0, t1, t2, t3, t4);

        Ok(BeaverTripleBatch32 {
            n,
            count,
            a_values,
            b_values,
            c_values,
        })
    }

    pub fn generate_single_party<R: Rng>(
        &self,
        party: usize,
        count: usize,
        ot_correlations: &[ExpandedCorrelations32],
        rng: &mut R,
    ) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>)> {
        let t0 = std::time::Instant::now();
        let n = self.constants.n as usize;
        let spr = self.constants.spr as usize;
        let num_rounds = count.div_ceil(spr);
        let round_seeds: Vec<u32> = (0..num_rounds).map(|_| rng.gen::<u32>()).collect();

        let mut ot_packed = Vec::with_capacity(num_rounds * n);
        let mut ot_offsets = [0u32; 8];
        for i in 0..n {
            ot_offsets[i] = ot_packed.len() as u32;
            ot_packed.extend_from_slice(&ot_correlations[i].random_values[..num_rounds]);
        }
        let t1 = std::time::Instant::now();

        let mut consts = self.constants;
        consts.count = count as u32;
        consts.num_rounds = num_rounds as u32;
        consts.party = party as u32;
        consts.ot_offsets = ot_offsets;

        let ot_buf = self.new_buffer_from_u32_slice(&ot_packed);
        let seed_buf = self.new_buffer_from_u32_slice(&round_seeds);

        let out_bytes = (count * 4) as u64;
        let a_buf = self
            .device
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
        let b_buf = self
            .device
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
        let c_buf = self
            .device
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
        let t2 = std::time::Instant::now();

        self.dispatch(
            &self.pipeline_single,
            &[&ot_buf, &a_buf, &b_buf, &c_buf],
            &consts,
            &seed_buf,
            num_rounds,
        );
        let t3 = std::time::Instant::now();

        let result = (
            self.read_u32_buffer(&a_buf, count),
            self.read_u32_buffer(&b_buf, count),
            self.read_u32_buffer(&c_buf, count),
        );
        let t4 = std::time::Instant::now();
        let _ = (t0, t1, t2, t3, t4);

        Ok(result)
    }

    /// Generate triples on GPU and process them via a consumer callback,
    /// without copying the output. The consumer reads directly from the GPU's
    /// shared memory buffer — true zero-copy.
    ///
    /// Returns the consumer's accumulated result.
    pub fn generate_streaming<R: Rng, F>(
        &self,
        count: usize,
        ot_correlations: &[ExpandedCorrelations32],
        rng: &mut R,
        consumer: F,
    ) -> Result<u64>
    where
        F: Fn(usize, &[u32], &[u32], &[u32]) -> u64,
    {
        let n = self.constants.n as usize;
        let spr = self.constants.spr as usize;
        let num_rounds = count.div_ceil(spr);
        let round_seeds: Vec<u32> = (0..num_rounds).map(|_| rng.gen::<u32>()).collect();

        let mut ot_packed = Vec::with_capacity(num_rounds * n);
        let mut ot_offsets = [0u32; 8];
        for i in 0..n {
            ot_offsets[i] = ot_packed.len() as u32;
            ot_packed.extend_from_slice(&ot_correlations[i].random_values[..num_rounds]);
        }

        let mut consts = self.constants;
        consts.count = count as u32;
        consts.num_rounds = num_rounds as u32;
        consts.ot_offsets = ot_offsets;

        let ot_buf = self.new_buffer_from_u32_slice(&ot_packed);
        let seed_buf = self.new_buffer_from_u32_slice(&round_seeds);

        let total = n * count;
        let out_bytes = (total * 4) as u64;
        let a_buf = self
            .device
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
        let b_buf = self
            .device
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
        let c_buf = self
            .device
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

        self.dispatch(
            &self.pipeline_all,
            &[&ot_buf, &a_buf, &b_buf, &c_buf],
            &consts,
            &seed_buf,
            num_rounds,
        );

        // Zero-copy: read directly from GPU's shared memory
        let a_slice = unsafe { std::slice::from_raw_parts(a_buf.contents() as *const u32, total) };
        let b_slice = unsafe { std::slice::from_raw_parts(b_buf.contents() as *const u32, total) };
        let c_slice = unsafe { std::slice::from_raw_parts(c_buf.contents() as *const u32, total) };

        // Process all triples through consumer (CPU reads from GPU memory, no copy)
        let mut acc: u64 = 0;
        for k in 0..count {
            acc = acc.wrapping_add(consumer(k, &a_slice[k..], &b_slice[k..], &c_slice[k..]));
        }

        Ok(acc)
    }

    /// Pre-allocate GPU buffers for a given count, reusable across dispatches.
    /// Returns (ot_buf, seed_buf, a_buf, b_buf, c_buf).
    pub fn preallocate_single_party(&self, count: usize) -> PreallocatedBuffers {
        let n = self.constants.n as usize;
        let spr = self.constants.spr as usize;
        let num_rounds = count.div_ceil(spr);

        let ot_bytes = (num_rounds * n * 4) as u64;
        let seed_bytes = (num_rounds * 4) as u64;
        let out_bytes = (count * 4) as u64;

        PreallocatedBuffers {
            ot_buf: self
                .device
                .new_buffer(ot_bytes, MTLResourceOptions::StorageModeShared),
            seed_buf: self
                .device
                .new_buffer(seed_bytes, MTLResourceOptions::StorageModeShared),
            a_buf: self
                .device
                .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared),
            b_buf: self
                .device
                .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared),
            c_buf: self
                .device
                .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared),
            count,
            num_rounds,
        }
    }

    /// Generate single-party triples into pre-allocated buffers (zero-copy).
    /// Returns slices into GPU shared memory — no memcpy.
    pub fn generate_single_party_prealloc<'a, R: Rng>(
        &self,
        party: usize,
        ot_correlations: &[ExpandedCorrelations32],
        rng: &mut R,
        bufs: &'a PreallocatedBuffers,
    ) -> (&'a [u32], &'a [u32], &'a [u32]) {
        let n = self.constants.n as usize;
        let count = bufs.count;
        let num_rounds = bufs.num_rounds;

        // Write OT data into pre-allocated GPU buffer
        let ot_ptr = bufs.ot_buf.contents() as *mut u32;
        let mut ot_offsets = [0u32; 8];
        for i in 0..n {
            ot_offsets[i] = (i * num_rounds) as u32;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ot_correlations[i].random_values.as_ptr(),
                    ot_ptr.add(i * num_rounds),
                    num_rounds,
                );
            }
        }

        // Write seeds into pre-allocated GPU buffer
        let seed_ptr = bufs.seed_buf.contents() as *mut u32;
        for r in 0..num_rounds {
            unsafe {
                *seed_ptr.add(r) = rng.gen::<u32>();
            }
        }

        let mut consts = self.constants;
        consts.count = count as u32;
        consts.num_rounds = num_rounds as u32;
        consts.party = party as u32;
        consts.ot_offsets = ot_offsets;

        self.dispatch(
            &self.pipeline_single,
            &[&bufs.ot_buf, &bufs.a_buf, &bufs.b_buf, &bufs.c_buf],
            &consts,
            &bufs.seed_buf,
            num_rounds,
        );

        unsafe {
            let a = std::slice::from_raw_parts(bufs.a_buf.contents() as *const u32, count);
            let b = std::slice::from_raw_parts(bufs.b_buf.contents() as *const u32, count);
            let c = std::slice::from_raw_parts(bufs.c_buf.contents() as *const u32, count);
            (a, b, c)
        }
    }

    /// Dispatch kernel only on pre-loaded buffers (no OT copy, no readback).
    /// Used for kernel-only throughput measurement. The OT and seed data must
    /// already be in the buffers from a previous `generate_single_party_prealloc` call.
    pub fn dispatch_kernel_only(&self, party: usize, bufs: &PreallocatedBuffers) {
        let n = self.constants.n as usize;
        let mut ot_offsets = [0u32; 8];
        for i in 0..n {
            ot_offsets[i] = (i * bufs.num_rounds) as u32;
        }
        let mut consts = self.constants;
        consts.count = bufs.count as u32;
        consts.num_rounds = bufs.num_rounds as u32;
        consts.party = party as u32;
        consts.ot_offsets = ot_offsets;

        self.dispatch(
            &self.pipeline_single,
            &[&bufs.ot_buf, &bufs.a_buf, &bufs.b_buf, &bufs.c_buf],
            &consts,
            &bufs.seed_buf,
            bufs.num_rounds,
        );
    }

    fn new_buffer_from_u32_slice(&self, data: &[u32]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            (data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn new_buffer_from_u64_slice(&self, data: &[u64]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            (data.len() * 8) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn read_u32_buffer(&self, buf: &Buffer, count: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(count);
        unsafe {
            out.set_len(count);
            std::ptr::copy_nonoverlapping(buf.contents() as *const u32, out.as_mut_ptr(), count);
        }
        out
    }

    fn dispatch(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[&Buffer],
        consts: &GpuConstants32,
        seed_buf: &Buffer,
        num_threads: usize,
    ) {
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);

        for (i, buf) in buffers.iter().enumerate() {
            enc.set_buffer(i as u64, Some(buf), 0);
        }
        let const_idx = buffers.len() as u64;
        enc.set_bytes(
            const_idx,
            std::mem::size_of::<GpuConstants32>() as u64,
            consts as *const GpuConstants32 as *const _,
        );
        enc.set_buffer(const_idx + 1, Some(seed_buf), 0);

        // SIMD kernel: one thread per triple, threadgroup = SIMD width for shuffle ops
        let grid = MTLSize::new(num_threads as u64, 1, 1);
        let tg_width = pipeline.thread_execution_width().min(num_threads as u64);
        let tg = MTLSize::new(tg_width, 1, 1);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_setup_ot_correlations32() {
        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let n = 5;
        let t = 1;
        let num_ots = 32;

        let corr = setup_ot_correlations32(n, t, num_ots, &mut rng).unwrap();
        assert_eq!(corr.len(), n);

        for (i, c) in corr.iter().enumerate() {
            assert_eq!(c.party_id, i);
            assert_eq!(c.num_ots, num_ots);
            assert_eq!(c.random_values.len(), num_ots);
        }
    }
}
