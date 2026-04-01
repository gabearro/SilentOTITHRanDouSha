#include <metal_stdlib>
using namespace metal;

// FP4 E2M1 lookup table (all 16 nibble values)
constant float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,       // positive
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f  // negative
};

// Decode E8M0 scale byte: value = 2^(byte - 127)
inline float decode_e8m0(uint8_t byte) {
    if (byte == 0xFF) return 0.0f; // NaN → treat as zero
    int exp = int(byte) - 127;
    return as_type<float>(uint((exp + 127) << 23)); // construct IEEE 754 directly
}

// Dequantize one FP4 element from packed data
inline float dequant_fp4(device const uint8_t* data, device const uint8_t* scales,
                          uint elem_idx) {
    uint byte_idx = elem_idx / 2;
    uint block_idx = elem_idx / 32;
    uint8_t packed_byte = data[byte_idx];
    uint8_t nibble = (elem_idx % 2 == 0) ? (packed_byte & 0x0F) : (packed_byte >> 4);
    float scale = decode_e8m0(scales[block_idx]);
    return FP4_LUT[nibble] * scale;
}

// SiLU activation: x * sigmoid(x)
inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

struct MoeConstants {
    uint hidden_size;    // 2880
    uint inter_size;     // 2880
    float swiglu_clamp;  // 7.0
};

// Phase 1: Compute gate*up intermediate values
// Each thread computes one element of the intermediate [inter_size] vector
kernel void moe_gate_up(
    device const uint8_t* gate_up_blocks  [[buffer(0)]],
    device const uint8_t* gate_up_scales  [[buffer(1)]],
    device const float*   gate_up_bias    [[buffer(2)]],
    device const float*   input           [[buffer(3)]],
    device float*         gated_output    [[buffer(4)]],  // [inter_size]
    constant MoeConstants& C              [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= C.inter_size) return;

    uint hs = C.hidden_size;
    uint inter = C.inter_size;

    // gate_up is fused: first inter rows are gate, next inter rows are up
    // gate row `tid`: elements at [tid * hs .. (tid+1) * hs] in the original [2*inter, hs] matrix
    // up row `tid`: elements at [(inter + tid) * hs .. (inter + tid + 1) * hs]

    float gate_acc = gate_up_bias[tid]; // gate bias
    float up_acc = gate_up_bias[inter + tid]; // up bias

    uint gate_row_start = tid * hs;
    uint up_row_start = (inter + tid) * hs;

    for (uint i = 0; i < hs; i++) {
        float x = input[i];
        float g = dequant_fp4(gate_up_blocks, gate_up_scales, gate_row_start + i);
        float u = dequant_fp4(gate_up_blocks, gate_up_scales, up_row_start + i);
        gate_acc += g * x;
        up_acc += u * x;
    }

    // SiLU + clamp on gate, then multiply with up
    float gate_val = clamp(silu(gate_acc), -C.swiglu_clamp, C.swiglu_clamp);
    gated_output[tid] = gate_val * up_acc;
}

// Phase 2: Down projection from intermediate to hidden
// Each thread computes one element of the output [hidden_size] vector
kernel void moe_down_proj(
    device const uint8_t* down_blocks    [[buffer(0)]],
    device const uint8_t* down_scales    [[buffer(1)]],
    device const float*   down_bias      [[buffer(2)]],
    device const float*   gated_input    [[buffer(3)]],  // [inter_size]
    device float*         output         [[buffer(4)]],   // [hidden_size]
    constant MoeConstants& C             [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= C.hidden_size) return;

    uint inter = C.inter_size;
    float acc = down_bias[tid];

    uint row_start = tid * inter;
    for (uint i = 0; i < inter; i++) {
        float w = dequant_fp4(down_blocks, down_scales, row_start + i);
        acc += w * gated_input[i];
    }

    output[tid] = acc;
}

// ═══════════════════════════════════════════════════════════════
// LM head: simple matmul weight[vocab, hidden] @ hidden → logits[vocab]
// Each thread computes one logit (one row of the weight matrix)
// ═══════════════════════════════════════════════════════════════

struct LmConstants {
    uint vocab_size;
    uint hidden_size;
};

kernel void lm_head_matmul(
    device const float* weight    [[buffer(0)]],  // [vocab_size, hidden_size] row-major
    device const float* input     [[buffer(1)]],  // [hidden_size]
    device float*       output    [[buffer(2)]],  // [vocab_size]
    constant LmConstants& C       [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= C.vocab_size) return;

    uint hs = C.hidden_size;
    float acc = 0.0f;
    uint row_start = tid * hs;

    for (uint i = 0; i < hs; i++) {
        acc += weight[row_start + i] * input[i];
    }

    output[tid] = acc;
}
