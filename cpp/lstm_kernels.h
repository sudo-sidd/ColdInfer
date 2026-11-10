// lstm_kernels.h — INT8 GEMM, sigmoid, tanh, and sparse GEMM kernels
// Part of ColdInfer — bare-metal LSTM inference engine
//
// All kernels operate on statically-allocated buffers.
// No dynamic allocation anywhere in this file.

#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------
// Activation functions (fast inline approximations)
// ---------------------------------------------------------------------------
inline float fast_sigmoid(float x) {
    // Numerically stable sigmoid: 1/(1+exp(-x))
    // Clamp to prevent extreme exp values on embedded targets
    if (x >  10.0f) return 1.0f;
    if (x < -10.0f) return 0.0f;
    return 1.0f / (1.0f + std::exp(-x));
}

inline float fast_tanh(float x) {
    // Standard tanh — clamped for embedded stability
    if (x >  10.0f) return  1.0f;
    if (x < -10.0f) return -1.0f;
    return std::tanh(x);
}

inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// ---------------------------------------------------------------------------
// INT8 dense GEMM: y += W * x (dequantized on the fly)
//
//   W_q:   (rows, cols) INT8 quantized weight matrix  [Flash]
//   scale: dequantization scale factor                [Flash]
//   x:     (cols,)      FP32 input activations        [SRAM]
//   y:     (rows,)      FP32 accumulator (in-place)   [SRAM]
//
// Each W_q[i][j] dequantizes to float via: W_q[i][j] * scale
// Accumulation in FP32 preserves precision despite INT8 weights.
// ---------------------------------------------------------------------------
template <int ROWS, int COLS>
inline void int8_gemm_accum(
    const int8_t W_q[ROWS][COLS],
    float        scale,
    const float  x[COLS],
    float        y[ROWS]
) {
    for (int i = 0; i < ROWS; ++i) {
        float acc = 0.0f;
        const int8_t* row = W_q[i];
        const float* xp   = x;
        // Manual unroll hint for the inner loop
        for (int j = 0; j < COLS; ++j) {
            acc += (float)row[j] * xp[j];
        }
        y[i] += acc * scale;
    }
}

// ---------------------------------------------------------------------------
// INT8 sparse GEMM: same as above but skips zero-valued weights
// (produced by magnitude pruning in export_chatbot.py)
//
// On pruned models (50%+ sparsity), this halves effective compute.
// The branch predictor on Xtensa LX6 (ESP32) handles this well.
// ---------------------------------------------------------------------------
template <int ROWS, int COLS>
inline void int8_sparse_gemm_accum(
    const int8_t W_q[ROWS][COLS],
    float        scale,
    const float  x[COLS],
    float        y[ROWS]
) {
    for (int i = 0; i < ROWS; ++i) {
        float acc = 0.0f;
        const int8_t* row = W_q[i];
        for (int j = 0; j < COLS; ++j) {
            if (row[j] != 0) {            // skip pruned (zero) weights
                acc += (float)row[j] * x[j];
            }
        }
        y[i] += acc * scale;
    }
}

// ---------------------------------------------------------------------------
// Add bias vector into accumulator: y += bias
// ---------------------------------------------------------------------------
template <int SIZE>
inline void add_bias(const float bias[SIZE], float y[SIZE]) {
    for (int i = 0; i < SIZE; ++i) {
        y[i] += bias[i];
    }
}

// ---------------------------------------------------------------------------
// Copy float array: dst = src
// ---------------------------------------------------------------------------
template <int SIZE>
inline void copy_vec(const float src[SIZE], float dst[SIZE]) {
    memcpy(dst, src, SIZE * sizeof(float));
}

// ---------------------------------------------------------------------------
// Zero float array
// ---------------------------------------------------------------------------
template <int SIZE>
inline void zero_vec(float v[SIZE]) {
    memset(v, 0, SIZE * sizeof(float));
}

// ---------------------------------------------------------------------------
// LSTM cell forward pass (single time step, single layer)
//
// Computes one step of:
//   i_t = σ(W_ii * x + W_hi * h + b_i)   [input gate]
//   f_t = σ(W_if * x + W_hf * h + b_f)   [forget gate]
//   g_t = tanh(W_ig * x + W_hg * h + b_g) [cell gate]
//   o_t = σ(W_io * x + W_ho * h + b_o)   [output gate]
//   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
//   h_t = o_t ⊙ tanh(c_t)
//
// All weight matrices are INT8; biases stay FP32 (tiny, 4×hidden floats).
// Uses sparse GEMM to skip pruned zero weights.
//
// Template params:
//   INPUT_DIM  — width of x (embed_dim for layer 1, hidden_size for layer 2)
//   HIDDEN     — hidden state size
// ---------------------------------------------------------------------------
template <int INPUT_DIM, int HIDDEN>
struct LSTMCell {
    // Compute one timestep. Reads h_prev/c_prev, writes h_next/c_next.
    static void step(
        // Input-hidden weight matrices (4 gates × INPUT_DIM)
        const int8_t wih_i[HIDDEN][INPUT_DIM], float scale_wih_i,
        const int8_t wih_f[HIDDEN][INPUT_DIM], float scale_wih_f,
        const int8_t wih_g[HIDDEN][INPUT_DIM], float scale_wih_g,
        const int8_t wih_o[HIDDEN][INPUT_DIM], float scale_wih_o,
        // Hidden-hidden weight matrices (4 gates × HIDDEN)
        const int8_t whh_i[HIDDEN][HIDDEN], float scale_whh_i,
        const int8_t whh_f[HIDDEN][HIDDEN], float scale_whh_f,
        const int8_t whh_g[HIDDEN][HIDDEN], float scale_whh_g,
        const int8_t whh_o[HIDDEN][HIDDEN], float scale_whh_o,
        // Biases (FP32, already merged ih+hh)
        const float bias_i[HIDDEN],
        const float bias_f[HIDDEN],
        const float bias_g[HIDDEN],
        const float bias_o[HIDDEN],
        // I/O
        const float x[INPUT_DIM],
        const float h_prev[HIDDEN],
        const float c_prev[HIDDEN],
        float h_next[HIDDEN],
        float c_next[HIDDEN],
        // Scratch buffers — must be HIDDEN floats, provided by caller
        float gate_i[HIDDEN],
        float gate_f[HIDDEN],
        float gate_g[HIDDEN],
        float gate_o[HIDDEN]
    ) {
        // --- Compute gate pre-activations: gate = Wih*x + Whh*h + b ---
        copy_vec<HIDDEN>(bias_i, gate_i);
        copy_vec<HIDDEN>(bias_f, gate_f);
        copy_vec<HIDDEN>(bias_g, gate_g);
        copy_vec<HIDDEN>(bias_o, gate_o);

        // Input→hidden contributions
        int8_sparse_gemm_accum<HIDDEN, INPUT_DIM>(wih_i, scale_wih_i, x, gate_i);
        int8_sparse_gemm_accum<HIDDEN, INPUT_DIM>(wih_f, scale_wih_f, x, gate_f);
        int8_sparse_gemm_accum<HIDDEN, INPUT_DIM>(wih_g, scale_wih_g, x, gate_g);
        int8_sparse_gemm_accum<HIDDEN, INPUT_DIM>(wih_o, scale_wih_o, x, gate_o);

        // Hidden→hidden contributions
        int8_sparse_gemm_accum<HIDDEN, HIDDEN>(whh_i, scale_whh_i, h_prev, gate_i);
        int8_sparse_gemm_accum<HIDDEN, HIDDEN>(whh_f, scale_whh_f, h_prev, gate_f);
        int8_sparse_gemm_accum<HIDDEN, HIDDEN>(whh_g, scale_whh_g, h_prev, gate_g);
        int8_sparse_gemm_accum<HIDDEN, HIDDEN>(whh_o, scale_whh_o, h_prev, gate_o);

        // --- Apply activations + gating ---
        for (int i = 0; i < HIDDEN; ++i) {
            float gi = fast_sigmoid(gate_i[i]);
            float gf = fast_sigmoid(gate_f[i]);
            float gg = fast_tanh(gate_g[i]);
            float go = fast_sigmoid(gate_o[i]);

            c_next[i] = gf * c_prev[i] + gi * gg;
            h_next[i] = go * fast_tanh(c_next[i]);
        }
    }
};

// ---------------------------------------------------------------------------
// Softmax with temperature scaling and numerical stability (max subtraction)
// ---------------------------------------------------------------------------
template <int SIZE>
inline void softmax_temperature(
    const float logits[SIZE],
    float       probs[SIZE],
    float       temperature = 1.0f
) {
    // Find max for numerical stability
    float max_val = logits[0];
    for (int i = 1; i < SIZE; ++i) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    // Compute exp(x - max) / temperature
    float inv_temp = (temperature > 1e-6f) ? (1.0f / temperature) : 100.0f;
    float sum = 0.0f;
    for (int i = 0; i < SIZE; ++i) {
        probs[i] = std::exp((logits[i] - max_val) * inv_temp);
        sum += probs[i];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < SIZE; ++i) {
        probs[i] *= inv_sum;
    }
}

// ---------------------------------------------------------------------------
// Greedy argmax over logits (deterministic, fastest)
// ---------------------------------------------------------------------------
template <int SIZE>
inline int argmax(const float v[SIZE]) {
    int best = 0;
    for (int i = 1; i < SIZE; ++i) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

// ---------------------------------------------------------------------------
// Embedding lookup: copy row embed_idx from the embedding table into out[]
// Dequantizes INT8 embedding on the fly.
// ---------------------------------------------------------------------------
template <int VOCAB_SIZE, int EMBED_DIM>
inline void embedding_lookup(
    const int8_t emb_weight[VOCAB_SIZE][EMBED_DIM],
    float        emb_scale,
    int          embed_idx,
    float        out[EMBED_DIM]
) {
    const int8_t* row = emb_weight[embed_idx];
    for (int i = 0; i < EMBED_DIM; ++i) {
        out[i] = (float)row[i] * emb_scale;
    }
}
