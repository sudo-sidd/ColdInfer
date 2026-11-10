// lstm_infer.cpp — ColdInfer LSTM Chatbot Inference Engine
//
// Bare-metal character-level LSTM autoregressive text generation.
// Compiles for:
//   - Linux desktop (benchmarking + webapp backend)
//   - ESP32 via ESP-IDF (define ESP32 at compile time)
//
// Compile (desktop):
//   g++ -O3 -std=c++17 -DINPUT_PROMPT="Hello" -o build/lstm_infer cpp/lstm_infer.cpp
//
// Build flags:
//   -DMAX_GEN_LEN=200     max characters to generate (default 200)
//   -DTEMPERATURE=0.8f    sampling temperature (default 0.8)
//   -DWINDOW_SIZE=8       rolling context window depth (default 8)
//   -DSPARSE_GEMM=1       use sparse GEMM (default on)

#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <fstream>

#ifdef ESP32
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#else
#include <sys/resource.h>
#endif

// -----------------------------------------------------------------------
// Generated headers — produced by export_chatbot.py
// -----------------------------------------------------------------------
#include "chatbot_weights.h"   // INT8 weights + scales
#include "chatbot_vocab.h"     // idx2char[], char2idx[], VOCAB_SIZE

// -----------------------------------------------------------------------
// Kernels and buffers
// -----------------------------------------------------------------------
#include "lstm_kernels.h"
#include "rolling_kv.h"

// -----------------------------------------------------------------------
// Compile-time configuration
// -----------------------------------------------------------------------
#ifndef MAX_GEN_LEN
#define MAX_GEN_LEN 200
#endif

#ifndef TEMPERATURE_F
#define TEMPERATURE_F 0.01f
#endif

#ifndef WINDOW_SZ
#define WINDOW_SZ 8
#endif

// -----------------------------------------------------------------------
// Static SRAM arena — ALL runtime state lives here, no heap
// -----------------------------------------------------------------------
// Embedding output
static float embed_out[CHATBOT_EMBED_DIM];

// LSTM layer scratch buffers (ping-pong style)
static float h1[CHATBOT_HIDDEN_SIZE];
static float c1[CHATBOT_HIDDEN_SIZE];
static float h2[CHATBOT_HIDDEN_SIZE];
static float c2[CHATBOT_HIDDEN_SIZE];

// Gate computation scratch (4 × hidden per layer)
static float gate_i[CHATBOT_HIDDEN_SIZE];
static float gate_f[CHATBOT_HIDDEN_SIZE];
static float gate_g[CHATBOT_HIDDEN_SIZE];
static float gate_o[CHATBOT_HIDDEN_SIZE];

// LSTM output at each step (= h2 passed to fc_out)
static float lstm_out[CHATBOT_HIDDEN_SIZE];

// Output projection: logits + probabilities over vocab
static float logits[CHATBOT_VOCAB_SIZE];
static float probs[CHATBOT_VOCAB_SIZE];

// Rolling hidden state window (two layers)
static RollingStateBuffer<CHATBOT_HIDDEN_SIZE, WINDOW_SZ> state1;
static RollingStateBuffer<CHATBOT_HIDDEN_SIZE, WINDOW_SZ> state2;

// Generated output buffer
static char output_buf[MAX_GEN_LEN + 1];

// -----------------------------------------------------------------------
// SRAM usage accounting (compile-time)
// -----------------------------------------------------------------------
constexpr int SRAM_BUFFERS_BYTES =
    sizeof(embed_out) +
    sizeof(h1) + sizeof(c1) +
    sizeof(h2) + sizeof(c2) +
    sizeof(gate_i) + sizeof(gate_f) + sizeof(gate_g) + sizeof(gate_o) +
    sizeof(lstm_out) +
    sizeof(logits) + sizeof(probs) +
    sizeof(state1) + sizeof(state2) +
    sizeof(output_buf);

// -----------------------------------------------------------------------
// Memory helpers
// -----------------------------------------------------------------------
long get_rss_kb() {
#ifdef ESP32
    return heap_caps_get_free_size(MALLOC_CAP_8BIT) / 1024;
#else
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            long kb = 0;
            sscanf(line.c_str(), "VmRSS: %ld", &kb);
            return kb;
        }
    }
    struct rusage u;
    getrusage(RUSAGE_SELF, &u);
    return u.ru_maxrss;
#endif
}

// -----------------------------------------------------------------------
// Reset all hidden/cell states to zero (clears conversation context)
// -----------------------------------------------------------------------
void reset_context() {
    state1.reset();
    state2.reset();
    memset(h1, 0, sizeof(h1));
    memset(c1, 0, sizeof(c1));
    memset(h2, 0, sizeof(h2));
    memset(c2, 0, sizeof(c2));
}

// -----------------------------------------------------------------------
// Single-step forward pass through the LSTM + output projection
//
// Given a character index, updates hidden states and fills logits[].
// -----------------------------------------------------------------------
void forward_step(int char_idx) {
    // Step 1: Embedding lookup (INT8 → FP32 dequantize)
    embedding_lookup<CHATBOT_VOCAB_SIZE, CHATBOT_EMBED_DIM>(
        emb_weight, emb_scale, char_idx, embed_out
    );

    // Step 2: LSTM layer 1
    const float* h1_prev = (state1.count > 0) ? state1.last_h() : h1;
    const float* c1_prev = (state1.count > 0) ? state1.last_c() : c1;

    float h1_new[CHATBOT_HIDDEN_SIZE];
    float c1_new[CHATBOT_HIDDEN_SIZE];

    LSTMCell<CHATBOT_EMBED_DIM, CHATBOT_HIDDEN_SIZE>::step(
        lstm1_wih_i, lstm1_wih_i_scale,
        lstm1_wih_f, lstm1_wih_f_scale,
        lstm1_wih_g, lstm1_wih_g_scale,
        lstm1_wih_o, lstm1_wih_o_scale,
        lstm1_whh_i, lstm1_whh_i_scale,
        lstm1_whh_f, lstm1_whh_f_scale,
        lstm1_whh_g, lstm1_whh_g_scale,
        lstm1_whh_o, lstm1_whh_o_scale,
        lstm1_bias_i, lstm1_bias_f, lstm1_bias_g, lstm1_bias_o,
        embed_out, h1_prev, c1_prev,
        h1_new, c1_new,
        gate_i, gate_f, gate_g, gate_o
    );
    state1.push(h1_new, c1_new);

    // Step 3: LSTM layer 2
    const float* h2_prev = (state2.count > 0) ? state2.last_h() : h2;
    const float* c2_prev = (state2.count > 0) ? state2.last_c() : c2;

    float h2_new[CHATBOT_HIDDEN_SIZE];
    float c2_new[CHATBOT_HIDDEN_SIZE];

    LSTMCell<CHATBOT_HIDDEN_SIZE, CHATBOT_HIDDEN_SIZE>::step(
        lstm2_wih_i, lstm2_wih_i_scale,
        lstm2_wih_f, lstm2_wih_f_scale,
        lstm2_wih_g, lstm2_wih_g_scale,
        lstm2_wih_o, lstm2_wih_o_scale,
        lstm2_whh_i, lstm2_whh_i_scale,
        lstm2_whh_f, lstm2_whh_f_scale,
        lstm2_whh_g, lstm2_whh_g_scale,
        lstm2_whh_o, lstm2_whh_o_scale,
        lstm2_bias_i, lstm2_bias_f, lstm2_bias_g, lstm2_bias_o,
        h1_new, h2_prev, c2_prev,
        h2_new, c2_new,
        gate_i, gate_f, gate_g, gate_o
    );
    state2.push(h2_new, c2_new);

    // Step 4: Output projection — logits = fc_out_weight * h2_new + fc_out_bias
    memcpy(logits, fc_out_bias, CHATBOT_VOCAB_SIZE * sizeof(float));
    for (int i = 0; i < CHATBOT_VOCAB_SIZE; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < CHATBOT_HIDDEN_SIZE; ++j) {
            if (fc_out_weight[i][j] != 0) {
                acc += (float)fc_out_weight[i][j] * h2_new[j];
            }
        }
        logits[i] += acc * fc_out_scale;
    }
}

// -----------------------------------------------------------------------
// Autoregressive generation
//
// Warm up with the prompt, then generate up to max_len characters.
// Stops early if EOS_IDX is sampled.
// -----------------------------------------------------------------------
int generate(const char* prompt, int max_len, float temperature) {
    reset_context();

    // Warm up: run each prompt character through the forward pass
    int prompt_len = (int)strlen(prompt);
    for (int i = 0; i < prompt_len; ++i) {
        unsigned char c = (unsigned char)prompt[i];
        int idx = (c < 128) ? char2idx[(int)c] : UNK_IDX;
        forward_step(idx);
    }

    // Autoregressive generation loop
    int out_len = 0;
    static uint32_t rng = 42;

    for (int step = 0; step < max_len; ++step) {
        // Sample next token directly from the current logits
        int next_idx;
        if (temperature <= 0.01f) {
            next_idx = argmax<CHATBOT_VOCAB_SIZE>(logits);
        } else {
            softmax_temperature<CHATBOT_VOCAB_SIZE>(logits, probs, temperature);
            rng = rng * 1664525u + 1013904223u;
            float r = (float)(rng >> 8) / (float)(1 << 24);
            float cumulative = 0.0f;
            next_idx = CHATBOT_VOCAB_SIZE - 1;
            for (int i = 0; i < CHATBOT_VOCAB_SIZE; ++i) {
                cumulative += probs[i];
                if (r < cumulative) { next_idx = i; break; }
            }
        }

        if (next_idx == EOS_IDX || next_idx == PAD_IDX) break;

        const char* ch_str = idx2char[next_idx];
        if (ch_str[0] == '\n' || (ch_str[0] == '<' && strstr(ch_str, "EOS"))) {
            break;
        }

        if (ch_str[0] != '<' && out_len < max_len - 1) {
            output_buf[out_len++] = ch_str[0];
            if (out_len >= 2 && output_buf[out_len - 2] == 'Q' && output_buf[out_len - 1] == ':') {
                out_len -= 2;
                break;
            }
        }

        // Feed sampled token to update hidden states for the next token
        forward_step(next_idx);
    }

    // Trim trailing whitespace
    while (out_len > 0 && (output_buf[out_len - 1] == ' ' || output_buf[out_len - 1] == '\r')) {
        out_len--;
    }
    output_buf[out_len] = '\0';
    return out_len;
}

// -----------------------------------------------------------------------
#ifndef ESP32
int main(int argc, char** argv) {
    float temp = TEMPERATURE_F;
    if (argc > 1) {
        try { temp = std::stof(argv[1]); } catch (...) {}
    }
#else
extern "C" void app_main() {
    float temp = TEMPERATURE_F;
#endif

#ifndef ESP32
    // Read prompt from stdin for subprocess-mode (called by the FastAPI backend)
    std::string prompt_str;
    if (!std::getline(std::cin, prompt_str)) {
        prompt_str = "hello";
    }
    while (!prompt_str.empty() && (prompt_str.back() == '\r' || prompt_str.back() == ' ')) {
        prompt_str.pop_back();
    }
    if (prompt_str.rfind("Q:", 0) != 0) {
        prompt_str = "Q: " + prompt_str + "\nA: ";
    }
    const char* prompt = prompt_str.c_str();
#else
    const char* prompt = "Q: hello\nA: ";
#endif

#ifdef ESP32
    int64_t t_start = esp_timer_get_time();
#else
    auto t_start = std::chrono::high_resolution_clock::now();
#ifndef NDEBUG
    struct rusage ru_before, ru_after;
    getrusage(RUSAGE_SELF, &ru_before);
#endif
#endif

    int n_tokens = generate(prompt, MAX_GEN_LEN, temp);

#ifdef ESP32
    int64_t t_end = esp_timer_get_time();
    long wall_us = (long)(t_end - t_start);
#else
    auto t_end = std::chrono::high_resolution_clock::now();
    long wall_us = (long)std::chrono::duration_cast<std::chrono::microseconds>(
        t_end - t_start
    ).count();
#ifndef NDEBUG
    getrusage(RUSAGE_SELF, &ru_after);
#endif
#endif

    long mem_kb = get_rss_kb();

    // Structured output — parsed by the FastAPI backend and simulate_constraints.py
    std::cout << "GENERATED:" << output_buf      << std::endl;
    std::cout << "INFERENCE_TIME_US:" << wall_us  << std::endl;
    std::cout << "TOKENS_GENERATED:"  << n_tokens << std::endl;
#ifdef ESP32
    std::cout << "FREE_HEAP_KB:"      << mem_kb   << std::endl;
#else
    std::cout << "PEAK_RSS_KB:"       << mem_kb   << std::endl;
    std::cout << "SRAM_BUFFERS_BYTES:" << SRAM_BUFFERS_BYTES << std::endl;
#endif

#ifndef ESP32
    return 0;
#endif
}
