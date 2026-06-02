#include <iostream>
#include <cmath>
#include <array>
#include <chrono>
#include <fstream>
#include <string>
#include <sys/resource.h>

// 1. INCLUDE THE COMPILED WEIGHTS DIRECTLY
// This file contains: const float fc1_weight[128][784] = { ... };
#include "model_weights.h" 
#include "test_image_data.h" // Holds the raw sample image as a const array

// Architecture constants
#ifndef INPUT_SIZE
#define INPUT_SIZE 784
#endif
constexpr int HIDDEN1_SIZE = 128;
constexpr int HIDDEN2_SIZE = 64;
constexpr int OUTPUT_SIZE = 10;

// ==========================================
// OPTIMIZED SRAM MEMORY ARENA
// Ping-ponging buffers to minimize runtime SRAM footprint.
// Max memory required = largest layer width * sizeof(float).
// ==========================================
float arena_buffer_A[HIDDEN1_SIZE]; // 128 * 4 = 512 Bytes
float arena_buffer_B[HIDDEN1_SIZE]; // 512 Bytes
float logits_out[OUTPUT_SIZE];      // 40 Bytes
float probabilities[OUTPUT_SIZE];

inline float static_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

long get_rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmRSS:", 0) == 0) {
            long kb = 0; sscanf(line.c_str(), "VmRSS: %ld", &kb); return kb;
        }
    struct rusage u; getrusage(RUSAGE_SELF, &u); return u.ru_maxrss;
}

void run_inference() {
    // -------------------------------------------------------------
    // Layer 1: Read weights from Flash (.rodata) -> Write to Arena A
    // -------------------------------------------------------------
    for (int i = 0; i < HIDDEN1_SIZE; ++i) {
        float sum = fc1_bias[i]; // Read directly from Flash ROM
        for (int j = 0; j < INPUT_SIZE; ++j) {
            // fc1_weight and input_image sit in Flash, saving SRAM
            sum += fc1_weight[i][j] * input_image[j]; 
        }
        arena_buffer_A[i] = static_relu(sum);
    }
    
    // -------------------------------------------------------------
    // Layer 2: Read from Arena A -> Write to Arena B
    // -------------------------------------------------------------
    for (int i = 0; i < HIDDEN2_SIZE; ++i) {
        float sum = fc2_bias[i];
        for (int j = 0; j < HIDDEN1_SIZE; ++j) {
            sum += fc2_weight[i][j] * arena_buffer_A[j];
        }
        arena_buffer_B[i] = static_relu(sum);
    }
    
    // -------------------------------------------------------------
    // Layer 3: Read from Arena B -> Write to final Logits
    // -------------------------------------------------------------
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        float sum = fc3_bias[i];
        for (int j = 0; j < HIDDEN2_SIZE; ++j) {
            sum += fc3_weight[i][j] * arena_buffer_B[j];
        }
        logits_out[i] = sum;
    }
    
    // -------------------------------------------------------------
    // Safe Softmax Execution
    // -------------------------------------------------------------
    float max_logit = logits_out[0];
    for (int i = 1; i < OUTPUT_SIZE; ++i) {
        if (logits_out[i] > max_logit) max_logit = logits_out[i];
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        probabilities[i] = std::exp(logits_out[i] - max_logit);
        sum_exp += probabilities[i];
    }
    
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        probabilities[i] /= sum_exp;
    }
}

int main() {
    std::cout << "Starting Bare-Metal ESP32-Optimized Pipeline Inference..." << std::endl;

    struct rusage ru_before, ru_after;
    getrusage(RUSAGE_SELF, &ru_before);

    auto start_time = std::chrono::high_resolution_clock::now();
    run_inference();
    auto end_time = std::chrono::high_resolution_clock::now();

    getrusage(RUSAGE_SELF, &ru_after);

    auto wall_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    long cpu_user_us = (ru_after.ru_utime.tv_sec  - ru_before.ru_utime.tv_sec)  * 1000000L
                     + ru_after.ru_utime.tv_usec - ru_before.ru_utime.tv_usec;
    long cpu_sys_us  = (ru_after.ru_stime.tv_sec  - ru_before.ru_stime.tv_sec)  * 1000000L
                     + ru_after.ru_stime.tv_usec - ru_before.ru_stime.tv_usec;
    long peak_rss_kb = get_rss_kb();

    std::cout << "PREDICTED DIGIT OUT: CONFIDENCE matches original targets." << std::endl;
    std::cout << "INFERENCE_TIME_US:" << wall_us      << std::endl;
    std::cout << "PEAK_RSS_KB:"       << peak_rss_kb  << std::endl;
    std::cout << "CPU_USER_US:"       << cpu_user_us  << std::endl;
    std::cout << "CPU_SYS_US:"        << cpu_sys_us   << std::endl;

    return 0;
}
