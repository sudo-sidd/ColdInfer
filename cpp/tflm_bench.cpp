/**
 * tflm_bench.cpp — TFLite Micro benchmark.
 * Compiled automatically by run_tests.sh after setup_tflm.sh.
 *
 * Key TFLM note: input/output data is accessed via GetTensor() + GetAllocations(),
 * NOT via interpreter.input(0)->data.f which is a persistent metadata-only tensor.
 *
 * Outputs (key:value per line, grep-friendly):
 *   TFLM_TIME_US      avg inference over 1000 runs (µs)
 *   TFLM_PEAK_RSS_KB  resident RAM after inference
 *   TFLM_CPU_USER_US  CPU userspace time per run
 *   TFLM_CPU_SYS_US   CPU kernel time per run
 *   TFLM_ARENA_KB     minimum viable static arena (binary searched)
 */

#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <cstring>
#include <sys/resource.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#ifndef MODEL_HEADER
  #define MODEL_HEADER "MNIST_MLP_FP32_tflm_model.h"
#endif
#include MODEL_HEADER

#ifndef MODEL_VAR
  #define MODEL_VAR MNIST_MLP_FP32_tflm_model
#endif
#ifndef INPUT_SIZE
  #define INPUT_SIZE 784
#endif

// ── Arenas ───────────────────────────────────────────────────────────────────
// Separate arena for the binary-search pass so it never touches infer_arena.
constexpr int kSearchArenaKB = 256;
constexpr int kInferArenaKB  = 256;
static uint8_t search_arena[kSearchArenaKB * 1024];
static uint8_t infer_arena [kInferArenaKB  * 1024];

// ── Helpers ───────────────────────────────────────────────────────────────────
long get_rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmRSS:", 0) == 0) {
            long kb = 0; sscanf(line.c_str(), "VmRSS: %ld", &kb); return kb;
        }
    struct rusage u; getrusage(RUSAGE_SELF, &u); return u.ru_maxrss;
}

// ── Build a fresh resolver (ops used by our MLP) ──────────────────────────────
static void fill_resolver(tflite::MicroMutableOpResolver<3>& r) {
    r.AddFullyConnected();
    r.AddRelu();
    r.AddSoftmax();
}

// ── Arena size binary search (uses search_arena, not infer_arena) ─────────────
static bool try_arena_kb(int kb) {
    if (kb > kSearchArenaKB) return false;
    const tflite::Model* m = tflite::GetModel(MODEL_VAR);
    if (m->version() != TFLITE_SCHEMA_VERSION) return false;
    tflite::MicroMutableOpResolver<3> res; fill_resolver(res);
    memset(search_arena, 0, kb * 1024);
    tflite::MicroInterpreter interp(m, res, search_arena, kb * 1024);
    return interp.AllocateTensors() == kTfLiteOk;
}

static int find_min_arena_kb() {
    int lo = 1, hi = kSearchArenaKB, best = kSearchArenaKB;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (try_arena_kb(mid)) { best = mid; hi = mid - 1; }
        else                    { lo  = mid + 1; }
    }
    return best;
}

int main() {
    tflite::InitializeTarget();

    const tflite::Model* model = tflite::GetModel(MODEL_VAR);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        std::cerr << "TFLM schema mismatch\n"; return 1;
    }

    // preserve_all_tensors=true lets us use GetTensor() to access live arena
    // data pointers. The persistent TfLiteTensor from input() has data.f=null
    // because TFLM separates metadata tensors from eval-time allocations.
    tflite::MicroMutableOpResolver<3> resolver; fill_resolver(resolver);
    tflite::MicroInterpreter interpreter(
        model, resolver, infer_arena, kInferArenaKB * 1024,
        /*resource_variables=*/nullptr, /*profiler=*/nullptr,
        /*preserve_all_tensors=*/true);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        std::cerr << "AllocateTensors failed\n"; return 1;
    }

    // Get the input tensor index from the flatbuffer subgraph
    int input_idx = model->subgraphs()->Get(0)->inputs()->Get(0);

    // GetTensor() returns the live TfLiteEvalTensor with a valid arena pointer
    TfLiteEvalTensor* eval_input = interpreter.GetTensor(input_idx, 0);
    if (!eval_input || !eval_input->data.f) {
        std::cerr << "ERROR: GetTensor(" << input_idx << ") returned null data\n";
        return 1;
    }
    memset(eval_input->data.f, 0, INPUT_SIZE * sizeof(float));

    // Warmup
    for (int i = 0; i < 10; ++i) interpreter.Invoke();

    // ── Timed hot loop ───────────────────────────────────────────────────────
    constexpr int kRuns = 1000;
    struct rusage ru_before, ru_after;
    getrusage(RUSAGE_SELF, &ru_before);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kRuns; ++i) interpreter.Invoke();
    auto t1 = std::chrono::high_resolution_clock::now();

    getrusage(RUSAGE_SELF, &ru_after);
    long rss = get_rss_kb();

    long avg_us      = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / kRuns;
    long cpu_user_us = ((ru_after.ru_utime.tv_sec  - ru_before.ru_utime.tv_sec)  * 1000000L
                        + ru_after.ru_utime.tv_usec - ru_before.ru_utime.tv_usec) / kRuns;
    long cpu_sys_us  = ((ru_after.ru_stime.tv_sec  - ru_before.ru_stime.tv_sec)  * 1000000L
                        + ru_after.ru_stime.tv_usec - ru_before.ru_stime.tv_usec) / kRuns;
    int min_arena_kb = find_min_arena_kb();

    std::cout << "TFLM_TIME_US:"     << avg_us      << "\n"
              << "TFLM_PEAK_RSS_KB:" << rss          << "\n"
              << "TFLM_CPU_USER_US:" << cpu_user_us  << "\n"
              << "TFLM_CPU_SYS_US:"  << cpu_sys_us   << "\n"
              << "TFLM_ARENA_KB:"    << min_arena_kb << "\n";
    return 0;
}
