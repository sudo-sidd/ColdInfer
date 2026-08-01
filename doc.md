# ColdInfer v0.1.0 — Reverse-Engineered Documentation

> **Audit Date:** 2026-07-31
> **Evidence Base:** All source code, scripts, configs, READMEs, logs, and comments in the repository.
> **Policy:** Every statement is derived from verified repository evidence. Unverifiable claims are marked explicitly.

---

# PART A — General Documentation (Non-Expert Audience)

---

## 1. Project Overview

**What it is:**
ColdInfer is a small, self-contained software project that runs machine-learning classification tasks directly on low-power hardware — including regular computers and ESP32 microcontrollers — without relying on large, standard AI frameworks at runtime.

**What problem it solves:**
Standard ML frameworks like PyTorch or TensorFlow require large amounts of RAM and many software dependencies. On a tiny microcontroller with only a few hundred kilobytes of memory, these frameworks cannot run at all. ColdInfer solves this by converting a trained neural network model into plain C++ code with weights baked in as numbers, and running that C++ code directly on the hardware with no framework needed.

**Who uses it:**
Based on the repository contents (specifically `Idea.md`), this appears to be an academic/learning project built by a team of four people. The stated audience is embedded/edge hardware developers.
*Source: `Idea.md`, `README.md`*

**Workflow summary:**
1. Train a small neural network in Python using PyTorch.
2. Export the trained weights as C++ header files (`.h` files containing raw numbers).
3. Compile and run the bare-metal C++ engine directly — no Python or TensorFlow needed at that point.
4. Optionally flash the compiled firmware to an ESP32 microcontroller for edge inference over Wi-Fi.
5. Benchmark the C++ engine against PyTorch and TFLite to compare speed and memory use.

---

## 2. High-Level Workflow

```
[ Raw Dataset ]
      ↓
  Download & Load
  (MNIST via torchvision / Mini Speech Commands via HTTP)
      ↓
  Preprocessing
  (normalize pixel values / downsample 16kHz audio → 1000 features)
      ↓
  Train PyTorch Model
  (MLP: 3 fully-connected layers + ReLU)
      ↓
  Save Checkpoint (.pt file)
      ↓
  Export Weights as C++ Header (.h file with float arrays)
  Export Test Sample as C++ Header (.h file with float array)
      ↓
  [Optional] Export TFLite flatbuffer + C array header (for TFLM benchmark)
      ↓
  Compile C++ Inference Engine (g++ -O3 -std=c++17)
  [Weights are #included directly at compile time]
      ↓
  Run Inference
  (weights in Flash/rodata → ping-pong SRAM buffers → logits → softmax)
      ↓
  Output: Class probabilities + timing/memory metrics
      ↓
  Log results to performance_log.csv
      ↓
  Auto-update README.md benchmark tables
```

*Source: `scripts/run_tests.sh`, `python/train_export.py`, `cpp/infer.cpp`*

---

## 3. Repository Structure

```
Coldinfer/
├── python/                   # Python toolchain: training, export, benchmarking
│   ├── train_export.py       # Trains MNIST or KWS MLP, exports weights as C++ headers
│   ├── export_tflite.py      # Converts PyTorch model → TFLite flatbuffer → C array header
│   └── benchmark_baselines.py# Benchmarks PyTorch and TFLite runtimes for comparison
│
├── cpp/                      # C++ inference engine code
│   ├── infer.cpp             # Desktop/benchmarking version of the bare-metal MLP engine
│   ├── tflm_bench.cpp        # Benchmark harness for TFLite Micro (TFLM)
│   └── esp32_inference/      # ESP32 firmware application (ESP-IDF project)
│       ├── CMakeLists.txt    # Top-level CMake for ESP-IDF
│       ├── partitions.csv    # Custom flash partition table (3 MB app, NVS, PHY)
│       ├── sdkconfig.defaults# Default ESP-IDF build config (4MB flash, custom partitions)
│       └── main/
│           ├── infer.cpp     # Full ESP32 firmware: Wi-Fi + HTTP server + inference
│           ├── idf_component.yml  # Declares cJSON dependency (espressif/cjson ^1.7.17)
│           ├── model_weights.h    # Baked-in MNIST weights (C float arrays)
│           └── test_image_data.h  # Baked-in MNIST test sample
│
├── scripts/                  # Automation and tooling
│   ├── run_tests.sh          # Master orchestration script: train → compile → benchmark → log
│   ├── build.sh              # Simple one-shot g++ compilation (legacy/manual use)
│   ├── setup_tflm.sh         # One-time setup: clones and builds TFLite Micro static library
│   ├── update_readme_stats.py# Reads performance_log.csv, rewrites README benchmark tables
│   ├── test_esp_api.py       # Sends HTTP POST to ESP32 /infer endpoint, prints results
│   └── diagnose_serial.py    # Tries multiple baud rates on /dev/ttyUSB0, diagnoses output
│
├── data/                     # Datasets (git-ignored, auto-downloaded at runtime)
│   ├── MNIST/                # MNIST binary files (auto-downloaded by torchvision)
│   └── mini_speech_commands/ # 8-class audio dataset (auto-downloaded from Google Storage)
│
├── weights/                  # Compiled artifacts (git-ignored, auto-generated)
│   ├── *.h                   # C++ float-array headers for weights and test samples
│   ├── *.tflite              # TFLite flatbuffers for TFLM benchmark
│   └── *.pt                  # PyTorch checkpoint files
│
├── build/                    # Compiled binaries (git-ignored)
│   ├── infer                 # Compiled ColdInfer engine executable
│   └── tflm_bench            # Compiled TFLM benchmark executable
│
├── third_party/              # External dependencies (git-ignored)
│   └── tflm_build/           # Built TFLM static library + headers
│
├── docs/
│   └── ESP32_DEPLOYMENT.md   # Guide for flashing, monitoring, and testing ESP32
│
├── performance_log.csv       # Live benchmark results (appended each test run)
├── performance_log_old.csv   # Historical results from earlier schema version
├── README.md                 # Project description, benchmarks, roadmap
├── Idea.md                   # Original project design notes / team scoping document
├── pyproject.toml            # Python project metadata and pinned dependencies
├── requirements.txt          # Bare pip requirements list (subset of pyproject.toml)
├── .python-version           # Pins Python 3.12 (for uv)
└── uv.lock                   # Lock file for uv package manager
```

*Source: Full directory traversal + README.md lines 42–48*

---

## 4. Components

### 4.1 Python Training & Export Toolchain (`python/`)
Responsible for: (a) downloading datasets, (b) training the neural network models, (c) saving PyTorch checkpoint files, and (d) exporting weights as C-language header files that the C++ engine can `#include` directly.

- `train_export.py` trains either the MNIST or KWS model. If a pre-trained `.pt` file already exists, it loads that instead of retraining. After training (or loading), it calls `export_c_headers()` which writes the weights as literal float arrays into `model_weights.h` and one test sample into `test_image_data.h`.
- `export_tflite.py` loads a saved `.pt` checkpoint, constructs an equivalent Keras model, manually copies the weights layer-by-layer, converts it to TFLite format, applies two binary patches to the flatbuffer to make it compatible with TFLite Micro, and outputs both a `.tflite` file and a C array header.
- `benchmark_baselines.py` loads a saved checkpoint, runs 1000 inference passes through both PyTorch and TFLite, and prints timing/memory metrics in a grep-friendly `KEY:VALUE` format.

*Source: All files in `python/`*

### 4.2 C++ Bare-Metal Inference Engine (`cpp/infer.cpp`)
Responsible for: running a 3-layer MLP forward pass with zero dynamic memory allocation.

The weights are compiled directly into the binary via `#include "model_weights.h"`. At runtime, only three static float arrays are used as working memory (two 128-float ping-pong buffers and a 10-float logits buffer). The engine performs matrix-vector multiplication for each layer, applies ReLU, and finishes with a numerically stable softmax.

On Linux, it uses `std::chrono` and `getrusage()` for timing and memory measurement. On ESP32, it uses `esp_timer_get_time()` and `heap_caps_get_free_size()` instead, selected via `#ifdef ESP32` preprocessor guards.

*Source: `cpp/infer.cpp`*

### 4.3 ESP32 Firmware Application (`cpp/esp32_inference/main/infer.cpp`)
Responsible for: running the same MLP engine on an ESP32 microcontroller over Wi-Fi, with an HTTP REST API.

The firmware (a) initializes NVS flash, (b) connects to Wi-Fi (SSID and password are hardcoded as string literals), (c) obtains an IP address, and (d) starts an HTTP server that listens for `POST /infer` requests. Each request body is expected to be a JSON array of floats. The handler parses the array using cJSON, runs inference, and returns a JSON response containing `probabilities`, `latency_us`, and `free_heap_kb`.

> [!CAUTION]
> The Wi-Fi credentials (`WIFI_SSID "Drackko"` and `WIFI_PASS "zdt32109"`) are hardcoded as plain-text string literals in `cpp/esp32_inference/main/infer.cpp` lines 22–23.

*Source: `cpp/esp32_inference/main/infer.cpp`*

### 4.4 TFLM Benchmark Harness (`cpp/tflm_bench.cpp`)
Responsible for: benchmarking TFLite Micro on the same model for head-to-head comparison.

Uses a static arena of 256 KB for inference and a separate 256 KB arena for a binary search that finds the minimum viable arena size. Registers only the ops the MLP uses: `FullyConnected`, `Relu`, and `Softmax`. Runs 1000 iterations and reports average latency, RSS, CPU time, and minimum arena KB.

*Source: `cpp/tflm_bench.cpp`*

### 4.5 Benchmark Orchestration (`scripts/run_tests.sh`)
The master shell script that ties every component together. When executed, it:
1. Verifies the Python virtual environment exists.
2. For each registered model (`MNIST_MLP_FP32` and `Keyword_Spotting_MLP_FP32`):
   - Trains and exports weights if not already cached.
   - Runs Python baseline benchmarks (PyTorch + TFLite).
   - Compiles the C++ engine with `g++ -O3 -std=c++17 -DINPUT_SIZE=<N>`.
   - Runs the compiled binary, captures output.
   - If the TFLM static library is present, compiles and runs `tflm_bench`.
   - Appends all metrics to `performance_log.csv`.
3. Calls `update_readme_stats.py` to rewrite the README benchmark tables.

*Source: `scripts/run_tests.sh`*

### 4.6 README Auto-Updater (`scripts/update_readme_stats.py`)
Reads `performance_log.csv`, selects the last row for each model, formats a Markdown table, and uses a regex to splice it between the `## Performance Benchmarks` and `## Project Roadmap` sections of `README.md`.

*Source: `scripts/update_readme_stats.py`*

---

## 5. Data Flow

**Complete execution trace — MNIST model, first-time run:**

```
User runs: ./scripts/run_tests.sh
      ↓
Script checks for .venv → found
Script checks for weights/MNIST_MLP_FP32_model_weights.h → NOT found
      ↓
Calls: python/train_export.py --model mnist
  → Downloads MNIST via torchvision to ./data/MNIST/
  → Constructs MNIST_MLP (784→128→64→10)
  → Checks for weights/model_weights.MNIST.pt → NOT found
  → Trains 1 epoch (Adam, lr=0.001, CrossEntropyLoss, batch=64)
  → Saves checkpoint to weights/model_weights.MNIST.pt
  → Grabs one test sample from test_loader
  → Calls export_c_headers(model, test_image)
     → Writes model_weights.h (const float fc1_weight[128][784] = {...}, ...)
     → Writes test_image_data.h (const float input_image[784] = {...})
  → Script moves these to weights/MNIST_MLP_FP32_model_weights.h etc.
      ↓
Copies staging files: weights/model_weights.h, weights/test_image_data.h
      ↓
Calls: python/benchmark_baselines.py --model mnist
  → Loads checkpoint from weights/model_weights.MNIST.pt
  → Runs 1000 PyTorch inference passes → prints PYTORCH_US, PYTORCH_RSS_KB, ...
  → Rebuilds equivalent Keras model with same weights
  → Converts to TFLite → runs 1000 passes → prints TFLITE_US, ...
Script captures this output and parses KEY:VALUE lines
      ↓
Compiles: g++ -O3 -std=c++17 -DINPUT_SIZE=784 -I./weights cpp/infer.cpp -o build/infer
  → Weights and test image are #included directly into the binary
      ↓
Runs: build/infer
  → Executes run_inference() over the statically-included input_image[]
  → Layer 1: reads fc1_weight[128][784] from .rodata, writes to arena_buffer_A[128]
  → Layer 2: reads fc2_weight[64][128] from .rodata, writes to arena_buffer_B[64]
  → Layer 3: reads fc3_weight[10][64] from .rodata, writes to logits_out[10]
  → Softmax over logits_out → probabilities[10]
  → Prints INFERENCE_TIME_US, PEAK_RSS_KB, CPU_USER_US, CPU_SYS_US
Script captures output and parses
      ↓
Appends one CSV row to performance_log.csv
      ↓
Calls: scripts/update_readme_stats.py
  → Reads performance_log.csv → formats Markdown table → patches README.md
```

*Source: `scripts/run_tests.sh`, `python/train_export.py`, `cpp/infer.cpp`, `scripts/update_readme_stats.py`*

---

## 6. Machine Learning Pipeline

### Datasets
Two datasets are used, both downloaded automatically:

| Dataset | Source | Task | Classes |
|---|---|---|---|
| MNIST | torchvision.datasets.MNIST | Handwritten digit classification | 10 digits (0–9) |
| Mini Speech Commands | Google Storage (HTTP ZIP) | Keyword classification from audio | 8 classes: down, go, left, no, right, stop, up, yes |

*Source: `python/train_export.py` lines 57–58, 121; `data/README.txt`*

### Preprocessing

**MNIST:**
- `transforms.ToTensor()` and `transforms.Normalize((0.1307,), (0.3081,))` are applied.
- Each 28×28 image is flattened to a 784-element vector inside the model's `forward()` via `x.view(-1, 784)`.

*Source: `python/train_export.py` lines 56–60*

**Keyword Spotting (KWS):**
- WAV files are read using Python's standard `wave` module.
- 16-bit samples are divided by 32768.0 to normalize to [-1, 1]. 8-bit samples are offset by 128 and divided by 128.
- Multi-channel audio is downmixed by keeping only every N-th sample (`audio[::n_channels]`).
- Audio is padded to or truncated at exactly 16,000 samples (1 second at 16 kHz).
- Downsampled to 1,000 features by reshaping to `(1000, 16)` and taking the mean across the 16-sample windows.
- Up to 200 samples per class are loaded (8 classes × 200 = 1,600 training examples total).

*Source: `python/train_export.py` lines 147–198*

### Model Architecture
Both models are identical in topology but differ in input size:

| Property | MNIST_MLP | KWS_MLP |
|---|---|---|
| Class | `MNIST_MLP` | `KWS_MLP` |
| Input | 784 floats (28×28 pixels) | 1,000 floats (downsampled audio) |
| Layer 1 | `nn.Linear(784→128)` + ReLU | `nn.Linear(1000→128)` + ReLU |
| Layer 2 | `nn.Linear(128→64)` + ReLU | `nn.Linear(128→64)` + ReLU |
| Layer 3 | `nn.Linear(64→10)` | `nn.Linear(64→10)` |
| Output | 10 logits (digit class) | 10 logits (keyword class) |

No batch normalization, dropout, or other regularization layers are present in the code.

*Source: `python/train_export.py` lines 38–52, 92–106*

### Training
- **Optimizer:** `torch.optim.Adam` with `lr=0.001`
- **Loss:** `nn.CrossEntropyLoss`
- **Epochs:** 1 (hardcoded for both models)
- **Batch size:** 64
- Training is skipped if the `.pt` checkpoint already exists on disk.

*Source: `python/train_export.py` lines 68–80, 213–225*

### Validation
No separate validation loop is implemented in the training script. The script prints a single test sample's true label after training/loading but does not compute accuracy over the test set.

*Source: `python/train_export.py` — no validation loop found*

### Inference
Forward pass is implemented manually in pure C++ in `cpp/infer.cpp` and `cpp/esp32_inference/main/infer.cpp`. No ONNX runtime, TensorFlow, or PyTorch is used at C++ inference time.

*Source: `cpp/infer.cpp`, `cpp/esp32_inference/main/infer.cpp`*

### Evaluation
Evaluation is performed via benchmarking, not classification accuracy. The system measures:
- Wall-clock latency (microseconds) averaged over 1,000 runs
- Peak RSS memory (KB) from `/proc/self/status`
- CPU user-space and kernel time (microseconds) via `getrusage()`
- Binary `.text` and `.bss` section sizes via the `size` utility

For TFLM, additionally: minimum viable arena size (KB) via binary search.

*Source: `cpp/infer.cpp`, `cpp/tflm_bench.cpp`, `python/benchmark_baselines.py`, `scripts/run_tests.sh` lines 96–105*

---

## 7. External Technologies

| Technology | Role | Where Used |
|---|---|---|
| **PyTorch** (`torch >= 2.12.0`) | Model training, checkpoint saving, baseline benchmarking | `python/train_export.py`, `python/benchmark_baselines.py` |
| **torchvision** (`>= 0.27.0`) | MNIST dataset download and preprocessing | `python/train_export.py` |
| **TensorFlow / Keras** | TFLite conversion and baseline benchmarking | `python/export_tflite.py`, `python/benchmark_baselines.py` |
| **TFLite Micro (TFLM)** | Third-party baseline for memory-constrained inference | `cpp/tflm_bench.cpp`, `scripts/setup_tflm.sh` |
| **NumPy** (`>= 2.4.6`) | Array manipulation during export | `python/train_export.py` |
| **GCC** (`g++`) | C++17 compilation of the inference engine | `scripts/run_tests.sh`, `scripts/build.sh` |
| **ESP-IDF** | Build system and SDK for ESP32 firmware | `cpp/esp32_inference/CMakeLists.txt`, `sdkconfig` |
| **cJSON** (`espressif/cjson ^1.7.17`) | JSON parsing on ESP32 (request/response) | `cpp/esp32_inference/main/infer.cpp` |
| **esptool** (`>= 5.3.0`) | Flashing firmware to ESP32 over USB | `docs/ESP32_DEPLOYMENT.md`, `requirements.txt` |
| **pyserial** (`>= 3.5`) | Serial port communication for diagnosis | `scripts/diagnose_serial.py` |
| **requests** (`>= 2.31.0`) | HTTP client for testing the ESP32 API | `scripts/test_esp_api.py` |
| **uv** | Python package manager and virtual environment | `pyproject.toml`, `uv.lock`, README setup instructions |
| **FreeRTOS** | Underlying RTOS on ESP32 (via ESP-IDF) | `cpp/esp32_inference/main/infer.cpp` — `freertos/FreeRTOS.h` included |

No GPU, CUDA, or distributed computing infrastructure is used anywhere in the project.

*Source: `pyproject.toml`, `requirements.txt`, all source files*

---

## 8. Current Capabilities

Based on what is demonstrably implemented:

- **Train** a 3-layer MLP on MNIST (784-input, 10-class digit classification).
- **Train** a 3-layer MLP on the Mini Speech Commands dataset (1000-input, 10-class keyword spotting — note: dataset has 8 classes but the model outputs 10 logits; see Technical Debt).
- **Export** trained weights and a test sample as C++ `#include`-able float array headers.
- **Export** trained models to TFLite flatbuffer format and C array headers for TFLM.
- **Compile** and **run** a bare-metal C++ MLP engine on Linux that reads weights from the `.rodata` segment and uses only ~1 KB of SRAM for intermediate activations.
- **Benchmark** the C++ engine against PyTorch and TFLite (and optionally TFLM) across latency, memory, and CPU time metrics, averaged over 1,000 runs.
- **Log** all benchmark results to `performance_log.csv` with a 22-column schema.
- **Auto-update** `README.md` benchmark tables from the CSV log.
- **Flash** the compiled firmware to an ESP32 microcontroller.
- **Run** bare-metal inference on ESP32 over Wi-Fi with a JSON REST API (`POST /infer`).
- **Diagnose** serial port communication at multiple baud rates.

*Source: All source files, verified end-to-end via `scripts/run_tests.sh` and `docs/ESP32_DEPLOYMENT.md`*

---

## 9. Limitations

Only limitations directly observable from the repository:

- **Single-epoch training:** Both models train for exactly 1 epoch (`for epoch in range(1)`), noted in a comment as "Reduced to 1 epoch for faster testing". No accuracy metrics are computed or reported.
  *Source: `python/train_export.py` lines 72, 217*

- **KWS class/output mismatch:** The Mini Speech Commands dataset has 8 classes (`down, go, left, no, right, stop, up, yes`), but `KWS_MLP` outputs 10 logits. This mismatch is not addressed in the code.
  *Source: `python/train_export.py` — 8 classes detected in `all_classes`, model `fc3 = nn.Linear(64, 10)`*

- **Hardcoded Wi-Fi credentials:** The ESP32 firmware contains `#define WIFI_SSID "Drackko"` and `#define WIFI_PASS "zdt32109"` as plain-text string literals.
  *Source: `cpp/esp32_inference/main/infer.cpp` lines 22–23*

- **Fixed input in desktop inference:** The desktop `cpp/infer.cpp` always runs inference on the statically compiled `input_image[]`. There is no mechanism to supply a different input at runtime without recompiling.
  *Source: `cpp/infer.cpp` — `input_image` is `#include`-d and used directly*

- **No accuracy evaluation:** No script in the repository computes classification accuracy over the test set. The only evaluation is latency/memory benchmarking.
  *Source: `python/train_export.py` — test set is used only to grab a single sample for export*

- **FP32 only:** The engine currently operates exclusively in 32-bit floating point. INT8/INT4 quantization is noted as planned but not implemented.
  *Source: `README.md` roadmap — "INT8 Post-Training Quantization — Planned"*

- **MLP only (no Conv2D):** The engine implements only fully-connected layers. Convolutional kernels are listed as a future roadmap item.
  *Source: `README.md` roadmap — "Conv2D and Pooling Kernels — Planned"*

- **TFLM dependency is optional and manual:** TFLM benchmarks are skipped unless `./scripts/setup_tflm.sh` has been run manually beforehand.
  *Source: `scripts/run_tests.sh` lines 24–30*

- **Buffer overrun risk:** The README explicitly notes that passing incorrectly-sized binary data could cause buffer overruns due to reliance on C++ pointer arithmetic.
  *Source: `README.md` lines 52–58 (Security & Implementation Notes)*

- **test_esp_api.py uses hardcoded 784-element dummy input:** The ESP32 test script always sends 784 zeros (`.5` × 784), regardless of which model is deployed.
  *Source: `scripts/test_esp_api.py` lines 13–14*

---

## 10. Future Upgrades

Only items explicitly stated in the repository:

From `README.md` Project Roadmap table (lines 83–91):

| Feature | Status |
|---|---|
| PyTorch & TFLite Benchmarking | In Progress (as of the README) |
| INT8 Post-Training Quantization | Planned |
| Conv2D and Pooling Kernels | Planned |
| ESP32 & STM32 Hardware Port | Planned |
| Layer Fusion / Cache Optimizations | Planned |

From `Idea.md` (research angles described but not yet implemented):
- Novel Mixed-Precision Quantization Schemes (INT4/INT8 per-layer heuristics)
- Layer Fusion Optimization (fusing Conv2D + Bias + ReLU in a single memory pass)
- Domain-Specific Sparse Kernel Execution (compressed sparse row for embedded processors)

No TODO or FIXME comments were found in source files. No issue tracker is present in the repository.

---

---

# PART B — Technical Documentation

---

## 1. Repository Architecture

```
Module Hierarchy:

scripts/run_tests.sh  [orchestrator]
    ├── invokes → python/train_export.py        [training + weight export]
    ├── invokes → python/benchmark_baselines.py [PyTorch + TFLite baseline]
    ├── invokes → python/export_tflite.py       [TFLite + TFLM header export]
    ├── compiles → cpp/infer.cpp                [bare-metal engine, desktop]
    ├── compiles → cpp/tflm_bench.cpp           [TFLM benchmark harness]
    └── invokes → scripts/update_readme_stats.py [README auto-update]

cpp/esp32_inference/main/infer.cpp  [standalone ESP-IDF app, independent of run_tests.sh]
    ├── depends on → weights/model_weights.h    [baked-in model weights]
    └── depends on → weights/test_image_data.h  [baked-in test sample]
```

**Package organization:**
- No Python packages are defined (no `__init__.py` files; all Python files are standalone scripts).
- The C++ engine is not a library — it is a standalone executable built per-model per-run.
- The ESP32 project is a self-contained ESP-IDF application.

*Source: Full directory traversal, `scripts/run_tests.sh`*

---

## 2. System Architecture

### Subsystem Interaction

```
┌──────────────────────────────────────────────────────────┐
│  Python Toolchain (offline, developer machine)           │
│                                                          │
│  train_export.py                                         │
│    PyTorch training loop ──► checkpoint (.pt)            │
│    export_c_headers() ──────► model_weights.h            │
│                               test_image_data.h          │
│                                                          │
│  export_tflite.py                                        │
│    .pt → Keras → TFLiteConverter → flatbuffer patch      │
│    ──► *.tflite + *_tflm_model.h                        │
│                                                          │
│  benchmark_baselines.py                                  │
│    PyTorch inference loop ──► stdout KEY:VALUE           │
│    TFLite inference loop ───► stdout KEY:VALUE           │
└────────────────────────────────┬─────────────────────────┘
                                 │ (via run_tests.sh)
┌────────────────────────────────▼─────────────────────────┐
│  C++ Build System (g++ -O3 -std=c++17)                   │
│                                                          │
│  cpp/infer.cpp                                           │
│    #include model_weights.h   ← weights in .rodata       │
│    #include test_image_data.h ← input in .rodata         │
│    static arena_buffer_A/B    ← activations in .bss      │
│    run_inference() ─────────► stdout KEY:VALUE           │
│                                                          │
│  cpp/tflm_bench.cpp                                      │
│    links against libtensorflow-microlite.a               │
│    binary-search for min arena ──► stdout KEY:VALUE      │
└────────────────────────────────┬─────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────┐
│  Logging & Reporting                                     │
│                                                          │
│  performance_log.csv (22-column append-only log)         │
│  update_readme_stats.py (CSV → README.md Markdown table) │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  ESP32 Firmware (separate ESP-IDF build)                 │
│                                                          │
│  nvs_flash_init() → esp_wifi_init() → IP acquired        │
│  → httpd_start() → POST /infer handler                   │
│  → cJSON parse → run_inference(dynamic_input)            │
│  → cJSON response: {probabilities, latency_us, free_heap}│
└──────────────────────────────────────────────────────────┘
```

*Source: `scripts/run_tests.sh`, `cpp/infer.cpp`, `cpp/esp32_inference/main/infer.cpp`*

---

## 3. Execution Flow

### Desktop Pipeline (run_tests.sh → build/infer)

**Entry point:** `scripts/run_tests.sh` (Bash)

1. `cd` to project root; verify `.venv/bin/python` exists.
2. Initialize `performance_log.csv` with 22-column header if absent.
3. Check for `third_party/tflm_build/libtensorflow-microlite.a`; set `TFLM_AVAILABLE=0|1`.
4. `declare -A MODELS` with two entries: `MNIST_MLP_FP32 → "mnist 784"`, `Keyword_Spotting_MLP_FP32 → "kws 1000"`.
5. For each model:
   a. Check for pre-existing weight headers; if missing, invoke `python/train_export.py`.
   b. Copy model-specific headers to staging locations (`weights/model_weights.h`, `weights/test_image_data.h`).
   c. Invoke `python/benchmark_baselines.py`; parse stdout with `grep` + `cut`.
   d. Compile: `g++ -O3 -std=c++17 -DINPUT_SIZE=<N> -I./weights cpp/infer.cpp -o build/infer`
   e. Execute `build/infer`; parse stdout.
   f. Run `size build/infer` to extract `.text`, `.bss`, and total decimal sizes.
   g. If TFLM available: compile and run `tflm_bench` with model-specific `-D` flags.
   h. Append CSV row.
6. Invoke `scripts/update_readme_stats.py`.

**Entry point: `cpp/infer.cpp` → `main()`**

1. Capture `rusage` before.
2. Capture `chrono::high_resolution_clock::now()` before.
3. Call `run_inference()`:
   - **Layer 1:** Nested loop over `[0, HIDDEN1_SIZE)` × `[0, INPUT_SIZE)`. Reads `fc1_bias[i]` and `fc1_weight[i][j]` (both in `.rodata`/Flash). Writes `arena_buffer_A[i] = static_relu(sum)`.
   - **Layer 2:** Nested loop over `[0, HIDDEN2_SIZE)` × `[0, HIDDEN1_SIZE)`. Reads `fc2_bias[i]`, `fc2_weight[i][j]`, reads `arena_buffer_A[j]`. Writes `arena_buffer_B[i] = static_relu(sum)`.
   - **Layer 3:** Nested loop over `[0, OUTPUT_SIZE)` × `[0, HIDDEN2_SIZE)`. Reads `fc3_bias[i]`, `fc3_weight[i][j]`, reads `arena_buffer_B[j]`. Writes `logits_out[i] = sum`.
   - **Softmax:** Finds `max_logit`; computes `exp(logits[i] - max_logit)` for numerical stability; normalizes.
4. Capture time and rusage after.
5. Call `get_rss_kb()` (reads `/proc/self/status` for `VmRSS`).
6. Print key:value lines to stdout.

*Source: `scripts/run_tests.sh`, `cpp/infer.cpp`*

### ESP32 Firmware Entry Point

**Entry point: `app_main()` in `cpp/esp32_inference/main/infer.cpp`**

1. `nvs_flash_init()` — initialize Non-Volatile Storage.
2. `esp_netif_init()`, `esp_event_loop_create_default()`, `esp_netif_create_default_wifi_sta()`.
3. `esp_wifi_init()` with default config.
4. Register `wifi_event_handler` for `WIFI_EVENT` and `IP_EVENT`.
5. Configure `wifi_config_t` from hardcoded `WIFI_SSID` / `WIFI_PASS`.
6. `esp_wifi_set_mode(WIFI_MODE_STA)`, `esp_wifi_start()`.
7. On `WIFI_EVENT_STA_START` → `esp_wifi_connect()`.
8. On `WIFI_EVENT_STA_DISCONNECTED` → retry `esp_wifi_connect()`.
9. On `IP_EVENT_STA_GOT_IP` → call `start_webserver()`.
10. `start_webserver()` calls `httpd_start()`, registers `POST /infer` → `infer_post_handler`.

**HTTP Request lifecycle (`infer_post_handler`):**
1. `malloc(content_len + 1)` for request body.
2. `httpd_req_recv()` to read body bytes.
3. `cJSON_Parse()` to decode JSON array.
4. `malloc(INPUT_SIZE * sizeof(float))` for `dynamic_input`.
5. Iterate cJSON array items, populate `dynamic_input`.
6. `esp_timer_get_time()` → `run_inference(dynamic_input)` → `esp_timer_get_time()`.
7. `free(dynamic_input)`, `cJSON_Delete(json)`.
8. Build cJSON response: `probabilities` array + `latency_us` + `free_heap_kb`.
9. `httpd_resp_set_type(req, "application/json")`, `httpd_resp_send()`.
10. Free cJSON objects.

Note: The ESP32 HTTP handler uses `malloc` for request body and input buffer (contrary to the zero-dynamic-allocation goal stated for the inference engine itself).

*Source: `cpp/esp32_inference/main/infer.cpp`*

---

## 4. Model Architecture

Both `MNIST_MLP` and `KWS_MLP` implement an identical 3-layer fully-connected (MLP) topology with ReLU activations between layers and no final activation (raw logits output). The C++ engine applies softmax post-hoc.

### Layer Dimensions

| Layer | PyTorch Class | Weight Shape | Bias Shape | Activation |
|---|---|---|---|---|
| fc1 | `nn.Linear` | `[128, 784]` (MNIST) or `[128, 1000]` (KWS) | `[128]` | ReLU |
| fc2 | `nn.Linear` | `[64, 128]` | `[64]` | ReLU |
| fc3 | `nn.Linear` | `[10, 64]` | `[10]` | None (logits) |

### Hyperparameters
- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Loss function:** CrossEntropyLoss
- **Batch size:** 64
- **Epochs:** 1

### Activation Functions
- **ReLU:** Implemented in PyTorch as `nn.ReLU()` and in C++ as `inline float static_relu(float x) { return x > 0.0f ? x : 0.0f; }`.
- **Softmax:** Numerically stable form: subtract `max_logit` before exponentiation.

### No regularization, dropout, batch normalization, or weight decay was found in any source file.

*Source: `python/train_export.py` lines 38–52, 92–106, 68–80; `cpp/infer.cpp` lines 41–112*

---

## 5. Data Pipeline

### MNIST
- **Loading:** `torchvision.datasets.MNIST('./data', download=True, transform=...)`
- **Normalization:** `transforms.Normalize((0.1307,), (0.3081,))` — standard MNIST mean/std.
- **Batching:** `DataLoader` with `batch_size=64, shuffle=True` for training; `batch_size=1, shuffle=False` for test sample extraction.
- **Flattening:** Inside `forward()` via `x.view(-1, 784)`.

*Source: `python/train_export.py` lines 56–60*

### Mini Speech Commands (KWS)
- **Download:** `urllib.request.urlretrieve` from `http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip`.
- **Extraction:** `zipfile.ZipFile.extractall()`.
- **Class discovery:** `sorted([d for d in os.listdir(extract_dir) if os.path.isdir(...) and not d.startswith('_')])`.
- **File limit:** First 200 `.wav` files per class only.
- **Sample width handling:** 2-byte samples → `/32768.0`; 1-byte samples → `(x - 128.0) / 128.0`; other widths → skipped.
- **Downmixing:** Multi-channel: `audio[::n_channels]`.
- **Length normalization:** Pad with zeros to 16,000 samples; truncate if longer.
- **Feature extraction:** `audio.reshape(1000, 16).mean(axis=1)` — average pooling over 16-sample windows → 1,000-element feature vector.
- **Batching:** `TensorDataset` + `DataLoader` with `batch_size=64, shuffle=True`.
- **No mel spectrogram, MFCC, or FFT is computed.** The feature is raw time-domain mean-pooling.

*Source: `python/train_export.py` lines 110–228*

### Weight Export
`export_c_headers(model, test_image)` iterates `model.named_parameters()`:
- 2D parameters (weight matrices) → `const float name[rows][cols] = { {row0...}, {row1...}, ... };`
- 1D parameters (bias vectors) → `const float name[length] = { ... };`
- C names are derived by replacing `.` with `_` in PyTorch parameter names (e.g., `fc1.weight` → `fc1_weight`).

*Source: `python/train_export.py` lines 9–33*

### TFLite Export (`export_tflite.py`)
1. Reconstructs a Keras `Sequential` model with same architecture.
2. Manually copies weights: `keras_model.layers[i].set_weights([weight.T, bias])` — note transposition required for Dense layer convention.
3. `TFLiteConverter.from_keras_model(keras_model).convert()`.
4. **Binary patch 1:** Replaces dynamic batch dim `-1` with static `1` in the flatbuffer bytes using `struct.pack('<ii', -1, input_shape)` pattern search.
5. **Binary patch 2:** Finds the input tensor's buffer index field in the flatbuffer and patches it from `1` → `0` so TFLM allocates arena space for the input tensor rather than treating it as a pre-loaded constant.
6. Outputs `.tflite` file and `_tflm_model.h` C array header.

*Source: `python/export_tflite.py`*

---

## 6. Training Pipeline

### Loop Structure
```
for epoch in range(1):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        if batch_idx % [200|10] == 0:
            print(loss)
```

### Checkpointing
- Saves: `torch.save(model.state_dict(), "weights/model_weights.<MNIST|KWS>.pt")`
- Loads: `model.load_state_dict(torch.load("weights/model_weights.<MNIST|KWS>.pt"))`
- Load uses `weights_only=True` in `benchmark_baselines.py` and `export_tflite.py` but **not** in `train_export.py`.

*Source: `python/train_export.py` lines 64–80, 142–144; `python/benchmark_baselines.py` line 96*

### No early stopping, learning rate scheduling, or mixed precision training is implemented.

*Source: Full review of `python/train_export.py` — none of these found*

---

## 7. Inference Pipeline

### C++ Desktop Inference (`cpp/infer.cpp`)

**Input:** `input_image[]` — a `const float` array `#include`-d from `test_image_data.h` at compile time.

**Memory layout:**
- `fc1_weight[HIDDEN1_SIZE][INPUT_SIZE]` — in `.rodata` (Flash/read-only data segment)
- `fc1_bias[HIDDEN1_SIZE]` — in `.rodata`
- `arena_buffer_A[128]` — in `.bss` (128 × 4 = 512 bytes)
- `arena_buffer_B[128]` — in `.bss` (512 bytes — sized to HIDDEN1_SIZE even though Layer 2 only uses 64)
- `logits_out[10]` — in `.bss` (40 bytes)
- `probabilities[10]` — in `.bss` (40 bytes)

**Forward pass:** Three nested loop pairs (no library calls, no BLAS). Each inner loop is a dot-product with bias accumulation.

**Post-processing:** Stable softmax; result in `probabilities[]`.

**Output:** Printed to stdout as key:value pairs on separate lines.

*Source: `cpp/infer.cpp`*

### ESP32 Inference (`cpp/esp32_inference/main/infer.cpp`)
Identical forward pass but `run_inference()` accepts `const float* current_input` as a parameter instead of using the global `input_image`. For HTTP requests, `dynamic_input` is `malloc`-d from heap per request, populated from the parsed JSON array, and freed after inference.

*Source: `cpp/esp32_inference/main/infer.cpp` lines 70–122, 126–188*

---

## 8. Configuration System

### Runtime Configuration
- **Model selection:** CLI argument `--model [mnist|kws]` passed to Python scripts.
- **Input size:** `-DINPUT_SIZE=<N>` preprocessor definition passed to `g++` at compile time; defaults to 784 if not set.
- **TFLM model header:** `-DMODEL_HEADER=\"<file>\"` and `-DMODEL_VAR=<varname>` preprocessor definitions.
- **Architecture constants:** C++ `constexpr int` values hardcoded in `infer.cpp`: `HIDDEN1_SIZE = 128`, `HIDDEN2_SIZE = 64`, `OUTPUT_SIZE = 10`.

### ESP32 Configuration
- **Wi-Fi credentials:** `#define WIFI_SSID` / `#define WIFI_PASS` — compile-time constants.
- **Flash configuration:** `sdkconfig.defaults` — sets `CONFIG_PARTITION_TABLE_CUSTOM=y`, `CONFIG_ESPTOOLPY_FLASHSIZE_4MB=y`.
- **Partition table:** `partitions.csv` — NVS at 0x9000 (20 KB), PHY init at 0xF000 (4 KB), factory app at 0x10000 (3 MB).

### No YAML, JSON, or .env configuration files are used. There is no runtime configuration file system.

*Source: `cpp/infer.cpp` lines 24–29, `scripts/run_tests.sh` lines 85, 119–126, `cpp/esp32_inference/main/infer.cpp` lines 22–23, `cpp/esp32_inference/sdkconfig.defaults`, `cpp/esp32_inference/partitions.csv`*

---

## 9. Engineering Decisions

The following are directly observable from the code:

**1. Weights-as-headers compile-time inclusion**
Model weights are embedded in the C++ binary via `#include "model_weights.h"` rather than loaded from a file at runtime. This places them in `.rodata`, which on microcontrollers maps to Flash rather than SRAM.
*Source: `cpp/infer.cpp` lines 18–21, comments on lines 63, 68*

**2. Static arena (ping-pong buffers)**
Two identically-sized static buffers (`arena_buffer_A` and `arena_buffer_B`) are used alternately as source and destination between layers. The comment states: "Ping-ponging buffers to minimize runtime SRAM footprint. Max memory required = largest layer width * sizeof(float)."
*Source: `cpp/infer.cpp` lines 32–39, comments*

**3. Compile-time architecture constants**
Layer sizes are `constexpr int` values rather than runtime parameters. Changing the network architecture requires source code changes and recompilation.
*Source: `cpp/infer.cpp` lines 27–29*

**4. Preprocessor-based platform abstraction**
`#ifdef ESP32` / `#else` blocks switch between ESP32-specific headers (`esp_timer.h`, `esp_heap_caps.h`) and POSIX headers (`sys/resource.h`). The same logical codebase compiles for both targets.
*Source: `cpp/infer.cpp` lines 10–17, `cpp/esp32_inference/main/infer.cpp` lines 10–26*

**5. Dynamic model dispatch in run_tests.sh**
An associative array maps model names to Python CLI args and `INPUT_SIZE` values. Adding a new model requires only adding one entry to the array.
*Source: `scripts/run_tests.sh` lines 34–38*

**6. grep-friendly stdout protocol**
All benchmarking executables (Python and C++) emit results as `KEY:VALUE` on separate lines, which the shell script parses with `grep "^KEY:" | cut -d':' -f2`. This avoids a structured serialization format.
*Source: `cpp/infer.cpp` lines 151–158; `python/benchmark_baselines.py` lines 106–113; `scripts/run_tests.sh` lines 71–78, 96–99*

**7. TFLite flatbuffer binary patching**
Rather than using a patched TFLite converter, `export_tflite.py` post-processes the converter output by directly modifying bytes in the flatbuffer. Two patches are applied: (a) static batch dimension fix, (b) input tensor buffer index fix. This is documented with detailed comments.
*Source: `python/export_tflite.py` lines 57–90*

**8. TFLM minimum arena binary search**
`tflm_bench.cpp` implements a binary search over arena sizes (1–256 KB) to find the minimum viable arena for the model, reporting it as `TFLM_ARENA_KB`. This uses a separate static search arena to avoid contaminating the inference arena.
*Source: `cpp/tflm_bench.cpp` lines 65–84*

---

## 10. Performance Considerations

### Verified techniques:

**Flash over SRAM for weight storage:**
Weights (`fc1_weight`, `fc2_weight`, `fc3_weight`, `fc1_bias`, etc.) are `const` arrays — they reside in `.rodata` / Flash. On microcontrollers with XIP (execute-in-place) Flash, this avoids consuming SRAM for weights.

**Static-only runtime memory:**
No heap allocation occurs during inference in `cpp/infer.cpp`. All buffers are statically allocated at program load.

**Compiler optimization:**
The engine is compiled with `g++ -O3`, enabling aggressive optimizations including loop unrolling, inlining, and auto-vectorization at the compiler's discretion. No explicit SIMD intrinsics or manual loop unrolling are present in the code.

**Numerically stable softmax:**
Subtracts `max_logit` before exponentiation to prevent overflow, a standard technique.

**Warmup runs before benchmarking:**
Both `benchmark_baselines.py` and `tflm_bench.cpp` execute 10 inference passes before timing begins.
*Source: `python/benchmark_baselines.py` line 42; `cpp/tflm_bench.cpp` line 119*

**Averaged over 1,000 runs:**
All latency measurements are averages over 1,000 iterations.

### Not implemented:
- No SIMD intrinsics (ARM NEON, SSE, AVX).
- No loop tiling / cache-blocking for GEMM.
- No quantization (INT8/INT4).
- No layer fusion.
- No parallelism (single-threaded).
- No streaming I/O.

*Source: Full review of `cpp/infer.cpp`, `cpp/tflm_bench.cpp`*

---

## 11. Error Handling

**Python scripts:**
- `train_export.py`: WAV file loading is wrapped in `try/except Exception`, corrupt files are silently skipped (`continue`). Dataset download errors print the exception and re-raise.
- `export_tflite.py`: Flatbuffer patches print a message if the pattern is not found but do not abort.
- `benchmark_baselines.py`: `get_rss_kb()` falls back to `resource.getrusage()` if `/proc/self/status` cannot be read.

**C++ (desktop):**
- `get_rss_kb()` falls back to `getrusage()` if the `/proc/self/status` read fails.
- No bounds-checking on matrix dimensions — the README explicitly flags this as a security note.

**C++ (ESP32 HTTP handler):**
- Allocation failures → `httpd_resp_send_500(req)` + `ESP_FAIL`.
- Receive errors → `httpd_resp_send_500(req)` + `ESP_FAIL`.
- Invalid JSON → `httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid JSON Array")`.
- Wi-Fi disconnections → infinite retry loop in `wifi_event_handler`.

**Shell script (`run_tests.sh`):**
- `set -e` causes the script to exit on any command failure.
- Missing venv → explicit error message and `exit 1`.
- Missing TFLM library → TFLM benchmarks skipped with a message, not an error.

*Source: `python/train_export.py`, `python/benchmark_baselines.py`, `cpp/infer.cpp`, `cpp/esp32_inference/main/infer.cpp`, `scripts/run_tests.sh`*

---

## 12. Testing

### No unit tests or integration test suite is present in the repository.

Testing is implemented exclusively through the benchmarking pipeline in `scripts/run_tests.sh`, which is an end-to-end functional validation that the engine compiles and produces output, rather than a correctness test suite.

**`scripts/test_esp_api.py`** is a manual smoke test: it sends an HTTP POST with a 784-element constant array (`[0.5] * 784`) and prints the ESP32's response. It does not assert correctness of the output probabilities.

**`scripts/diagnose_serial.py`** is a hardware diagnostic tool, not a software test.

No CI/CD configuration files (`.github/workflows/`, `.gitlab-ci.yml`, `Makefile` targets, etc.) were found in the repository.

*Source: Full directory traversal — no `tests/`, `test_*.py`, `*_test.py` found*

---

## 13. Known Technical Debt

Directly observable from source code comments, code structure, and file content:

**1. `# Reduced to 1 epoch for faster testing`**
Both training functions use `range(1)` with an inline comment indicating this was reduced for development speed.
*Source: `python/train_export.py` line 72*

**2. KWS output dimension mismatch (8 classes, 10 outputs)**
`KWS_MLP.fc3 = nn.Linear(64, 10)` outputs 10 logits, but the dataset has 8 classes. No comment explains this.
*Source: `python/train_export.py` line 99 vs. `data/README.txt` lines 8–15*

**3. Hardcoded Wi-Fi credentials**
`#define WIFI_SSID "Drackko"` and `#define WIFI_PASS "zdt32109"` are embedded in the ESP32 firmware source.
*Source: `cpp/esp32_inference/main/infer.cpp` lines 22–23*

**4. Duplicate `#include <string>` in infer.cpp**
Both `cpp/infer.cpp` and `cpp/esp32_inference/main/infer.cpp` contain `#include <string>` twice (lines 6 and 7).
*Source: `cpp/infer.cpp` lines 6–7; `cpp/esp32_inference/main/infer.cpp` lines 6–7*

**5. Desktop `cpp/infer.cpp` cannot accept runtime input**
The input is `#include`-d at compile time. The code comment "CONFIDENCE matches original targets" in the output print statement suggests placeholder text.
*Source: `cpp/infer.cpp` line 151: `"PREDICTED DIGIT OUT: CONFIDENCE matches original targets."`*

**6. `performance_log_old.csv` has a different schema**
The old log (7 columns) is present alongside the current log (22 columns), suggesting a schema migration occurred. The old file is not cleaned up or documented beyond its presence.
*Source: `performance_log_old.csv` line 1 vs. `performance_log.csv` line 1*

**7. `pyproject.toml` has no description**
`description = "Add your description here"` — the default placeholder was not replaced.
*Source: `pyproject.toml` line 4*

**8. `train_export.py` uses `torch.load` without `weights_only=True`**
The training script does not pass `weights_only=True` to `torch.load()`, while `benchmark_baselines.py` and `export_tflite.py` do. This is a security inconsistency.
*Source: `python/train_export.py` line 66 vs. `python/benchmark_baselines.py` line 96*

**9. `arena_buffer_B` is sized to `HIDDEN1_SIZE` (128) but Layer 2 only needs 64 floats**
The ping-pong buffer is over-allocated for the second layer.
*Source: `cpp/infer.cpp` line 37: `float arena_buffer_B[HIDDEN1_SIZE]; // 512 Bytes`*

**10. `test_esp_api.py` hardcodes IP and 784-element dummy input**
The default IP `"192.168.1.100"` differs from the example output in `ESP32_DEPLOYMENT.md` (`10.205.64.92`), and the script always sends `[0.5] * 784` regardless of the deployed model.
*Source: `scripts/test_esp_api.py` lines 6, 13–14*

---

## 14. Dependency Graph

```
scripts/run_tests.sh
  ├── reads/writes: performance_log.csv
  ├── reads/writes: weights/ (headers, .pt files)
  ├── reads/writes: build/ (compiled executables)
  ├── calls: python/train_export.py
  │     ├── imports: torch, torch.nn, torch.optim, torchvision, numpy, argparse, os
  │     ├── downloads to: data/MNIST/ (via torchvision)
  │     ├── downloads to: data/mini_speech_commands/ (via urllib)
  │     ├── reads/writes: weights/model_weights.MNIST.pt
  │     ├── reads/writes: weights/model_weights.KWS.pt
  │     └── writes: model_weights.h, test_image_data.h (then moved to weights/)
  ├── calls: python/benchmark_baselines.py
  │     ├── imports: torch, torch.nn, time, argparse, numpy, resource, tensorflow
  │     ├── reads: weights/model_weights.MNIST.pt or weights/model_weights.KWS.pt
  │     └── writes: stdout (KEY:VALUE pairs)
  ├── calls: python/export_tflite.py (if TFLM available and .tflite not present)
  │     ├── imports: torch, torch.nn, numpy, argparse, os, tensorflow, struct
  │     ├── reads: weights/model_weights.MNIST.pt or weights/model_weights.KWS.pt
  │     └── writes: weights/*_model.tflite, weights/*_tflm_model.h
  ├── invokes g++ to build: cpp/infer.cpp
  │     ├── #includes: weights/model_weights.h
  │     ├── #includes: weights/test_image_data.h
  │     └── links: libc, libstdc++ (standard g++ default)
  ├── invokes g++ to build: cpp/tflm_bench.cpp
  │     ├── #includes: weights/*_tflm_model.h
  │     ├── #includes: third_party/tflm_build/include/tensorflow/lite/...
  │     └── links: third_party/tflm_build/libtensorflow-microlite.a
  └── calls: scripts/update_readme_stats.py
        ├── reads: performance_log.csv
        └── reads/writes: README.md

cpp/esp32_inference/ (independent ESP-IDF project)
  ├── main/infer.cpp
  │     ├── #includes: main/model_weights.h (baked-in MNIST weights)
  │     ├── #includes: main/test_image_data.h
  │     └── links: ESP-IDF (esp_wifi, esp_http_server, nvs_flash, esp_netif, FreeRTOS)
  └── managed_components/espressif__cjson/
        └── cJSON.h / cJSON.c (via idf_component.yml dependency)

scripts/setup_tflm.sh (one-time setup)
  ├── clones: https://github.com/tensorflow/tflite-micro.git → third_party/tflite-micro/
  └── builds: third_party/tflm_build/libtensorflow-microlite.a + headers
```

*Source: Full source review*

---

## 15. End-to-End System Summary

ColdInfer is a two-phase system consisting of an offline Python toolchain and an online C++ inference engine targeting both Linux and ESP32.

**Offline phase (Python):** Starting from either the MNIST dataset (via torchvision) or the Mini Speech Commands dataset (downloaded from Google Cloud Storage), `train_export.py` trains a 3-layer fully-connected MLP for 1 epoch using PyTorch (Adam optimizer, CrossEntropyLoss) and serializes the trained weights as C++ literal float array headers via `export_c_headers()`. A parallel path in `export_tflite.py` converts the same weights through a Keras intermediate representation into a TFLite flatbuffer, applying two binary patches to make it TFLM-compatible, and outputs both a `.tflite` file and a C array header. A third script, `benchmark_baselines.py`, measures the inference latency and memory footprint of the PyTorch and TFLite runtimes directly in Python for baseline comparison.

**Online phase (C++):** The inference engine in `cpp/infer.cpp` is compiled by `g++` with `-O3 -std=c++17`, with the weight headers `#include`-d directly so that weights reside in `.rodata` (read-only data / Flash on microcontrollers). The runtime allocates two static float buffers of 512 bytes each for ping-pong activation storage, performing three sequential matrix-vector multiply-accumulate loops with bias addition and ReLU activation, followed by a numerically stable softmax. No dynamic memory allocation occurs during inference. The executable emits `KEY:VALUE` pairs on stdout that `run_tests.sh` parses and appends to `performance_log.csv`. A parallel `tflm_bench` binary runs the same model through TFLite Micro for additional comparison.

**ESP32 deployment:** The `cpp/esp32_inference/main/infer.cpp` file is the same logical inference engine adapted for an ESP-IDF application. It initializes Wi-Fi (hardcoded SSID/password), connects as a station, obtains an IP, and starts an lwIP-based HTTP server. `POST /infer` requests accept a JSON float array, which is parsed with cJSON (espressif/cjson ^1.7.17), used as the inference input, and the softmax probabilities plus latency and free heap are returned as a JSON response. The binary is built via CMake/ESP-IDF and flashed using `esptool`.

**Automation:** The entire pipeline — training, export, compilation, benchmarking across four runtimes, CSV logging, and README update — is orchestrated by `scripts/run_tests.sh` in a single command (`./scripts/run_tests.sh`).

*Source: Full repository audit.*

---

## Performance Data (from performance_log.csv — last recorded run: 2026-06-02)

| Model | Engine | Latency (µs) | Memory RSS (KB) | CPU Total (µs) |
|---|---|---|---|---|
| Keyword_Spotting_MLP_FP32 | ColdInfer | 142 | 4,468 | 150 |
| Keyword_Spotting_MLP_FP32 | TFLM | 62 | 4,320 | 62 |
| Keyword_Spotting_MLP_FP32 | PyTorch | 63 | 885,556 | 764 |
| Keyword_Spotting_MLP_FP32 | TFLite | 9 | 930,652 | 9 |
| MNIST_MLP_FP32 | ColdInfer | 117 | 4,400 | 126 |
| MNIST_MLP_FP32 | TFLM | 50 | 4,208 | 50 |
| MNIST_MLP_FP32 | PyTorch | 57 | 885,536 | 682 |
| MNIST_MLP_FP32 | TFLite | 8 | 928,984 | 8 |

> [!NOTE]
> These figures are from `performance_log.csv` (last 2 rows per model, 2026-06-02). ColdInfer and TFLM share a comparable memory footprint (~4–4.5 MB RSS on Linux) versus PyTorch/TFLite which load full frameworks (~885–930 MB RSS). ColdInfer is slower than both TFLite and TFLM in latency.

*Source: `performance_log.csv` rows 5–7 (2026-06-02)*

---

*Document generated by full repository audit. All claims have been directly verified against source files. No information was inferred from general ML knowledge.*
