─────────────────────────────────────────────── ⊹ ࣪ ˖ ૮( ˶ᵔ ᵕ ᵔ˶ )っ  ColdInfer vers. 0.1.0 ───────────────────────────────────────────────

A lightweight, dependency-free runtime library written in pure C++ that takes pre-trained machine learning models, compresses their weights, and runs forward-pass inference directly on bare-metal hardware. No standard AI bloat, no dynamic allocation, just optimized math kernels and static memory pools.

## Features
* **Zero Dynamic Allocation** -- Uses a static memory arena for predictable SRAM usage, entirely avoiding heap fragmentation.
* **Graph Compiler & Parser** -- Python toolchain that traverses PyTorch topological graphs and serializes them into custom flat binary formats.
* **Bare-Metal Optimized Kernels** -- Manual execution loops for GEMM, Conv2D, pooling, and ReLU, heavily optimized for cache alignment and pointer arithmetic.
* **Automated Benchmarking** -- Built-in Bash testing suite that compiles the engine dynamically based on input size and benchmarks against standard PyTorch and TFLite runtimes.
* **FP32 & Quantization Ready** -- Currently supports flat FP32 exports, heavily structured to support INT8 and INT4 post-training quantization in the upcoming roadmap.

## Quick Start
Defaults work out of the box. Ensure you have a standard GCC toolchain and Python 3.12+ installed. 

### Native Linux / macOS
```bash
git clone [https://github.com/sudo-sidd/ColdInfer.git](https://github.com/sudo-sidd/ColdInfer.git)
cd ColdInfer

# Install dependencies and create the virtual environment automatically
uv sync

# Run the benchmark suite
./scripts/run_tests.sh
```

### Live Simulation Dashboard

The showcase dashboard runs the native C++ classifier in a simulated SRAM budget:

```bash
cd webapp && npm run dev -- --host 0.0.0.0
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:3000](http://localhost:3000) and use **Simulation Lab** to
switch between the MNIST digit model and keyword spotter. The lab renders the
same baked input consumed by the C++ engine, class probabilities, prediction,
latency, and the configured simulated SRAM cap. This is the recommended demo
flow; the experimental chatbot is no longer part of the dashboard.

### YOLO Smoke Test

The repository includes the small public COCO8 detection smoke set under
`data/yolo-coco8/` and a downloaded YOLO11n checkpoint (`yolo11n.pt`, ignored
from source control). Run the same target used by the dashboard with:

```bash
uv run python scripts/smoke_test_yolo.py
```

The dashboard's default YOLO target is a quantized 160x160 simulated ESP32
profile. It uses uint8 input, int8 detector output, fixed scratch accounting,
and refuses an arena that cannot hold the tensors. The fixed envelope is about
312 KB, so it fits within the 520 KB internal-SRAM profile. The reproducible artifact is generated with
`uv run python scripts/quantize_yolo.py` and stored as an INT8-weight ONNX
model. The current simulated execution retains float activations through
ONNX Runtime; full integer ESP32 execution still requires wiring a
Micro-compatible model plus convolution, detection-head, and NMS operators
into the firmware.

### ESP32 Deployment

ColdInfer supports flashing the engine and model to an ESP32 for bare-metal edge inference, exposing the runtime over a local HTTP server.

Please see the [ESP32 Deployment Guide](docs/ESP32_DEPLOYMENT.md) for full instructions on flashing the firmware, monitoring serial output, and testing the inference API over Wi-Fi.

### Architecture

The project is split into two strict halves to enforce hardware independence on the execution side:

    - The Compiler / Converter (Python): Parses the model graph, handles quantization (WIP), and serializes weights into flat .h arrays (soon to be .bin).

    - The Inference Engine Runtime (Pure C++): Allocates a single static scratchpad memory buffer, reads the structural topology, and executes optimized kernels over the data.

```
app/ 
├── cpp/               # Bare-metal C++ kernel execution
├── data/              # Raw training data (MNIST, Audio samples)
├── python/            # PyTorch training, model parsing, and TFLite baselines
├── scripts/           # CI/CD, build tools, and automated performance logging
└── weights/           # Serialized model exports waiting for compilation
```

### Security & Implementation Notes

Treat this engine as an isolated execution environment. Because it relies entirely on C++ pointer arithmetic for speed, passing maliciously crafted or incorrectly sized binary blobs could cause buffer overruns.

    - Never dynamically allocate memory at runtime; respect the arena_buffer boundaries.

    - Do not expose the compiler pipeline to untrusted user input without strict bounds checking on matrix dimensions.

    - Track memory footprints inside performance_log.csv to ensure updates don't silently blow past your microcontroller's strict SRAM limits.

## Performance Benchmarks

Latest benchmark results comparing ColdInfer against standard runtimes. Lower is better for all metrics.

### Keyword_Spotting_MLP_FP32
| Engine | Memory RSS (KB) | Latency (µs) | CPU Time (µs) |
|---|---|---|---|
| **TFLM (Micro)** | 4320 | 62 | 62 |
| **ColdInfer** | 4468 | 142 | 150 |
| **PyTorch** | 885556 | 63 | 764 |
| **TFLite** | 930652 | 9 | 9 |

### MNIST_MLP_FP32
| Engine | Memory RSS (KB) | Latency (µs) | CPU Time (µs) |
|---|---|---|---|
| **TFLM (Micro)** | 4208 | 50 | 50 |
| **ColdInfer** | 4400 | 117 | 126 |
| **PyTorch** | 885536 | 57 | 682 |
| **TFLite** | 928984 | 8 | 8 |


## Project Roadmap & Timeline

| Feature | Status | Target Date |
| --- | --- | --- |
| **FP32 MLP Engine** | Complete | June 2026 |
| **PyTorch & TFLite Benchmarking** | In Progress | June 2026 |
| **INT8 Post-Training Quantization** | Planned | July 2026 |
| **Conv2D and Pooling Kernels** | Planned | August 2026 |
| **ESP32 & STM32 Hardware Port** | Planned | October 2026 |
| **Layer Fusion / Cache Optimizations** | Planned | December 2026 |

### License
MIT -- see [LICENSE](LICENSE) for details.

```                                                     
                       \\-----//-   
                    \\-\\\\\\-/////               
                    \\/\----//|/////-\\\\                
                   \\-\\\   ||||||:/ \|///\-                
                  \\.\\     ///\\\-\\////|\/0/////                 
                  \\||      //-\\\\     |//-/00////                  
                  \|||                  //-//000////                  
                 \\\////                 //||0000///                  
                 ||/.////   \-            \\\0000//|                  
                 ///../////\\     /      \\\\0000/||/                 
                  ////-\.-// \    //   \\\\\\0000\|////       
                   ////   .|||  /\//--\\\\\//0000\\|////      
                      \\-..  /  ///.\\\\|//////0\\:/|-.//             
                     .//../////  /////////////--\\\/////||             
                /--\-/////\\-///    ////////---\--|\/// /\\    
               /       ///\\-///      /////////\\\/-|//                
              |        /////\||          ////////-/\||\               
               \        |||/\\\\      |\\-- /////-///|||               
                \\-//   \||\\\      --\\\\----./|\\\||||               
                                    ..-\\|\--//-\\\\\|||               
                                    //../////c\--\\\-/||                 
                                     //////////|\\\\?\\\                 
                                       //////////----\\\                  
              
```
