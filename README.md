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
python -m venv .venv
source .venv/bin/activate
pip install uv
uv pip install -r pyproject.toml
./scripts/run_tests.sh
```

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
                \    .//../////  /////////////--\\\/////||             
               |  \-/////\\-///    ////////---\--|\/// /\\    
               /       ///\\-///      /////////\\\/-|//                
                        /////\||          ////////-/\||\               
                        |||/\\\\      |\\-- /////-///|||               
                \\-//   \||\\\      --\\\\----./|\\\||||               
                //\//// \\\||       ..-\\|\--//-\\\\\|||               
                ////|//\\\\/|      //../////c\--\\\-/||                 
                  ///:/\\\\||       //////////|\\\\?\\\                 
                   ///:\\\|||         //////////----\\\                  
                   \|//\\/\\\    //         ///-\\\-\\                    
                \\/ //|\/\\\    \-// \--\--.\//-\\\                        
                |/\\////\\\    //-\  \\\\:|\                               
                //\\ ///\\     //\\///./-\|//                                      
                                    ////...//                                      
                                     /// /-\                                 
                                      //                                     
```
