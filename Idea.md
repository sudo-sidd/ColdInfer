## ***ColdInfer: Custom TinyML Inference Engine (Low-Level Systems)***

### **Feasibility & Core Scope**

Building a custom TinyML inference engine from scratch is entirely feasible for a team of four over one year. In fact, it is an exceptional project choice because it avoids the typical trap of "training a model on a dataset," which offers low research value. Instead, it positions your team in **Systems Programming and AI Tools Engineering**.

You do not need a single GPU for this project. The entire development cycle takes place in standard **C++17/20** and **Python**. You can write, debug, and profile the engine entirely on your local CPU before compiling it for a low-cost target like an ARM Cortex-M microcontroller (e.g., STM32) or an ESP32.

---

### **The Project Goal: What Are You Actually Building?**

The goal is to create a lightweight, dependency-free runtime library (written in pure C/C++) that takes a pre-trained machine learning model (like a Convolutional Neural Network or a small Multi-Layer Perceptron), parses its structure, compresses its weights, and runs forward-pass inference on bare-metal hardware.

To do this, your project will split into two distinct halves:

```
[ PyTorch / ONNX Model ] 
         │
         ▼
 ┌────────────────────────────────────────────────────────┐
 │ 1. THE COMPILER / CONVERTER (Python)                   │
 │   - Parses the model graph                             │
 │   - Quantizes weights (e.g., Float32 -> Int8)          │
 │   - Serializes into a custom flat binary format (.bin) │
 └────────────────────────────────────────────────────────┘
         │
         ▼
 ┌────────────────────────────────────────────────────────┐
 │ 2. THE INFERENCE ENGINE RUNTIME (Pure C++ / Bare-Metal)│
 │   - Allocates a static memory pool (zero dynamic malloc)│
 │   - Reads the custom binary                            │
 │   - Executes optimized math kernels (Gemm, Conv2D, ReLU)│
 └────────────────────────────────────────────────────────┘
         │
         ▼
[ Low-Power Microcontroller Execution ]

```

---

### **Division of Labor (4-Person Team)**

Over a year, your team can break down the engineering architecture into four clear, parallel specializations:

1. **Graph Compiler & Parser Engineer (Python):** Responsible for taking an export format like ONNX or PyTorch `safetensors`, traversing the computational graph, extracting the structural topology, and writing the offline compilation tool that outputs your custom engine's proprietary model format.
2. **Quantization & Compression Engineer (Python/C++):** Focuses on math optimization. This role designs the algorithms that convert floating-point parameters into low-precision integers (INT8, INT4, or even binary/ternary weights) and implements Post-Training Quantization (PTQ) or Quantization-Aware Training (QAT) utilities to minimize accuracy loss.
3. **Low-Level Kernel Developer (C++):** The heart of the engine performance. This engineer writes the actual execution loops for matrix multiplication (GEMM), 2D convolutions, pooling layers, and activation functions. They focus on optimization techniques like loop unrolling, pointer arithmetic, cache alignment, and writing SIMD architecture intrinsics (such as ARM CMSIS-NN or NEON instructions).
4. **Systems & Hardware Integration Engineer (C++/Embedded):** Manages the target runtime. This role ensures the entire engine works with a zero-dynamic-allocation constraint (allocating everything in a single static scratchpad memory buffer), hooks up the physical sensors (like a microphone for wake-word detection or an accelerometer for gesture recognition), and builds the profiling system to measure exact clock cycles and SRAM consumption.

---

### **Where is the Room for New Research? (The Paper)**

To publish in a national journal, you cannot just rewrite a worse version of existing tools. You need a specific, isolated research thesis. Here are three highly publishable research angles your team can choose from:

* **Novel Mixed-Precision Quantization Schemes:** Instead of quantizing the entire model to INT8, your paper could investigate a dynamic heuristic that determines which critical layers must remain at 8-bit to preserve accuracy, while compressing non-critical layers down to INT4 or INT2 to save memory.
* **Layer Fusion Optimization Paths:** Look into memory-bound performance bottlenecks. You can design an engine that fuses operations natively in cache (e.g., performing Conv2D, Bias Addition, and ReLU activation in a single pass over a memory row), drastically reducing read/write cycles to the chip's limited SRAM.
* **Domain-Specific Sparse Kernel Execution:** Create an engine optimized specifically for sparse networks (models where a high percentage of weights are pruned to zero). Standard engines waste cycles multiplying by zero; your paper can introduce a novel compressed sparse row storage format tailored specifically for embedded microprocessors.

---

### **The Existing Landscape: Prior Art & Resources**

To build an engine from scratch, you must study how the industry currently solves these problems. Do not copy their code, but analyze their architectural patterns:

* **TensorFlow Lite for Microcontrollers (TFLM):** The current industry standard. It uses a static arena allocator to prevent heap fragmentation. Study its `LinearMemoryPlanner` source code to understand how it reuses tensor buffers to minimize memory footprints.
* **Apache TVM (MicroTVM):** An open-source machine learning compiler framework. Instead of interpreting a model file at runtime, TVM compiles the model directly into raw C code. This "ahead-of-time" (AOT) compilation pattern is an incredibly strong alternative route for your project.
* **ARM CMSIS-NN:** A collection of highly optimized neural network kernels for ARM Cortex-M processors. This provides the gold standard for how manual assembly or hardware intrinsics can accelerate low-level operations like fixed-point matrix multiplication.

