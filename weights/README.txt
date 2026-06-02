This directory is NOT tracked by git.

What goes here:
  weights/
  ├── MNIST_MLP_FP32_model_weights.h        ← Exported C float array of trained MNIST weights
  ├── MNIST_MLP_FP32_test_image_data.h      ← One MNIST test sample as a C float array
  ├── Keyword_Spotting_MLP_FP32_model_weights.h   ← Exported C float array of trained KWS weights
  ├── Keyword_Spotting_MLP_FP32_test_image_data.h ← One KWS audio sample as a C float array
  ├── model_weights.h                       ← Staging copy used by the C++ compiler (auto-managed)
  └── test_image_data.h                     ← Staging copy used by the C++ compiler (auto-managed)

These files are compiled directly into the C++ inference engine via:
  #include "model_weights.h"
  #include "test_image_data.h"

How to populate:
  Run `uv run bash scripts/run_tests.sh` from the project root.
  The pipeline trains the PyTorch model, exports weights as C headers, and
  places them here automatically. Do NOT commit these files — they are large,
  model-specific, and fully reproducible from source.

  PyTorch checkpoint files (model_weights.MNIST.pt, model_weights.KWS.pt)
  are saved to the project root and are also gitignored.
