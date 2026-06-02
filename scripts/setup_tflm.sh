#!/bin/bash
# One-time setup: clone TFLite Micro and build the static library for Linux.
# Safe to re-run — skips if already built.
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/third_party/tflite-micro"
LIB_DIR="$ROOT_DIR/third_party/tflm_build"
LIB_FILE="$LIB_DIR/libtensorflow-microlite.a"
INC_DIR="$LIB_DIR/include"

if [ -f "$LIB_FILE" ]; then
    echo "TFLM already built at $LIB_FILE — skipping."
    exit 0
fi

echo "=== Setting up TensorFlow Lite Micro ==="

# ── 1. Clone (shallow) ─────────────────────────────────────────────────────
if [ ! -d "$SRC_DIR/.git" ]; then
    echo "Cloning tflite-micro (shallow)..."
    git clone --depth=1 https://github.com/tensorflow/tflite-micro.git "$SRC_DIR"
else
    echo "Source already present at $SRC_DIR"
fi

# ── 2. Build microlite static library for Linux ────────────────────────────
echo "Building TFLM static library (this may take a few minutes)..."
cd "$SRC_DIR"

make -j"$(nproc)" \
    -f tensorflow/lite/micro/tools/make/Makefile \
    microlite \
    TARGET=linux \
    OPTIMIZED_KERNEL_DIR= \
    2>&1 | tail -20

# ── 3. Copy library + headers to a clean output directory ──────────────────
GEN_LIB=$(find tensorflow/lite/micro/tools/make/gen -name "libtensorflow-microlite.a" 2>/dev/null | head -1)
if [ -z "$GEN_LIB" ]; then
    echo "ERROR: Library not found after build. Check make output above."
    exit 1
fi

mkdir -p "$LIB_DIR" "$INC_DIR"
cp "$GEN_LIB" "$LIB_FILE"

# Copy only the headers needed by tflm_bench.cpp
rsync -a --include="*/" --include="*.h" --exclude="*" \
    tensorflow/lite/     "$INC_DIR/tensorflow/lite/"
rsync -a --include="*/" --include="*.h" --exclude="*" \
    tensorflow/core/     "$INC_DIR/tensorflow/core/"
rsync -a --include="*/" --include="*.h" --exclude="*" \
    third_party/flatbuffers/include/ "$INC_DIR/"
rsync -a --include="*/" --include="*.h" --exclude="*" \
    third_party/gemmlowp/ "$INC_DIR/"

echo ""
echo "=== TFLM build complete ==="
echo "  Library : $LIB_FILE"
echo "  Headers : $INC_DIR"
echo ""
echo "Run ./scripts/run_tests.sh — TFLM column will now be populated."
