#!/bin/bash
# One-time TFLM setup: clone tflite-micro and build the Linux static library.
# Safe to re-run — skips if already built.
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/third_party/tflite-micro"
LIB_DIR="$ROOT_DIR/third_party/tflm_build"
LIB_FILE="$LIB_DIR/libtensorflow-microlite.a"

if [ -f "$LIB_FILE" ]; then
    echo "TFLM already built at $LIB_FILE — skipping."
    exit 0
fi

echo "=== Setting up TensorFlow Lite Micro ==="

# 1. Shallow clone
if [ ! -d "$SRC_DIR/.git" ]; then
    echo "Cloning tflite-micro (shallow)..."
    git clone --depth=1 https://github.com/tensorflow/tflite-micro.git "$SRC_DIR"
else
    echo "Source already present at $SRC_DIR"
fi

# 2. Build microlite static library
echo "Building TFLM static library (takes a few minutes)..."
cd "$SRC_DIR"
make -j"$(nproc)" \
    -f tensorflow/lite/micro/tools/make/Makefile \
    microlite \
    TARGET=linux \
    OPTIMIZED_KERNEL_DIR= \
    2>&1 | tail -5

# 3. Locate the generated library
GEN_LIB=$(find gen -name "libtensorflow-microlite.a" 2>/dev/null | head -1)
if [ -z "$GEN_LIB" ]; then
    echo "ERROR: Library not found after build."
    exit 1
fi

# 4. Copy lib + headers to clean output directory
INC_DIR="$LIB_DIR/include"
mkdir -p "$LIB_DIR" "$INC_DIR"
cp "$GEN_LIB" "$LIB_FILE"

rsync -a --include="*/" --include="*.h" --exclude="*" \
    tensorflow/lite/      "$INC_DIR/tensorflow/lite/"
# FlatBuffers (downloaded by the Makefile into tools/make/downloads/)
FLATBUF_SRC="tensorflow/lite/micro/tools/make/downloads/flatbuffers/include"
if [ -d "$FLATBUF_SRC" ]; then
    cp -r "$FLATBUF_SRC/flatbuffers" "$INC_DIR/"
fi

echo ""
echo "=== TFLM build complete ==="
echo "  Library : $LIB_FILE"
echo "  Headers : $INC_DIR"
echo "Now run: ./scripts/run_tests.sh"
