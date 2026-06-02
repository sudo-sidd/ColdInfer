# #!/bin/bash

# # Exit immediately if a command exits with a non-zero status
# set -e

# # Move string execution to root dir of the engine
# cd "$(dirname "$0")/.."

# LOG_FILE="performance_log.csv"

# # Initialize CSV if it doesn't exist
# if [ ! -f $LOG_FILE ]; then
#     echo "Date,Model_Name,Inference_Time_us,Text_Size_bytes,BSS_Size_bytes(RAM),Total_Dec_Size_bytes" > $LOG_FILE
# fi

# # Make sure weights dir exists
# mkdir -p weights

# # Define the models and their configurations
# declare -A MODELS
# MODELS=(
#     ["MNIST_MLP_FP32"]="mnist 784"
#     ["Keyword_Spotting_MLP_FP32"]="kws 1000"
# )

# for MODEL_NAME in "${!MODELS[@]}"; do
#     CFG=(${MODELS[$MODEL_NAME]})
#     PY_MODEL_ARG=${CFG[0]}
#     INPUT_SIZE=${CFG[1]}
    
#     echo -e "\n============================================="
#     echo "1. Training PyTorch Model: $MODEL_NAME"
#     echo "============================================="
    
#     # Path where this model's weights will be saved
#     TARGET_WEIGHTS="weights/${MODEL_NAME}_model_weights.h"
#     TARGET_IMAGE="weights/${MODEL_NAME}_test_image_data.h"
    
#     if [ -f "$TARGET_WEIGHTS" ]; then
#         echo "Valid weights file ($TARGET_WEIGHTS) detected. Skipping training for $MODEL_NAME."
#     else
#         echo "No existing weights found. Initiating training pipeline..."
#         python3 python/train_export.py --model $PY_MODEL_ARG
        
#         # Move the newly exported weight files from the root to the weights directory with the model name prefix
#         mv model_weights.h "$TARGET_WEIGHTS"
#         mv test_image_data.h "$TARGET_IMAGE"
#     fi

#     # The C++ script looks for EXACTLY "model_weights.h" and "test_image_data.h" via #include
#     echo "Staging weights for C++ Compiler..."
#     cp "$TARGET_WEIGHTS" weights/model_weights.h
#     cp "$TARGET_IMAGE" weights/test_image_data.h

#     echo -e "\n============================================="
#     echo "2. Compiling C++ Inference Engine for $MODEL_NAME"
#     echo "============================================="
#     # Compile with the specific input size for static allocation
#     # Added -I./weights to tell the compiler to look in the weights folder for the headers
#     g++ -O3 -std=c++17 -DINPUT_SIZE=$INPUT_SIZE -I./weights cpp/infer.cpp -o infer
#     echo "Compilation Successful."

#     echo -e "\n============================================="
#     echo "3. Running C++ Inference ($MODEL_NAME)"
#     echo "============================================="
#     INFER_OUT=$(./infer)
#     echo "$INFER_OUT"

#     echo -e "\n============================================="
#     echo "4. Generating Performance Stats & Logging"
#     echo "============================================="

#     # Fixed extraction text format logic (matches "INFERENCE EXECUTION TIME:")
#     INF_TIME=$(echo "$INFER_OUT" | grep -i "INFERENCE EXECUTION TIME" | grep -o -E '[0-9]+' | head -1)

#     SIZE_OUT=$(size ./infer | tail -n 1)
#     TEXT_SIZE=$(echo $SIZE_OUT | awk '{print $1}')
#     BSS_SIZE=$(echo $SIZE_OUT | awk '{print $3}')
#     DEC_SIZE=$(echo $SIZE_OUT | awk '{print $4}')

#     CURRENT_DATE=$(date '+%Y-%m-%d %H:%M:%S')

#     echo "$CURRENT_DATE,$MODEL_NAME,$INF_TIME,$TEXT_SIZE,$BSS_SIZE,$DEC_SIZE" >> $LOG_FILE

#     echo "Metrics logged to $LOG_FILE for $MODEL_NAME:"
#     echo " - Inference Time: ${INF_TIME} us"
#     echo " - Code Size (Text): ${TEXT_SIZE} bytes"
#     echo " - Static RAM (BSS): ${BSS_SIZE} bytes"
# done

#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Move string execution to root dir of the engine
cd "$(dirname "$0")/.."

# Always use the venv Python so packages are available regardless of whether
# the caller has activated the environment in their shell.
PYTHON="$(pwd)/.venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: .venv not found. Run 'uv sync' first."
    exit 1
fi

LOG_FILE="performance_log.csv"

# Initialize CSV — 22-column schema
if [ ! -f $LOG_FILE ]; then
    echo "Date,Model_Name,Custom_Infer_us,PT_Infer_us,TFLite_Infer_us,TFLM_Infer_us,Custom_RSS_KB,PT_RSS_KB,TFLite_RSS_KB,TFLM_RSS_KB,Custom_CPU_User_us,PT_CPU_User_us,TFLite_CPU_User_us,TFLM_CPU_User_us,Custom_CPU_Sys_us,PT_CPU_Sys_us,TFLite_CPU_Sys_us,TFLM_CPU_Sys_us,TFLM_Arena_KB,Binary_Text_B,Binary_BSS_B,Binary_Total_B" > $LOG_FILE
fi

TFLM_LIB="$(pwd)/third_party/tflm_build/libtensorflow-microlite.a"
TFLM_INC="$(pwd)/third_party/tflm_build/include"
if [ -f "$TFLM_LIB" ]; then
    TFLM_AVAILABLE=1; echo "TFLM library found — TFLM benchmarks enabled."
else
    TFLM_AVAILABLE=0; echo "TFLM not built. Run ./scripts/setup_tflm.sh once to enable."
fi

mkdir -p weights build

declare -A MODELS
MODELS=(
    ["MNIST_MLP_FP32"]="mnist 784"
    ["Keyword_Spotting_MLP_FP32"]="kws 1000"
)

for MODEL_NAME in "${!MODELS[@]}"; do
    CFG=(${MODELS[$MODEL_NAME]})
    PY_MODEL_ARG=${CFG[0]}
    INPUT_SIZE=${CFG[1]}
    
    echo -e "\n============================================="
    echo "1. Training PyTorch Model: $MODEL_NAME"
    echo "============================================="
    
    TARGET_WEIGHTS="weights/${MODEL_NAME}_model_weights.h"
    TARGET_IMAGE="weights/${MODEL_NAME}_test_image_data.h"
    
    PT_FILE="weights/model_weights.MNIST.pt"
    if [ "$PY_MODEL_ARG" == "kws" ]; then
        PT_FILE="weights/model_weights.KWS.pt"
    fi
    
    if [ ! -f "$TARGET_WEIGHTS" ] || [ ! -f "$PT_FILE" ]; then
        echo "No existing weights or model checkpoint found. Initiating training pipeline..."
        $PYTHON python/train_export.py --model $PY_MODEL_ARG
        mv model_weights.h "$TARGET_WEIGHTS"
        mv test_image_data.h "$TARGET_IMAGE"
    fi

    cp "$TARGET_WEIGHTS" weights/model_weights.h
    cp "$TARGET_IMAGE" weights/test_image_data.h

    echo -e "\n============================================="
    echo "2. Running Baselines (PyTorch & TFLite)"
    echo "============================================="
    BASELINE_OUT=$($PYTHON python/benchmark_baselines.py --model $PY_MODEL_ARG 2>/dev/null)
    PT_TIME=$(echo "$BASELINE_OUT"     | grep "^PYTORCH_US:"          | cut -d':' -f2)
    PT_RSS=$(echo "$BASELINE_OUT"      | grep "^PYTORCH_RSS_KB:"      | cut -d':' -f2)
    PT_CPU_USER=$(echo "$BASELINE_OUT" | grep "^PYTORCH_CPU_USER_US:" | cut -d':' -f2)
    PT_CPU_SYS=$(echo "$BASELINE_OUT"  | grep "^PYTORCH_CPU_SYS_US:"  | cut -d':' -f2)
    TF_TIME=$(echo "$BASELINE_OUT"     | grep "^TFLITE_US:"           | cut -d':' -f2)
    TF_RSS=$(echo "$BASELINE_OUT"      | grep "^TFLITE_RSS_KB:"       | cut -d':' -f2)
    TF_CPU_USER=$(echo "$BASELINE_OUT" | grep "^TFLITE_CPU_USER_US:"  | cut -d':' -f2)
    TF_CPU_SYS=$(echo "$BASELINE_OUT"  | grep "^TFLITE_CPU_SYS_US:"   | cut -d':' -f2)
    echo "PyTorch : ${PT_TIME} us | RSS ${PT_RSS} KB | CPU user ${PT_CPU_USER} us sys ${PT_CPU_SYS} us"
    echo "TFLite  : ${TF_TIME} us | RSS ${TF_RSS} KB | CPU user ${TF_CPU_USER} us sys ${TF_CPU_SYS} us"

    echo -e "\n============================================="
    echo "3. Compiling C++ Inference Engine for $MODEL_NAME"
    echo "============================================="
    g++ -O3 -std=c++17 -DINPUT_SIZE=$INPUT_SIZE -I./weights cpp/infer.cpp -o build/infer
    
    echo -e "\n============================================="
    echo "4. Running Custom C++ Inference ($MODEL_NAME)"
    echo "============================================="
    INFER_OUT=$(build/infer)
    echo "$INFER_OUT"

    echo -e "\n============================================="
    echo "5. Performance Stats & Logging"
    echo "============================================="
    INF_TIME=$(echo "$INFER_OUT"     | grep "^INFERENCE_TIME_US:" | cut -d':' -f2)
    INF_RSS=$(echo "$INFER_OUT"      | grep "^PEAK_RSS_KB:"       | cut -d':' -f2)
    INF_CPU_USER=$(echo "$INFER_OUT" | grep "^CPU_USER_US:"       | cut -d':' -f2)
    INF_CPU_SYS=$(echo "$INFER_OUT"  | grep "^CPU_SYS_US:"        | cut -d':' -f2)
    echo "ColdInfer: ${INF_TIME} us | RSS ${INF_RSS} KB | CPU user ${INF_CPU_USER} us sys ${INF_CPU_SYS} us"

    SIZE_OUT=$(size build/infer | tail -n 1)
    TEXT_SIZE=$(echo $SIZE_OUT | awk '{print $1}')
    BSS_SIZE=$(echo $SIZE_OUT  | awk '{print $3}')
    DEC_SIZE=$(echo $SIZE_OUT  | awk '{print $4}')

    # ── 6. TFLM (optional) ────────────────────────────────────────────────
    TFLM_TIME="" TFLM_RSS="" TFLM_CPU_USER="" TFLM_CPU_SYS="" TFLM_ARENA=""
    if [ "$TFLM_AVAILABLE" -eq 1 ]; then
        echo -e "\n============================================="
        echo "6. TFLM Benchmark for $MODEL_NAME"
        echo "============================================="
        TFLM_MODEL_H="${MODEL_NAME}_tflm_model.h"
        TFLM_MODEL_VAR="${MODEL_NAME}_tflm_model"
        if [ ! -f "weights/$TFLM_MODEL_H" ]; then
            echo "Exporting .tflite + C array..."
            $PYTHON python/export_tflite.py --model $PY_MODEL_ARG 2>/dev/null
        fi
        g++ -O3 -std=c++17 \
            -DINPUT_SIZE=$INPUT_SIZE \
            "-DMODEL_HEADER=\"${TFLM_MODEL_H}\"" \
            "-DMODEL_VAR=${TFLM_MODEL_VAR}" \
            -I./weights -I"$TFLM_INC" \
            cpp/tflm_bench.cpp \
            -L"$(dirname $TFLM_LIB)" -ltensorflow-microlite \
            -o build/tflm_bench
        TFLM_OUT=$(build/tflm_bench)
        TFLM_TIME=$(echo "$TFLM_OUT"     | grep "^TFLM_TIME_US:"     | cut -d':' -f2)
        TFLM_RSS=$(echo "$TFLM_OUT"      | grep "^TFLM_PEAK_RSS_KB:" | cut -d':' -f2)
        TFLM_CPU_USER=$(echo "$TFLM_OUT" | grep "^TFLM_CPU_USER_US:" | cut -d':' -f2)
        TFLM_CPU_SYS=$(echo "$TFLM_OUT"  | grep "^TFLM_CPU_SYS_US:"  | cut -d':' -f2)
        TFLM_ARENA=$(echo "$TFLM_OUT"    | grep "^TFLM_ARENA_KB:"    | cut -d':' -f2)
        echo "TFLM    : ${TFLM_TIME} us | RSS ${TFLM_RSS} KB | CPU user ${TFLM_CPU_USER} us | Arena ${TFLM_ARENA} KB"
    fi

    CURRENT_DATE=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$CURRENT_DATE,$MODEL_NAME,$INF_TIME,$PT_TIME,$TF_TIME,$TFLM_TIME,$INF_RSS,$PT_RSS,$TF_RSS,$TFLM_RSS,$INF_CPU_USER,$PT_CPU_USER,$TF_CPU_USER,$TFLM_CPU_USER,$INF_CPU_SYS,$PT_CPU_SYS,$TF_CPU_SYS,$TFLM_CPU_SYS,$TFLM_ARENA,$TEXT_SIZE,$BSS_SIZE,$DEC_SIZE" >> $LOG_FILE
    echo "Metrics logged to $LOG_FILE"
done

echo -e "\n============================================="
echo "7. Updating README.md Benchmark Tables"
echo "============================================="
$PYTHON scripts/update_readme_stats.py