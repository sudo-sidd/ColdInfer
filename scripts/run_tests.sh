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

LOG_FILE="performance_log.csv"

# Initialize CSV with new baseline columns if it doesn't exist
if [ ! -f $LOG_FILE ]; then
    echo "Date,Model_Name,Custom_Inference_Time_us,PyTorch_Time_us,TFLite_Time_us,Text_Size_bytes,BSS_Size_bytes(RAM),Total_Dec_Size_bytes" > $LOG_FILE
fi

mkdir -p weights

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
    
    PT_FILE="model_weights.MNIST.pt"
    if [ "$PY_MODEL_ARG" == "kws" ]; then
        PT_FILE="model_weights.KWS.pt"
    fi
    
    if [ ! -f "$TARGET_WEIGHTS" ] || [ ! -f "$PT_FILE" ]; then
        echo "No existing weights or model checkpoint found. Initiating training pipeline..."
        python python/train_export.py --model $PY_MODEL_ARG
        mv model_weights.h "$TARGET_WEIGHTS"
        mv test_image_data.h "$TARGET_IMAGE"
    fi

    cp "$TARGET_WEIGHTS" weights/model_weights.h
    cp "$TARGET_IMAGE" weights/test_image_data.h

    echo -e "\n============================================="
    echo "2. Running Baselines (PyTorch & TFLite)"
    echo "============================================="
    BASELINE_OUT=$(python python/benchmark_baselines.py --model $PY_MODEL_ARG)
    PT_TIME=$(echo "$BASELINE_OUT" | grep "PYTORCH_US" | cut -d':' -f2)
    TF_TIME=$(echo "$BASELINE_OUT" | grep "TFLITE_US" | cut -d':' -f2)
    echo "PyTorch Time: ${PT_TIME} us | TFLite Time: ${TF_TIME} us"

    echo -e "\n============================================="
    echo "3. Compiling C++ Inference Engine for $MODEL_NAME"
    echo "============================================="
    g++ -O3 -std=c++17 -DINPUT_SIZE=$INPUT_SIZE -I./weights cpp/infer.cpp -o infer
    
    echo -e "\n============================================="
    echo "4. Running Custom C++ Inference ($MODEL_NAME)"
    echo "============================================="
    INFER_OUT=$(./infer)
    echo "$INFER_OUT"

    echo -e "\n============================================="
    echo "5. Generating Performance Stats & Logging"
    echo "============================================="
    INF_TIME=$(echo "$INFER_OUT" | grep -i "INFERENCE EXECUTION TIME" | grep -o -E '[0-9]+' | head -1)

    SIZE_OUT=$(size ./infer | tail -n 1)
    TEXT_SIZE=$(echo $SIZE_OUT | awk '{print $1}')
    BSS_SIZE=$(echo $SIZE_OUT | awk '{print $3}')
    DEC_SIZE=$(echo $SIZE_OUT | awk '{print $4}')

    CURRENT_DATE=$(date '+%Y-%m-%d %H:%M:%S')

    echo "$CURRENT_DATE,$MODEL_NAME,$INF_TIME,$PT_TIME,$TF_TIME,$TEXT_SIZE,$BSS_SIZE,$DEC_SIZE" >> $LOG_FILE
    echo "Metrics logged to $LOG_FILE"
done