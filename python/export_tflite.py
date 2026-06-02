"""
Export trained PyTorch MLP → .tflite flatbuffer → C array header for TFLM.

Usage:
    python python/export_tflite.py --model mnist
    python python/export_tflite.py --model kws

Outputs (into weights/):
    <MODEL_NAME>_model.tflite
    <MODEL_NAME>_tflm_model.h   ← uint8_t C array for tflm_bench.cpp
"""

import torch
import torch.nn as nn
import numpy as np
import argparse, os
import tensorflow as tf

class MNIST_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128); self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64);  self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = x.view(-1, 784)
        return self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(x)))))

class KWS_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1000, 128); self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64);   self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = x.view(-1, 1000)
        return self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(x)))))

def pytorch_to_tflite(model, input_shape, prefix):
    keras_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64,  activation='relu'),
        tf.keras.layers.Dense(10),
    ])
    with torch.no_grad():
        keras_model.layers[0].set_weights([model.fc1.weight.numpy().T, model.fc1.bias.numpy()])
        keras_model.layers[1].set_weights([model.fc2.weight.numpy().T, model.fc2.bias.numpy()])
        keras_model.layers[2].set_weights([model.fc3.weight.numpy().T, model.fc3.bias.numpy()])

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_bytes = bytearray(converter.convert())

    import struct

    # Patch 1: Change dynamic batch dim (-1) to static (1) for TFLM.
    dyn_pattern  = struct.pack('<ii', -1, input_shape)
    static_patch = struct.pack('<ii',  1, input_shape)
    idx = tflite_bytes.find(dyn_pattern)
    if idx != -1:
        tflite_bytes[idx:idx+8] = static_patch
        print(f"  Patched batch dim → [1, {input_shape}] at offset {idx}")

    # Patch 2: Set input tensor buffer_idx to 0.
    # Keras converter sets it to 1 (non-zero), which makes TFLM treat the
    # input as a pre-loaded constant and skip arena allocation → data.f = null.
    # Buffer index 0 signals "no pre-loaded data" → TFLM allocates arena space.
    # The buffer field for tensor 0 is a uint32 LE = 0x01000000 immediately
    # following the tensor's table offset. We find it by locating the input
    # tensor name and scanning for the buffer field.
    tensor_name = b"serving_default_keras_tensor:0\x00"
    name_idx = tflite_bytes.find(tensor_name)
    if name_idx == -1:
        # fallback: try original name
        tensor_name = b"keras_tensor:0\x00"
        name_idx = tflite_bytes.find(tensor_name)
    if name_idx != -1:
        # Buffer index field (uint32=1) appears near the tensor name in the flatbuffer.
        # Search backward from the name offset for the pattern \x01\x00\x00\x00
        # within a 256-byte window (flatbuffer tables are compact).
        window_start = max(0, name_idx - 256)
        window = tflite_bytes[window_start:name_idx]
        buf_pat = struct.pack('<I', 1)  # buffer_idx = 1
        for i in range(len(window) - 4, -1, -1):
            if window[i:i+4] == buf_pat:
                abs_pos = window_start + i
                tflite_bytes[abs_pos:abs_pos+4] = struct.pack('<I', 0)
                print(f"  Patched input buffer_idx 1→0 at offset {abs_pos}")
                break

    tflite_bytes = bytes(tflite_bytes)

    tflite_path = f"{prefix}_model.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_bytes)
    print(f"Saved: {tflite_path}  ({len(tflite_bytes):,} bytes)")

    # Convert to C array for tflm_bench.cpp
    var_name    = os.path.basename(prefix) + "_tflm_model"
    header_path = f"{prefix}_tflm_model.h"
    with open(header_path, 'w') as f:
        f.write("#pragma once\n#include <cstdint>\n\n")
        hex_vals = [f"0x{b:02x}" for b in tflite_bytes]
        rows = [", ".join(hex_vals[i:i+12]) for i in range(0, len(hex_vals), 12)]
        f.write(f"const uint8_t {var_name}[] = {{\n  " + ",\n  ".join(rows) + "\n};\n")
        f.write(f"const unsigned int {var_name}_len = {len(tflite_bytes)};\n")
    print(f"Saved: {header_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mnist", "kws"], required=True)
    args = parser.parse_args()
    os.makedirs("weights", exist_ok=True)

    if args.model == "mnist":
        model = MNIST_MLP()
        model.load_state_dict(torch.load("weights/model_weights.MNIST.pt", weights_only=True))
        pytorch_to_tflite(model, 784, "weights/MNIST_MLP_FP32")
    else:
        model = KWS_MLP()
        model.load_state_dict(torch.load("weights/model_weights.KWS.pt", weights_only=True))
        pytorch_to_tflite(model, 1000, "weights/Keyword_Spotting_MLP_FP32")
