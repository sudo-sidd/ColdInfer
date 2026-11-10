"""
export_chatbot.py — INT8 quantization + magnitude pruning + C++ header export
for the ColdInfer LSTM chatbot.

Pipeline:
  1. Load trained FP32 checkpoint
  2. Calibrate INT8 scale/zero-point per tensor (post-training quantization)
  3. Apply magnitude pruning (zero out lowest-abs weights below threshold)
  4. Export all tensors as C++ headers for the bare-metal inference engine

Usage:
    uv run python/export_chatbot.py --checkpoint weights/chatbot_best.pt --prune 0.5
    uv run python/export_chatbot.py --dry-run   # show stats without writing files
"""

import argparse
import os
import struct

import numpy as np
import torch

from vocab import CharVocab
from train_chatbot import TinyLSTM


# ---------------------------------------------------------------------------
# INT8 Post-Training Quantization (symmetric, per-tensor)
# ---------------------------------------------------------------------------
def quantize_tensor_int8(tensor: np.ndarray):
    """
    Symmetric per-tensor INT8 quantization.
    Returns (q_data: int8 ndarray, scale: float32)
    Dequantize via: float_val = q_data * scale
    """
    max_abs = np.max(np.abs(tensor))
    if max_abs == 0.0:
        return np.zeros_like(tensor, dtype=np.int8), 1.0

    scale = max_abs / 127.0
    q = np.clip(np.round(tensor / scale), -127, 127).astype(np.int8)
    return q, float(scale)


def dequantize_int8(q_data: np.ndarray, scale: float) -> np.ndarray:
    return q_data.astype(np.float32) * scale


def quantization_error(original: np.ndarray, q_data: np.ndarray, scale: float) -> float:
    """Mean absolute error between original FP32 and dequantized INT8."""
    recon = dequantize_int8(q_data, scale)
    return float(np.mean(np.abs(original - recon)))


# ---------------------------------------------------------------------------
# Magnitude pruning
# ---------------------------------------------------------------------------
def prune_tensor(tensor: np.ndarray, threshold_ratio: float) -> tuple[np.ndarray, float]:
    """
    Zero out all weights whose absolute value is below the given percentile threshold.
    threshold_ratio: 0.0 = no pruning, 0.5 = prune bottom 50% by magnitude.
    Returns (pruned tensor, sparsity fraction achieved).
    """
    if threshold_ratio <= 0.0:
        return tensor, 0.0
    threshold = float(np.percentile(np.abs(tensor), threshold_ratio * 100.0))
    mask = np.abs(tensor) >= threshold
    pruned = tensor * mask
    sparsity = 1.0 - float(np.count_nonzero(pruned)) / pruned.size
    return pruned, sparsity


# ---------------------------------------------------------------------------
# C++ header writers
# ---------------------------------------------------------------------------
def write_int8_array_2d(f, name: str, data: np.ndarray):
    """Write a 2D int8 array as a C++ const int8_t."""
    rows, cols = data.shape
    f.write(f"const int8_t {name}[{rows}][{cols}] = {{\n")
    for r in range(rows):
        row = ", ".join(str(int(v)) for v in data[r])
        f.write(f"  {{{row}}},\n")
    f.write("};\n\n")


def write_int8_array_1d(f, name: str, data: np.ndarray):
    """Write a 1D int8 array as a C++ const int8_t."""
    vals = ", ".join(str(int(v)) for v in data)
    f.write(f"const int8_t {name}[{len(data)}] = {{{vals}}};\n\n")


def write_float_array_1d(f, name: str, data: np.ndarray):
    """Write a 1D float32 array (for scales and biases)."""
    vals = ", ".join(f"{float(v):.8e}f" for v in data)
    f.write(f"const float {name}[{len(data)}] = {{{vals}}};\n\n")


def write_float_scalar(f, name: str, val: float):
    f.write(f"const float {name} = {val:.8e}f;\n\n")


def write_int_scalar(f, name: str, val: int):
    f.write(f"constexpr int {name} = {val};\n")


# ---------------------------------------------------------------------------
# LSTM weight layout
# PyTorch LSTM stores weights as combined gate matrices:
#   weight_ih: (4*hidden, input)   — input-hidden gates [i, f, g, o]
#   weight_hh: (4*hidden, hidden)  — hidden-hidden gates
#   bias_ih, bias_hh: (4*hidden,)
# We split and export each gate separately for cleaner C++ indexing.
# ---------------------------------------------------------------------------
GATE_NAMES = ["i", "f", "g", "o"]  # input, forget, cell, output


def split_lstm_gates(weight: np.ndarray, hidden_size: int):
    """Split a (4*H, *) weight matrix into 4 gate matrices of shape (H, *)."""
    return np.split(weight, 4, axis=0)


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------
def export(args):
    os.makedirs("weights", exist_ok=True)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load vocab
    vocab = CharVocab.load("weights/chatbot_vocab.json")
    print(f"[export] Vocab size: {vocab.size}")

    # Load model
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_state = state["model"] if "model" in state else state

    # Infer architecture from weights
    embed_weight = model_state["embedding.weight"].numpy()
    vocab_size, embed_dim = embed_weight.shape
    hidden_size = model_state["lstm1.weight_hh_l0"].shape[1]
    print(f"[export] Model: vocab={vocab_size}, embed={embed_dim}, hidden={hidden_size}")

    if args.dry_run:
        print("[export] Dry run — no files written.")

    # -----------------------------------------------------------------------
    # Collect all tensors, apply pruning + quantization
    # -----------------------------------------------------------------------
    tensors = {}
    stats = []

    def process(name: str, fp32: np.ndarray, prune_ratio: float = 0.0):
        """Prune → quantize → record stats."""
        pruned, sparsity = prune_tensor(fp32, prune_ratio)
        q_data, scale = quantize_tensor_int8(pruned)
        err = quantization_error(pruned, q_data, scale)
        stats.append({
            "name": name,
            "shape": list(fp32.shape),
            "params": fp32.size,
            "sparsity": sparsity,
            "scale": scale,
            "mae": err,
        })
        return q_data, scale

    prune = args.prune

    # Embedding (don't prune — it's already small and lookup accuracy matters)
    emb_q, emb_scale = process("embedding", embed_weight, prune_ratio=0.0)
    tensors["embedding"] = (emb_q, emb_scale)

    # LSTM 1
    wih1 = model_state["lstm1.weight_ih_l0"].numpy()
    whh1 = model_state["lstm1.weight_hh_l0"].numpy()
    bih1 = model_state["lstm1.bias_ih_l0"].numpy()
    bhh1 = model_state["lstm1.bias_hh_l0"].numpy()
    bias1 = bih1 + bhh1  # merge biases into one (standard practice)

    for gate_idx, gate in enumerate(GATE_NAMES):
        gate_wih = wih1[gate_idx * hidden_size : (gate_idx + 1) * hidden_size]
        gate_whh = whh1[gate_idx * hidden_size : (gate_idx + 1) * hidden_size]
        q_ih, s_ih = process(f"lstm1_wih_{gate}", gate_wih, prune)
        q_hh, s_hh = process(f"lstm1_whh_{gate}", gate_whh, prune)
        tensors[f"lstm1_wih_{gate}"] = (q_ih, s_ih)
        tensors[f"lstm1_whh_{gate}"] = (q_hh, s_hh)

    bias1_gates = np.split(bias1, 4)
    for gate_idx, gate in enumerate(GATE_NAMES):
        tensors[f"lstm1_bias_{gate}"] = bias1_gates[gate_idx]

    # LSTM 2
    wih2 = model_state["lstm2.weight_ih_l0"].numpy()
    whh2 = model_state["lstm2.weight_hh_l0"].numpy()
    bias2 = model_state["lstm2.bias_ih_l0"].numpy() + model_state["lstm2.bias_hh_l0"].numpy()

    for gate_idx, gate in enumerate(GATE_NAMES):
        gate_wih = wih2[gate_idx * hidden_size : (gate_idx + 1) * hidden_size]
        gate_whh = whh2[gate_idx * hidden_size : (gate_idx + 1) * hidden_size]
        q_ih, s_ih = process(f"lstm2_wih_{gate}", gate_wih, prune)
        q_hh, s_hh = process(f"lstm2_whh_{gate}", gate_whh, prune)
        tensors[f"lstm2_wih_{gate}"] = (q_ih, s_ih)
        tensors[f"lstm2_whh_{gate}"] = (q_hh, s_hh)

    bias2_gates = np.split(bias2, 4)
    for gate_idx, gate in enumerate(GATE_NAMES):
        tensors[f"lstm2_bias_{gate}"] = bias2_gates[gate_idx]

    # Output projection
    fc_weight = model_state["fc_out.weight"].numpy()   # (vocab_size, hidden)
    fc_bias   = model_state["fc_out.bias"].numpy()     # (vocab_size,)
    q_fc, s_fc = process("fc_out_weight", fc_weight, prune)
    tensors["fc_out_weight"] = (q_fc, s_fc)
    tensors["fc_out_bias"] = fc_bias

    # -----------------------------------------------------------------------
    # Print compression stats
    # -----------------------------------------------------------------------
    total_params = sum(s["params"] for s in stats)
    avg_sparsity = np.mean([s["sparsity"] for s in stats])
    fp32_kb = total_params * 4 / 1024
    int8_kb  = total_params * 1 / 1024 * (1 - avg_sparsity)

    print(f"\n{'='*60}")
    print(f"  Compression Report")
    print(f"{'='*60}")
    print(f"  Total parameters : {total_params:,}")
    print(f"  FP32 size        : {fp32_kb:.1f} KB")
    print(f"  INT8 size (est.) : {int8_kb:.1f} KB (incl. {avg_sparsity*100:.0f}% sparsity)")
    print(f"  Compression ratio: {fp32_kb/max(int8_kb,0.01):.1f}x")
    print(f"{'='*60}")
    print(f"  {'Tensor':<30} {'Shape':<18} {'Sparsity':>9} {'Scale':>12} {'MAE':>10}")
    print(f"  {'-'*80}")
    for s in stats:
        shape_str = "x".join(str(d) for d in s["shape"])
        print(
            f"  {s['name']:<30} {shape_str:<18} {s['sparsity']*100:>8.1f}% "
            f"{s['scale']:>12.4e} {s['mae']:>10.4e}"
        )

    if args.dry_run:
        return

    # -----------------------------------------------------------------------
    # Write C++ header: chatbot_weights.h
    # -----------------------------------------------------------------------
    header_path = os.path.join(out_dir, "chatbot_weights.h")
    with open(header_path, "w") as f:
        f.write("#pragma once\n\n")
        f.write("// ColdInfer LSTM Chatbot — INT8 quantized weights\n")
        f.write(f"// Generated from: {args.checkpoint}\n")
        f.write(f"// Prune ratio: {args.prune}\n\n")
        f.write(f"// Architecture\n")
        write_int_scalar(f, "CHATBOT_VOCAB_SIZE",  vocab_size)
        write_int_scalar(f, "CHATBOT_EMBED_DIM",   embed_dim)
        write_int_scalar(f, "CHATBOT_HIDDEN_SIZE", hidden_size)
        f.write("\n")

        # Embedding
        f.write("// Embedding: (vocab_size, embed_dim) — INT8\n")
        write_int8_array_2d(f, "emb_weight", tensors["embedding"][0])
        write_float_scalar(f, "emb_scale", tensors["embedding"][1])

        # LSTM layers
        for layer_idx in [1, 2]:
            f.write(f"// LSTM Layer {layer_idx}\n")
            for gate in GATE_NAMES:
                q_ih, s_ih = tensors[f"lstm{layer_idx}_wih_{gate}"]
                q_hh, s_hh = tensors[f"lstm{layer_idx}_whh_{gate}"]
                write_int8_array_2d(f, f"lstm{layer_idx}_wih_{gate}", q_ih)
                write_float_scalar(f, f"lstm{layer_idx}_wih_{gate}_scale", s_ih)
                write_int8_array_2d(f, f"lstm{layer_idx}_whh_{gate}", q_hh)
                write_float_scalar(f, f"lstm{layer_idx}_whh_{gate}_scale", s_hh)
            for gate in GATE_NAMES:
                bias = tensors[f"lstm{layer_idx}_bias_{gate}"]
                write_float_array_1d(f, f"lstm{layer_idx}_bias_{gate}", bias)

        # Output projection
        f.write("// Output projection\n")
        q_fc, s_fc = tensors["fc_out_weight"]
        write_int8_array_2d(f, "fc_out_weight", q_fc)
        write_float_scalar(f, "fc_out_scale", s_fc)
        write_float_array_1d(f, "fc_out_bias", tensors["fc_out_bias"])

    print(f"\n[export] Wrote {header_path}")

    # Also write vocab header
    vocab_header = os.path.join(out_dir, "chatbot_vocab.h")
    vocab.export_c_header(vocab_header)
    print(f"[export] Wrote {vocab_header}")

    # Write a machine-readable compression summary JSON for the webapp
    import json
    summary = {
        "total_params": total_params,
        "fp32_kb": round(fp32_kb, 2),
        "int8_kb": round(int8_kb, 2),
        "compression_ratio": round(fp32_kb / max(int8_kb, 0.01), 2),
        "avg_sparsity": round(avg_sparsity, 4),
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "hidden_size": hidden_size,
        "prune_ratio": args.prune,
        "layers": stats,
    }
    summary_path = os.path.join(out_dir, "compression_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[export] Wrote {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ColdInfer LSTM chatbot to INT8 C++ headers")
    parser.add_argument("--checkpoint", type=str, default="weights/chatbot_best.pt",
                        help="Path to PyTorch checkpoint")
    parser.add_argument("--prune",  type=float, default=0.5,
                        help="Magnitude pruning threshold ratio (0=none, 0.5=prune bottom 50%%)")
    parser.add_argument("--out-dir", type=str, default="weights",
                        help="Output directory for C++ headers")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing any files")
    args = parser.parse_args()
    export(args)
