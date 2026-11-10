"""
engine.py — Subprocess runner for ColdInfer C++ binaries.
Called by the FastAPI backend to run inference and benchmark jobs.
"""

import asyncio
import csv
import json
import os
import resource
import subprocess
import time
import re
from functools import lru_cache
from pathlib import Path
from typing import AsyncIterator, Optional

PROJECT_ROOT = Path(__file__).parent.parent
BUILD_DIR    = PROJECT_ROOT / "build"
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
WEIGHTS_DIR  = PROJECT_ROOT / "weights"
PERF_LOG     = PROJECT_ROOT / "performance_log.csv"


# ---------------------------------------------------------------------------
# Binary paths
# ---------------------------------------------------------------------------
def _binary(name: str) -> str:
    return str(BUILD_DIR / name)


MODEL_CONFIG = {
    "mnist": {
        "binary": "infer_mnist",
        "weights": "MNIST_MLP_FP32_model_weights.h",
        "sample": "MNIST_MLP_FP32_test_image_data.h",
        "labels": [str(i) for i in range(10)],
        "shape": [28, 28],
    },
    "kws": {
        "binary": "infer_kws",
        "weights": "Keyword_Spotting_MLP_FP32_model_weights.h",
        "sample": "Keyword_Spotting_MLP_FP32_test_image_data.h",
        "labels": ["down", "go", "left", "no", "right", "stop", "up", "yes", "unknown", "silence"],
        "shape": [50, 20],
    },
}


@lru_cache(maxsize=1)
def get_yolo_model():
    """Load the small local YOLO model once for repeated dashboard requests."""
    from ultralytics import YOLO

    return YOLO(str(PROJECT_ROOT / "yolo11n.pt"))


def run_yolo_image(image, confidence: float = 0.25) -> dict:
    """Run the local YOLO target used by the ColdInfer dashboard."""
    started = time.perf_counter()
    model = get_yolo_model()
    inference_started = time.perf_counter()
    prediction = model.predict(source=image, conf=confidence, verbose=False, device="cpu")[0]
    inference_ms = (time.perf_counter() - inference_started) * 1000
    detections = []
    for box in prediction.boxes:
        class_index = int(box.cls.item())
        detections.append({
            "label": prediction.names[class_index],
            "confidence": round(float(box.conf.item()), 4),
            "x1": round(float(box.xyxy[0][0].item()), 1),
            "y1": round(float(box.xyxy[0][1].item()), 1),
            "x2": round(float(box.xyxy[0][2].item()), 1),
            "y2": round(float(box.xyxy[0][3].item()), 1),
        })
    return {
        "model": "coldinfer-yolo11n",
        "detections": detections,
        "timings_ms": {
            "preprocess": round(max(0, (inference_started - started) * 1000), 2),
            "inference": round(inference_ms, 2),
            "postprocess": round(max(0, (time.perf_counter() - inference_started) * 1000 - inference_ms), 2),
        },
    }


COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


@lru_cache(maxsize=1)
def get_quantized_yolo_session():
    import onnxruntime as ort

    path = PROJECT_ROOT / "weights" / "yolo11n_int8_160" / "yolo11n_160_dynamic_int8.onnx"
    if not path.exists():
        raise FileNotFoundError(f"Quantized YOLO artifact not found: {path}")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _iou(left, right) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_left = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    area_right = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    return intersection / (area_left + area_right - intersection + 1e-8)


def run_quantized_yolo_image(image, confidence: float = 0.25) -> dict:
    import numpy as np

    started = time.perf_counter()
    width, height = image.size
    resized = image.resize((160, 160))
    tensor = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    preprocess_ms = (time.perf_counter() - started) * 1000
    inference_started = time.perf_counter()
    session = get_quantized_yolo_session()
    raw = session.run(None, {session.get_inputs()[0].name: tensor})[0][0].T
    inference_ms = (time.perf_counter() - inference_started) * 1000
    scale_x, scale_y = width / 160, height / 160
    candidates = []
    for row in raw:
        class_index = int(np.argmax(row[4:]))
        score = float(row[4 + class_index])
        if score < confidence:
            continue
        x, y, box_width, box_height = row[:4]
        box = (
            max(0.0, (x - box_width / 2) * scale_x),
            max(0.0, (y - box_height / 2) * scale_y),
            min(float(width), (x + box_width / 2) * scale_x),
            min(float(height), (y + box_height / 2) * scale_y),
        )
        candidates.append((score, class_index, box))

    detections = []
    for score, class_index, box in sorted(candidates, reverse=True):
        if any(class_index == existing[1] and _iou(box, existing[2]) > 0.45 for existing in detections):
            continue
        detections.append((score, class_index, box))
    postprocess_started = time.perf_counter()
    result = {
        "model": "coldinfer-yolo11n-int8",
        "detections": [
            {
                "label": COCO_NAMES[class_index],
                "confidence": float(round(float(score), 4)),
                "x1": float(round(float(box[0]), 1)),
                "y1": float(round(float(box[1]), 1)),
                "x2": float(round(float(box[2]), 1)),
                "y2": float(round(float(box[3]), 1)),
            }
            for score, class_index, box in detections
        ],
    }
    result["timings_ms"] = {
        "preprocess": round(preprocess_ms, 2),
        "inference": round(inference_ms, 2),
        "postprocess": round((time.perf_counter() - postprocess_started) * 1000, 2),
    }
    return result


def run_yolo_simulation(image, confidence: float = 0.25, arena_kb: int = 520) -> dict:
    """Run YOLO with an embedded-style fixed-buffer accounting envelope.

    The detector kernel is still the host YOLO implementation until a
    Micro-compatible model and convolution operators are added to the ESP32
    firmware. No success-shaped fallback is returned when the configured
    arena cannot hold the fixed tensors.
    """
    if arena_kb < 128:
        return {"success": False, "error": "YOLO simulation arena must be at least 128 KB"}

    # Quantized deployment profile: 160x160 model with INT8 weights and
    # compact detector tensors. The current ONNX Runtime path retains float
    # activations; a full integer ESP32 kernel path is a separate task.
    input_bytes = 160 * 160 * 3
    output_bytes = 2100 * 84
    scratch_bytes = 64 * 1024
    required_bytes = input_bytes + output_bytes + scratch_bytes
    if required_bytes > arena_kb * 1024:
        return {
            "success": False,
            "error": f"Quantized YOLO fixed buffers require {required_bytes // 1024} KB; configured arena is {arena_kb} KB",
            "arena_kb": arena_kb,
            "required_kb": round(required_bytes / 1024, 1),
        }

    result = run_quantized_yolo_image(image, confidence)
    result.update({
        "target": "simulated-esp32",
        "arena_kb": arena_kb,
        "required_kb": round(required_bytes / 1024, 1),
        "memory": {
            "input_tensor_kb": round(input_bytes / 1024, 1),
            "output_tensor_kb": round(output_bytes / 1024, 1),
            "scratch_kb": round(scratch_bytes / 1024, 1),
            "free_arena_kb": round((arena_kb * 1024 - required_bytes) / 1024, 1),
        },
        "pipeline": [
            "decode image into fixed RGB input buffer",
            "resize and quantize to 160x160 uint8",
            "run INT8 YOLO11n detector",
            "decode boxes and apply confidence filtering",
        ],
        "console": [
            f"[input] received {image.width}x{image.height} RGB image",
            "[memory] fixed arena check passed",
            f"[memory] required {round(required_bytes / 1024, 1)} KB / configured {arena_kb} KB",
            "[preprocess] resized and normalized image to 160x160",
            "[model] loaded INT8-weight YOLO11n ONNX target",
            f"[model] inference completed in {result['timings_ms']['inference']} ms",
            f"[postprocess] {len(result['detections'])} detections above confidence {confidence:.2f}",
            f"[done] simulated ESP32 target has {round((arena_kb * 1024 - required_bytes) / 1024, 1)} KB arena headroom",
        ],
    })
    return result


def run_yolo_full_simulation(image, confidence: float = 0.25, arena_kb: int = 4096) -> dict:
    """Run the desktop FP32 model inside an explicit large simulated arena."""
    required_kb = 3120
    if arena_kb < required_kb:
        return {
            "success": False,
            "error": f"Full FP32 YOLO requires approximately {required_kb} KB; configured arena is {arena_kb} KB",
            "arena_kb": arena_kb,
            "required_kb": required_kb,
        }
    result = run_yolo_image(image, confidence)
    result.update({
        "target": "simulated-esp32-full",
        "arena_kb": arena_kb,
        "required_kb": required_kb,
        "memory": {
            "model_profile": "FP32 full-size YOLO11n",
            "free_arena_kb": arena_kb - required_kb,
        },
        "console": [
            f"[memory] full FP32 arena check passed: {required_kb} KB / {arena_kb} KB",
            "[model] loaded full-size FP32 YOLO11n reference target",
            f"[model] slow simulated ESP32 inference completed in {result['timings_ms']['inference']} ms",
            f"[postprocess] {len(result['detections'])} detections above confidence {confidence:.2f}",
        ],
    })
    return result


def compare_yolo_models(image, confidence: float = 0.25) -> dict:
    normal = run_yolo_image(image, confidence)
    quantized = run_quantized_yolo_image(image, confidence)
    matches = []
    used_quantized = set()
    for normal_box in sorted(normal["detections"], key=lambda item: item["confidence"], reverse=True):
        best_index = None
        best_iou = 0.5
        for index, quantized_box in enumerate(quantized["detections"]):
            if index in used_quantized or normal_box["label"] != quantized_box["label"]:
                continue
            quantized_coords = (
                quantized_box["x1"], quantized_box["y1"],
                quantized_box["x2"], quantized_box["y2"],
            )
            normal_coords = (
                normal_box["x1"], normal_box["y1"],
                normal_box["x2"], normal_box["y2"],
            )
            overlap = _iou(normal_coords, quantized_coords)
            if overlap >= best_iou:
                best_index, best_iou = index, overlap
        if best_index is not None:
            used_quantized.add(best_index)
            matched = quantized["detections"][best_index]
            matches.append({
                "label": normal_box["label"],
                "iou": round(best_iou, 4),
                "confidence_delta": round(matched["confidence"] - normal_box["confidence"], 4),
                "bbox_delta": {
                    key: round(matched[key] - normal_box[key], 1)
                    for key in ("x1", "y1", "x2", "y2")
                },
            })
    return {
        "normal": normal,
        "quantized": quantized,
        "matches": matches,
        "matched_labels": [item["label"] for item in matches],
        "confidence_delta": [item["confidence_delta"] for item in matches],
        "normal_count": len(normal["detections"]),
        "quantized_count": len(quantized["detections"]),
        "count_delta": len(quantized["detections"]) - len(normal["detections"]),
    }


def benchmark_yolo_dataset(confidence: float = 0.25) -> dict:
    from PIL import Image

    image_dir = PROJECT_ROOT / "data" / "yolo-coco8" / "images" / "val"
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No YOLO benchmark images found in {image_dir}")
    rows = []
    for image_path in images:
        with Image.open(image_path) as source:
            comparison = compare_yolo_models(source.convert("RGB"), confidence)
        rows.append({
            "image": image_path.name,
            "normal_ms": comparison["normal"]["timings_ms"]["inference"],
            "quantized_ms": comparison["quantized"]["timings_ms"]["inference"],
            "normal_detections": comparison["normal_count"],
            "quantized_detections": comparison["quantized_count"],
            "count_delta": comparison["count_delta"],
            "confidence_delta_avg": round(
                sum(comparison["confidence_delta"]) / len(comparison["confidence_delta"]), 4
            ) if comparison["confidence_delta"] else 0,
        })
    return {
        "dataset": "COCO8 validation",
        "images": len(rows),
        "rows": rows,
        "summary": {
            "normal_avg_ms": round(sum(row["normal_ms"] for row in rows) / len(rows), 2),
            "quantized_avg_ms": round(sum(row["quantized_ms"] for row in rows) / len(rows), 2),
            "normal_total_detections": sum(row["normal_detections"] for row in rows),
            "quantized_total_detections": sum(row["quantized_detections"] for row in rows),
        },
    }


def build_classifier_binary(model: str) -> dict:
    config = MODEL_CONFIG.get(model)
    if not config:
        return {"success": False, "error": f"Unsupported model: {model}"}
    output = BUILD_DIR / config["binary"]
    cmd = [
        "g++", "-O3", "-std=c++17",
        f'-DMODEL_WEIGHTS_HEADER="{config["weights"]}"',
        f'-DTEST_IMAGE_HEADER="{config["sample"]}"',
        f"-I{WEIGHTS_DIR}",
        str(PROJECT_ROOT / "cpp" / "infer.cpp"),
        "-o", str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    return {
        "success": result.returncode == 0,
        "command": " ".join(cmd),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "binary": str(output),
    }


def read_sample_input(model: str) -> list[float]:
    config = MODEL_CONFIG[model]
    text = (WEIGHTS_DIR / config["sample"]).read_text()
    values = text[text.index("{") + 1:text.rindex("}")]
    return [float(value.strip()) for value in values.split(",") if value.strip()]


def run_classifier(model: str, sram_cap_kb: Optional[int] = None, input_variant: int = 0) -> dict:
    config = MODEL_CONFIG.get(model)
    if not config:
        return {"success": False, "error": f"Unsupported model: {model}"}
    built = build_classifier_binary(model)
    if not built["success"]:
        return {"success": False, "error": built["stderr"] or "Classifier build failed"}
    sample = read_sample_input(model)
    if input_variant == 1:
        sample = [-value for value in sample]
    elif input_variant == 2:
        shift = 1 if model == "mnist" else 7
        sample = sample[shift:] + sample[:shift]
    elif input_variant == 3:
        sample = [value * (0.65 + ((index * 17) % 31) / 100) for index, value in enumerate(sample)]
    result = run_binary(
        config["binary"],
        stdin_text=" ".join(f"{value:.8g}" for value in sample),
        sram_cap_kb=sram_cap_kb,
    )
    if not result.get("success"):
        return result
    probabilities = [
        float(result.get(f"PROB_{index}", 0))
        for index in range(len(config["labels"]))
    ]
    predicted_index = int(result.get("PREDICTED_CLASS", 0))
    return {
        "success": True,
        "model": model,
        "labels": config["labels"],
        "shape": config["shape"],
        "input": sample,
        "probabilities": probabilities,
        "predicted_index": predicted_index,
        "predicted_label": config["labels"][predicted_index],
        "inference_us": int(result.get("INFERENCE_TIME_US", 0)),
        "peak_rss_kb": int(result.get("PEAK_RSS_KB", 0)),
        "latency_ms": result.get("latency_ms", 0),
        "sram_cap_kb": sram_cap_kb,
        "stdout": result.get("stdout", ""),
    }


def run_classifier_input(model: str, values: list[float], sram_cap_kb: Optional[int] = None) -> dict:
    config = MODEL_CONFIG.get(model)
    if not config or len(values) != config["shape"][0] * config["shape"][1]:
        expected = config["shape"][0] * config["shape"][1] if config else 0
        return {"success": False, "error": f"{model} input must contain {expected} values"}
    built = build_classifier_binary(model)
    if not built["success"]:
        return {"success": False, "error": built["stderr"] or "Classifier build failed"}
    result = run_binary(
        config["binary"],
        stdin_text=" ".join(f"{value:.8g}" for value in values),
        sram_cap_kb=sram_cap_kb,
    )
    if not result.get("success"):
        return result
    probabilities = [float(result.get(f"PROB_{index}", 0)) for index in range(len(config["labels"]))]
    predicted_index = int(result.get("PREDICTED_CLASS", 0))
    return {
        "success": True,
        "model": model,
        "labels": config["labels"],
        "shape": config["shape"],
        "input": values,
        "probabilities": probabilities,
        "predicted_index": predicted_index,
        "predicted_label": config["labels"][predicted_index],
        "inference_us": int(result.get("INFERENCE_TIME_US", 0)),
        "peak_rss_kb": int(result.get("PEAK_RSS_KB", 0)),
        "latency_ms": result.get("latency_ms", 0),
        "sram_cap_kb": sram_cap_kb,
        "stdout": result.get("stdout", ""),
    }


# ---------------------------------------------------------------------------
# Run a C++ binary synchronously (for single inference calls)
# ---------------------------------------------------------------------------
def run_binary(
    binary_name: str,
    stdin_text: str = "",
    timeout: float = 30.0,
    sram_cap_kb: Optional[int] = None,
    extra_args: Optional[list[str]] = None,
) -> dict:
    """
    Run a compiled ColdInfer binary.
    Returns parsed key:value output as a dict.
    """
    binary_path = _binary(binary_name)
    if not os.path.exists(binary_path):
        return {"success": False, "error": f"Binary not found: {binary_path}. Build first."}

    def preexec():
        if sram_cap_kb:
            limit = sram_cap_kb * 1024
            resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))

    t0 = time.perf_counter()
    cmd = [binary_path] + (extra_args or [])
    try:
        result = subprocess.run(
            cmd,
            input=stdin_text + "\n",
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=preexec if sram_cap_kb else None,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if result.returncode != 0:
            return {
                "success": False,
                "error": result.stderr.strip() or f"Exit code {result.returncode}",
                "stdout": result.stdout,
                "latency_ms": elapsed_ms,
            }

        parsed = {"success": True, "latency_ms": elapsed_ms, "stdout": result.stdout}
        for line in result.stdout.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                parsed[key.strip()] = val.strip()
        return parsed

    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Async streaming runner (for training/benchmark jobs with SSE)
# ---------------------------------------------------------------------------
async def run_script_streaming(
    cmd: list[str],
    cwd: Optional[str] = None,
) -> AsyncIterator[str]:
    """
    Run a shell command and yield output lines as they arrive.
    Used for streaming training/benchmark progress via SSE.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd or str(PROJECT_ROOT),
    )

    async for line in proc.stdout:
        yield line.decode("utf-8", errors="replace").rstrip()

    await proc.wait()
    yield f"__EXIT__:{proc.returncode}"


# ---------------------------------------------------------------------------
# Build the LSTM chatbot binary
# ---------------------------------------------------------------------------
def build_lstm_binary(hidden_size: int = 64, embed_dim: int = 64, max_gen: int = 200) -> dict:
    """
    Compile lstm_infer.cpp against the exported weights headers.
    Returns success/error and compiler output.
    """
    BUILD_DIR.mkdir(exist_ok=True)
    weights_dir = str(WEIGHTS_DIR)

    cmd = [
        "g++", "-O3", "-std=c++17",
        f"-I{weights_dir}",
        f"-Icpp",
        f"-DCHATBOT_HIDDEN_SIZE={hidden_size}",
        f"-DCHATBOT_EMBED_DIM={embed_dim}",
        f"-DMAX_GEN_LEN={max_gen}",
        "cpp/lstm_infer.cpp",
        "-o", str(BUILD_DIR / "lstm_infer"),
        "-lm",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )

    return {
        "success": result.returncode == 0,
        "command": " ".join(cmd),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# ---------------------------------------------------------------------------
# Read performance log
# ---------------------------------------------------------------------------
def read_performance_log() -> list[dict]:
    """Parse performance_log.csv into a list of dicts."""
    if not PERF_LOG.exists():
        return []
    rows = []
    with open(PERF_LOG, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Coerce numeric fields
            cleaned = {}
            for k, v in row.items():
                try:
                    cleaned[k] = float(v) if v else None
                except (ValueError, TypeError):
                    cleaned[k] = v
            rows.append(cleaned)
    return rows


# ---------------------------------------------------------------------------
# Read compression summary (produced by export_chatbot.py)
# ---------------------------------------------------------------------------
def read_compression_summary() -> Optional[dict]:
    path = WEIGHTS_DIR / "compression_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# List available model checkpoints
# ---------------------------------------------------------------------------
def list_models() -> list[dict]:
    models = []
    for pt_file in WEIGHTS_DIR.glob("*.pt"):
        stat = pt_file.stat()
        models.append({
            "name": pt_file.stem,
            "path": str(pt_file),
            "size_kb": round(stat.st_size / 1024, 1),
        })
    return models
