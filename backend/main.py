"""
main.py — ColdInfer FastAPI Backend

Serves the web dashboard and bridges frontend ↔ C++ engine ↔ ESP32.

Run:
    cd /data/projects/Coldinfer
    uv run uvicorn backend.main:app --reload --port 8000
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import io
from pathlib import Path
from typing import Optional, AsyncIterator

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path so we can import python/ modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from backend.engine import (
    run_binary,
    run_script_streaming,
    build_lstm_binary,
    read_performance_log,
    read_compression_summary,
    list_models,
    run_classifier,
    run_classifier_input,
    run_yolo_image,
    run_yolo_simulation,
    run_yolo_full_simulation,
    compare_yolo_models,
    benchmark_yolo_dataset,
    BUILD_DIR,
    WEIGHTS_DIR,
    SCRIPTS_DIR,
)
from backend.esp32_proxy import get_client, set_client, clear_client

# ---------------------------------------------------------------------------
app = FastAPI(
    title="ColdInfer Dashboard API",
    description="Bare-metal TinyML inference engine control plane",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the compiled React SPA
WEBAPP_DIST = PROJECT_ROOT / "webapp" / "dist"
if (WEBAPP_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(WEBAPP_DIST / "assets")), name="assets")
if WEBAPP_DIST.exists():
    app.mount("/app", StaticFiles(directory=str(WEBAPP_DIST), html=True), name="webapp")

@app.get("/", include_in_schema=False)
def root():
    index_file = WEBAPP_DIST / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return RedirectResponse(url="/docs")

# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class InferRequest(BaseModel):
    model: str = "mnist"           # "mnist" | "kws"
    input_data: Optional[list[float]] = None

class SimulationRequest(BaseModel):
    model: str = "mnist"
    sram_cap_kb: int = 520
    input_variant: int = 0
    input_data: Optional[list[float]] = None

class TrainRequest(BaseModel):
    model: str = "chatbot"         # "chatbot" | "mnist" | "kws"
    epochs: int = 60
    hidden: int = 64
    embed: int = 64
    lr: float = 3e-3
    batch_size: int = 16

class QuantizeRequest(BaseModel):
    checkpoint: str = "weights/chatbot_best.pt"
    prune: float = 0.5
    out_dir: str = "weights"

class BuildRequest(BaseModel):
    hidden_size: int = 64
    embed_dim: int = 64
    max_gen: int = 200

class ESP32ConnectRequest(BaseModel):
    ip: str
    port: int = 80


@app.post("/api/yolo/detect")
async def api_yolo_detect(
    image: UploadFile = File(...),
    confidence: float = 0.25,
    target: str = "simulated-esp32",
    arena_kb: int = 520,
):
    """Run the ColdInfer YOLO target and return boxes for the visual demo."""
    if not 0 <= confidence <= 1:
        raise HTTPException(400, "confidence must be between 0 and 1")
    try:
        from PIL import Image
        from ultralytics import YOLO  # noqa: F401 - validates optional runtime
    except ImportError as exc:
        raise HTTPException(503, "YOLO support is not installed. Run uv sync first.") from exc

    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if target == "simulated-esp32":
            detection = run_yolo_simulation(pil_image, confidence, arena_kb)
        elif target == "simulated-esp32-full":
            detection = run_yolo_full_simulation(pil_image, confidence, arena_kb)
        elif target == "local":
            detection = run_yolo_image(pil_image, confidence)
        else:
            raise HTTPException(400, "target must be simulated-esp32, simulated-esp32-full, or local")
        if not detection.get("success", True):
            raise HTTPException(507, detection["error"])
        return {
            **detection,
            "width": pil_image.width,
            "height": pil_image.height,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"YOLO inference failed: {exc}") from exc


@app.post("/api/yolo/compare")
async def api_yolo_compare(image: UploadFile = File(...), confidence: float = 0.25):
    """Run normal and quantized YOLO on the same image."""
    try:
        from PIL import Image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return {
            **compare_yolo_models(pil_image, confidence),
            "width": pil_image.width,
            "height": pil_image.height,
        }
    except Exception as exc:
        raise HTTPException(422, f"YOLO comparison failed: {exc}") from exc


@app.post("/api/yolo/benchmark")
def api_yolo_benchmark(confidence: float = 0.25):
    """Benchmark normal and quantized YOLO on the bundled COCO8 validation set."""
    try:
        return benchmark_yolo_dataset(confidence)
    except Exception as exc:
        raise HTTPException(500, f"YOLO benchmark failed: {exc}") from exc

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
def api_health():
    return {
        "status": "ok",
        "version": "0.2.0",
        "build_dir_exists": BUILD_DIR.exists(),
        "weights_dir_exists": WEIGHTS_DIR.exists(),
    }

# ---------------------------------------------------------------------------
# Generic inference (MNIST / KWS)
# ---------------------------------------------------------------------------

@app.post("/api/infer")
async def api_infer(req: InferRequest):
    """Run the MLP inference engine for MNIST or KWS."""
    result = run_classifier(req.model)
    if not result.get("success"):
        raise HTTPException(500, result.get("error", "Inference failed"))
    return result


@app.post("/api/simulation/infer")
async def api_simulation_infer(req: SimulationRequest):
    """Run a visualizable classifier inference under a simulated SRAM budget."""
    if req.input_variant not in range(4):
        raise HTTPException(400, "input_variant must be between 0 and 3")
    if req.input_data is not None:
        result = run_classifier_input(req.model, req.input_data, sram_cap_kb=req.sram_cap_kb)
    else:
        result = run_classifier(req.model, sram_cap_kb=req.sram_cap_kb, input_variant=req.input_variant)
    if not result.get("success"):
        raise HTTPException(500, result.get("error", "Simulation failed"))
    return result

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

@app.get("/api/benchmarks")
def api_benchmarks():
    """Return all rows from performance_log.csv."""
    return {"data": read_performance_log()}

@app.post("/api/benchmarks/run")
async def api_run_benchmarks():
    """
    Kick off run_tests.sh and stream its output via SSE.
    """
    async def event_stream():
        cmd = ["bash", str(SCRIPTS_DIR / "run_tests.sh")]
        yield f"data: {json.dumps({'event': 'log', 'line': '[start] benchmark suite'})}\n\n"
        async for line in run_script_streaming(cmd, cwd=str(PROJECT_ROOT)):
            if line.startswith("__EXIT__:"):
                code = line.split(":")[1]
                yield f"data: {{\"event\":\"done\",\"exit_code\":{code}}}\n\n"
            else:
                yield f"data: {json.dumps({'event': 'log', 'line': line})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@app.post("/api/train")
async def api_train(req: TrainRequest):
    """
    Start a training job and stream progress via SSE.
    """
    async def event_stream():
        if req.model == "chatbot":
            cmd = [
                sys.executable, str(PROJECT_ROOT / "python" / "train_chatbot.py"),
                "--epochs", str(req.epochs),
                "--hidden", str(req.hidden),
                "--embed",  str(req.embed),
                "--lr",     str(req.lr),
                "--batch-size", str(req.batch_size),
            ]
        else:
            cmd = [
                sys.executable, "-u", str(PROJECT_ROOT / "python" / "train_export.py"),
                "--model", req.model, "--epochs", str(req.epochs), "--force-retrain",
            ]

        start_line = "[start] " + " ".join(cmd)
        yield f"data: {json.dumps({'event': 'log', 'line': start_line})}\n\n"
        async for line in run_script_streaming(cmd, cwd=str(PROJECT_ROOT)):
            if line.startswith("__EXIT__:"):
                code = line.split(":")[1]
                yield f"data: {{\"event\":\"done\",\"exit_code\":{code}}}\n\n"
            else:
                yield f"data: {json.dumps({'event': 'log', 'line': line})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/api/models")
def api_models():
    return {"models": list_models()}

# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

@app.post("/api/quantize")
async def api_quantize(req: QuantizeRequest):
    """
    Run the quantization + pruning pipeline via export_chatbot.py.
    Streams output, returns compression summary when done.
    """
    async def event_stream():
        cmd = [
            sys.executable, str(PROJECT_ROOT / "python" / "export_chatbot.py"),
            "--checkpoint", req.checkpoint,
            "--prune", str(req.prune),
            "--out-dir", req.out_dir,
        ]
        async for line in run_script_streaming(cmd, cwd=str(PROJECT_ROOT)):
            if line.startswith("__EXIT__:"):
                code = line.split(":")[1]
                # Attach compression summary to done event
                summary = read_compression_summary() or {}
                payload = json.dumps({"event": "done", "exit_code": int(code), "summary": summary})
                yield f"data: {payload}\n\n"
            else:
                safe = line.replace('"', '\\"')
                yield f"data: {{\"event\":\"log\",\"line\":\"{safe}\"}}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/api/quantize/summary")
def api_quantize_summary():
    """Return the latest compression summary JSON."""
    summary = read_compression_summary()
    if not summary:
        raise HTTPException(404, "No compression summary found. Run quantization first.")
    return summary

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

@app.post("/api/build")
def api_build(req: BuildRequest):
    """Compile the LSTM inference binary."""
    result = build_lstm_binary(req.hidden_size, req.embed_dim, req.max_gen)
    if not result["success"]:
        raise HTTPException(500, f"Build failed:\n{result['stderr']}")
    return result

# ---------------------------------------------------------------------------
# ESP32 Panel
# ---------------------------------------------------------------------------

@app.post("/api/esp32/connect")
def api_esp32_connect(req: ESP32ConnectRequest):
    client = set_client(req.ip, req.port)
    result = client.health()
    return result

@app.get("/api/esp32/health")
def api_esp32_health():
    client = get_client()
    if not client:
        return {"connected": False, "error": "No ESP32 configured"}
    return client.health()

@app.delete("/api/esp32/connect")
def api_esp32_disconnect():
    clear_client()
    return {"connected": False}
