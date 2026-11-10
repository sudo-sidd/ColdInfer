"""
simulate_constraints.py — Simulate ESP32 SRAM constraints on a Linux laptop.

Sets RLIMIT_AS (virtual address space) to mimic the ESP32's 520 KB SRAM ceiling,
then runs the ColdInfer C++ chatbot binary inside that constraint.

This lets you validate that the inference engine works within hardware limits
without physically having the board attached.

Usage:
    # Simulate ESP32 (520 KB SRAM)
    uv run python/simulate_constraints.py --binary build/lstm_infer --prompt "Hello"

    # Custom limit (e.g., 1 MB for a more capable board)
    uv run python/simulate_constraints.py --binary build/lstm_infer --sram-kb 1024 --prompt "What is TinyML?"

    # Benchmark mode: run N inferences and report avg latency
    uv run python/simulate_constraints.py --binary build/lstm_infer --benchmark --n 20
"""

import argparse
import os
import resource
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# Memory limit helper
# ---------------------------------------------------------------------------
def set_memory_limit_bytes(limit_bytes: int):
    """
    Set the data segment limit (heap + static data) for the current process.
    Called as the preexec_fn in subprocess so only the child is limited to the target SRAM budget.
    """
    resource.setrlimit(resource.RLIMIT_DATA, (limit_bytes, limit_bytes))


# ---------------------------------------------------------------------------
# Run the C++ binary with an SRAM cap
# ---------------------------------------------------------------------------
def run_constrained(
    binary: str,
    prompt: str,
    sram_kb: int,
    timeout: float = 30.0,
    extra_args: list = None,
) -> dict:
    """
    Run the inference binary with a virtual memory cap of sram_kb * 1024 bytes.

    The binary receives the prompt via stdin and writes its output to stdout
    in the same key:value format as infer.cpp.

    Returns dict with keys: output, latency_us, success, error
    """
    limit_bytes = sram_kb * 1024

    cmd = [binary] + (extra_args or [])

    t_start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            input=prompt + "\n",
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=lambda: set_memory_limit_bytes(limit_bytes),
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        if result.returncode != 0:
            return {
                "success": False,
                "error": result.stderr.strip() or f"Exit code {result.returncode}",
                "output": result.stdout.strip(),
                "latency_ms": elapsed_ms,
                "sram_cap_kb": sram_kb,
            }

        # Parse structured output
        parsed = {}
        for line in result.stdout.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                parsed[key.strip()] = val.strip()

        return {
            "success": True,
            "error": None,
            "output": result.stdout.strip(),
            "generated_text": parsed.get("GENERATED", ""),
            "inference_us": int(parsed.get("INFERENCE_TIME_US", 0)),
            "tokens_generated": int(parsed.get("TOKENS_GENERATED", 0)),
            "latency_ms": elapsed_ms,
            "sram_cap_kb": sram_kb,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Timed out after {timeout}s",
            "output": "",
            "latency_ms": timeout * 1000,
            "sram_cap_kb": sram_kb,
        }
    except MemoryError:
        return {
            "success": False,
            "error": "OOM: Binary exceeded SRAM cap — model too large for target hardware",
            "output": "",
            "latency_ms": 0,
            "sram_cap_kb": sram_kb,
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": f"Binary not found: {binary}. Run scripts/build.sh first.",
            "output": "",
            "latency_ms": 0,
            "sram_cap_kb": sram_kb,
        }


# ---------------------------------------------------------------------------
# Benchmark mode
# ---------------------------------------------------------------------------
def run_benchmark(binary: str, prompt: str, sram_kb: int, n: int) -> dict:
    latencies = []
    inference_times = []
    failures = 0

    print(f"[sim] Running {n} iterations at {sram_kb} KB SRAM cap...")
    for i in range(n):
        result = run_constrained(binary, prompt, sram_kb)
        if result["success"]:
            latencies.append(result["latency_ms"])
            if "inference_us" in result:
                inference_times.append(result["inference_us"])
        else:
            failures += 1
            print(f"  Run {i+1}: FAILED — {result['error']}")

    if not latencies:
        return {"success": False, "error": "All runs failed", "failures": n}

    return {
        "success": True,
        "n": n,
        "failures": failures,
        "sram_cap_kb": sram_kb,
        "latency_ms": {
            "mean": sum(latencies) / len(latencies),
            "min": min(latencies),
            "max": max(latencies),
        },
        "inference_us": {
            "mean": sum(inference_times) / max(len(inference_times), 1),
            "min": min(inference_times) if inference_times else 0,
            "max": max(inference_times) if inference_times else 0,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Simulate ESP32 SRAM constraints and run ColdInfer chatbot"
    )
    parser.add_argument(
        "--binary", type=str, default="build/lstm_infer",
        help="Path to compiled ColdInfer LSTM binary"
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello",
        help="Input prompt to send to the chatbot"
    )
    parser.add_argument(
        "--sram-kb", type=int, default=520,
        help="Simulated SRAM cap in KB (default: 520 = ESP32 classic)"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run benchmark mode instead of single inference"
    )
    parser.add_argument(
        "--n", type=int, default=10,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0,
        help="Per-run timeout in seconds"
    )
    args = parser.parse_args()

    print(f"[sim] ColdInfer ESP32 Constraint Simulator")
    print(f"[sim] Binary  : {args.binary}")
    print(f"[sim] SRAM cap: {args.sram_kb} KB")
    print(f"[sim] Prompt  : {repr(args.prompt)}")
    print()

    if args.benchmark:
        result = run_benchmark(args.binary, args.prompt, args.sram_kb, args.n)
        if result["success"]:
            print(f"\n{'='*50}")
            print(f"  Benchmark Results ({result['n']} runs, {result['failures']} failures)")
            print(f"{'='*50}")
            print(f"  Wall latency (ms):")
            print(f"    mean = {result['latency_ms']['mean']:.2f}")
            print(f"    min  = {result['latency_ms']['min']:.2f}")
            print(f"    max  = {result['latency_ms']['max']:.2f}")
            if result["inference_us"]["mean"] > 0:
                print(f"  C++ inference (µs):")
                print(f"    mean = {result['inference_us']['mean']:.1f}")
                print(f"    min  = {result['inference_us']['min']:.1f}")
                print(f"    max  = {result['inference_us']['max']:.1f}")
        else:
            print(f"[sim] FAILED: {result['error']}")
            sys.exit(1)
    else:
        result = run_constrained(args.binary, args.prompt, args.sram_kb, args.timeout)
        if result["success"]:
            print(f"[sim] ✅ PASSED — fits within {args.sram_kb} KB SRAM cap")
            print(f"[sim] Wall time : {result['latency_ms']:.2f} ms")
            if result.get("inference_us"):
                print(f"[sim] C++ infer : {result['inference_us']} µs")
            print(f"\n--- Output ---")
            print(result["output"])
        else:
            print(f"[sim] ❌ FAILED — {result['error']}")
            sys.exit(1)


if __name__ == "__main__":
    main()
