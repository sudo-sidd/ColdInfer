"""
esp32_proxy.py — HTTP client that proxies requests to an ESP32 over Wi-Fi.
"""

import time
from typing import Optional

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False


class ESP32Client:
    """
    Thin HTTP client for the ESP32 inference server.
    The ESP32 firmware exposes:
      GET  /health  → {"status":"ok","free_heap_kb":N,"uptime_s":N}
      POST /infer   → {"input":[...]} → {"output":"...", "latency_us":N}
      POST /chat    → {"prompt":"..."} → {"response":"...", "latency_us":N}
    """

    def __init__(self, ip: str, port: int = 80, timeout: float = 10.0):
        self.base_url = f"http://{ip}:{port}"
        self.timeout = timeout
        self._ip = ip

    def _check_httpx(self):
        if not _HAS_HTTPX:
            raise RuntimeError("httpx not installed. Run: uv add httpx")

    def health(self) -> dict:
        """Ping the ESP32 and return status info."""
        self._check_httpx()
        t0 = time.perf_counter()
        try:
            r = httpx.get(f"{self.base_url}/health", timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            data["ping_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            data["connected"] = True
            return data
        except Exception as e:
            return {"connected": False, "error": str(e), "ip": self._ip}

    def chat(self, prompt: str, temperature: float = 0.8) -> dict:
        """Send a chat prompt to the ESP32 and return the response."""
        self._check_httpx()
        t0 = time.perf_counter()
        try:
            r = httpx.post(
                f"{self.base_url}/chat",
                json={"prompt": prompt, "temperature": temperature},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            data["wall_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            data["success"] = True
            return data
        except Exception as e:
            return {"success": False, "error": str(e)}

    def infer(self, input_data: list[float]) -> dict:
        """Send a raw float array to the ESP32 /infer endpoint."""
        self._check_httpx()
        try:
            r = httpx.post(
                f"{self.base_url}/infer",
                json={"input": input_data},
                timeout=self.timeout,
            )
            r.raise_for_status()
            return {"success": True, **r.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Module-level singleton — set via /api/esp32/connect
_client: Optional[ESP32Client] = None


def get_client() -> Optional[ESP32Client]:
    return _client


def set_client(ip: str, port: int = 80, timeout: float = 10.0):
    global _client
    _client = ESP32Client(ip, port, timeout)
    return _client


def clear_client():
    global _client
    _client = None
