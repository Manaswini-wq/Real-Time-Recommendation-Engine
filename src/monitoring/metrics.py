import threading
import time
from collections import deque

import numpy as np


class MetricsCollector:
    """Tracks recommendation latency and request metrics."""

    def __init__(self, window_size: int = 10000) -> None:
        self._latencies: deque = deque(maxlen=window_size)
        self._counts: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def record_latency(self, latency_ms: float) -> None:
        with self._lock:
            self._latencies.append((time.time(), latency_ms))

    def record_request(self, endpoint: str, status: str) -> None:
        with self._lock:
            if endpoint not in self._counts:
                self._counts[endpoint] = {}
            self._counts[endpoint][status] = self._counts[endpoint].get(status, 0) + 1

    def get_summary(self) -> dict:
        with self._lock:
            latencies = [v for _, v in self._latencies]
            return {
                "latency": self._pct(latencies),
                "total_requests": sum(
                    sum(s.values()) for s in self._counts.values()
                ),
                "endpoints": {k: dict(v) for k, v in self._counts.items()},
            }

    @staticmethod
    def _pct(vals: list[float]) -> dict:
        if not vals:
            return {"p50": 0, "p95": 0, "p99": 0}
        a = np.array(vals)
        return {
            "p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2),
            "p99": round(float(np.percentile(a, 99)), 2),
        }
