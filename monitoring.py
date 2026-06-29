"""Prometheus monitoring helpers for MediScan."""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import psutil
from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest


REQUEST_COUNT = Counter(
    "mediscan_http_requests_total",
    "Total HTTP requests handled by the MediScan API",
    ["method", "path", "status"],
)

REQUEST_LATENCY = Histogram(
    "mediscan_http_request_duration_seconds",
    "HTTP request latency for the MediScan API",
    ["method", "path"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

PROCESS_MEMORY_BYTES = Gauge(
    "mediscan_process_memory_bytes",
    "Resident memory usage for the MediScan backend process",
)

PROCESS_CPU_PERCENT = Gauge(
    "mediscan_process_cpu_percent",
    "CPU usage percentage for the MediScan backend process",
)

SYSTEM_MEMORY_PERCENT = Gauge(
    "mediscan_system_memory_percent",
    "System-wide memory utilization percentage",
)

SYSTEM_CPU_PERCENT = Gauge(
    "mediscan_system_cpu_percent",
    "System-wide CPU utilization percentage",
)

_metrics_thread: Optional[threading.Thread] = None
_metrics_started = False
_metrics_lock = threading.Lock()


def metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _collect_system_metrics(interval_seconds: float = 5.0) -> None:
    process = psutil.Process(os.getpid())
    process.cpu_percent(None)
    psutil.cpu_percent(None)

    while True:
        try:
            PROCESS_MEMORY_BYTES.set(process.memory_info().rss)
            PROCESS_CPU_PERCENT.set(process.cpu_percent(None))
            SYSTEM_MEMORY_PERCENT.set(psutil.virtual_memory().percent)
            SYSTEM_CPU_PERCENT.set(psutil.cpu_percent(None))
        except Exception as exc:  # pragma: no cover - background telemetry only
            print(f"Warning: failed to update Prometheus metrics: {exc}")

        time.sleep(interval_seconds)


def start_background_metrics(interval_seconds: float = 5.0) -> None:
    global _metrics_started, _metrics_thread

    with _metrics_lock:
        if _metrics_started:
            return

        _metrics_thread = threading.Thread(
            target=_collect_system_metrics,
            kwargs={"interval_seconds": interval_seconds},
            daemon=True,
            name="mediscan-prometheus-metrics",
        )
        _metrics_thread.start()
        _metrics_started = True


def track_request(request: Request, status_code: int, duration_seconds: float) -> None:
    path = request.url.path
    if path == "/metrics":
        return

    method = request.method
    REQUEST_COUNT.labels(method=method, path=path, status=str(status_code)).inc()
    REQUEST_LATENCY.labels(method=method, path=path).observe(duration_seconds)
