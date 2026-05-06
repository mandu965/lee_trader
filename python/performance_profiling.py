from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import ctypes
    from ctypes import wintypes
except Exception:  # pragma: no cover - optional Windows fallback
    ctypes = None
    wintypes = None

try:
    import resource
except Exception:  # pragma: no cover - not available on Windows
    resource = None

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
PROFILE_EVENTS_JSONL = OUTPUTS_DIR / "pipeline_profile_events.jsonl"


@dataclass
class ProfileSample:
    started_at_iso: str
    wall_start: float
    process_cpu_start: float
    psutil_available: bool


def profile_begin() -> ProfileSample:
    if psutil:
        try:
            psutil.Process(os.getpid()).cpu_percent(interval=None)
        except Exception:
            pass
    return ProfileSample(
        started_at_iso=datetime.now().isoformat(timespec="seconds"),
        wall_start=time.perf_counter(),
        process_cpu_start=time.process_time(),
        psutil_available=bool(psutil),
    )


def _get_rss_mb_psutil() -> float | None:
    if not psutil:
        return None
    try:
        rss_bytes = psutil.Process(os.getpid()).memory_info().rss
        return float(rss_bytes) / (1024.0 * 1024.0)
    except Exception:
        return None


def _get_rss_mb_windows() -> float | None:
    if os.name != "nt" or ctypes is None or wintypes is None:
        return None
    try:
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            return None
        return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
    except Exception:
        return None


def _get_rss_mb_resource() -> float | None:
    if resource is None:
        return None
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = float(usage.ru_maxrss)
        if sys.platform == "darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except Exception:
        return None


def _get_rss_mb_proc_status() -> float | None:
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / 1024.0
    except Exception:
        return None
    return None


def get_rss_mb() -> float | None:
    for getter in (
        _get_rss_mb_psutil,
        _get_rss_mb_windows,
        _get_rss_mb_resource,
        _get_rss_mb_proc_status,
    ):
        value = getter()
        if value is not None:
            return value
    return None


def get_cpu_percent(sample: ProfileSample, elapsed_sec: float) -> float | None:
    if elapsed_sec <= 0:
        return None
    if psutil:
        try:
            return float(psutil.Process(os.getpid()).cpu_percent(interval=None))
        except Exception:
            pass
    try:
        cpu_delta = max(0.0, time.process_time() - sample.process_cpu_start)
        return float(cpu_delta / elapsed_sec * 100.0)
    except Exception:
        return None


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.2f}"


def _append_profile_event(payload: dict[str, Any]) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with PROFILE_EVENTS_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def log_profile_result(
    step: str,
    elapsed_sec: float,
    *,
    sample: ProfileSample | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effective_sample = sample or profile_begin()
    payload: dict[str, Any] = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "step": str(step),
        "elapsed_sec": round(float(elapsed_sec), 6),
        "memory_mb": None,
        "cpu_percent": None,
        "psutil_available": bool(psutil),
        "started_at": effective_sample.started_at_iso,
    }
    if extra:
        payload.update(extra)

    try:
        payload["memory_mb"] = get_rss_mb()
        payload["cpu_percent"] = get_cpu_percent(effective_sample, float(elapsed_sec))
        logging.info("[PROFILE] step=%s elapsed_sec=%.2f", step, elapsed_sec)
        logging.info(
            "[PROFILE] memory_mb=%s cpu_percent=%s",
            _fmt_float(payload["memory_mb"]),
            _fmt_float(payload["cpu_percent"]),
        )
        if not psutil:
            logging.info("[PROFILE] psutil_available=0 fallback=process_cpu_time")
        _append_profile_event(payload)
    except Exception:
        logging.exception("Profiling output failed for step=%s", step)
    return payload
