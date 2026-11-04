"""Run batched GP leave-one-patient-out fits on one or more GPUs (CPU fallback).

One worker process per device; tasks are split round-robin. Each worker logs its progress to
<log_dir>/gp_worker_<k>.log so a long run can be followed without the parent's console.
"""
from __future__ import annotations

import time
from pathlib import Path

from joblib import Parallel, delayed


def available_devices(requested: list[str]) -> list[str]:
    try:
        import torch
    except ImportError:
        return ["cpu"]
    if not torch.cuda.is_available():
        return ["cpu"]
    n = torch.cuda.device_count()
    devices = [d for d in requested if d.startswith("cuda") and int(d.split(":")[1] if ":" in d else 0) < n]
    return devices or ["cpu"]


def _run_chunk(chunk, device, settings, log_path):
    import torch
    from .gp_torch import fit_lopo

    out = {}
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"worker on {device}: {len(chunk)} tasks\n")
        log.flush()
        for i, (key, X, y, patients) in enumerate(chunk, start=1):
            t0 = time.time()
            out[key] = fit_lopo(X, y, patients, metric="full", use_re=True, objective="map",
                                restarts=settings["restarts"], steps=settings["steps"], lr=settings["lr"],
                                jitter=settings["jitter"], device=device, dtype=torch.float64)
            log.write(f"{time.strftime('%H:%M:%S')} {i}/{len(chunk)} {key} {time.time() - t0:.0f}s\n")
            log.flush()
    return out


def run_gp_tasks(tasks, settings: dict, log_dir: Path) -> dict:
    """tasks: list of (key, X, y, patients). Returns {key: per-patient result arrays}."""
    devices = available_devices(settings.get("devices", ["cuda:0"]))
    log_dir.mkdir(parents=True, exist_ok=True)
    chunks = [tasks[i::len(devices)] for i in range(len(devices))]
    parts = Parallel(n_jobs=len(devices), backend="loky")(
        delayed(_run_chunk)(chunk, device, settings, log_dir / f"gp_worker_{k}.log")
        for k, (chunk, device) in enumerate(zip(chunks, devices)) if chunk
    )
    merged = {}
    for part in parts:
        merged.update(part)
    return merged
