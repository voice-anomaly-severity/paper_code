"""Run the statistical analyses.

    python run.py                      # all steps, settings from config.yaml next to this file
    python run.py --steps table3,age_correlation
    python run.py --config other.yaml --quick    # fewer resamples/replicates, for a smoke test

Outputs (aggregate tables, run metadata) go to `paths.output_dir` of the config.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # one BLAS thread per worker
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.config import load_config  # noqa: E402

PACKAGES = ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "joblib", "PyYAML", "openpyxl", "torch"]


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"))
    parser.add_argument("--steps", default="all", help="comma-separated subset of steps")
    parser.add_argument("--quick", action="store_true", help="smoke test: few bootstrap draws and replicates")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.quick:
        config.analysis.update(cluster_bootstrap_draws=50, bayesian_bootstrap_draws=500)
        config.analysis["calibration_check"].update(lmm_replicates=4, gp_replicates=1)
        config.analysis["gp"].update(steps=100)
        config.raw["paths"]["output_dir"] = str(config.path("output_dir")) + "_quick"

    from pipeline import analyses
    from pipeline.data import load_recordings

    steps = list(analyses.STEPS) if args.steps == "all" else [s.strip() for s in args.steps.split(",")]
    out = config.output_dir
    started = time.time()
    df = load_recordings(config)
    print(f"loaded {len(df)} recordings ({int(df['healthy'].sum())} healthy) from {df['patient'].nunique()} individuals", flush=True)

    step_log = {}
    for step in steps:
        t0 = time.time()
        print(f"[{step}] running", flush=True)
        analyses.STEPS[step](df, config, out)
        step_log[step] = {"seconds": round(time.time() - t0, 1),
                          "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        print(f"[{step}] done in {step_log[step]['seconds']} s", flush=True)

    input_keys = ["analysis_table", "analysis_workbook_table", "score_cache", "gmm_scores", "rbh_csv", "metadata_xlsx"]
    inputs = {k: config.path(k) for k in input_keys if config.raw["paths"].get(k) and config.path(k).is_file()}
    meta_path = out / "run_metadata.json"
    previous = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    if "steps" not in previous and "step_seconds" in previous:   # metadata written by an earlier version of this script
        previous["steps"] = {s: {"seconds": sec, "finished_utc": previous.get("finished_utc")} for s, sec in previous["step_seconds"].items()}
    meta = {
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "steps_run_last": steps,
        "steps": {**previous.get("steps", {}), **step_log},   # latest run of every step that produced the outputs
        "total_seconds_last_run": round(time.time() - started, 1),
        "quick": args.quick,
        "config_file": str(config.config_path),
        "config": config.raw,
        "inputs_sha256": {k: _sha256(p) for k, p in inputs.items()},
        "paper_code": ({"path": str(config.path("paper_code")), "git_head": _git_head(config.path("paper_code")),
                        "voice_ai/data.py sha256": _sha256(config.path("paper_code") / "voice_ai" / "data.py")}
                       if (config.path("paper_code") / "voice_ai" / "data.py").is_file() else "not used (pseudonymized inputs)"),
        "python": platform.python_version(),
        "packages": {p: _version(p) for p in PACKAGES},
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"metadata written to {meta_path}")


if __name__ == "__main__":
    main()
