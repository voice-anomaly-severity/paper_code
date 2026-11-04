from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("VOICE_DATA_ROOT") or PROJECT_ROOT)   # data root (default: repository root)
DATA_DIR = DATA_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def get_dataset_dir(dataset_name: str, recording_type: str, split: str) -> Path:
    if dataset_name == "ukl_wavs":
        if split == "healthy":
            return DATA_DIR / "ukl_wavs" / "healthy_studis" / recording_type
        if split == "unhealthy":
            return DATA_DIR / "ukl_wavs" / "unhealthy" / recording_type
    raise ValueError(
        f"Unsupported dataset/split combination for paper release: "
        f"dataset_name={dataset_name!r}, split={split!r}"
    )


def get_default_metadata_xlsx() -> Path | None:
    env_value = os.environ.get("VOICE_AI_METADATA_XLSX", "").strip()
    if env_value:
        return Path(env_value).expanduser()

    candidates = [
        DATA_DIR / "metadata" / "metadata.xlsx",
        DATA_ROOT / "Voice_KI_RBH_full.xlsx",
        DATA_ROOT / "Voice_KI_ICCAS_RBH.xlsx",
        DATA_ROOT / "Voice_KI_ICCAS_110725_RBH.xlsx",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
