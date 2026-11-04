from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    raw: dict
    root: Path
    config_path: Path

    def path(self, key: str) -> Path:
        """Absolute paths as given; './' or '../' relative to this config file; anything else relative to root."""
        raw = str(self.raw["paths"][key])
        value = Path(raw)
        if value.is_absolute():
            return value
        if raw.startswith(("./", "../")):
            return (self.config_path.parent / value).resolve()
        return (self.root / value).resolve()

    @property
    def data(self) -> dict:
        return self.raw["data"]

    @property
    def analysis(self) -> dict:
        return self.raw["analysis"]

    @property
    def output_dir(self) -> Path:
        out = self.path("output_dir")
        out.mkdir(parents=True, exist_ok=True)
        return out


def load_config(config_path: str | Path) -> Config:
    config_path = Path(config_path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    root_value = os.environ.get("VOICE_DATA_ROOT") or os.environ.get("VOICE_REVISION_ROOT") or raw["paths"]["root"]
    root = Path(root_value)
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    config = Config(raw=raw, root=root, config_path=config_path)

    paper_code = str(config.path("paper_code"))
    if paper_code not in sys.path:
        sys.path.insert(0, paper_code)
    return config
