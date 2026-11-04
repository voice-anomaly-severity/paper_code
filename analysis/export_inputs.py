"""Export the pseudonymized analysis inputs (run on the server, where the raw data are).

    python export_inputs.py --out ../inputs

Reads the raw sources configured in config.yaml (score cache, GMM scores, RBH table, clinical
workbook) and writes recordings_pseudonymized.csv and workbook_recordings_pseudonymized.csv:
no names, birth dates or recording IDs; recording dates shifted per patient. Point
`paths.analysis_table` and `paths.analysis_workbook_table` in config.yaml at these files to run all
analyses without the raw data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.config import load_config  # noqa: E402
from pipeline.data import export_pseudonymized  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ("analysis_table", "analysis_workbook_table"):    # always export from the raw sources
        config.raw["paths"].pop(key, None)
    for key, path in export_pseudonymized(config, args.out).items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
