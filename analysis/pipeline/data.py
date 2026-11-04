"""Assemble the recording-level analysis table in memory.

One row per recording (`Dateiname`). The recording score is the mean of its word-level
scores, exactly as in the paper code (voice_ai.data.build_patient_scores_df). Patients are
identified by a hash of the identity columns of the clinical workbook; the hash and the
identity columns never leave memory. Nothing in this module writes to disk.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .config import Config


def _person_and_date_maps(config: Config) -> tuple[dict[str, str], dict[str, pd.Timestamp]]:
    xlsx = config.path("metadata_xlsx")
    identity_columns = config.data["identity_columns"]
    date_column = config.data["date_column"]
    person, date = {}, {}
    for sheet in pd.ExcelFile(xlsx).sheet_names:
        frame = pd.read_excel(xlsx, sheet_name=sheet)
        if not {"Dateiname", date_column, *identity_columns}.issubset(frame.columns):
            continue
        frame = frame[frame["Dateiname"].notna()]
        for _, row in frame.iterrows():
            key = "|".join(str(row[c]).strip().lower() for c in identity_columns)
            recording = str(row["Dateiname"])
            person[recording] = hashlib.sha256(key.encode("utf-8")).hexdigest()
            date[recording] = pd.to_datetime(row[date_column], errors="coerce")
    return person, date


def _workbook_column(xlsx, column: str) -> dict[str, float]:
    """Numeric values of one workbook column by recording ID ('-' and empty cells are missing)."""
    values = {}
    for sheet in pd.ExcelFile(xlsx).sheet_names:
        frame = pd.read_excel(xlsx, sheet_name=sheet)
        if not {"Dateiname", column}.issubset(frame.columns):
            continue
        frame = frame[frame["Dateiname"].notna()]
        numeric = pd.to_numeric(frame[column], errors="coerce")
        for recording, value in zip(frame["Dateiname"].astype(str), numeric):
            if pd.notna(value):
                values[recording] = float(value)
    return values


def _pseudonymized(config: Config, key: str):
    """Path of a configured pseudonymized table, or None (then the raw sources are used)."""
    if not config.raw["paths"].get(key):
        return None
    path = config.path(key)
    return path if path.is_file() else None


def load_recordings(config: Config) -> pd.DataFrame:
    """Return one row per scored recording with scores, ratings, covariates and targets.

    If `paths.analysis_table` is configured, the pseudonymized export (see export_pseudonymized) is read
    instead of the raw sources; the analyses give identical results.
    """
    table = _pseudonymized(config, "analysis_table")
    if table is not None:
        return pd.read_csv(table, parse_dates=["date"])
    from voice_ai.data import (  # paper code, imported unmodified
        build_patient_scores_df,
        get_score_map,
        load_age_map_from_xlsx,
        load_sex_map_from_xlsx,
        load_voice_measure_maps_from_xlsx,
    )

    with config.path("score_cache").open(encoding="utf-8") as handle:
        word_scores = {str(k): float(v) for k, v in json.load(handle)["reconstruction_map"].items()}
    _, rbh_map, _ = get_score_map(csv_path=config.path("rbh_csv"), encoding=config.raw["paths"]["rbh_csv_encoding"])

    df = build_patient_scores_df(word_scores, rbh_map, score_column="score")
    df = df.rename(columns={"patient_id": "recording_id", "mean_score": "score", "num_recordings": "n_words"})

    gmm = pd.read_csv(config.path("gmm_scores"))
    df["gmm_score"] = df["recording_id"].map(dict(zip(gmm["patient_id"].astype(str), gmm["mean_gmm_score"])))

    xlsx = config.path("metadata_xlsx")
    df["age"] = df["recording_id"].map(load_age_map_from_xlsx(xlsx))
    df["sex_female"] = df["recording_id"].map(load_sex_map_from_xlsx(xlsx))
    measures = load_voice_measure_maps_from_xlsx(xlsx)
    for label, column in config.data["targets"].items():
        df[label] = df["recording_id"].map(measures[column])

    for label, column in config.data.get("extra_workbook_targets", {}).items():
        df[label] = df["recording_id"].map(_workbook_column(xlsx, column))

    person, date = _person_and_date_maps(config)
    df["date"] = df["recording_id"].map(date)
    person_key = df["recording_id"].map(person)
    if person_key.isna().any():
        raise ValueError(f"{int(person_key.isna().sum())} recordings could not be matched to a person")
    df["patient"] = pd.factorize(person_key)[0]

    df["healthy"] = df["recording_id"].str.contains(config.data["healthy_marker"], na=False)
    return df.drop(columns=["recording_id"]).reset_index(drop=True)


def load_workbook_recordings(config: Config) -> pd.DataFrame:
    """All recordings listed in the clinical workbook (the basis of the manuscript's Table 1)."""
    table = _pseudonymized(config, "analysis_workbook_table")
    if table is not None:
        return pd.read_csv(table, parse_dates=["date"])
    from voice_ai.data import load_age_map_from_xlsx

    person, date = _person_and_date_maps(config)
    ages = load_age_map_from_xlsx(config.path("metadata_xlsx"))
    wb = pd.DataFrame({"recording_id": list(person)})
    wb["age"] = wb["recording_id"].map(ages)
    wb["patient"] = pd.factorize(wb["recording_id"].map(person))[0]
    wb["date"] = wb["recording_id"].map(date)
    wb["healthy"] = wb["recording_id"].str.contains(config.data["healthy_marker"], na=False)
    return wb.drop(columns=["recording_id"])


def patients_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["healthy"]].copy()


def complete(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.dropna(subset=columns).copy()


def earliest_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Earliest recording (by recording date) of each patient among the rows given."""
    order = df.sort_values(["patient", "date"], na_position="last")
    return order.groupby("patient", sort=False).head(1).copy()


def counts(df: pd.DataFrame) -> dict:
    return {"n_recordings": int(len(df)), "n_patients": int(df["patient"].nunique())}


def age_strata(df: pd.DataFrame, bounds: list[float]) -> pd.Series:
    labels = [f"{int(lo)}-{int(hi)}" if hi < 150 else f">={int(lo)}" for lo, hi in zip(bounds[:-1], bounds[1:])]
    labels[0] = f"<{int(bounds[1])}"
    return pd.cut(df["age"], bins=bounds, right=False, labels=labels)


def summarize_age(values: pd.Series) -> dict:
    a = values.dropna().to_numpy(dtype=float)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "p05": float(np.quantile(a, 0.05)),
        "p95": float(np.quantile(a, 0.95)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _shift_dates(df: pd.DataFrame) -> pd.Series:
    """Shift each patient's dates so that the first recording falls on 2000-01-01 (intervals preserved)."""
    first = df.groupby("patient")["date"].transform("min")
    return pd.Timestamp("2000-01-01") + (df["date"] - first)


def export_pseudonymized(config: Config, out_dir) -> dict:
    """Write the analysis inputs without identifying information.

    Rows and the patient numbers are exactly those used by the analyses; names, birth dates and
    recording IDs are not included, and recording dates are shifted per patient (intervals between
    a patient's visits are preserved). Ages are kept unrounded so that all results reproduce exactly.
    """
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = load_recordings(config)
    rec = rec.assign(date=_shift_dates(rec))
    wb = load_workbook_recordings(config)
    wb = wb.assign(date=_shift_dates(wb))
    paths = {"analysis_table": out_dir / "recordings_pseudonymized.csv",
             "analysis_workbook_table": out_dir / "workbook_recordings_pseudonymized.csv"}
    rec.to_csv(paths["analysis_table"], index=False)
    wb.to_csv(paths["analysis_workbook_table"], index=False)
    return paths
