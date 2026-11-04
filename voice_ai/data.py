from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from voice_ai.diagnoses import diagnoses_map_eng
from voice_ai.paths import DATA_DIR


DEFAULT_METADATA_CSV = DATA_DIR / "metadata" / "rbh.csv"
DEFAULT_METADATA_ENCODING = "ISO-8859-1"


def to_num(score):
    if isinstance(score, str):
        return float(score.replace(",", "."))
    return score


def normalize_recording_key(key: str) -> str:
    return key


def extract_patient_id(key: str) -> str:
    normalized = normalize_recording_key(key)
    parts = normalized.rsplit("_", 2)
    if len(parts) == 3:
        return parts[0]
    return normalized


def extract_condition(key: str) -> str:
    normalized = normalize_recording_key(key)
    return normalized.split("_", 1)[0]


def get_score_map(csv_path: Path = DEFAULT_METADATA_CSV, encoding: str = DEFAULT_METADATA_ENCODING):
    fname_to_score_map = {}
    patients_map = {}
    if not csv_path.is_file():
        return [], fname_to_score_map, patients_map

    metadata = pd.read_csv(csv_path, encoding=encoding)
    scores = []

    for _, row in metadata.iterrows():
        fname = row["Dateiname"]
        pid = row["Dateiname"]
        recording_date = row["Aufnahmedatum"]
        r = row["R"]
        b = row["B"]
        h = row["H"]

        if pd.isna(r) or pd.isna(b) or pd.isna(h):
            continue
        if "-" in [r, b, h]:
            continue

        r = to_num(r)
        b = to_num(b)
        h = to_num(h)
        cumul = r + b + h
        scores.append(cumul)
        fname_to_score_map[fname] = (cumul, r, b, h)
        patients_map[fname] = (recording_date, pid)

    return scores, fname_to_score_map, patients_map


def build_recording_scores_df(score_map: dict[str, float], score_column: str) -> pd.DataFrame:
    rows = []
    for recording_id, score in score_map.items():
        rows.append(
            {
                "recording_id": recording_id,
                "patient_id": extract_patient_id(recording_id),
                "condition": extract_condition(recording_id),
                score_column: float(score),
            }
        )
    return pd.DataFrame(rows)


def build_patient_scores_df(
    score_map: dict[str, float],
    rbh_map: dict[str, tuple[float, float, float, float]] | None,
    score_column: str,
) -> pd.DataFrame:
    patient_rows = []
    patient_scores = defaultdict(list)
    patient_conditions = {}
    mean_score_column = f"mean_{score_column}"

    for recording_id, score in score_map.items():
        patient_id = extract_patient_id(recording_id)
        patient_scores[patient_id].append(score)
        patient_conditions[patient_id] = extract_condition(recording_id)

    rbh_map = rbh_map or {}
    for patient_id, values in patient_scores.items():
        row = {
            "patient_id": patient_id,
            "condition": patient_conditions.get(patient_id),
            mean_score_column: float(np.mean(values)),
            "num_recordings": int(len(values)),
        }
        if patient_id in rbh_map:
            row["RBH"] = float(rbh_map[patient_id][0])
            row["R"] = float(rbh_map[patient_id][1])
            row["B"] = float(rbh_map[patient_id][2])
            row["H"] = float(rbh_map[patient_id][3])
        patient_rows.append(row)

    return pd.DataFrame(patient_rows)


def build_reverse_diagnoses_map() -> dict[int, str]:
    reverse_map = {}
    for name, value in diagnoses_map_eng.items():
        if value not in reverse_map:
            reverse_map[value] = name
    return reverse_map


def load_age_map_from_xlsx(xlsx_path: Path) -> dict[str, float]:
    if not xlsx_path.is_file():
        return {}

    xl = pd.ExcelFile(xlsx_path)
    age_map = {}
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
        required_cols = {"Dateiname", "Geburtsdatum", "Aufnahmedatum"}
        if not required_cols.issubset(df.columns):
            continue

        dates_df = df[["Dateiname", "Geburtsdatum", "Aufnahmedatum"]].copy()
        dates_df["Geburtsdatum"] = pd.to_datetime(dates_df["Geburtsdatum"], errors="coerce")
        dates_df["Aufnahmedatum"] = pd.to_datetime(dates_df["Aufnahmedatum"], errors="coerce")
        dates_df["age_years"] = (dates_df["Aufnahmedatum"] - dates_df["Geburtsdatum"]).dt.days / 365.2425

        for row in dates_df.itertuples(index=False):
            if pd.isna(row.age_years):
                continue
            recording_key = str(row.Dateiname)
            normalized_key = normalize_recording_key(recording_key)
            patient_key = extract_patient_id(normalized_key)
            age_value = float(row.age_years)
            age_map[recording_key] = age_value
            age_map[normalized_key] = age_value
            age_map[patient_key] = age_value

    return age_map


def load_voice_measure_maps_from_xlsx(xlsx_path: Path) -> dict[str, dict[str, float]]:
    if not xlsx_path.is_file():
        return {}

    column_aliases = {
        "Stimmstärke": "Stimmstärke dB(A)",
        "Haltedauer": "Tonhaltedauer (s)",
    }
    measure_maps = {name: {} for name in column_aliases}

    xl = pd.ExcelFile(xlsx_path)
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
        required_cols = {"Dateiname", *column_aliases.values()}
        if not required_cols.issubset(df.columns):
            continue

        subset = df[["Dateiname", *column_aliases.values()]].copy()
        for _, row in subset.iterrows():
            recording_key = str(row["Dateiname"])
            if not recording_key or recording_key == "nan":
                continue

            normalized_key = normalize_recording_key(recording_key)
            patient_key = extract_patient_id(normalized_key)
            for short_name, column_name in column_aliases.items():
                value = pd.to_numeric(pd.Series([row[column_name]]), errors="coerce").iloc[0]
                if pd.isna(value):
                    continue
                numeric_value = float(value)
                measure_maps[short_name][recording_key] = numeric_value
                measure_maps[short_name][normalized_key] = numeric_value
                measure_maps[short_name][patient_key] = numeric_value

    return measure_maps


def load_sex_map_from_xlsx(xlsx_path: Path) -> dict[str, float]:
    if not xlsx_path.is_file():
        return {}

    sex_map = {}
    xl = pd.ExcelFile(xlsx_path)
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
        required_cols = {"Dateiname", "Geschlecht"}
        if not required_cols.issubset(df.columns):
            continue

        subset = df[["Dateiname", "Geschlecht"]].copy()
        for _, row in subset.iterrows():
            recording_key = str(row["Dateiname"])
            if not recording_key or recording_key == "nan":
                continue

            normalized_key = normalize_recording_key(recording_key)
            patient_key = extract_patient_id(normalized_key)
            raw_value = str(row["Geschlecht"]).strip().lower()
            if raw_value in {"w", "weiblich", "female", "f", "frau"}:
                encoded = 1.0
            elif raw_value in {"m", "männlich", "maennlich", "male", "mann"}:
                encoded = 0.0
            else:
                continue

            sex_map[recording_key] = encoded
            sex_map[normalized_key] = encoded
            sex_map[patient_key] = encoded

    return sex_map
