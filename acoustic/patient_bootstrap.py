"""Patient-level bootstrap for the acoustic association results.

The original analysis (analyze_associations.py) resampled source recordings. Several
patients contributed recordings from follow-up visits, so this script recomputes the
confidence intervals by resampling patients, as in ../analysis. Point estimates are unchanged.

Patients are identified by a hash of (surname, first name, birth date) from the clinical
workbook. The identity columns and the hash stay in memory; only aggregates are written.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyze_associations import ACOUSTICS, EXPLORATORY, LABELS, pearson, rank_model, zrank

import os

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get('VOICE_DATA_ROOT') or Path(__file__).resolve().parents[1])   # data root (default: repository root)
ASSOC = HERE / 'outputs' / 'associations'
OUT = HERE / 'outputs' / 'patient_bootstrap'
WORKBOOK = DATA_ROOT / 'Voice_KI_RBH_full.xlsx'
IDENTITY = ['Nachname', 'Vorname', 'Geburtsdatum']


def patient_map(xlsx):
    person = {}
    for sheet in pd.ExcelFile(xlsx).sheet_names:
        frame = pd.read_excel(xlsx, sheet_name=sheet)
        if not {'Dateiname', *IDENTITY}.issubset(frame.columns):
            continue
        frame = frame[frame['Dateiname'].notna()]
        for _, row in frame.iterrows():
            key = '|'.join(str(row[c]).strip().lower() for c in IDENTITY)
            person[str(row['Dateiname'])] = hashlib.sha256(key.encode('utf-8')).hexdigest()
    return person


def attach_patients(frame, person):
    key = frame['source_recording_id'].astype(str).map(person)
    if key.isna().any():
        raise ValueError(f'{int(key.isna().sum())} recordings could not be matched to a patient')
    frame = frame.copy()
    frame['patient'] = pd.factorize(key)[0]
    return frame


def cluster_bootstrap(values, patient, n_boot, rng, partial=False):
    """Percentile CI for (partial) rank correlation, resampling whole patients."""
    values = np.asarray(values, dtype=float)
    rows_of = [np.flatnonzero(patient == p) for p in np.unique(patient)]
    result = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(rows_of), len(rows_of))
        b = values[np.concatenate([rows_of[i] for i in pick])]
        try:
            if partial:
                r = rank_model(b, robust=False)
            else:
                z = zrank(b)
                r = pearson(z[:, 0], z[:, 1])
            if np.isfinite(r):
                result.append(r)
        except ValueError:
            pass
    if len(result) < n_boot * .95:
        raise ValueError('Too many invalid bootstrap samples')
    lo, hi = np.quantile(result, [.025, .975])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261028)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    person = patient_map(WORKBOOK)

    original = pd.read_csv(ASSOC / 'score_feature_correlations.csv').set_index('feature')
    rows = []
    for feat in ACOUSTICS + EXPLORATORY:
        frame = attach_patients(pd.read_csv(ASSOC / f'score_feature_dataset_{feat}.csv'), person)
        vals = frame[['score', feat]].to_numpy(float)
        lo, hi = cluster_bootstrap(vals, frame['patient'].to_numpy(), args.bootstrap, rng)
        rows.append({'feature': feat, 'label': LABELS[feat],
                     'n_recordings': len(frame), 'n_patients': int(frame['patient'].nunique()),
                     'rho': float(stats.spearmanr(vals[:, 0], vals[:, 1]).statistic),
                     'ci95_low_patient': lo, 'ci95_high_patient': hi,
                     'ci95_low_recording': float(original.loc[feat, 'ci95_low']),
                     'ci95_high_recording': float(original.loc[feat, 'ci95_high']),
                     'q_bh_recording_permutation': float(original.loc[feat, 'q_bh'])})
    features = pd.DataFrame(rows)
    features.to_csv(OUT / 'score_feature_correlations_patient_bootstrap.csv', index=False)

    frame = attach_patients(pd.read_csv(ASSOC / 'analysis_dataset_primary_common_clips.csv'), person)
    specs = {'unadjusted': [], 'acoustic': ACOUSTICS, 'age_sex': ['age_years', 'sex'],
             'acoustic_age_sex': ACOUSTICS + ['age_years', 'sex']}
    rows = []
    for outcome in ['R', 'B', 'H']:
        complete = frame.dropna(subset=['score', outcome] + ACOUSTICS + ['age_years', 'sex'])
        for model, covars in specs.items():
            vals = complete[['score', outcome] + covars].to_numpy(float)
            res = rank_model(vals)
            lo, hi = cluster_bootstrap(vals, complete['patient'].to_numpy(), args.bootstrap, rng, partial=True)
            rows.append({'outcome': outcome, 'model': model, 'n_recordings': len(complete),
                         'n_patients': int(complete['patient'].nunique()), 'rho': res['rho'],
                         'ci95_low_patient': lo, 'ci95_high_patient': hi})
    rbh = pd.DataFrame(rows)
    rbh.to_csv(OUT / 'rbh_adjusted_associations_patient_bootstrap.csv', index=False)

    (OUT / 'config.json').write_text(json.dumps({
        'created_utc': datetime.now(timezone.utc).isoformat(), 'bootstrap': args.bootstrap,
        'seed': args.seed, 'resampling_unit': 'patient (hashed identity, in memory only)',
        'inputs': 'outputs/associations/score_feature_dataset_*.csv, analysis_dataset_primary_common_clips.csv'},
        indent=2))
    print(features.round(3).to_string(index=False))
    print(rbh.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
