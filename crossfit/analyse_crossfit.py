"""Cross-fitted comparison of held-out healthy speakers with patients grouped by perceived severity.

Inputs: runs/scores_p{partition}_fold{fold}.csv from run_fold.py (5 partitions x 5 folds by person),
the RBH table and the clinical workbook (age and person identity, used in memory only).
Each healthy recording is scored by the 5 models (one per partition) that did not see its speaker;
each patient recording by all 25 models. Recording score = mean over these models.
95% CIs: bootstrap over people (healthy speakers and patients resampled separately).

    python analyse_crossfit.py --runs runs --results results

Writes only aggregates (results/crossfit_summary.json); recording IDs and identities stay in memory.
"""
import argparse
import os
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get('VOICE_DATA_ROOT') or HERE.parent)   # data root (default: repository root)
GROUPS = [('R+B+H < 3', lambda s: s < 3), ('R+B+H 3-6', lambda s: (s >= 3) & (s <= 6)),
          ('R+B+H > 6', lambda s: s > 6), ('All patients', lambda s: s.notna() | s.isna())]


def load(data_root, runs):
    rbh = pd.read_csv(data_root / 'audio_img/speech/ICCAS_RBH_DATA_with_healthy_full.csv', encoding='ISO-8859-1')
    for c in 'RBH':
        rbh[c] = pd.to_numeric(rbh[c].astype(str).str.replace(',', '.'), errors='coerce')
    rbh = rbh.dropna(subset=['Dateiname', 'R', 'B', 'H']).drop_duplicates('Dateiname')
    severity = pd.Series((rbh.R + rbh.B + rbh.H).values, index=rbh.Dateiname.astype(str).str.strip())

    age, person = {}, {}
    for frame in pd.read_excel(data_root / 'Voice_KI_RBH_full.xlsx', sheet_name=None).values():
        if not {'Dateiname', 'Geburtsdatum', 'Aufnahmedatum', 'Nachname', 'Vorname'}.issubset(frame.columns):
            continue
        frame = frame[frame.Dateiname.notna()]
        years = (pd.to_datetime(frame.Aufnahmedatum, errors='coerce')
                 - pd.to_datetime(frame.Geburtsdatum, errors='coerce')).dt.days / 365.25
        for (_, row), a in zip(frame.iterrows(), years):
            key = str(row.Dateiname).replace('.wav', '').strip()
            identity = '|'.join(str(row[c]).strip().lower() for c in ['Nachname', 'Vorname', 'Geburtsdatum'])
            age.setdefault(key, a)
            person.setdefault(key, hashlib.sha256(identity.encode()).hexdigest())

    scores = pd.concat(pd.read_csv(f) for f in glob.glob(str(Path(runs) / 'scores_p*_fold*.csv')))
    model = scores.groupby(['partition', 'fold', 'recording', 'role'], as_index=False).score.mean()
    healthy = model[model.role == 'healthy_heldout']
    patients = model[model.role == 'patient']
    per_partition = {
        p: (healthy[healthy.partition == p].set_index('recording').score,
            patients[patients.partition == p].groupby('recording').score.mean())
        for p in sorted(model.partition.unique())}
    rec = pd.concat([healthy.groupby('recording').score.mean().to_frame().assign(healthy=True),
                     patients.groupby('recording').score.mean().to_frame().assign(healthy=False)])
    rec['S'] = rec.index.map(severity)
    rec['age'] = rec.index.map(age)
    rec['person'] = [person.get(r, 'unmatched_' + r) for r in rec.index]
    n_models = model[model.role != 'healthy_train'].groupby('recording').size()   # out-of-sample scores per recording
    return rec, per_partition, n_models


def auc(h, p):
    return roc_auc_score(np.r_[np.zeros(len(h)), np.ones(len(p))], np.r_[h, p])


def equal_rating_effect(d):
    """OLS score ~ R+B+H + patient on rated recordings: (per rating point, patient vs. healthy)."""
    X = np.c_[np.ones(len(d)), d.S, ~d.healthy]
    beta = np.linalg.lstsq(X, d.score.values, rcond=None)[0]
    return beta[1], beta[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--runs', type=Path, default=HERE / 'runs')
    parser.add_argument('--results', type=Path, default=HERE / 'results')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261103)
    args = parser.parse_args()
    rec, per_partition, n_models = load(args.data_root, args.runs)
    healthy, patients = rec[rec.healthy], rec[~rec.healthy]
    rng = np.random.default_rng(args.seed)
    people_h = {k: v for k, v in healthy.groupby('person')}
    people_p = {k: v for k, v in patients.groupby('person')}

    boot = []
    for _ in range(args.bootstrap):
        h = pd.concat([people_h[k] for k in rng.choice(list(people_h), len(people_h))])
        p = pd.concat([people_p[k] for k in rng.choice(list(people_p), len(people_p))])
        row = {name: auc(h.score, p[sel(p.S)].score) for name, sel in GROUPS}
        rated = pd.concat([h, p]).dropna(subset=['S'])
        row['per_point'], row['patient_offset'] = equal_rating_effect(rated)
        boot.append(row)
    boot = pd.DataFrame(boot)
    ci = lambda col: [float(boot[col].quantile(.025)), float(boot[col].quantile(.975))]

    out = {'n_models_per_recording': {'healthy': sorted(set(n_models[healthy.index])),
                                      'patients': sorted(set(n_models[patients.index]))},
           'groups': [], 'equal_rating': {}, 'by_rating_bin': []}
    out['groups'].append({'group': 'Healthy, held out', 'recordings': len(healthy), 'people': healthy.person.nunique(),
                          'age_median': float(healthy.age.median()), 'age_mean': float(healthy.age.mean()),
                          'score_median': float(healthy.score.median())})
    for name, sel in GROUPS:
        g = patients[sel(patients.S)]
        partition_aucs = [auc(h, p.reindex(g.index).dropna()) for h, p in per_partition.values()]
        out['groups'].append({'group': name, 'recordings': len(g), 'people': g.person.nunique(),
                              'age_median': float(g.age.median()), 'age_mean': float(g.age.mean()),
                              'score_median': float(g.score.median()), 'auroc': auc(healthy.score, g.score),
                              'auroc_ci': ci(name), 'auroc_partitions': [min(partition_aucs), max(partition_aucs)]})
    rated = rec.dropna(subset=['S'])
    per_point, offset = equal_rating_effect(rated)
    out['equal_rating'] = {'recordings_rated': {'healthy': int(rated.healthy.sum()), 'patients': int((~rated.healthy).sum())},
                           'per_point': per_point, 'per_point_ci': ci('per_point'),
                           'patient_offset': offset, 'patient_offset_ci': ci('patient_offset')}
    for label, lo, hi in [('0', -.1, 0), ('0.5-1.5', .25, 1.5), ('2-3', 1.75, 3)]:
        h = healthy[(healthy.S > lo) & (healthy.S <= hi)].score
        p = patients[(patients.S > lo) & (patients.S <= hi)].score
        out['by_rating_bin'].append({'rating': label, 'healthy_n': len(h), 'healthy_median': float(h.median()),
                                     'patients_n': len(p), 'patients_median': float(p.median()), 'auroc': auc(h, p)})
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / 'crossfit_summary.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
