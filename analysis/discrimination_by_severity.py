"""Discrimination by perceived severity within the pathological cohort (main model).

Recordings rated at most mildly impaired (sum of R, B and H below 3) vs. moderately to severely impaired (above 6):
AUC of the anomaly score and of the GMM baseline, a less extreme split (below 3.5 vs. above 5.5), each patient's
first recording, age alone, and the score with the linear effects of age and sex removed. 95% CIs from a bootstrap
over patients. Uses the data loader of the analyses (pipeline/data.py); only aggregates are printed.

    python discrimination_by_severity.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline.config import load_config  # noqa: E402
from pipeline.data import load_recordings  # noqa: E402

BOOTSTRAP, SEED = 2000, 1


def auc_ci(d, col, low, high):
    """AUC for rating sum < low vs. > high, with a patient-level bootstrap CI."""
    g = d[(d.S < low) | (d.S > high)].dropna(subset=[col])
    auc = roc_auc_score(g.S > high, g[col])
    rng = np.random.default_rng(SEED)
    by = {k: v for k, v in g.groupby('patient')}
    patients, draws = g.patient.unique(), []          # order of appearance, as for the reported CIs
    for _ in range(BOOTSTRAP):
        s = pd.concat([by[k] for k in rng.choice(patients, len(patients))])
        if (s.S > high).nunique() == 2:
            draws.append(roc_auc_score(s.S > high, s[col]))
    return auc, np.percentile(draws, 2.5), np.percentile(draws, 97.5), int((g.S < low).sum()), int((g.S > high).sum())


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    df = load_recordings(load_config(HERE / 'config.yaml'))
    p = df[~df.healthy & df[['R', 'B', 'H']].notna().all(axis=1)].copy()
    p['S'] = p.R + p.B + p.H
    X = np.c_[np.ones(len(p)), p.age, p.sex_female]
    ok = np.isfinite(X).all(axis=1)
    beta = np.linalg.lstsq(X[ok], p.score[ok], rcond=None)[0]
    p['score_adjusted'] = p.score - X[:, 1:] @ beta[1:]          # linear age and sex effect removed

    print(f'pathological recordings with R, B and H: {len(p)} from {p.patient.nunique()} patients')
    for low, high in [(3, 6), (3.5, 5.5)]:
        print(f'\nsum of R, B and H < {low} vs. > {high}')
        for col in ['score', 'gmm_score', 'age', 'score_adjusted']:
            auc, lo, hi, n0, n1 = auc_ci(p, col, low, high)
            print(f'  {col:15s} AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]  ({n0} vs. {n1} recordings)')
    first = p.sort_values('date').groupby('patient').head(1)
    auc, lo, hi, n0, n1 = auc_ci(first, 'score', 3, 6)
    print(f'\nfirst recording per patient, < 3 vs. > 6: AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]  ({n0} vs. {n1})')
    g = p[(p.S < 3) | (p.S > 6)]
    print(f'median age: {g.age[g.S < 3].median():.1f} (< 3) vs. {g.age[g.S > 6].median():.1f} years (> 6)')


if __name__ == '__main__':
    main()
