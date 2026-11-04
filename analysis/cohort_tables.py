"""Tables 1 and 2 of the manuscript, recomputed on the recordings that were scored and analysed.

The submitted Tables 1 and 2 were counted from the clinical workbook (364 patient and
107 healthy recordings), but only 322 patient and 90 healthy recordings have anomaly
scores. This script uses the data loader of the analyses (pipeline/data.py):
patients are identified by a hash of their identity columns, kept in memory only. Only
aggregates are written.

Run from this directory:
    python cohort_tables.py
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline.config import load_config  # noqa: E402
from pipeline.data import earliest_per_patient, load_recordings  # noqa: E402

OUT = HERE / 'results' / 'cohort_tables'

# Recording-ID prefix -> Table 2 row; everything else is "Others".
TABLE2_GROUPS = {
    'Stimmlippenpolyp': 'Vocal fold polyp',
    'chronische': 'Chronic laryngitis',
    'Larynxkarzinom': 'Laryngeal carcinoma',
    'Stimmlippenparese': 'Unilateral vocal fold paralysis',
    'Larynxpapillomatose': 'Laryngeal papillomatosis',
    'Reinke': "Reinke's oedema",
    'funktionelle': 'Functional dysphonia',
    'Stimmlippenzyste': 'Vocal fold cyst',
    'Sulcus': 'Sulcus vocalis',
}


def table1(df):
    rows = []
    for label, cohort in [('Healthy cohort', df[df.healthy]), ('Pathological cohort', df[~df.healthy])]:
        persons = earliest_per_patient(cohort)
        sex = persons.sex_female
        rows.append({
            'cohort': label,
            'individuals': int(cohort.patient.nunique()),
            'recordings': len(cohort),
            'pct_male': 100 * float((sex == 0).mean()),
            'pct_female': 100 * float((sex == 1).mean()),
            'pct_unspecified': 100 * float(sex.isna().mean()),
            'age_median_person': float(persons.age.median()),
            'age_p05_person': float(persons.age.quantile(.05)),
            'age_p95_person': float(persons.age.quantile(.95)),
        })
        rated = cohort.dropna(subset=['R', 'B', 'H'])          # perceptual ratings, recording level
        rows[-1]['recordings_rated'] = len(rated)
        for name, values in [('R', rated.R), ('B', rated.B), ('H', rated.H), ('RBH_sum', rated.R + rated.B + rated.H)]:
            rows[-1][f'{name}_median'] = float(values.median())
            rows[-1][f'{name}_q25'] = float(values.quantile(.25))
            rows[-1][f'{name}_q75'] = float(values.quantile(.75))
    return pd.DataFrame(rows)


def table2(df):
    patients = df[~df.healthy].copy()
    patients['group'] = patients.condition.map(TABLE2_GROUPS).fillna('Others')
    order = list(TABLE2_GROUPS.values()) + ['Others']
    rows = []
    for group in order:
        g = patients[patients.group == group]
        rows.append({'group': group, 'recordings': len(g), 'patients': int(g.patient.nunique()),
                     'age_median': float(g.age.median()), 'age_sd': float(g.age.std(ddof=1)),
                     'number_words': int(g.n_words.sum())})
    others = patients[patients.group == 'Others'].condition.value_counts()
    return pd.DataFrame(rows), others


def main():
    config = load_config(HERE / 'config.yaml')
    df = load_recordings(config)
    OUT.mkdir(exist_ok=True)
    t1 = table1(df)
    t2, others = table2(df)
    t1.to_csv(OUT / 'table1_analysed.csv', index=False)
    t2.to_csv(OUT / 'table2_analysed.csv', index=False)
    others.rename('recordings').to_csv(OUT / 'table2_others_composition.csv')
    print(t1.round(1).to_string(index=False))
    print(t2.round(2).to_string(index=False))
    print(others.to_string())
    print('total patient recordings', int((~df.healthy).sum()), 'words', int(df[~df.healthy].n_words.sum()))


if __name__ == '__main__':
    main()
