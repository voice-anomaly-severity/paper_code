"""Figure: score distributions of held-out healthy speakers and of patients grouped by R+B+H.

Uses the cross-fitted recording scores of analyse_crossfit.py (5 partitions x 5 folds by person) and
the AUROCs with 95% CIs from results/crossfit_summary.json (run analyse_crossfit.py first).

    python plot_distributions.py --runs runs --results results --out figure3_crossfit.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import gaussian_kde

from analyse_crossfit import DATA_ROOT, GROUPS, HERE, load

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e1', '#ffffff'
COLORS = {'Healthy, held out': '#eb6834', 'R+B+H < 3': '#86b6ef', 'R+B+H 3-6': '#2a78d6', 'R+B+H > 6': '#104281'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--runs', type=Path, default=HERE / 'runs')
    parser.add_argument('--results', type=Path, default=HERE / 'results')
    parser.add_argument('--out', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    rec, _, _ = load(args.data_root, args.runs)
    summary = {g['group']: g for g in json.loads((args.results / 'crossfit_summary.json').read_text())['groups']}
    healthy, patients = rec[rec.healthy], rec[~rec.healthy]
    series = [('Healthy, held out', healthy)] + [(name, patients[sel(patients.S)]) for name, sel in GROUPS[:3]]

    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                         'xtick.color': INK2, 'ytick.color': INK2})
    fig, ax = plt.subplots(figsize=(7.5, 5.0), facecolor=SURFACE)
    x = np.linspace(25, 95, 600)
    for name, g in series:
        s = summary[name]
        label = (f"{'Healthy speakers, held out' if name.startswith('Healthy') else 'Patients, ' + name.replace('3-6', '3–6')}"
                 f" (n = {s['recordings']}; age {s['age_median']:.0f} y)")
        if 'auroc' in s:
            label += f" · AUROC {s['auroc']:.2f} [{s['auroc_ci'][0]:.2f}, {s['auroc_ci'][1]:.2f}]"
        density = gaussian_kde(g.score.values)(x)
        if name.startswith('Healthy'):
            ax.fill_between(x, density, color=COLORS[name], alpha=.18, lw=0)
        ax.plot(x, density, lw=2, color=COLORS[name], label=label)
    ax.set_xlim(25, 95)
    ax.set_ylim(0, None)
    ax.set_xlabel('Anomaly score (cross-fitted, recording mean)')
    ax.set_ylabel('Density')
    ax.grid(axis='y', color=GRID, lw=.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8.3, loc='upper left', bbox_to_anchor=(0, -0.14), labelcolor=INK)
    fig.tight_layout()
    for out in args.out:
        fig.savefig(out, dpi=200, facecolor=SURFACE)
        print(out)


if __name__ == '__main__':
    main()
