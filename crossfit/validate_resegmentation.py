"""Check that the re-segmented clips reproduce the reported recording scores.

Score the re-segmented clips with the epoch-30 checkpoint of the main model
(run_fold.py --fold -1 --checkpoint <epoch-30 checkpoint>), then compare the recording means with the word-level
score cache written by voice_ai/evaluate_student.py for the reported run.

    python validate_resegmentation.py --scores runs/scores_p0_fold-1.csv --cache <evaluation_cache.json>
"""
import argparse
import json

import pandas as pd
from scipy.stats import pearsonr, spearmanr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scores', required=True, help='scores CSV of run_fold.py scored with the reported checkpoint')
    ap.add_argument('--cache', required=True, help='evaluation_cache.json of the reported run')
    args = ap.parse_args()

    words = pd.Series(json.load(open(args.cache))['reconstruction_map'])
    reported = words.groupby(words.index.str.replace(r'_[a-zäöüß]+_\d+$', '', regex=True)).mean()
    rescored = pd.read_csv(args.scores).groupby('recording').score.mean()
    both = pd.concat([reported.rename('reported'), rescored.rename('rescored')], axis=1).dropna()
    print(f'recordings: {len(both)}; Spearman {spearmanr(both.reported, both.rescored)[0]:.3f}; '
          f'Pearson {pearsonr(both.reported, both.rescored)[0]:.3f}')
    for label, mask in [('healthy', both.index.str.startswith('Phon_GA')), ('patients', ~both.index.str.startswith('Phon_GA'))]:
        print(f'{label}: median reported {both.reported[mask].median():.1f}, rescored {both.rescored[mask].median():.1f}')


if __name__ == '__main__':
    main()
