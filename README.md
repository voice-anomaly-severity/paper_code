# Unsupervised anomaly-based severity proxy for voice pathology

Code for the paper "An unsupervised anomaly-based severity proxy for voice pathology assessment".

Innovation Center Computer Assisted Surgery (ICCAS), University of Leipzig, Leipzig, Germany.

| Folder | Content | Paper |
|---|---|---|
| `segmentation/` | Cut the counting-task recordings into number words (21–29) with a German wav2vec2 model | Methods |
| `voice_ai/` | Teacher–student model: training, anomaly scores, GMM baseline | Methods, Figure 2 |
| `analysis/` | Statistical analyses; `cohort_tables.py`; `discrimination_by_severity.py` | Tables 1–4, Supplementary Tables S4, S5, S8, S9 |
| `crossfit/` | Comparison with held-out healthy speakers (cross-fitting by speaker, 5 splits × 5 folds) | Figure 3 |
| `acoustic/` | F0, CPPS, spectral slope and intensity | Supplementary Tables S6–S7 |

Each folder has its own `requirements.txt`.

## Run

```bash
python segmentation/extract_number_words.py --input-dir <healthy wavs> --output-dir data/ukl_wavs/healthy_studis/number
python segmentation/extract_number_words.py --input-dir <patient wavs> --output-dir data/ukl_wavs/unhealthy/number --skip-a
python voice_ai/scripts/train_teacher_student.py --epochs 50        # the paper uses the epoch-30 checkpoint
python voice_ai/scripts/evaluate_teacher_student.py --epoch-to-load 30
python voice_ai/scripts/evaluate_gmm.py
python analysis/run.py && python analysis/cohort_tables.py && python analysis/discrimination_by_severity.py
crossfit/run_all.sh data/ukl_wavs/healthy_studis/number data/ukl_wavs/unhealthy/number 0,1
python crossfit/analyse_crossfit.py && python crossfit/plot_distributions.py --out figure3.pdf
python acoustic/extract_features.py --output-dir acoustic/outputs/available_clips
python acoustic/analyze_associations.py && python acoustic/patient_bootstrap.py
```

Input paths are set in `analysis/config.yaml`. By default, data are read relative to the repository
root; set `VOICE_DATA_ROOT` to use another location.

## License

MIT, see `LICENSE`.
