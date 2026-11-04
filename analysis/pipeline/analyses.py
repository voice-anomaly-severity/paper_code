"""Analysis steps: cohort, score-age association, Table 3, age effect, Table 4, within-patient follow-up.

Every step receives the in-memory recording table and writes aggregate tables
to the output directory. Recordings are rows; patients are the unit of resampling and of
leave-out. "earliest" = each patient's earliest recording with the variables an analysis needs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import stats as S
from .config import Config
from .models import compare, lopo, model_summary
from .report import write_table

SCORES = {"teacher_student": "score", "gmm": "gmm_score"}
RATINGS = ["R", "B", "H"]
ADJUSTMENTS = {"none": [], "age": ["age"], "age+sex": ["age", "sex_female"]}
FULL = ["score", "R", "B", "H", "sex_female", "age"]


def _rng(config: Config, offset: int) -> np.random.Generator:
    return np.random.default_rng(config.analysis["seed"] + offset)


def _ci(samples: np.ndarray) -> tuple[float, float]:
    return S.percentile_interval(samples, 0.95)


# ---------------------------------------------------------------- cohort ages
def cohort_ages(df: pd.DataFrame, config: Config, out) -> dict:
    band = config.analysis["overlap_band"]
    rows, bins = [], []
    for set_name, frame in [("workbook (Table 1 basis)", D.load_workbook_recordings(config)), ("scored recordings", df)]:
        for healthy, sub in frame.groupby("healthy"):
            sub = sub.dropna(subset=["age"])
            first = D.earliest_per_patient(sub)
            rec, per = D.summarize_age(sub["age"]), D.summarize_age(first["age"])
            in_band = sub[(sub["age"] >= band[0]) & (sub["age"] < band[1])]
            rows.append({
                "set": set_name, "cohort": "healthy" if healthy else "patients",
                "recordings": len(sub), "individuals": sub["patient"].nunique(),
                "age_median_rec": rec["median"], "age_p05_rec": rec["p05"], "age_p95_rec": rec["p95"],
                "age_min": rec["min"], "age_max": rec["max"],
                "age_median_person": per["median"], "age_p05_person": per["p05"], "age_p95_person": per["p95"],
                f"recordings_{band[0]}-{band[1]}": len(in_band), f"individuals_{band[0]}-{band[1]}": in_band["patient"].nunique(),
                f"recordings_age>={band[0]}": int((sub["age"] >= band[0]).sum()),
                f"recordings_age<{band[1]}": int((sub["age"] < band[1]).sum()),
            })
            if set_name == "scored recordings":
                edges = list(range(10, 95, 5))
                hist = pd.cut(sub["age"], bins=edges, right=False).value_counts().sort_index()
                bins.append(pd.Series(hist.to_numpy(), index=[f"{int(i.left)}-{int(i.right)}" for i in hist.index],
                                      name="healthy" if healthy else "patients"))
    table = pd.DataFrame(rows)
    write_table(table, out, "cohort_ages", "Age at recording by cohort (years) and counts in the overlap band",
                digits={c: 1 for c in table.columns if c.startswith("age")})
    hist = pd.concat(bins, axis=1).fillna(0).astype(int).reset_index(names="age_bin")
    write_table(hist, out, "cohort_age_bins", "Scored recordings per 5-year age bin and cohort")
    return {"table": table}


# ---------------------------------------------------------------- score-age correlation
def age_correlation(df: pd.DataFrame, config: Config, out) -> dict:
    draws = config.analysis["cluster_bootstrap_draws"]
    rng = _rng(config, 1)
    rows = []
    for method, col in SCORES.items():
        pooled = D.complete(df, [col, "age"])
        pats = D.complete(D.patients_only(df), [col, "age"])
        early = D.earliest_per_patient(pats)
        for label, frame, p_valid in [("healthy and patients pooled", pooled, True),
                                      ("patients, all recordings", pats, False),
                                      ("patients, earliest recording", early, True)]:
            stat = lambda f, c=col: S.adjusted_spearman(f, c, "age", [])
            boot = S.cluster_bootstrap(frame, stat, draws, rng)
            lo, hi = _ci(boot)
            rows.append({
                "score": method, "sample": label, **D.counts(frame),
                "n_healthy_recordings": int(frame["healthy"].sum()),
                "rho": stat(frame), "ci_low": lo, "ci_high": hi,
                "p_value": S.adjusted_spearman_p(frame, col, "age", []) if p_valid else np.nan,
            })
    table = pd.DataFrame(rows)
    write_table(table, out, "age_correlation",
                "Spearman correlation of anomaly score with age. CI: patient-level bootstrap",
                digits={"rho": 3, "ci_low": 3, "ci_high": 3})

    pats = D.complete(D.patients_only(df), ["age", *RATINGS])
    rating_rows = []
    for r in RATINGS:
        stat = lambda f, r=r: S.adjusted_spearman(f, "age", r, [])
        lo, hi = _ci(S.cluster_bootstrap(pats, stat, draws, rng))
        rating_rows.append({"rating": r, **D.counts(pats), "rho_age_rating": stat(pats), "ci_low": lo, "ci_high": hi})
    rating_table = pd.DataFrame(rating_rows)
    write_table(rating_table, out, "age_vs_ratings", "Spearman correlation of age with R, B, H within patients (all recordings)")
    return {"table": table, "ratings": rating_table}


# ---------------------------------------------------------------- Table 3
def table3(df: pd.DataFrame, config: Config, out) -> dict:
    draws = config.analysis["cluster_bootstrap_draws"]
    rng = _rng(config, 2)
    rows = []
    for method, col in SCORES.items():
        base = D.complete(D.patients_only(df), [col, *RATINGS, "age", "sex_female"])
        for sample, frame in [("all recordings", base), ("earliest recording", D.earliest_per_patient(base))]:
            for rating in RATINGS:
                for adj, covs in ADJUSTMENTS.items():
                    stat = lambda f, c=col, r=rating, v=covs: S.adjusted_spearman(f, c, r, v)
                    lo, hi = _ci(S.cluster_bootstrap(frame, stat, draws, rng))
                    rows.append({
                        "score": method, "sample": sample, "rating": rating, "adjustment": adj, **D.counts(frame),
                        "rho": stat(frame), "ci_low": lo, "ci_high": hi,
                        "p_value": S.adjusted_spearman_p(frame, col, rating, covs) if sample == "earliest recording" else np.nan,
                    })
    long = pd.DataFrame(rows)
    write_table(long, out, "table3_long", "Table 3 revised, all variants (patients only). CI: patient-level bootstrap; p only for one recording per patient")

    wide_rows = []
    for method in SCORES:
        for rating in RATINGS:
            sel = long[(long.score == method) & (long["sample"] == "all recordings") & (long.rating == rating)].set_index("adjustment")
            fmt = lambda a: f"{sel.loc[a, 'rho']:.3f} [{sel.loc[a, 'ci_low']:.3f}, {sel.loc[a, 'ci_high']:.3f}]"
            wide_rows.append({
                "score": method, "rating": rating,
                "ρ": fmt("none"), "age-adj. ρ": fmt("age"), "age+sex-adj. ρ": fmt("age+sex"),
                "recordings / patients": f"{int(sel.loc['none', 'n_recordings'])} / {int(sel.loc['none', 'n_patients'])}",
            })
    wide = pd.DataFrame(wide_rows)
    write_table(wide, out, "table3_revised",
                "Spearman correlation of anomaly score with R, B, H within patients (all recordings; 95% CI by patient-level bootstrap)")

    # within age strata (teacher-student score; age at recording)
    base = D.complete(D.patients_only(df), ["score", *RATINGS, "age", "sex_female"])
    base = base.assign(stratum=D.age_strata(base, config.analysis["age_strata"]))
    strata_rows = []
    for stratum, frame in base.groupby("stratum", observed=True):
        for rating in RATINGS:
            for adj, covs in [("none", []), ("age", ["age"])]:
                stat = lambda f, r=rating, v=covs: S.adjusted_spearman(f, "score", r, v)
                lo, hi = _ci(S.cluster_bootstrap(frame, stat, draws, rng))
                strata_rows.append({"age stratum": str(stratum), "rating": rating, "adjustment": adj, **D.counts(frame),
                                    "age_median": float(frame["age"].median()), "rho": stat(frame), "ci_low": lo, "ci_high": hi})
    strata = pd.DataFrame(strata_rows)
    write_table(strata, out, "table3_age_strata",
                "Spearman correlation of the teacher–student score with R, B, H within age strata (patients, all recordings in the stratum; 95% CI by patient-level bootstrap)",
                digits={"age_median": 1})
    return {"long": long, "wide": wide, "strata": strata}


# ---------------------------------------------------------------- age effect at equal severity
def age_effect(df: pd.DataFrame, config: Config, out) -> dict:
    draws = config.analysis["cluster_bootstrap_draws"]
    rng = _rng(config, 3)
    frame = D.complete(D.patients_only(df), ["score", "age", "sex_female", *RATINGS]).assign(age10=lambda f: f["age"] / 10.0)
    # diagnosis indicators: groups with >= 10 recordings, smaller groups pooled as "other";
    # reference = the largest group
    sizes = frame["condition"].value_counts()
    diag = [c for c in sizes.index if sizes[c] >= 10]
    reference = diag[0]
    grouped = frame["condition"].where(frame["condition"].isin(diag), "other")
    diag_cols = []
    for c in [*diag[1:], "other"]:
        if (grouped == c).any():
            frame[f"dx_{c}"] = (grouped == c).astype(float)
            diag_cols.append(f"dx_{c}")
    specs = {
        "score ~ age": ["age10"],
        "score ~ age + sex": ["age10", "sex_female"],
        "score ~ age + sex + H": ["age10", "sex_female", "H"],
        "score ~ age + sex + R + B + H": ["age10", "sex_female", "R", "B", "H"],
        f"score ~ age + sex + H + diagnosis ({len(diag_cols) + 1} groups)": ["age10", "sex_female", "H", *diag_cols],
    }
    rows = []
    for name, preds in specs.items():
        beta = S.ols(frame, "score", preds)
        boot = S.cluster_bootstrap(frame, lambda f, p=preds: S.ols(f, "score", p), draws, rng)
        for i, p in enumerate(preds, start=1):
            if p.startswith("dx_"):
                continue
            lo, hi = S.percentile_interval(boot[:, i])
            rows.append({"model": name, "term": {"age10": "age (per 10 years)", "sex_female": "female sex"}.get(p, p),
                         **D.counts(frame), "coefficient": beta[i], "ci_low": lo, "ci_high": hi})
    coef = pd.DataFrame(rows)
    write_table(coef, out, "age_effect_regression",
                "Score points per unit of each term within patients (OLS, all recordings; 95% CI by patient-level bootstrap)")

    partial = []
    for covs in [[], ["sex_female"], ["sex_female", "H"], ["sex_female", "R", "B", "H"]]:
        stat = lambda f, c=covs: S.adjusted_spearman(f, "score", "age", c)
        lo, hi = _ci(S.cluster_bootstrap(frame, stat, draws, rng))
        partial.append({"adjusted for": " + ".join(covs) if covs else "nothing", **D.counts(frame),
                        "rho_score_age": stat(frame), "ci_low": lo, "ci_high": hi})
    partial = pd.DataFrame(partial)
    write_table(partial, out, "age_effect_partial", "Spearman correlation of score with age within patients after residualizing on the listed covariates")
    return {"coef": coef, "partial": partial}


# ---------------------------------------------------------------- Table 4
FAMILY_LABEL = {"lmm": "linear mixed model", "gp": "GP (Methods kernel, MAP)", "linear": "linear (earliest recording)"}
PERCEPTUAL_TARGETS = ["R", "B", "H"]


def _t4_spec(target: str):
    """Full predictor set and predictor blocks of a Table 4 target."""
    if target in PERCEPTUAL_TARGETS:             # perceptual target: RBH cannot be a predictor
        full = ["score", "sex_female", "age"]
        blocks = {"score": ["score"], "sex+age": ["sex_female", "age"], "sex": ["sex_female"], "age": ["age"]}
    else:
        full = list(FULL)
        blocks = {"score": ["score"], "RBH": ["R", "B", "H"], "sex+age": ["sex_female", "age"],
                  "sex": ["sex_female"], "age": ["age"]}
    return full, blocks


def _predictor_sets(full, blocks):
    sets = {(): None, tuple(full): None}
    for cols in blocks.values():
        sets[tuple(cols)] = None
        sets[tuple(p for p in full if p not in cols)] = None
    return [list(s) for s in sets]


def _replicate(frame, seed, rep):
    """Calibration-check replicate: a pure-noise column and a random permutation of the score."""
    rng = np.random.default_rng([seed, 7, rep])
    return frame.assign(noise=rng.normal(size=len(frame)), score_perm=rng.permutation(frame["score"].to_numpy()))


def _task(key, frame, target, preds):
    return (key, frame[preds].to_numpy(float), frame[target].to_numpy(float), frame["patient"].to_numpy())


def _comparisons(fits, sample, target, family, full, blocks, draws, rng):
    rows, fit_rows = [], []
    null = fits[()]
    for block, cols in blocks.items():
        ablated = tuple(p for p in full if p not in cols)
        for kind, base, top in [("single", null, fits[tuple(cols)]), ("added", fits[ablated], fits[tuple(full)])]:
            rows.append({"sample": sample, "target": target, "model": family, "predictor": block, "comparison": kind,
                         **compare(base, top, draws, rng)})
    base_sa, top_sa = ("sex_female", "age"), ("score", "sex_female", "age")
    if base_sa in fits and top_sa in fits:
        rows.append({"sample": sample, "target": target, "model": family, "predictor": "score", "comparison": "added to sex+age",
                     **compare(fits[base_sa], fits[top_sa], draws, rng)})
    for preds, res in fits.items():
        fit_rows.append({"sample": sample, "target": target,
                         "model": family if preds else ("intercept only" if family == "linear" else "intercept + patient effect"),
                         "predictors": " + ".join(preds) if preds else "(none)", **model_summary(res)})
    return rows, fit_rows


def _lmm_check_rep(frame, target, full, rep, seed):
    rep_frame = _replicate(frame, seed, rep)
    ablated = [p for p in full if p != "score"]
    return (rep, lopo(rep_frame, target, full + ["noise"], "lmm")["logpdf"].to_numpy(),
            lopo(rep_frame, target, ablated + ["score_perm"], "lmm")["logpdf"].to_numpy())


def table4(df: pd.DataFrame, config: Config, out) -> dict:
    from joblib import Parallel, delayed

    from .gpu import run_gp_tasks

    a = config.analysis
    draws, seed, n_jobs = a["bayesian_bootstrap_draws"], a["seed"], a["n_jobs"]
    targets = a["table4"]["targets"]
    gp_reps, lmm_reps = a["calibration_check"]["gp_replicates"], a["calibration_check"]["lmm_replicates"]
    rng = _rng(config, 4)
    specs = {t: _t4_spec(t) for t in targets}
    frames = {t: D.complete(D.patients_only(df), [t, *specs[t][0]]) for t in targets}

    # secondary model: all GP fits (main predictor sets + calibration replicates) on the GPUs
    tasks = []
    for t in targets:
        full, blocks = specs[t]
        tasks += [_task(("main", t, tuple(p)), frames[t], t, p) for p in _predictor_sets(full, blocks) if p]
    for t in targets:
        full, _ = specs[t]
        ablated = [p for p in full if p != "score"]
        for r in range(gp_reps):
            rep = _replicate(frames[t], seed, r)
            tasks += [_task(("noise", t, r), rep, t, full + ["noise"]), _task(("perm", t, r), rep, t, ablated + ["score_perm"])]
    print(f"  GP: {len(tasks)} leave-one-patient-out fits (progress in {out / 'logs'})", flush=True)
    gp = run_gp_tasks(tasks, a["gp"], out / "logs")

    rows, fit_rows, lmm_fits, gp_fits = [], [], {}, {}
    for t in targets:
        full, blocks = specs[t]
        frame = frames[t]
        sets = _predictor_sets(full, blocks)
        lmm_fits[t] = {tuple(p): lopo(frame, t, p, "lmm") for p in sets}
        gp_fits[t] = {tuple(p): (lopo(frame, t, p, "gp", gp[("main", t, tuple(p))]) if p else lmm_fits[t][()]) for p in sets}
        early = D.earliest_per_patient(frame)
        early_fits = {tuple(p): lopo(early, t, p, "linear") for p in sets}
        for sample, family, fits in [("all recordings", "lmm", lmm_fits[t]), ("all recordings", "gp", gp_fits[t]),
                                     ("earliest recording", "linear", early_fits)]:
            r, f = _comparisons(fits, sample, t, family, full, blocks, draws, rng)
            rows += r
            fit_rows += f
        print(f"  table4 {t}: comparisons done", flush=True)
    long = pd.DataFrame(rows)
    fit_table = pd.DataFrame(fit_rows).drop_duplicates(subset=["sample", "target", "model", "predictors"]).reset_index(drop=True)
    write_table(long, out, "table4_long",
                "Table 4, all comparisons: Δelpd (leave-one-patient-out; Bayesian bootstrap over patients) for every predictor block, alone (single) and beyond all other predictors (added)")
    write_table(fit_table, out, "table4_model_fit",
                "Leave-one-patient-out fit of every model: elpd, RMSE, R² and mean squared standardized error (calibrated ≈ 1)")

    wide = []
    for t in targets:
        for family in ["lmm", "gp"]:
            sel = long[(long["sample"] == "all recordings") & (long.target == t) & (long.model == family)].set_index(["predictor", "comparison"])
            cell = lambda b, k: (f"{sel.loc[(b, k), 'delta_elpd']:.1f} [{sel.loc[(b, k), 'bb_q025']:.1f}, {sel.loc[(b, k), 'bb_q975']:.1f}]"
                                 if (b, k) in sel.index else "–")
            wide.append({
                "target": t, "model": FAMILY_LABEL[family],
                "recordings / patients": f"{int(sel.loc[('score', 'single'), 'n_recordings'])} / {int(sel.loc[('score', 'single'), 'n_patients'])}",
                "score alone": cell("score", "single"),
                "score beyond sex + age": cell("score", "added to sex+age"),
                "Pr(> 0)": float(sel.loc[("score", "added to sex+age"), "bb_pr_positive"]),
                "score beyond sex + age + RBH": cell("score", "added") if t not in PERCEPTUAL_TARGETS else "–",
                "Pr(> 0) ": float(sel.loc[("score", "added"), "bb_pr_positive"]) if t not in PERCEPTUAL_TARGETS else np.nan,
                "sex + age beyond others": cell("sex+age", "added"),
                "RBH beyond others": cell("RBH", "added"),
            })
    wide = pd.DataFrame(wide)
    write_table(wide, out, "table4_revised",
                "Predictive contribution of the anomaly score to severity proxies, compared with the confounders (sex + age) and RBH. "
                "Δelpd summed over patients [95% Bayesian-bootstrap interval]; leave-one-patient-out; all recordings with patient random effect. "
                "'Beyond others' = beyond all other predictors of the target's model; the perceptual targets (R, B, H) "
                "are predicted from score, sex and age only",
                digits={"Pr(> 0)": 3, "Pr(> 0) ": 3})

    # sensitivity: score within age strata (primary model)
    strata_rows = []
    for t in targets:
        full, _ = specs[t]
        frame = frames[t].assign(stratum=D.age_strata(frames[t], a["age_strata"]))
        for stratum, sub in frame.groupby("stratum", observed=True):
            ablated = [p for p in full if p != "score"]
            fits = {tuple(p): lopo(sub, t, p, "lmm") for p in [[], ["score"], ablated, full]}
            r, _ = _comparisons(fits, f"age {stratum}", t, "lmm", full, {"score": ["score"]}, draws, rng)
            strata_rows += r
    strata = pd.DataFrame(strata_rows)
    write_table(strata, out, "table4_age_strata", "Contribution of the anomaly score within age strata (linear mixed model, leave-one-patient-out)")

    # calibration check: noise column / permuted score through the identical pipeline
    check_rows = []
    for t in targets:
        full, _ = specs[t]
        ablated = tuple(p for p in full if p != "score")
        reps = Parallel(n_jobs=n_jobs)(delayed(_lmm_check_rep)(frames[t], t, full, r, seed) for r in range(lmm_reps))
        base_full, base_abl = lmm_fits[t][tuple(full)]["logpdf"].to_numpy(), lmm_fits[t][ablated]["logpdf"].to_numpy()
        for r, noise, perm in reps:
            check_rows += [{"target": t, "model": "lmm", "check": "noise column added", "replicate": r, "delta_elpd": float((noise - base_full).sum())},
                           {"target": t, "model": "lmm", "check": "permuted score added", "replicate": r, "delta_elpd": float((perm - base_abl).sum())}]
        g_full, g_abl = gp_fits[t][tuple(full)]["logpdf"].to_numpy(), gp_fits[t][ablated]["logpdf"].to_numpy()
        for r in range(gp_reps):
            check_rows += [{"target": t, "model": "gp", "check": "noise column added", "replicate": r, "delta_elpd": float((gp[("noise", t, r)]["logpdf"] - g_full).sum())},
                           {"target": t, "model": "gp", "check": "permuted score added", "replicate": r, "delta_elpd": float((gp[("perm", t, r)]["logpdf"] - g_abl).sum())}]
    reps_df = pd.DataFrame(check_rows)
    summary = []
    for (t, family, check), g in reps_df.groupby(["target", "model", "check"]):
        obs = long[(long["sample"] == "all recordings") & (long.target == t) & (long.model == family)
                   & (long.predictor == "score") & (long.comparison == "added")].delta_elpd.iloc[0]
        d = g["delta_elpd"].to_numpy()
        summary.append({"target": t, "model": FAMILY_LABEL[family], "check": check, "replicates": len(d),
                        "mean Δelpd": d.mean(), "sd": d.std(ddof=1), "95th pct": np.quantile(d, 0.95), "max": d.max(),
                        "share > 0": float(np.mean(d > 0)), "observed score beyond others": obs,
                        "empirical p": (1 + np.sum(d >= obs)) / (len(d) + 1)})
    summary = pd.DataFrame(summary)
    write_table(reps_df, out, "table4_calibration_check_replicates", "Calibration check, every replicate")
    write_table(summary, out, "table4_calibration_check",
                "Calibration check: Δelpd when a pure-noise column, or a permuted copy of the score, is added to the other predictors (identical pipeline)")
    return {"long": long, "wide": wide, "fit": fit_table, "strata": strata, "check": summary}


# ---------------------------------------------------------------- within-patient follow-up
def longitudinal(df: pd.DataFrame, config: Config, out) -> dict:
    draws = config.analysis["cluster_bootstrap_draws"]
    rng = _rng(config, 6)
    pats = D.complete(D.patients_only(df), ["score", "age", *RATINGS, "date"])
    multi = pats[pats.groupby("patient")["patient"].transform("size") > 1].sort_values(["patient", "date"]).copy()
    for c in ["score", "age", *RATINGS]:
        multi[f"{c}_w"] = multi[c] - multi.groupby("patient")[c].transform("mean")
        multi[f"d_{c}"] = multi.groupby("patient")[c].diff()
    pairs = multi.dropna(subset=["d_score"])
    span = multi.groupby("patient")["date"].agg(lambda s: (s.max() - s.min()).days)
    rows = []
    for r in RATINGS:
        within_rho = lambda f, r=r: S.adjusted_spearman(f, "score_w", f"{r}_w", [])
        change_rho = lambda f, r=r: S.adjusted_spearman(f.dropna(subset=["d_score"]), "d_score", f"d_{r}", [])
        slope = lambda f, r=r: S.ols(f, "score_w", [f"{r}_w", "age_w"])[1]
        for name, fn in [("Spearman ρ, within-patient deviations", within_rho),
                         ("Spearman ρ, change between consecutive visits", change_rho),
                         ("score points per rating point, within patient (adjusted for age)", slope)]:
            lo, hi = _ci(S.cluster_bootstrap(multi, fn, draws, rng))
            rows.append({"rating": r, "measure": name, "patients": multi["patient"].nunique(),
                         "recordings": len(multi), "visit pairs": len(pairs), "estimate": fn(multi), "ci_low": lo, "ci_high": hi})
    table = pd.DataFrame(rows)
    write_table(table, out, "longitudinal_within_patient",
                "Within-patient association of score with R, B, H across visits (patients with ≥2 recordings; 95% CI by patient-level bootstrap)")
    # between-patient reference: earliest recordings, slope adjusted for age
    early = D.earliest_per_patient(pats)
    ref_rows = []
    for r in RATINGS:
        b = S.ols(early, "score", [r, "age"])
        lo, hi = S.percentile_interval(S.cluster_bootstrap(early, lambda f, r=r: S.ols(f, "score", [r, "age"])[1], draws, rng))
        ref_rows.append({"rating": r, **D.counts(early), "score points per rating point, between patients (adjusted for age)": b[1],
                         "ci_low": lo, "ci_high": hi})
    ref = pd.DataFrame(ref_rows)
    write_table(ref, out, "longitudinal_between_patient_reference", "Between-patient reference: earliest recording per patient, OLS adjusted for age")
    timing = {"patients": int(multi["patient"].nunique()), "recordings": len(multi), "pairs": len(pairs),
              "span_days_median": float(span.median()), "span_days_q25": float(span.quantile(0.25)),
              "span_days_q75": float(span.quantile(0.75)),
              "age_change_years_median": float((multi.groupby("patient")["age"].agg(lambda s: s.max() - s.min())).median())}
    write_table(pd.DataFrame([timing]), out, "longitudinal_timing", "Follow-up structure of patients with ≥2 recordings")
    return {"table": table, "reference": ref, "timing": timing}


STEPS = {
    "cohort_ages": cohort_ages,
    "age_correlation": age_correlation,
    "table3": table3,
    "age_effect": age_effect,
    "table4": table4,
    "longitudinal": longitudinal,
}
