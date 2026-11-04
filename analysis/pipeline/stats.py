"""Correlations and regressions with patient-level (cluster) resampling.

Rows are recordings; several recordings can belong to one patient. Point estimates use all
rows. Uncertainty comes from resampling whole patients with replacement, so recordings of
the same patient always stay together.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def residualize(values: np.ndarray, covariates: np.ndarray | None) -> np.ndarray:
    """OLS residuals of `values` on an intercept plus covariates (the paper's adjustment)."""
    values = np.asarray(values, dtype=float)
    if covariates is None or covariates.size == 0:
        return values - values.mean()
    design = np.column_stack([np.ones(len(values)), covariates])
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ beta


def adjusted_spearman(df: pd.DataFrame, x: str, y: str, covariates: list[str]) -> float:
    """Spearman correlation of x and y after residualizing both on covariates.

    With no covariates this is the plain Spearman correlation. This is the method used for the
    age-adjusted column of the manuscript's Table 3 (evaluate_student.py:713-761).
    """
    if not covariates:
        return float(spearmanr(df[x], df[y]).statistic)
    cov = df[covariates].to_numpy(dtype=float)
    return float(spearmanr(residualize(df[x].to_numpy(), cov), residualize(df[y].to_numpy(), cov)).statistic)


def adjusted_spearman_p(df: pd.DataFrame, x: str, y: str, covariates: list[str]) -> float:
    """Classical p-value; valid only when rows are independent (one row per patient)."""
    if not covariates:
        return float(spearmanr(df[x], df[y]).pvalue)
    cov = df[covariates].to_numpy(dtype=float)
    return float(spearmanr(residualize(df[x].to_numpy(), cov), residualize(df[y].to_numpy(), cov)).pvalue)


def cluster_bootstrap(
    df: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float | np.ndarray],
    draws: int,
    rng: np.random.Generator,
    cluster: str = "patient",
) -> np.ndarray:
    """Statistic recomputed on `draws` patient-level bootstrap samples."""
    groups = {key: idx for key, idx in df.groupby(cluster).indices.items()}
    keys = np.array(list(groups))
    out = []
    for _ in range(draws):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        rows = np.concatenate([groups[k] for k in chosen])
        out.append(statistic(df.iloc[rows]))
    return np.asarray(out, dtype=float)


def percentile_interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    samples = samples[np.isfinite(samples)] if samples.ndim == 1 else samples
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def ols(df: pd.DataFrame, y: str, predictors: list[str]) -> np.ndarray:
    design = np.column_stack([np.ones(len(df)), df[predictors].to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(design, df[y].to_numpy(dtype=float), rcond=None)
    return beta
