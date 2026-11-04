"""Predictive models, leave-one-patient-out (LOPO) and the patient-level Bayesian bootstrap.

Model families for Table 4 (all with LOPO: every recording of a held-out patient is predicted
jointly from all other patients; the elpd unit is the patient):

* lmm (primary): linear mixed model y = X beta + u_patient + e, u ~ N(0, tau2), e ~ N(0, sigma2).
  sigma2 and tau2 by REML on the training patients (refitted per fold); beta by GLS. A held-out
  patient's rows are predicted jointly: N(X beta, sigma2 I + tau2 11' + X Cov(beta) X').
  Implemented with per-patient sufficient statistics (block-diagonal covariance), so a LOPO pass
  costs milliseconds.
* gp (secondary): the manuscript's Gaussian process kernel (RBF with full-rank Mahalanobis metric)
  plus the patient random effect, hyperparameters by maximum a posteriori with fixed weakly
  informative priors, refitted per fold; batched on GPU in pipeline.gp_torch.
* linear: Gaussian linear regression without patient effect (reference prior, Student-t
  predictive); used where every patient contributes one row (earliest-recording sensitivity).

The intercept-only model (null) is the lmm without predictors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import multivariate_t

LOG_2PI = np.log(2.0 * np.pi)


# ---------------------------------------------------------------- linear model (no patient effect)
def linear_predict_group(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray):
    design = np.column_stack([np.ones(len(X_tr)), X_tr])
    design_te = np.column_stack([np.ones(len(X_te)), X_te])
    n, p = design.shape
    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ design.T @ y_tr
    resid = y_tr - design @ beta
    dof = n - p
    s2 = float(resid @ resid / dof)
    loc = design_te @ beta
    shape = s2 * (np.eye(len(X_te)) + design_te @ xtx_inv @ design_te.T)
    logpdf = float(multivariate_t(loc=loc, shape=shape, df=dof).logpdf(y_te))
    var = np.diag(shape) * dof / (dof - 2.0)
    return logpdf, loc, var


def _linear_lopo(values, patients):
    rows = []
    for p in np.unique(patients):
        t = patients == p
        lp, mean, var = linear_predict_group(values[~t, 1:], values[~t, 0], values[t, 1:], values[t, 0])
        err = values[t, 0] - mean
        rows.append((p, lp, int(t.sum()), float(err @ err), float(np.sum(err**2 / var))))
    return rows


# ---------------------------------------------------------------- linear mixed model
class _GroupStats:
    """Per-patient sufficient statistics for the random-intercept model."""

    def __init__(self, X, y, groups):
        self.ids, inv = np.unique(groups, return_inverse=True)
        G, p = len(self.ids), X.shape[1]
        self.m = np.bincount(inv).astype(float)
        self.XtX = np.zeros((G, p, p))
        np.add.at(self.XtX, inv, X[:, :, None] * X[:, None, :])
        self.s = np.zeros((G, p))
        np.add.at(self.s, inv, X)
        self.Xty = np.zeros((G, p))
        np.add.at(self.Xty, inv, X * y[:, None])
        self.t = np.bincount(inv, weights=y)
        self.yty = np.bincount(inv, weights=y * y)
        self.inv = inv


def _gls_terms(log_s2, log_t2, st, keep):
    s2, t2 = np.exp(log_s2), np.exp(log_t2)
    m = st.m[keep]
    c = t2 / (s2 + m * t2)
    s, t = st.s[keep], st.t[keep]
    XtVX = (st.XtX[keep].sum(0) - np.einsum("g,gi,gj->ij", c, s, s)) / s2
    XtVy = (st.Xty[keep].sum(0) - (c * t) @ s) / s2
    ytVy = (st.yty[keep].sum() - (c * t * t).sum()) / s2
    logdetV = ((m - 1.0) * np.log(s2) + np.log(s2 + m * t2)).sum()
    return XtVX, XtVy, ytVy, logdetV


def _neg_reml(theta, st, keep):
    XtVX, XtVy, ytVy, logdetV = _gls_terms(theta[0], theta[1], st, keep)
    sign, logdet_xtvx = np.linalg.slogdet(XtVX)
    if sign <= 0:
        return 1e20
    beta = np.linalg.solve(XtVX, XtVy)
    return 0.5 * (logdetV + logdet_xtvx + ytVy - beta @ XtVy)


def _lmm_lopo(values, patients):
    """values[:, 0] = target, values[:, 1:] = predictors (intercept added here)."""
    y = values[:, 0]
    X = np.column_stack([np.ones(len(values)), values[:, 1:]])
    st = _GroupStats(X, y, patients)
    var_y = float(np.var(y))
    start = np.log([0.6 * var_y, 0.4 * var_y])
    rows = []
    for g, pid in enumerate(st.ids):
        keep = np.ones(len(st.ids), dtype=bool)
        keep[g] = False
        res = minimize(_neg_reml, start, args=(st, keep), method="L-BFGS-B", bounds=[(-12, 12), (-12, 12)])
        s2, t2 = np.exp(res.x)
        XtVX, XtVy, _, _ = _gls_terms(res.x[0], res.x[1], st, keep)
        cov_beta = np.linalg.inv(XtVX)
        beta = cov_beta @ XtVy
        rows_g = st.inv == g
        Xg, yg = X[rows_g], y[rows_g]
        k = len(yg)
        cov = s2 * np.eye(k) + t2 * np.ones((k, k)) + Xg @ cov_beta @ Xg.T
        r = yg - Xg @ beta
        chol = np.linalg.cholesky(cov)
        z = np.linalg.solve(chol, r)
        logpdf = -0.5 * (z @ z) - np.log(np.diag(chol)).sum() - 0.5 * k * LOG_2PI
        rows.append((pid, float(logpdf), k, float(r @ r), float(np.sum(r**2 / np.diag(cov)))))
    return rows


# ---------------------------------------------------------------- dispatch
def lopo(df: pd.DataFrame, target: str, predictors: list[str], model: str, gp_result: dict | None = None):
    """Per-patient LOPO results (index: patient) with columns logpdf, n_rows, sq_err, z2.

    model: 'lmm', 'linear', or 'gp' (then `gp_result` must hold the arrays returned by
    pipeline.gp_torch.fit_lopo for exactly these rows, ordered by sorted patient id).
    The intercept-only model is always the lmm (or linear) without predictors.
    """
    values = df[[target, *predictors]].to_numpy(dtype=float)
    patients = df["patient"].to_numpy()
    if model == "gp" and predictors:
        ids = np.unique(patients)
        rows = list(zip(ids, gp_result["logpdf"], gp_result["n_rows"].astype(int), gp_result["sq_err"], gp_result["z2"]))
    elif model == "linear":
        rows = _linear_lopo(values, patients)
    else:
        rows = _lmm_lopo(values, patients)
    out = pd.DataFrame(rows, columns=["patient", "logpdf", "n_rows", "sq_err", "z2"]).set_index("patient")
    out.attrs["sst"] = float(np.sum((df[target] - df[target].mean()) ** 2))
    return out


def model_summary(res: pd.DataFrame) -> dict:
    n_rows = int(res["n_rows"].sum())
    return {
        "elpd": float(res["logpdf"].sum()),
        "loo_rmse": float(np.sqrt(res["sq_err"].sum() / n_rows)),
        "loo_r2": float(1.0 - res["sq_err"].sum() / res.attrs["sst"]),
        "mean_z2": float(res["z2"].sum() / n_rows),
    }


def compare(base: pd.DataFrame, full: pd.DataFrame, draws: int, rng: np.random.Generator) -> dict:
    """Delta elpd (full - base) with a Bayesian bootstrap over patients."""
    delta = (full["logpdf"] - base["logpdf"]).to_numpy()
    n_pat = len(delta)
    weights = rng.dirichlet(np.ones(n_pat), size=draws)
    boot = n_pat * (weights @ delta)
    return {
        "n_patients": n_pat,
        "n_recordings": int(full["n_rows"].sum()),
        "delta_elpd": float(delta.sum()),
        "delta_elpd_se": float(np.sqrt(n_pat) * delta.std(ddof=1)),
        "delta_elpd_per_patient": float(delta.mean()),
        "bb_q025": float(np.quantile(boot, 0.025)),
        "bb_q975": float(np.quantile(boot, 0.975)),
        "bb_pr_positive": float(np.mean(boot > 0)),
    }
