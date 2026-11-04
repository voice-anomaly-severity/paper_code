"""Batched Gaussian-process leave-one-patient-out (LOPO) on GPU with PyTorch.

Many GPs ("runs") are fitted at once as one batch. A run is defined by a training mask over the
rows; masked-out rows are decoupled from the training covariance (unit variance, zero target), so
every run uses the full n x n shape. Inputs and target are standardised on each run's training
rows. Gradients come from autograd.

Kernel: k(x, x') = [x^T diag(v) x'] + s2 * exp(-0.5 |L^T (x - x')|^2) [+ tau2 * 1(same patient)] + noise * 1(same row).
The optional linear part (one slope variance v_d per input, "linear=True") makes the GP contain the
linear mixed model; the RBF part then models only departures from linearity.
L is lower triangular ("full", the manuscript's Mahalanobis metric) or diagonal ("diag", ARD).

Hyperparameter selection per run:
  "ml"   exact marginal likelihood
  "map"  marginal likelihood + weakly informative priors (penalises short length-scales):
         ARD: length-scale ~ LogNormal(sqrt(2) + log(D)/2, sqrt(3))  (Hvarfner et al., 2024);
         full-rank: L_ij ~ N(0, kappa^2) with kappa = 1 / (l0 * sqrt(D)), l0 = exp(sqrt(2) + log(D)/2),
         so a random direction has the same typical length-scale as under the ARD prior;
         log s2 ~ N(0, 1), log noise ~ N(log 0.5, 1), log tau2 ~ N(log 0.3, 1).
  "lopo" closed-form leave-one-patient-out predictive density on the run's training patients
  early stopping (fit_lopo_early_stopping): marginal likelihood from one smooth start, stopped at the
         step with the lowest validation NLL over k rotating patient folds of the training set.

A held-out patient shares no random effect with training patients; its rows share tau2 * 11^T.
"""
from __future__ import annotations

import math

import numpy as np
import torch

LOG2PI = math.log(2.0 * math.pi)
BOUNDS = {"log_s2": (-7.0, 5.0), "log_noise": (-7.0, 2.0), "L": (-20.0, 20.0), "log_inv_ls": (-10.0, 5.0), "log_tau2": (-7.0, 2.0), "log_v": (-10.0, 5.0)}
PRESETS = [(1.0, 0.5), (3.0, 0.8), (0.5, 0.2), (1.5, 0.3), (6.0, 0.9)]


# ---------------------------------------------------------------- data for a batch of runs
class _Runs:
    def __init__(self, X, y, patients, train_masks, device, dtype):
        """train_masks: (R, n) bool numpy array, one training mask per run."""
        self.device, self.dtype = device, dtype
        pats, inverse = np.unique(patients, return_inverse=True)
        self.n, self.dim, self.n_pat = len(y), X.shape[1], len(pats)
        t = lambda a: torch.as_tensor(a, device=device, dtype=dtype)
        self.pid = torch.as_tensor(inverse, device=device)
        self.train = t(train_masks.astype(float))
        cnt = self.train.sum(1, keepdim=True)
        Xt, yt = t(X), t(y)
        mx = self.train @ Xt / cnt
        sx = torch.sqrt(torch.clamp(self.train @ Xt**2 / cnt - mx**2, min=1e-12))
        self.Xs = (Xt[None] - mx[:, None]) / sx[:, None]
        self.my = (self.train @ yt) / cnt[:, 0]
        self.sy = torch.sqrt(torch.clamp(self.train @ yt**2 / cnt[:, 0] - self.my**2, min=1e-12))
        self.ys = (yt[None] - self.my[:, None]) / self.sy[:, None]
        self.y_raw = yt
        self.same = (self.pid[:, None] == self.pid[None, :]).to(dtype)
        sizes = np.bincount(inverse)
        self.max_g = int(sizes.max())
        idx = np.zeros((self.n_pat, self.max_g), dtype=np.int64)
        pad = np.zeros((self.n_pat, self.max_g), dtype=bool)
        for g in range(self.n_pat):
            rows = np.where(inverse == g)[0]
            idx[g, : len(rows)] = rows
            pad[g, len(rows):] = True
        self.gidx = torch.as_tensor(idx, device=device)
        self.gpad = torch.as_tensor(pad, device=device)
        self.gsize = t(sizes.astype(float))
        self.patient_index = inverse


def _n_metric(dim, metric):
    return dim * (dim + 1) // 2 if metric == "full" else dim


def _initial(dim, metric, use_re, presets, linear=False):
    out = []
    for ls, noise in presets:
        m = (np.eye(dim) / ls)[np.tril_indices(dim)] if metric == "full" else np.full(dim, -np.log(ls))
        lin = np.zeros(dim) if linear else []
        out.append(np.concatenate([[0.0, np.log(noise)], m, lin, [np.log(0.3)] if use_re else []]))
    return np.stack(out)


def _bounds(dim, metric, use_re, device, dtype, linear=False):
    mb = BOUNDS["L"] if metric == "full" else BOUNDS["log_inv_ls"]
    b = ([BOUNDS["log_s2"], BOUNDS["log_noise"]] + [mb] * _n_metric(dim, metric) + [BOUNDS["log_v"]] * (dim if linear else 0)
         + ([BOUNDS["log_tau2"]] if use_re else []))
    b = torch.as_tensor(b, device=device, dtype=dtype)
    return b[:, 0], b[:, 1]


def _metric_matrix(theta, dim, metric):
    p = theta[:, 2 : 2 + _n_metric(dim, metric)]
    if metric == "full":
        L = torch.zeros(theta.shape[0], dim, dim, device=theta.device, dtype=theta.dtype)
        r, c = torch.tril_indices(dim, dim, device=theta.device)
        L[:, r, c] = p
        return L
    return torch.diag_embed(torch.exp(p))


def _kernel_all(theta, Xs, dim, metric, linear=False):
    Z = Xs @ _metric_matrix(theta, dim, metric)
    sq = (Z**2).sum(-1)
    d = torch.clamp(sq[:, :, None] + sq[:, None, :] - 2.0 * Z @ Z.transpose(1, 2), min=0.0)
    K = torch.exp(theta[:, 0])[:, None, None] * torch.exp(-0.5 * d)
    if linear:
        nm = _n_metric(dim, metric)
        v = torch.exp(theta[:, 2 + nm : 2 + nm + dim])
        K = K + (Xs * v[:, None, :]) @ Xs.transpose(1, 2)
    return K


def _train_cov(theta, runs, metric, use_re, jitter, linear=False):
    K = _kernel_all(theta, runs.Xs, runs.dim, metric, linear)
    if use_re:
        K = K + torch.exp(theta[:, -1])[:, None, None] * runs.same[None]
    m = runs.train
    K = K * (m[:, :, None] * m[:, None, :])
    diag = (torch.exp(theta[:, 1])[:, None] + jitter) * m + (1.0 - m)
    return K + torch.diag_embed(diag)


def _neg_prior(theta, dim, metric, use_re):
    def normal(x, mu, sd):
        return 0.5 * ((x - mu) / sd) ** 2
    l0_log = math.sqrt(2.0) + 0.5 * math.log(dim)
    pen = normal(theta[:, 0], 0.0, 1.0) + normal(theta[:, 1], math.log(0.5), 1.0)
    p = theta[:, 2 : 2 + _n_metric(dim, metric)]
    if metric == "full":
        kappa = 1.0 / (math.exp(l0_log) * math.sqrt(dim))
        pen = pen + normal(p, 0.0, kappa).sum(-1)
    else:   # p = log inverse length-scale = -log l
        pen = pen + normal(-p, l0_log, math.sqrt(3.0)).sum(-1)
    if use_re:
        pen = pen + normal(theta[:, -1], math.log(0.3), 1.0)
    return pen


def _objective(theta, runs, metric, use_re, objective, jitter, linear=False):
    C = _train_cov(theta, runs, metric, use_re, jitter, linear)
    y = runs.ys * runs.train
    chol, info = torch.linalg.cholesky_ex(C)
    alpha = torch.cholesky_solve(y[..., None], chol)[..., 0]
    if objective in ("ml", "map"):
        val = 0.5 * (y * alpha).sum(-1) + torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        if objective == "map":
            val = val + _neg_prior(theta, runs.dim, metric, use_re)
        return torch.where(info == 0, val, torch.full_like(val, 1e10))
    # closed-form leave-one-patient-out on the run's training patients
    P = torch.cholesky_inverse(chol)
    gi, gp = runs.gidx, runs.gpad
    blocks = P[:, gi[:, :, None], gi[:, None, :]]
    keep = (~gp).to(P.dtype)
    eye = torch.eye(runs.max_g, device=P.device, dtype=P.dtype)
    blocks = blocks * (keep[:, :, None] * keep[:, None, :]) + eye * (1.0 - keep)[:, :, None]
    a = alpha[:, gi] * keep
    cb, binfo = torch.linalg.cholesky_ex(blocks)
    quad = (a * torch.cholesky_solve(a[..., None], cb)[..., 0]).sum(-1)
    logdet = 2.0 * torch.log(torch.diagonal(cb, dim1=-2, dim2=-1)).sum(-1)
    lp = -0.5 * (quad - logdet + runs.gsize * LOG2PI)
    in_train = (runs.train[:, gi[:, 0]] > 0.5)                       # patient-level training membership
    lp = torch.where(in_train, lp, torch.zeros_like(lp))
    total = -lp.sum(-1)
    bad = (info != 0) | (binfo != 0).any(-1)
    return torch.where(bad, torch.full_like(total, 1e10), total)


@torch.no_grad()
def _group_scores(theta, runs, eval_groups, metric, use_re, jitter, linear=False):
    """Joint predictive log density (original units) of every eval patient's rows given the run's
    training rows. eval_groups: (R, G) bool. Returns logpdf, sq_err, z2, n_rows as (R, G) tensors."""
    C = _train_cov(theta, runs, metric, use_re, jitter, linear)
    chol = torch.linalg.cholesky(C)
    m = runs.train
    alpha = torch.cholesky_solve((runs.ys * m)[..., None], chol)[..., 0]
    Kall = _kernel_all(theta, runs.Xs, runs.dim, metric, linear)
    Kcross = Kall * m[:, None, :]                                          # every row x training cols
    mean = (Kcross @ alpha[..., None])[..., 0]
    V = Kcross @ torch.cholesky_solve(Kcross.transpose(1, 2), chol)
    cov_all = Kall - V
    if use_re:
        cov_all = cov_all + torch.exp(theta[:, -1])[:, None, None] * runs.same[None]
    noise = torch.exp(theta[:, 1]) + jitter
    gi, gp = runs.gidx, runs.gpad
    keep = (~gp).to(C.dtype)[None].expand(theta.shape[0], -1, -1)                    # R x G x m
    S = cov_all[:, gi[:, :, None], gi[:, None, :]] + noise[:, None, None, None] * torch.eye(runs.max_g, device=C.device, dtype=C.dtype)
    eye = torch.eye(runs.max_g, device=C.device, dtype=C.dtype)
    S = S * (keep[..., :, None] * keep[..., None, :]) + eye * (1.0 - keep)[..., :, None]
    mu = mean[:, gi]
    z_res = ((runs.y_raw[gi][None] - runs.my[:, None, None]) / runs.sy[:, None, None] - mu) * keep
    cs = torch.linalg.cholesky(S)
    quad = (z_res * torch.cholesky_solve(z_res[..., None], cs)[..., 0]).sum(-1)
    logdet = 2.0 * torch.log(torch.diagonal(cs, dim1=-2, dim2=-1)).sum(-1)
    n_g = keep.sum(-1)
    logpdf = -0.5 * (quad + logdet + n_g * LOG2PI) - n_g * torch.log(runs.sy)[:, None]
    sq_err = ((z_res * runs.sy[:, None, None]) ** 2).sum(-1)
    z2 = ((z_res**2) / torch.diagonal(S, dim1=-2, dim2=-1) * keep).sum(-1)
    e = eval_groups.to(C.dtype)
    return logpdf * e, sq_err * e, z2 * e, n_g * e


def _optimise(runs, theta0, metric, use_re, objective, steps, lr, jitter, schedule=True, snapshot_every=None, linear=False):
    lo, hi = _bounds(runs.dim, metric, use_re, runs.device, runs.dtype, linear)
    theta = theta0.clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.02) if schedule else None
    best = torch.full((theta.shape[0],), float("inf"), device=runs.device, dtype=runs.dtype)
    best_theta = theta0.clone()
    snapshots = [theta0.clone()] if snapshot_every else None
    for step in range(1, steps + 1):
        opt.zero_grad()
        f = _objective(theta, runs, metric, use_re, objective, jitter, linear)
        with torch.no_grad():
            better = f < best
            best = torch.where(better, f, best)
            best_theta[better] = theta[better]
        f.sum().backward()
        opt.step()
        if sched:
            sched.step()
        with torch.no_grad():
            theta.copy_(torch.max(torch.min(theta, hi), lo))
            if snapshot_every and step % snapshot_every == 0:
                snapshots.append(theta.detach().clone())
    with torch.no_grad():
        f = _objective(theta, runs, metric, use_re, objective, jitter, linear)
        better = f < best
        best = torch.where(better, f, best)
        best_theta[better] = theta[better]
    return best, best_theta, snapshots


def _lopo_masks(patient_index, n_pat):
    return patient_index[None, :] != np.arange(n_pat)[:, None]


def _outputs(runs, theta, metric, use_re, jitter, linear=False):
    own = torch.eye(runs.n_pat, device=runs.device, dtype=torch.bool)               # run b evaluates patient b
    lp, se, z2, n = _group_scores(theta, runs, own, metric, use_re, jitter, linear)
    return {k: v.sum(-1).cpu().numpy() for k, v in {"logpdf": lp, "sq_err": se, "z2": z2, "n_rows": n}.items()}


def fit_lopo(X, y, patients, metric="full", use_re=False, objective="ml", restarts=3, steps=1500,
             lr=0.05, jitter=1e-6, device="cuda", dtype=torch.float64, linear=False):
    """LOPO with hyperparameters per fold by 'ml', 'map' or 'lopo'. Returns per-patient arrays."""
    X, y, patients = np.asarray(X, float), np.asarray(y, float), np.asarray(patients)
    _, inverse = np.unique(patients, return_inverse=True)
    n_pat = inverse.max() + 1
    masks = _lopo_masks(inverse, n_pat)
    runs = _Runs(X, y, patients, np.repeat(masks, restarts, axis=0), device, dtype)
    dim = X.shape[1]
    theta0 = torch.as_tensor(np.tile(_initial(dim, metric, use_re, PRESETS[:restarts], linear), (n_pat, 1)), device=device, dtype=dtype)
    best, best_theta, _ = _optimise(runs, theta0, metric, use_re, objective, steps, lr, jitter, linear=linear)
    choice = best.view(n_pat, restarts).argmin(1)
    theta = best_theta.view(n_pat, restarts, -1)[torch.arange(n_pat, device=device), choice]
    final = _Runs(X, y, patients, masks, device, dtype)
    return _outputs(final, theta, metric, use_re, jitter, linear)


def fit_lopo_early_stopping(X, y, patients, metric="full", use_re=False, k_inner=5, steps=1500, lr=0.05,
                            eval_every=25, start=(3.0, 0.8), jitter=1e-6, seed=0, device="cuda", dtype=torch.float64,
                            linear=False):
    """LOPO; hyperparameters by marginal likelihood from one smooth start, stopped early.

    For every outer fold (held-out patient b), the other patients are split into k_inner fixed
    patient folds. k_inner inner runs optimise the marginal likelihood on k_inner - 1 folds and
    record the validation NLL of the left-out fold every `eval_every` steps. The stopping step for
    fold b minimises the validation NLL summed over its inner runs; the final run (all training
    patients, same start, same optimiser) is stopped at that step and predicts patient b.
    """
    X, y, patients = np.asarray(X, float), np.asarray(y, float), np.asarray(patients)
    _, inverse = np.unique(patients, return_inverse=True)
    n_pat, dim = inverse.max() + 1, X.shape[1]
    rng = np.random.default_rng(seed)
    inner_fold_of_patient = rng.permutation(n_pat) % k_inner
    outer = _lopo_masks(inverse, n_pat)                                             # (B, n)
    row_fold = inner_fold_of_patient[inverse]
    inner_masks, inner_eval = [], []
    for b in range(n_pat):
        for k in range(k_inner):
            inner_masks.append(outer[b] & (row_fold != k))
            ev = (inner_fold_of_patient == k)
            ev[b] = False
            inner_eval.append(ev)
    inner_masks, inner_eval = np.stack(inner_masks), np.stack(inner_eval)
    theta0 = torch.as_tensor(_initial(dim, metric, use_re, [start], linear), device=device, dtype=dtype)

    # inner runs: record validation NLL along the optimisation path
    inner = _Runs(X, y, patients, inner_masks, device, dtype)
    ev_t = torch.as_tensor(inner_eval, device=device)
    _, _, snaps = _optimise(inner, theta0.expand(len(inner_masks), -1), metric, use_re, "ml", steps, lr, jitter,
                            schedule=False, snapshot_every=eval_every, linear=linear)
    val = torch.stack([-_group_scores(th, inner, ev_t, metric, use_re, jitter, linear)[0].sum(-1) for th in snaps])   # T x (B*k)
    val = val.view(len(snaps), n_pat, k_inner).sum(-1)                               # T x B
    stop = val.argmin(0)                                                             # per outer fold

    # final runs on all training patients, same start and optimiser, stopped at `stop`
    final = _Runs(X, y, patients, outer, device, dtype)
    _, _, fsnaps = _optimise(final, theta0.expand(n_pat, -1), metric, use_re, "ml", steps, lr, jitter,
                             schedule=False, snapshot_every=eval_every, linear=linear)
    fsnaps = torch.stack(fsnaps)                                                     # T x B x P
    theta = fsnaps[stop, torch.arange(n_pat, device=device)]
    out = _outputs(final, theta, metric, use_re, jitter, linear)
    out["stop_step"] = (stop.cpu().numpy() * eval_every)
    # validation NLL per patient-row along the path (mean over outer folds), for diagnostics
    n_eval_rows = float(np.sum(inner_eval[:, inverse], axis=1).sum()) / n_pat
    out["val_curve_steps"] = np.arange(len(snaps)) * eval_every
    out["val_curve"] = (val.mean(1) / n_eval_rows).cpu().numpy()
    out["theta"] = theta.cpu().numpy()
    return out
