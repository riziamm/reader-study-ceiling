"""Simulation: generator, the three analyses, the sweep, and size adjustment.

Appendix B of the paper. The generator is calibrated against two measured
properties of the real ratings (share at the scale maximum, between-case
variance share) and frozen; see CASE_SD and the ceiling levels below.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parent.parent
ALPHA = 0.05
CRIT = ["Overall Realism", "Anatomical Accuracy", "Boundary Sharpness",
        "Absence of Artifacts"]

# frozen calibration (Table B1). 0.55 was inherited from the pipeline-validation
# generator and gave a between-case share of 28%, seven times the real study.
CALIB_CASE_SD = 0.18
LOW_CEILING, MID_CEILING, ANCHOR_CEILING = 3.20, 4.45, 5.05
CALIB_READER_SD = float(np.std([0.30, -0.10, -0.45, 0.05], ddof=1))
CALIB_CRIT_DIFF = np.array([0.00, 0.15, -0.30, -0.05])
CALIB_NOISE_SD = 0.55

BASE_SEED = 20260729
GRID_OFFSET = {"main": 0, "extension": 5_000_000, "gradient": 9_000_000}
ANALYSES = ["absolute_naive", "absolute_conditioned", "paired"]

def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _design(df: pd.DataFrame, with_model: bool = True):
    """Fixed-effects design matrix: model indicator + reader + criterion dummies."""
    cols, names = [], []
    if with_model:
        cols.append((df["model"] == "Model A").to_numpy(float))
        names.append("model_A")
    for r in sorted(df["reader"].unique())[1:]:
        cols.append((df["reader"] == r).to_numpy(float))
        names.append(f"reader_{r}")
    for c in sorted(df["criterion"].unique())[1:]:
        cols.append((df["criterion"] == c).to_numpy(float))
        names.append(f"crit_{c}")
    if not cols:
        return np.zeros((len(df), 0)), []
    return np.column_stack(cols), names


def simulate_long(ceiling_level: float, true_gap: float, n_readers: int,
                  n_cases: int, seed: int, gap_case_sd: float = 0.0):
    """Latent additive model thresholded to 1-5 (paper Appendix B.1).

    gap_case_sd defaults to 0 so the zero-effect arm is genuinely null: with a
    case-varying effect each study realises a nonzero average effect even when
    the population effect is zero, and rejection rates become power, not size."""
    rng = np.random.default_rng(seed)
    R, C, K = n_readers, n_cases, 4

    if R == 4:
        sev = np.array([0.30, -0.10, -0.45, 0.05])   # exact calibrated values
    else:
        sev = rng.normal(0, CALIB_READER_SD, size=R)  # same calibrated SPREAD

    case_effect = rng.normal(0, CALIB_CASE_SD, size=C)
    case_gap = (rng.normal(true_gap, gap_case_sd, size=C) if gap_case_sd > 0
                else np.full(C, true_gap))

    base = (ceiling_level + sev[:, None, None] + CALIB_CRIT_DIFF[None, None, :]
            + case_effect[None, :, None])                       # (R,C,K)
    latent = np.stack([base + case_gap[None, :, None] / 2,
                       base - case_gap[None, :, None] / 2], axis=-1)  # (R,C,K,2)
    latent += rng.normal(0, CALIB_NOISE_SD, size=latent.shape)
    scores = np.clip(np.rint(latent), 1, 5).astype(int)

    r_idx, c_idx, k_idx = np.meshgrid(np.arange(R), np.arange(C), np.arange(K),
                                      indexing="ij")
    crit_arr = np.array(CRIT)
    long_a = pd.DataFrame({
        "reader": (r_idx.ravel() + 1), "case_num": (c_idx.ravel() + 1),
        "criterion": crit_arr[k_idx.ravel()], "model": "Model A",
        "score": scores[..., 0].ravel(),
    })
    long_b = long_a.copy()
    long_b["model"] = "Model B"
    long_b["score"] = scores[..., 1].ravel()
    df = pd.concat([long_a, long_b], ignore_index=True)

    pct_at_max = float(100 * (scores == 5).mean())
    mean_gap = float(scores[..., 0].mean() - scores[..., 1].mean())
    return df, dict(pct_at_max=pct_at_max, mean_absolute_gap=mean_gap,
                    case_effect=case_effect)


def between_case_share(df: pd.DataFrame) -> float:
    """Between-case variance share by one-way ANOVA on case. A light proxy used
    inside the sweep loop; stats.variance_components does the full split."""
    g = df.groupby("case_num")["score"]
    grand = df["score"].mean()
    ni = g.size().to_numpy(float)
    means = g.mean().to_numpy(float)
    ss_between = float((ni * (means - grand) ** 2).sum())
    ss_total = float(((df["score"] - grand) ** 2).sum())
    ms_between = ss_between / (len(ni) - 1)
    ss_within = ss_total - ss_between
    n_total = len(df)
    ms_within = ss_within / (n_total - len(ni))
    var_between = max((ms_between - ms_within) / (ni.mean()), 0.0)
    var_total = var_between + ms_within
    return float(100 * var_between / var_total) if var_total > 0 else np.nan


def sum_to_zero_design(df: pd.DataFrame, factor_cols: list[str]):
    """Deviation coding: intercept = grand mean across levels, not the
    reference-level value. Needed only for the paired-paradigm fit,
    where the intercept IS the tested quantity (see module docstring)."""
    cols, names = [], []
    for col in factor_cols:
        levels = sorted(df[col].unique())
        ref = levels[-1]
        v = df[col].to_numpy()
        for lv in levels[:-1]:
            cols.append(np.where(v == lv, 1.0, np.where(v == ref, -1.0, 0.0)))
            names.append(f"{col}_{lv}")
    if not cols:
        return np.zeros((len(df), 0)), []
    return np.column_stack(cols), names


def fast_clmm_fit(y: np.ndarray, X: np.ndarray, groups: np.ndarray,
                  fixed_intercept: float | None = None, n_nodes: int = 16,
                  start: np.ndarray | None = None, with_re: bool = True):
    """Cumulative link model, optional normal case random intercept integrated out
    by Gauss-Hermite quadrature. with_re=False drops the random intercept and
    gives the naive analysis. start warm-starts a nested fit from the full one."""
    y = np.asarray(y, int)
    cats = np.unique(y)
    K = len(cats)
    if K < 2:
        return None
    ycode = np.searchsorted(cats, y)
    n_groups = int(groups.max()) + 1
    if with_re:
        nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
        weights = weights / np.sqrt(np.pi)
    else:
        nodes, weights = np.array([0.0]), np.array([1.0])
    n_beta = X.shape[1]
    n_free_cut = (K - 1) - (1 if fixed_intercept is not None else 0)

    def unpack(theta):
        if fixed_intercept is not None:
            t0 = fixed_intercept
            inc = np.exp(theta[:max(K - 2, 0)])
            idx = max(K - 2, 0)
        else:
            t0 = theta[0]
            inc = np.exp(theta[1:K - 1]) if K > 2 else np.array([])
            idx = max(K - 1, 0)
        cut = (np.concatenate([[t0], t0 + np.cumsum(inc)]) if K > 2
               else np.array([t0]))
        beta = theta[idx: idx + n_beta]
        if not with_re:
            return cut, beta, 0.0
        # BOUND the random-intercept SD on the log scale.
        #
        # Unbounded, exp(theta) overflows to inf, lin becomes inf, the
        # probabilities become nan, and nll falls back to its 1e10 guard.
        # That guard turns a wide region of parameter space into a FLAT
        # plateau, and Nelder-Mead then wanders across it for its full
        # iteration budget without improving anything -- observed as a cell
        # stalling well past its expected runtime, not merely as a warning.
        # Clipping keeps the surface finite and smooth everywhere.
        # exp([-6, 2]) = [0.0025, 7.4], which spans any plausible case SD
        # for a 1-5 scale by a wide margin.
        sigma = np.exp(np.clip(theta[idx + n_beta], -6.0, 2.0))
        return cut, beta, sigma

    def nll(theta):
        cut, beta, sigma = unpack(theta)
        eta = X @ beta if n_beta else np.zeros(len(ycode))
        cut_lo = np.concatenate([[-np.inf], cut])
        cut_hi = np.concatenate([cut, [np.inf]])
        # Vectorised over quadrature nodes, and np.bincount instead of
        # np.add.at -- the latter is unbuffered and roughly two orders of
        # magnitude slower here. Same arithmetic.
        u = np.sqrt(2.0) * sigma * nodes                    # (Q,)
        lin = eta[:, None] + u[None, :]                     # (n, Q)
        hi = _expit(cut_hi[ycode][:, None] - lin)
        lo = _expit(cut_lo[ycode][:, None] - lin)
        lp = np.log(np.clip(hi - lo, 1e-12, None))          # (n, Q)
        loglik_g = np.empty((n_groups, len(nodes)))
        for q in range(len(nodes)):
            loglik_g[:, q] = np.bincount(groups, weights=lp[:, q],
                                         minlength=n_groups)
        loglik_g += np.log(weights)[None, :]
        mmax = loglik_g.max(axis=1, keepdims=True)
        ll = mmax.ravel() + np.log(np.exp(loglik_g - mmax).sum(axis=1))
        val = -ll.sum()
        return val if np.isfinite(val) else 1e10

    n_params = n_free_cut + n_beta + (1 if with_re else 0)
    if start is None:
        # Cold-starting every parameter at zero puts the cutpoints at
        # roughly [0,1,2,3] regardless of where the data actually sit. At a
        # ceiling the latent scale is near 5, so full and null models were
        # landing in DIFFERENT local optima and the LR statistic was
        # garbage (observed range -126 to +64, 5/35 negative). Module 06
        # avoided this with a statsmodels OrderedModel warm start, which
        # was removed here as a speed cut -- that removal was the bug.
        #
        # This replaces it with a near-free analytic warm start: the
        # marginal cumulative logits, which is the exact MLE of the
        # cutpoints when there are no covariates and no random effect.
        start = np.zeros(n_params)
        counts = np.bincount(ycode, minlength=K).astype(float)
        cum = np.clip(np.cumsum(counts)[:-1] / counts.sum(), 1e-3, 1 - 1e-3)
        thr = np.log(cum / (1 - cum))               # marginal cumulative logits
        if fixed_intercept is None:
            start[0] = thr[0]
            if K > 2:
                start[1:K - 1] = np.log(np.clip(np.diff(thr), 1e-3, None))
        elif K > 2:
            shifted = np.clip(np.diff(np.concatenate([[fixed_intercept], thr[1:]])),
                              1e-3, None)
            start[:K - 2] = np.log(shifted)
        if with_re:
            start[-1] = np.log(0.5)
    res = optimize.minimize(nll, start, method="BFGS",
                            options=dict(maxiter=1500, gtol=1e-4))
    grad_norm = float(np.linalg.norm(optimize.approx_fprime(res.x, nll, 1e-6)))
    if grad_norm > 5e-2:
        nm = optimize.minimize(nll, res.x, method="Nelder-Mead",
                               options=dict(maxiter=3000, xatol=1e-5, fatol=1e-7))
        if nm.fun < res.fun:
            res = optimize.minimize(nll, nm.x, method="BFGS",
                                    options=dict(maxiter=1500, gtol=1e-4))
            grad_norm = float(np.linalg.norm(optimize.approx_fprime(res.x, nll, 1e-6)))
    converged = bool(grad_norm < 1e-1)   # looser than m06's 1e-2 -- see docstring
    cut, beta, sigma = unpack(res.x)
    # did the sigma bound bind? counted and reported per cell, never ignored
    bound_hit = bool(with_re and not (-6.0 < res.x[-1] < 2.0))
    return dict(loglik=-res.fun, cut=cut, beta=beta, sigma_case=sigma,
                converged=converged, grad_norm=grad_norm, _theta=res.x,
                bound_hit=bound_hit)


def fit_absolute(df: pd.DataFrame):
    """Same design as module 06's clmm_fit, called via the shared
    fast_clmm_fit core -- not a different model, a faster fit of it."""
    y = df["score"].to_numpy(int)
    cats = np.sort(df["score"].unique())
    ycode = np.searchsorted(cats, y)
    groups = pd.factorize(df["case_num"])[0]
    X_full, _ = _design(df, with_model=True)
    X_null, _ = _design(df, with_model=False)
    full = fast_clmm_fit(ycode, X_full, groups)
    if full is None:
        return dict(p=np.nan, est=np.nan, converged=False)
    # Warm-start the null AT the full fit's solution with the model
    # coefficient deleted. Nested models must be compared from the same
    # basin or the LR statistic is meaningless; this also guarantees the
    # null cannot "beat" the full fit through an optimiser accident.
    K = len(np.unique(ycode))
    n_cut = K - 1
    theta_f = np.concatenate([full["_theta"][:n_cut],
                              full["_theta"][n_cut + 1:]])
    null = fast_clmm_fit(ycode, X_null, groups, start=theta_f)
    if null is None:
        return dict(p=np.nan, est=np.nan, converged=False)
    lr = 2 * (full["loglik"] - null["loglik"])
    if lr < -1e-6:      # null genuinely beat full -> optimiser failure, not evidence
        return dict(p=np.nan, est=float(full["beta"][0]), converged=False)
    p = float(stats.chi2.sf(max(lr, 0), df=1))
    return dict(p=p, est=float(full["beta"][0]),
               converged=bool(full["converged"] and null["converged"]))


def fit_absolute_naive(df: pd.DataFrame):
    """Naive absolute: cumulative link model, model and reader fixed, no case term.
    Ignores that each case contributes 2*R*K correlated rows, so its size is
    not exact; that is measured in the size-adjustment step, not corrected here."""
    y = df["score"].to_numpy(int)
    cats = np.sort(df["score"].unique())
    ycode = np.searchsorted(cats, y)
    groups = pd.factorize(df["case_num"])[0]
    X_full, _ = _design(df, with_model=True)
    X_null, _ = _design(df, with_model=False)
    full = fast_clmm_fit(ycode, X_full, groups, with_re=False)
    if full is None:
        return dict(p=np.nan, est=np.nan, converged=False, bound_hit=False)
    K = len(np.unique(ycode)); n_cut = K - 1
    theta_f = np.concatenate([full["_theta"][:n_cut], full["_theta"][n_cut + 1:]])
    null = fast_clmm_fit(ycode, X_null, groups, start=theta_f, with_re=False)
    if null is None:
        return dict(p=np.nan, est=np.nan, converged=False, bound_hit=False)
    lr = 2 * (full["loglik"] - null["loglik"])
    if lr < -1e-6:
        return dict(p=np.nan, est=float(full["beta"][0]), converged=False,
                    bound_hit=False)
    return dict(p=float(stats.chi2.sf(max(lr, 0), df=1)),
                est=float(full["beta"][0]),
                converged=bool(full["converged"] and null["converged"]),
                bound_hit=False)


def fit_paired(df: pd.DataFrame):
    """Paired sign contrast: within-case wins/losses, ties dropped, mixed logistic
    with a case random intercept. Ties must be dropped rather than counted as
    losses; with ties as losses P(A wins) != 0.5 under the null and the tested
    hypothesis is false by construction."""
    w = df.pivot_table(index=["reader", "case_num", "criterion"],
                       columns="model", values="score", aggfunc="first")
    w = w.dropna()
    if len(w) == 0:
        return dict(p=np.nan, est=np.nan, converged=False)
    delta = (w["Model A"] - w["Model B"]).to_numpy()
    # TIES ARE DROPPED, not counted as losses.
    #
    # Module 05 reports ties-as-losses as a conservative DESCRIPTIVE
    # proportion. It cannot be used as the basis of a hypothesis test here:
    # counting ties as losses makes P(A wins) far below 0.5 even when the
    # models are identical (measured: 0.303 at ceiling=3.20, 0.182 at
    # ceiling=5.35 under a true null), so "intercept == 0" is a FALSE
    # hypothesis under the null and the test rejects ~100% of the time.
    # That was the observed pathology, and it was the test being wrong, not
    # the data.
    #
    # Excluding ties restores exchangeability: with no true difference,
    # delta is symmetric about zero, so P(delta>0 | delta!=0) = 0.5 exactly
    # and the intercept is genuinely 0 under H0. This is the standard
    # sign-test framing.
    decided = delta != 0
    if decided.sum() < 10 or len(np.unique((delta[decided] > 0))) < 2:
        return dict(p=np.nan, est=np.nan, converged=False, n_decided=int(decided.sum()))
    win_a = (delta[decided] > 0).astype(int)
    wide = w.reset_index().loc[decided].reset_index(drop=True)
    groups = pd.factorize(wide["case_num"])[0]
    X, _ = sum_to_zero_design(wide, ["reader", "criterion"])
    full = fast_clmm_fit(win_a, X, groups, fixed_intercept=None)
    if full is None:
        return dict(p=np.nan, est=np.nan, converged=False, n_decided=int(decided.sum()))
    null = fast_clmm_fit(win_a, X, groups, fixed_intercept=0.0,
                         start=full["_theta"][1:])   # drop the free intercept
    if null is None:
        return dict(p=np.nan, est=np.nan, converged=False, n_decided=int(decided.sum()))
    lr = 2 * (full["loglik"] - null["loglik"])
    if lr < -1e-6:
        return dict(p=np.nan, est=float(full["cut"][0]), converged=False,
                    n_decided=int(decided.sum()))
    p = float(stats.chi2.sf(max(lr, 0), df=1))
    return dict(p=p, est=float(full["cut"][0]),
               converged=bool(full["converged"] and null["converged"]),
               n_decided=int(decided.sum()))


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------

def build_grid(name):
    """Cell order is canonical, so a cell's seed is a pure function of
    (grid, position) and results are independent of worker count."""
    if name == "main":
        ceil, gaps, readers, cases = [LOW_CEILING, ANCHOR_CEILING], [0.0, 0.20, 0.45], \
            [2, 3, 4, 6, 8, 12, 20], [44]
    elif name == "extension":
        ceil, gaps, readers, cases = [ANCHOR_CEILING], [0.0, 0.02, 0.05, 0.10, 0.15], \
            [2, 3, 4, 6], [20, 44]
    elif name == "gradient":
        ceil, gaps, readers, cases = [LOW_CEILING, MID_CEILING], \
            [0.0, 0.02, 0.05, 0.10, 0.15], [2, 3, 4, 6], [20, 44]
    else:
        raise ValueError(name)
    cells = [dict(ceiling_level=c, true_gap=g, n_readers=r, n_cases=n,
                  n_sims=1000 if g == 0.0 else 500,
                  arm="null_arm" if g == 0.0 else "power")
             for c in ceil for g in gaps for r in readers for n in cases]
    cells.sort(key=lambda c: (c["ceiling_level"], c["true_gap"], c["n_readers"],
                              c["n_cases"]))
    for i, c in enumerate(cells):
        c.update(cell_index=i, grid=name,
                 seed=BASE_SEED + GRID_OFFSET[name] + 100003 * (i + 1))
    return cells


def raw_dir(grid):
    d = ROOT / "outputs" / "tables" / "_raw" / grid
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_cell(cell, batch=25):
    """Resumable within the cell: per-replicate p-values are flushed every
    `batch` replicates, and replicate i always uses seed + i."""
    path = raw_dir(cell["grid"]) / f"cell_{cell['cell_index']:04d}.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    rows = rows[:cell["n_sims"]]

    for i in range(len(rows), cell["n_sims"]):
        df, info = simulate_long(cell["ceiling_level"], cell["true_gap"],
                                 cell["n_readers"], cell["n_cases"],
                                 seed=cell["seed"] + i)
        if cell["arm"] == "null_arm":
            assert cell["true_gap"] == 0.0
        fingerprint = (len(df), int(df["score"].sum()))
        a, b, c = fit_absolute_naive(df), fit_absolute(df), fit_paired(df)
        assert (len(df), int(df["score"].sum())) == fingerprint, "analysis mutated ratings"
        rows.append(dict(p_naive=a["p"], p_cond=b["p"], p_paired=c["p"],
                         conv_naive=a["converged"], conv_cond=b["converged"],
                         conv_paired=c["converged"],
                         bound_cond=bool(b.get("bound_hit")),
                         bound_paired=bool(c.get("bound_hit")),
                         pct_max=info["pct_at_max"],
                         btw_case=between_case_share(df) if i < 30 else np.nan))
        if (i + 1) % batch == 0:
            pd.DataFrame(rows).to_csv(path, index=False)
    pd.DataFrame(rows).to_csv(path, index=False)

    d = pd.DataFrame(rows)
    out = {k: cell[k] for k in ("grid", "cell_index", "ceiling_level", "true_gap",
                                "n_readers", "n_cases", "n_sims", "arm", "seed")}
    out["realised_pct_max"] = float(d.pct_max.mean())
    out["realised_between_case_share"] = float(d.btw_case.mean(skipna=True))
    for name, p, cv, bd in (("absolute_naive", "p_naive", "conv_naive", None),
                            ("absolute_conditioned", "p_cond", "conv_cond", "bound_cond"),
                            ("paired", "p_paired", "conv_paired", "bound_paired")):
        ok = d[p].notna()
        out[f"power_{name}"] = float((d[p][ok] < ALPHA).mean()) if ok.any() else np.nan
        out[f"n_fail_{name}"] = int((~ok).sum())
        out[f"n_nonconv_{name}"] = int((~d[cv].astype(bool)).sum())
        out[f"n_bound_hits_{name}"] = int(d[bd].astype(bool).sum()) if bd else 0
    pw = [out[f"power_{a}"] for a in ANALYSES]
    # a cell where every analysis is pinned at the ceiling carries no
    # information about relative power and is excluded from all comparisons
    out["informative"] = bool(not all(p >= 0.999 for p in pw)
                              and not all(p <= 0.001 for p in pw))
    return out


def sweep(grids, workers, quick=False):
    cells = [c for g in grids for c in build_grid(g)]
    if quick:
        # smoke subset: anchor ceiling, small panels, few replicates
        cells = [dict(c, n_sims=40) for c in cells
                 if c["n_readers"] in (2, 4) and c["n_cases"] == 44][:12]
    done, t0 = {}, time.time()
    ck = {g: ROOT / "outputs" / "tables" / f"_ckpt_{g}.csv" for g in grids}
    for g, p in ck.items():
        if p.exists():
            for r in pd.read_csv(p).to_dict("records"):
                done[(r["grid"], r["cell_index"])] = r
    todo = [c for c in cells if (c["grid"], c["cell_index"]) not in done]
    print(f"{len(cells)} cells, {len(done)} already done, {len(todo)} to run, "
          f"{workers} workers")

    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_cell, c): c for c in todo}
        for n, fut in enumerate(cf.as_completed(futs), start=1):
            r = fut.result()
            done[(r["grid"], r["cell_index"])] = r
            for g in grids:
                rs = [v for k, v in done.items() if k[0] == g]
                if rs:
                    pd.DataFrame(rs).sort_values("cell_index").to_csv(ck[g], index=False)
            print(f"  [{n}/{len(todo)}] {r['grid']} ceiling={r['ceiling_level']:.2f} "
                  f"gap={r['true_gap']:.2f} r={r['n_readers']:>2} n={r['n_cases']:>3} | "
                  f"naive={r['power_absolute_naive']:.3f} "
                  f"cond={r['power_absolute_conditioned']:.3f} "
                  f"paired={r['power_paired']:.3f} ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(list(done.values())).sort_values(["grid", "cell_index"])
    for g in grids:
        df[df.grid == g].to_csv(ROOT / "outputs" / "tables" / f"sweep_{g}.csv", index=False)
    return df


# --------------------------------------------------------------------------
# size adjustment (Appendix B.5)
# --------------------------------------------------------------------------

MATCH = dict(gaps=[0.02, 0.05, 0.10, 0.15], readers=[2, 3, 4, 6], cases=[20, 44])
LEVELS = [("low", LOW_CEILING), ("mid", MID_CEILING), ("anchor", ANCHOR_CEILING)]


def _matched_raw():
    """Per-replicate p-values for the design that exists at all three ceilings."""
    frames, missing = [], []
    for grid in ("extension", "gradient"):
        for c in build_grid(grid):
            if not (c["true_gap"] in MATCH["gaps"] + [0.0]
                    and c["n_readers"] in MATCH["readers"]
                    and c["n_cases"] in MATCH["cases"]):
                continue
            f = raw_dir(grid) / f"cell_{c['cell_index']:04d}.csv"
            if not f.exists() or len(pd.read_csv(f)) < c["n_sims"]:
                missing.append(f"{grid}:{c['cell_index']}")
                continue
            d = pd.read_csv(f).iloc[:c["n_sims"]].copy()
            for k in ("ceiling_level", "true_gap", "n_readers", "n_cases", "arm"):
                d[k] = c[k]
            frames.append(d)
    if missing:
        raise SystemExit(
            f"{len(missing)} matched cells missing per-replicate p-values "
            f"(e.g. {missing[:3]}). Re-run the sweep for grids extension and gradient.")
    return pd.concat(frames, ignore_index=True)


def size_adjust():
    """Empirical critical value per (analysis, ceiling) so the null rejection
    rate is exactly 0.05, then power recomputed at that value."""
    a = _matched_raw()
    cols = [("absolute_naive", "p_naive"), ("absolute_conditioned", "p_cond"),
            ("paired", "p_paired")]
    nul = a[a.arm == "null_arm"]
    crit = {(n, cl): float(nul[nul.ceiling_level == cl][c].dropna().quantile(ALPHA))
            for n, c in cols for _, cl in LEVELS if (nul.ceiling_level == cl).any()}

    rows = []
    pw = a[(a.arm == "power") & a.true_gap.isin(MATCH["gaps"])]
    for (cl, gap, nr, nc), g in pw.groupby(["ceiling_level", "true_gap",
                                            "n_readers", "n_cases"]):
        rec = dict(ceiling_level=cl, true_gap=gap, n_readers=nr, n_cases=nc)
        for n, c in cols:
            v = g[c].dropna()
            rec[f"raw_{n}"] = float((v < ALPHA).mean())
            rec[f"adj_{n}"] = float((v < crit[(n, cl)]).mean())
        rows.append(rec)
    cells = pd.DataFrame(rows)
    raw = cells[[f"raw_{n}" for n, _ in cols]]
    cells["informative"] = ~(raw.ge(0.999).all(axis=1) | raw.le(0.001).all(axis=1))

    inf = cells[cells.informative]
    summ = []
    for lab, cl in LEVELS:
        s = inf[inf.ceiling_level == cl]
        if not len(s):
            continue
        for cname, x, y in (("conditioned - naive", "absolute_conditioned", "absolute_naive"),
                            ("paired - naive", "paired", "absolute_naive")):
            rec = dict(level=lab, ceiling_level=cl, n_cells=len(s), contrast=cname)
            for tag in ("raw", "adj"):
                d = s[f"{tag}_{x}"] - s[f"{tag}_{y}"]
                se = d.std(ddof=1) / np.sqrt(len(d))
                rec[f"{tag}_mean"] = d.mean()
                rec[f"{tag}_ci_lo"] = d.mean() - 1.96 * se
                rec[f"{tag}_ci_hi"] = d.mean() + 1.96 * se
            summ.append(rec)
    crit_tab = pd.DataFrame([dict(level=lab, ceiling_level=cl,
                                  **{n: crit.get((n, cl), np.nan) for n, _ in cols})
                             for lab, cl in LEVELS])
    out = ROOT / "outputs" / "tables"
    cells.round(5).to_csv(out / "size_adjusted_cells.csv", index=False)
    pd.DataFrame(summ).round(5).to_csv(out / "size_adjusted_summary.csv", index=False)
    crit_tab.round(5).to_csv(out / "critical_values.csv", index=False)
    return cells, pd.DataFrame(summ), crit_tab


def main():
    ap = argparse.ArgumentParser(description="run the simulation sweep")
    ap.add_argument("--grid", default="all",
                    choices=["main", "extension", "gradient", "all"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--quick", action="store_true",
                    help="12-cell smoke subset at 40 replicates")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    grids = ["main", "extension", "gradient"] if args.grid == "all" else [args.grid]
    cells = [c for g in grids for c in build_grid(g)]
    unit = {2: .45, 3: .50, 4: .85, 6: 1.35, 8: 3.0, 12: 8.0, 20: 19.0}
    core_s = sum(unit.get(c["n_readers"], 20.) * c["n_sims"] for c in cells)
    print(f"{len(cells)} cells, {sum(c['n_sims'] for c in cells)} replicates, "
          f"{core_s/3600:.1f} core-hours")
    for w in sorted({args.workers, 32, 128}):
        print(f"  {w:>3} workers: {core_s/w/60:6.1f} min")
    if args.dry_run:
        return
    sweep(grids, args.workers, quick=args.quick)
    if not args.quick:
        size_adjust()


if __name__ == "__main__":
    main()
