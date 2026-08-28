"""Null distribution of Δα: how much of +0.158 is cancellation alone?

The frozen generator (simulate_long) at true_gap=0 has no between-model
signal, but reader and case effects still cancel in the within-case
difference exactly as in Section 5.2. Recoding null-arm ratings the same way
as analysis.recoding therefore isolates the mechanical contribution of that
cancellation from any real effect.

Run: python src/null_delta_alpha.py
"""
import numpy as np
import pandas as pd

import stats as st
from simulate import ANCHOR_CEILING, CALIB_CASE_SD, simulate_long
from utils import ROOT, load_config, save

N_REPS = 2000


def delta_alpha(df, n_readers, n_cases):
    """Same recoding as analysis.recoding: alpha on the within-case
    difference minus mean alpha on the two absolute strata, pooled over the
    four criteria (matching how the paper reports one number per criterion,
    then a median)."""
    out = []
    for crit in st.CRIT if hasattr(st, "CRIT") else sorted(df.criterion.unique()):
        g = df[df.criterion == crit]
        absolute = {m: g[g.model == m].pivot_table(
            index="case_num", columns="reader", values="score", aggfunc="first")
            for m in ("Model A", "Model B")}
        diff = (absolute["Model A"] - absolute["Model B"])
        a_abs = np.nanmean([st.alpha_matrix(absolute[m], "ordinal")
                            for m in ("Model A", "Model B")])
        a_dif = st.alpha_matrix(diff, "ordinal")
        out.append(a_dif - a_abs)
    return float(np.nanmedian(out))


def main():
    cfg = load_config()
    st_ = cfg["study"]
    n_readers, n_cases = st_["n_readers"], st_["n_cases"]

    print(f"generator: ceiling={ANCHOR_CEILING}, case_sd={CALIB_CASE_SD} (frozen), "
          f"true_gap=0, n_readers={n_readers}, n_cases={n_cases}, {N_REPS} reps")

    vals = []
    for i in range(N_REPS):
        df, _ = simulate_long(ANCHOR_CEILING, 0.0, n_readers, n_cases, seed=10_000 + i)
        v = delta_alpha(df, n_readers, n_cases)
        if v == v:
            vals.append(v)
    vals = np.asarray(vals)

    lo, hi = np.percentile(vals, [2.5, 97.5])
    observed = 0.158
    pct_above = float((vals >= observed).mean())
    print(f"\nnull median change of pairing/recoding alone: {np.median(vals):+.4f}")
    print(f"95% range: [{lo:+.4f}, {hi:+.4f}]  (n={len(vals)} valid reps)")
    print(f"share of the observed +{observed:.3f} attributable to the null "
          f"(median / observed): {np.median(vals) / observed:.1%}")
    print(f"fraction of null replicates reaching >= +{observed:.3f}: {pct_above:.4f}")

    t = pd.DataFrame([dict(n_reps=len(vals), median=np.median(vals), ci_lo=lo, ci_hi=hi,
                           observed=observed, pct_of_observed=np.median(vals) / observed,
                           pct_null_reps_ge_observed=pct_above)])
    save(t.round(4), "null_delta_alpha")


if __name__ == "__main__":
    main()