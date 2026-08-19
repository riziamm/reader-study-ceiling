"""Every analysis the paper runs on the ratings.

    saturation      share at the scale maximum, per stratum
    coefficients    Table C1 and the variance decomposition
    absolute_within Figure 2 and the within-case effect
    recoding        Table 2 and Figure 1: agreement on the difference
    loro            leave-one-reader-out, on both codings
    locality        Appendix D, pre-declared and triggered
    robustness      leave-one-case-out, position, tie conventions

Run all of them with `python src/analysis.py`.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats as sps

import stats as st
from utils import cluster_bootstrap, load_config, pairs, save, tidy

CATS = st.CATS


def _strata(d):
    return d.groupby(["criterion", "model"])


# --------------------------------------------------------------------------

def saturation(cfg, d):
    lo, hi = cfg["study"]["scale"]
    rows = [dict(criterion=c, model=m, n=len(g),
                 mean=g.score.mean(),
                 pct_at_max=100 * (g.score == hi).mean(),
                 pct_at_min=100 * (g.score == lo).mean())
            for (c, m), g in _strata(d)]
    t = pd.DataFrame(rows).sort_values(["criterion", "model"])
    overall = 100 * (d.score == hi).mean()
    print(f"share of ratings at the scale maximum: {overall:.1f}%")
    save(t.round(4), "saturation")
    return t, overall


def coefficients(cfg, d):
    rows, vc = [], []
    for (c, m), g in _strata(d):
        rows.append(dict(criterion=c, model=m, **st.stratum_coefficients(g)))
        v = st.variance_components(st.wide_matrix(g))
        vc.append(dict(stratum=f"{c} · {m}", criterion=c, model=m, **v))
    t = pd.DataFrame(rows).sort_values(["criterion", "model"]).reset_index(drop=True)
    v = pd.DataFrame(vc).sort_values(["criterion", "model"]).reset_index(drop=True)

    sens = t[st.SENSITIVE].to_numpy(float)
    med40 = float(np.nanmedian(sens))
    n_neg = int((t.fleiss < 0).sum())
    print(f"median observed agreement {t.p_o.median():.4f}, "
          f"median chance-expected {t.p_e.median():.4f}")
    print(f"median over all {sens.size} range-sensitive values {med40:+.4f}; "
          f"Fleiss kappa negative in {n_neg} of {len(t)} strata")
    print(f"median between-case variance share {v.pct_case.median():.2f}%")
    save(t.round(4), "coefficients_by_stratum")
    save(v.round(4), "variance_components")
    return t, v, med40, n_neg


def absolute_within(cfg, d, w):
    nb, ci, seed = cfg["analysis"]["n_boot"], cfg["analysis"]["ci"], cfg["seed"]
    rows = []
    for crit, g in d.groupby("criterion"):
        rec = dict(criterion=crit)
        for m in ("Model A", "Model B"):
            s = g[g.model == m]
            rec[f"mean_{m}"] = s.score.mean()
            _, lo, hi, _ = cluster_bootstrap(
                s, "case_num", lambda x: float(x.score.mean()), nb, ci, seed)
            rec[f"ci_lo_{m}"], rec[f"ci_hi_{m}"] = lo, hi
        rows.append(rec)
    absolute = pd.DataFrame(rows).sort_values("criterion").reset_index(drop=True)

    rows = []
    for crit, g in w.groupby("criterion"):
        est, lo, hi, _ = cluster_bootstrap(
            g, "case_num", lambda x: float(x.delta.mean()), nb, ci, seed)
        rows.append(dict(criterion=crit, n_pairs=len(g), mean_delta=est,
                         ci_lo=lo, ci_hi=hi,
                         pct_tied=100 * (g.delta == 0).mean(),
                         excludes_zero=bool(lo > 0 or hi < 0)))
    within = pd.DataFrame(rows).sort_values("criterion").reset_index(drop=True)

    pooled = float((within.mean_delta * within.n_pairs).sum() / within.n_pairs.sum())
    n_ex = int(within.excludes_zero.sum())
    print(f"pooled within-case difference {pooled:+.4f}; "
          f"{n_ex} of {len(within)} criteria exclude zero; "
          f"{100 * (w.delta == 0).mean():.1f}% of pairs tied")
    save(absolute.round(4), "absolute_means")
    save(within.round(4), "within_case_deltas")
    return absolute, within, pooled, n_ex


# --------------------------------------------------------------------------

def _difference_matrix(w, crit):
    return w[w.criterion == crit].pivot_table(index="case_num", columns="reader",
                                              values="delta", aggfunc="first")


def _criterion_matrices(d, w, crit):
    """cases x readers for each model, and for the within-case difference,
    all on the same case index so a bootstrap resamples them together."""
    mats = {}
    for m in ("Model A", "Model B"):
        g = d[(d.criterion == crit) & (d.model == m)]
        mats[m] = g.pivot_table(index="case_num", columns="reader", values="score",
                                aggfunc="first")
    mats["diff"] = _difference_matrix(w, crit)
    idx = mats["diff"].index
    return {k: v.reindex(idx) for k, v in mats.items()}


def _delta_alpha(mats, rows=None):
    """alpha on the difference minus mean alpha on the two absolute strata."""
    take = (lambda m: m.iloc[rows]) if rows is not None else (lambda m: m)
    absolute = np.nanmean([st.alpha_matrix(take(mats[m]), "ordinal")
                           for m in ("Model A", "Model B")])
    return st.alpha_matrix(take(mats["diff"]), "ordinal") - absolute


def recoding(cfg, d, w, coef):
    """Table 2. Agreement on the absolute ratings against agreement on the
    within-case difference, per criterion, with case-clustered bootstrap
    intervals on the change. Ordinal alpha on both sides is the like-for-like
    comparison; Fleiss kappa compares five categories on the ratings against
    three on the sign of the difference and depends on that count."""
    nb, ci, seed = cfg["analysis"]["n_boot"], cfg["analysis"]["ci"], cfg["seed"]
    rng = np.random.default_rng(seed)
    rows, boots = [], {}
    for crit in sorted(w.criterion.unique()):
        mats = _criterion_matrices(d, w, crit)
        n = len(mats["diff"])
        b = np.array([_delta_alpha(mats, rng.integers(0, n, n)) for _ in range(nb)])
        b = b[np.isfinite(b)]
        boots[crit] = b
        lo, hi = np.percentile(b, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])

        dm = mats["diff"]
        sign3 = np.sign(dm)
        abs_ = coef[coef.criterion == crit]
        rows.append(dict(
            criterion=crit,
            alpha_absolute=float(np.nanmean(abs_.alpha_ord)),
            alpha_difference=st.alpha_matrix(dm, "ordinal"),
            fleiss_absolute=float(np.nanmean(abs_.fleiss)),
            fleiss_difference=st.fleiss_matrix(sign3, [-1, 0, 1]),
            p_o_difference=st.pairwise_agreement(sign3),
            p_e_difference=st.expected_agreement(sign3),
            marginal_sign=st.marginal(sign3),
            change_ci_lo=lo, change_ci_hi=hi, n_cases=n))
    t = pd.DataFrame(rows)
    t["change_alpha"] = t.alpha_difference - t.alpha_absolute
    t["change_fleiss"] = t.fleiss_difference - t.fleiss_absolute

    # interval on the median change: resample cases once and take the median
    # over criteria within each replicate, so the criteria stay paired
    k = min(len(b) for b in boots.values())
    med = np.median(np.vstack([boots[c][:k] for c in sorted(boots)]), axis=0)
    med_lo, med_hi = np.percentile(med, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    t.attrs["median_change_ci"] = (float(med_lo), float(med_hi))

    n_up = int((t.change_alpha > 0).sum())
    print(f"agreement higher on the difference in {n_up} of {len(t)} criteria; "
          f"median change {t.change_alpha.median():+.4f} "
          f"[{med_lo:+.4f}, {med_hi:+.4f}] (alpha), "
          f"{t.change_fleiss.median():+.4f} (Fleiss)")
    print(f"median chance-expected on the difference {t.p_e_difference.median():.4f}")
    save(t.round(4), "recoding")
    return t


def loro(cfg, d, w, coef):
    """Leave-one-reader-out on both codings. Refits are ordered by resulting
    value and never labelled by reader: with four readers, a labelled row plus
    the full-sample value identifies that reader by subtraction."""
    band = cfg["analysis"]["marginal_band"]
    readers = sorted(d.reader.unique())

    def sens_median(dd):
        vals = [list(st.stratum_coefficients(g)[k] for k in st.SENSITIVE)
                for _, g in _strata(dd)]
        return float(np.nanmedian(np.asarray(vals, float)))

    def per_stratum(dd):
        return {f"{c} · {m}": float(np.nanmedian(
            [st.stratum_coefficients(g)[k] for k in st.SENSITIVE]))
            for (c, m), g in _strata(dd)}

    def change_counts(dd, ww):
        cf = pd.DataFrame([dict(criterion=c, model=m, **st.stratum_coefficients(g))
                           for (c, m), g in _strata(dd)])
        n = 0
        med = []
        for crit in sorted(ww.criterion.unique()):
            a = float(np.nanmean(cf[cf.criterion == crit].alpha_ord))
            b = st.alpha_matrix(_difference_matrix(ww, crit), "ordinal")
            n += int(b - a > 0)
            med.append(b - a)
        return n, float(np.nanmedian(med))

    full_ps = per_stratum(d)
    full_med = sens_median(d)
    full_pos = sum(v > 0 for v in full_ps.values())

    abs_rows, dif_rows, flips = [], [], []
    for held in readers:
        sub, subw = d[d.reader != held], w[w.reader != held]
        ps = per_stratum(sub)
        abs_rows.append(dict(median_sensitive=sens_median(sub),
                             n_strata_positive=sum(v > 0 for v in ps.values()),
                             n_strata=len(ps)))
        n_up, med_chg = change_counts(sub, subw)
        dif_rows.append(dict(n_criteria_higher=n_up,
                             n_criteria=subw.criterion.nunique(),
                             median_change=med_chg))
        for k, v in ps.items():
            if full_ps[k] <= 0 < v:
                flips.append(dict(stratum=k, before=full_ps[k], after=v,
                                  marginal=bool(abs(full_ps[k]) < band
                                                and abs(v) < band)))

    a = pd.DataFrame(abs_rows).sort_values("median_sensitive").reset_index(drop=True)
    a.insert(0, "refit", [f"refit {i+1}" for i in range(len(a))])
    b = pd.DataFrame(dif_rows).sort_values("median_change").reset_index(drop=True)
    b.insert(0, "refit", [f"refit {i+1}" for i in range(len(b))])
    f = pd.DataFrame(flips)

    print(f"absolute ratings, full sample median {full_med:+.4f}, "
          f"{full_pos} of {len(full_ps)} strata positive")
    print(f"  refit medians {a.median_sensitive.min():+.4f} to "
          f"{a.median_sensitive.max():+.4f}; strata positive "
          f"{a.n_strata_positive.min()} to {a.n_strata_positive.max()}")
    if len(f):
        print(f"  {int((~f.marginal).sum())} of {len(f)} sign changes exceed the "
              f"{band} band; largest stratum shift "
              f"{(f.after - f.before).abs().max():.3f}")
    print(f"difference, worst refit keeps {b.n_criteria_higher.min()} of "
          f"{b.n_criteria.iloc[0]} criteria higher; median change "
          f"{b.median_change.min():+.4f} to {b.median_change.max():+.4f}")
    save(a.round(4), "loro_absolute")
    save(b.round(4), "loro_difference")
    if len(f):
        save(f.round(4), "loro_stratum_flips")
    return a, b, f, full_med, full_pos


# --------------------------------------------------------------------------

def locality(cfg, w):
    """Pre-declared check of whether the difference concentrates where
    inpainting acts. Masks were never shown, so a positive result confounds
    spatial locality with breadth of construct; reported, not interpreted."""
    nb, ci, seed = cfg["analysis"]["n_boot"], cfg["analysis"]["ci"], cfg["seed"]
    w = w.assign(kind=w.criterion.map(cfg["criterion_type"])).dropna(subset=["kind"])
    rows = []
    for k, g in w.groupby("kind"):
        _, lo, hi, _ = cluster_bootstrap(
            g, "case_num", lambda x: float(x.delta.mean()), nb, ci, seed)
        rows.append(dict(kind=k, criteria=", ".join(sorted(g.criterion.unique())),
                         n_pairs=len(g), mean_delta=g.delta.mean(),
                         ci_lo=lo, ci_hi=hi))
    t = pd.DataFrame(rows)
    per_case = w.pivot_table(index="case_num", columns="kind",
                             values="delta", aggfunc="mean").dropna()
    ks = list(per_case.columns)
    p = sps.wilcoxon(per_case[ks[0]], per_case[ks[1]]).pvalue if len(ks) == 2 else np.nan
    t["contrast_p"] = p
    print("locality: " + "; ".join(f"{r.kind} {r.mean_delta:+.3f}"
                                   for r in t.itertuples()) + f"; p = {p:.4g}")
    save(t.round(4), "locality")
    return t


def robustness(cfg, d, w):
    full = float(w.delta.mean())

    # leave one case out
    loco = pd.DataFrame([dict(case_num=c,
                              estimate=float(w[w.case_num != c].delta.mean()))
                         for c in sorted(w.case_num.unique())])
    loco["influence"] = 100 * (loco.estimate - full) / full
    worst = float(loco["influence"].abs().max())
    order = loco.reindex(loco["influence"].abs().sort_values(ascending=False).index)
    flip = None
    for k in range(1, len(order)):
        if np.sign(w[~w.case_num.isin(order.case_num.iloc[:k])].delta.mean()) != np.sign(full):
            flip = k
            break

    # position: left-hand against right-hand score, ignoring which model was where
    pos_delta = pos_p = np.nan
    if "position" in d.columns:
        pv = d.pivot_table(index=["reader", "case_num", "criterion"],
                           columns="position", values="score", aggfunc="first").dropna()
        pos_delta = float((pv["A"] - pv["B"]).mean())
        pos_p = float(sps.wilcoxon(pv["A"], pv["B"]).pvalue)

    # tie conventions
    a, b = w["Model A"].to_numpy(float), w["Model B"].to_numpy(float)
    dd = a - b
    pos_n, neg_n, tie_n = int((dd > 0).sum()), int((dd < 0).sum()), int((dd == 0).sum())
    conv = []
    for zm in ("wilcox", "zsplit", "pratt"):
        try:
            conv.append((f"wilcoxon {zm}", float(sps.wilcoxon(a, b, zero_method=zm).pvalue)))
        except (ValueError, TypeError):
            conv.append((f"wilcoxon {zm}", np.nan))
    exact = lambda k, n: float(sps.binomtest(k, n, 0.5).pvalue) if n else np.nan
    conv += [("sign, ties as losses", exact(pos_n, pos_n + neg_n + tie_n)),
             ("sign, ties dropped", exact(pos_n, pos_n + neg_n)),
             ("sign, ties split", exact(pos_n + tie_n // 2, pos_n + neg_n + tie_n)),
             ("paired t", float(sps.ttest_rel(a, b).pvalue))]
    ties = pd.DataFrame(conv, columns=["convention", "p"])

    print(f"leave-one-case-out: largest single-case influence {worst:.1f}%; "
          f"sign flips after removing {flip if flip else '>' + str(len(order)-1)} cases")
    if pos_delta == pos_delta:
        print(f"position effect {pos_delta:+.4f}, p = {pos_p:.3g}")
    print(f"tie conventions: {len(ties)} compared, max p = {ties.p.max():.4g}")
    save(pd.DataFrame([dict(max_single_case_pct=worst, n_cases_to_flip=flip,
                            position_delta=pos_delta, position_p=pos_p)]).round(4),
         "robustness")
    save(ties.round(6), "tie_conventions")
    return worst, flip, pos_delta, pos_p, ties


# --------------------------------------------------------------------------

def main():
    cfg = load_config()
    d, w = tidy(cfg), pairs(cfg)
    print(f"dataset {cfg['dataset']}: {len(d)} valid ratings, {len(w)} complete pairs\n")

    print("-- saturation")
    saturation(cfg, d)
    print("\n-- coefficients")
    coef, _, _, _ = coefficients(cfg, d)
    print("\n-- absolute and within-case views")
    absolute_within(cfg, d, w)
    print("\n-- recoding")
    recoding(cfg, d, w, coef)
    print("\n-- leave one reader out")
    loro(cfg, d, w, coef)
    print("\n-- locality")
    locality(cfg, w)
    print("\n-- robustness")
    robustness(cfg, d, w)


if __name__ == "__main__":
    main()
