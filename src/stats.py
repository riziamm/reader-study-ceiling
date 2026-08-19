"""Agreement coefficients and variance decomposition.

Range-sensitive: Krippendorff alpha (ordinal, interval), ICC(2,1), ICC(3,1),
Fleiss kappa. Range-robust: Gwet AC1/AC2, PABAK, reported for context only
(Remark 1).
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

CATS = (1, 2, 3, 4, 5)

def wide_matrix(g: pd.DataFrame) -> pd.DataFrame:
    """cases x readers matrix of scores for one criterion x model stratum."""
    return g.pivot_table(index="case_num", columns="reader", values="score",
                         aggfunc="first")


def observed_expected_agreement(mat: pd.DataFrame):
    """Mean pairwise percent agreement, and agreement expected by chance under
    independent raters sharing the pooled marginal."""
    m = mat.dropna()
    if m.shape[0] == 0 or m.shape[1] < 2:
        return np.nan, np.nan
    cols = list(m.columns)
    obs = np.mean([(m[a].to_numpy() == m[b].to_numpy()).mean()
                   for a, b in itertools.combinations(cols, 2)])
    vals = m.to_numpy().ravel()
    cats, cnt = np.unique(vals, return_counts=True)
    p = cnt / cnt.sum()
    exp = float((p ** 2).sum())
    return float(obs), exp


def pabak(p_obs: float, k_cat: int) -> float:
    """Prevalence-and-bias-adjusted kappa, generalised to k categories:
    assumes a uniform chance distribution rather than the empirical one."""
    if np.isnan(p_obs) or k_cat < 2:
        return np.nan
    return (k_cat * p_obs - 1) / (k_cat - 1)


def fleiss(mat: pd.DataFrame, cats=(1, 2, 3, 4, 5)):
    from statsmodels.stats.inter_rater import fleiss_kappa

    m = mat.dropna()
    if m.shape[0] == 0:
        return np.nan
    table = np.zeros((m.shape[0], len(cats)))
    for i in range(m.shape[0]):
        for v in m.iloc[i]:
            table[i, int(v) - 1] += 1
    try:
        return float(fleiss_kappa(table, method="fleiss"))
    except Exception:
        return np.nan


def icc_family(g: pd.DataFrame):
    import pingouin as pg

    out = dict.fromkeys(["ICC1", "ICC2_1", "ICC2_k", "ICC3_1", "ICC3_k"], np.nan)
    sub = g.dropna(subset=["score"])
    try:
        r = pg.intraclass_corr(data=sub, targets="case_num", raters="reader",
                               ratings="score", nan_policy="omit")
        # pingouin >=0.5 labels: ICC(1,1) ICC(A,1) ICC(C,1) ICC(1,k) ICC(A,k) ICC(C,k)
        # A = absolute agreement (ICC2 family), C = consistency (ICC3 family).
        mp = {"ICC(1,1)": "ICC1", "ICC(A,1)": "ICC2_1", "ICC(A,k)": "ICC2_k",
              "ICC(C,1)": "ICC3_1", "ICC(C,k)": "ICC3_k"}
        for t, key in mp.items():
            v = r.loc[r["Type"] == t, "ICC"]
            if len(v):
                out[key] = float(v.iloc[0])
    except Exception as e:
        print(f"   ICC failed: {e}")
    return out


def kripp(mat: pd.DataFrame, level: str):
    import krippendorff

    m = mat.T.to_numpy(float)          # raters x units
    try:
        return float(krippendorff.alpha(reliability_data=m,
                                        level_of_measurement=level))
    except Exception:
        return np.nan


def gwet(mat: pd.DataFrame, cats=(1, 2, 3, 4, 5)):
    """Gwet AC1 and AC2. Categories pinned to the full 1-5 scale so
    strata using only part of it stay comparable."""
    from irrCAC.raw import CAC

    m = mat.dropna()
    if m.shape[0] < 2:
        return np.nan, np.nan
    ac1 = ac2 = np.nan
    try:
        ac1 = float(CAC(m.astype(int), categories=list(cats))
                    .gwet()["est"]["coefficient_value"])
    except Exception as e:
        print(f"   Gwet AC1 failed: {e}")
    try:
        ac2 = float(CAC(m.astype(int), weights="ordinal", categories=list(cats))
                    .gwet()["est"]["coefficient_value"])
    except Exception as e:
        print(f"   Gwet AC2 failed: {e}")
    return ac1, ac2


def variance_components(mat: pd.DataFrame):
    """Two-way random-effects ANOVA without replication (Shrout-Fleiss).
    Negative components are reported as computed, not truncated."""
    m = mat.dropna()
    n, k = m.shape
    if n < 2 or k < 2:
        return dict.fromkeys(["var_case", "var_reader", "var_residual",
                              "pct_case", "n_cases", "n_readers"], np.nan)
    x = m.to_numpy(float)
    grand = x.mean()
    ms_case = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_reader = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + grand
    ms_err = (resid ** 2).sum() / ((n - 1) * (k - 1))
    v_case = (ms_case - ms_err) / k
    v_read = (ms_reader - ms_err) / n
    tot = max(v_case, 0) + max(v_read, 0) + ms_err
    return {"var_case": v_case, "var_reader": v_read, "var_residual": ms_err,
            "pct_case": 100 * max(v_case, 0) / tot if tot > 0 else np.nan,
            "n_cases": n, "n_readers": k}


# --------------------------------------------------------------------------
# matrix-level variants, used for the recoded within-case difference where the
# category set is not the 1-5 rating scale
# --------------------------------------------------------------------------

def pairwise_agreement(mat):
    """Mean pairwise exact-match rate over reader pairs, on cells both scored."""
    vals = []
    for a, b in itertools.combinations(list(mat.columns), 2):
        s = mat[[a, b]].dropna()
        if len(s):
            vals.append(float((s[a].to_numpy() == s[b].to_numpy()).mean()))
    return float(np.mean(vals)) if vals else np.nan


def expected_agreement(mat):
    v = mat.to_numpy().ravel()
    v = v[~pd.isna(v)]
    if v.size == 0:
        return np.nan
    _, cnt = np.unique(v, return_counts=True)
    p = cnt / cnt.sum()
    return float((p ** 2).sum())


def fleiss_matrix(mat, cats):
    """Fleiss kappa tolerating missing cells; items with fewer than two raters
    are dropped, and unequal raters per item are levelled off, since Fleiss is
    undefined otherwise."""
    from statsmodels.stats.inter_rater import fleiss_kappa
    rows = []
    for _, r in mat.iterrows():
        v = r.dropna()
        if len(v) >= 2:
            rows.append([int((v == c).sum()) for c in cats])
    if len(rows) < 2:
        return np.nan
    tab = np.asarray(rows, float)
    tab = tab[tab.sum(axis=1) == tab.sum(axis=1).max()]
    if len(tab) < 2:
        return np.nan
    try:
        return float(fleiss_kappa(tab, method="fleiss"))
    except Exception:
        return np.nan


def alpha_matrix(mat, level):
    import krippendorff
    try:
        return float(krippendorff.alpha(reliability_data=mat.T.to_numpy(float),
                                        level_of_measurement=level))
    except Exception:
        return np.nan


def marginal(mat):
    v = mat.to_numpy().ravel()
    v = v[~pd.isna(v)]
    if v.size == 0:
        return "n/a"
    u, c = np.unique(v, return_counts=True)
    return " ".join(f"{int(k)}:{100 * n / c.sum():.0f}%" for k, n in zip(u, c))


def stratum_coefficients(g):
    """All coefficients for one criterion x model stratum (Table C1)."""
    mat = wide_matrix(g)
    p_o, p_e = observed_expected_agreement(mat)
    icc = icc_family(g)
    ac1, ac2 = gwet(mat, CATS)
    return dict(alpha_ord=kripp(mat, "ordinal"), alpha_int=kripp(mat, "interval"),
                ICC2_1=icc.get("ICC2_1", np.nan), ICC3_1=icc.get("ICC3_1", np.nan),
                fleiss=fleiss(mat, CATS), AC1=ac1, AC2=ac2,
                PABAK=pabak(p_o, len(CATS)), p_o=p_o, p_e=p_e)


SENSITIVE = ["alpha_ord", "alpha_int", "ICC2_1", "ICC3_1", "fleiss"]
ROBUST = ["AC1", "AC2", "PABAK"]
