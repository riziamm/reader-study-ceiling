"""Reader-centred absolute ratings: isolates βr, leaves everything else alone.

Subtract each reader's own mean (pooled over both models, within criterion)
from that reader's absolute scores, then recompute ordinal alpha on the
centred ratings. This removes the reader main effect while leaving the case
effect, the model contrast, the images and the support untouched -- the
counterfactual the paper says it cannot construct directly, because it is
the raw ratings with only one term removed.

If alpha_centred lands near alpha_absolute, centring buys nothing and the
recoding result in Section 5.2 is not a reader-centring artefact. If it lands
near alpha_difference, βr was doing more work than the paper's model implies
and that needs to be said plainly.

Run: DATASET=real python src/reader_centred_alpha.py
"""
import numpy as np
import pandas as pd

import stats as st
from utils import ROOT, TABLES, cluster_bootstrap, load_config, pairs, save, tidy


def centre_by_reader(mats):
    """Subtract each reader's mean, pooled over both models within this
    criterion. A scalar per reader, estimated once on the full sample --
    the standard treatment of a nuisance term being profiled out, not
    re-estimated inside the bootstrap below."""
    pooled = pd.concat(list(mats.values()), axis=0)
    reader_mean = pooled.mean(axis=0)
    return {m: v.subtract(reader_mean, axis=1) for m, v in mats.items()}


def alpha_centred(mats_centred):
    return float(np.nanmean([st.alpha_matrix(mats_centred[m], "ordinal")
                             for m in ("Model A", "Model B")]))


def main():
    cfg = load_config()
    nb, ci, seed = cfg["analysis"]["n_boot"], cfg["analysis"]["ci"], cfg["seed"]
    d, w = tidy(cfg), pairs(cfg)
    rng = np.random.default_rng(seed)

    rec = pd.read_csv(TABLES / "recoding.csv") if (TABLES / "recoding.csv").exists() else None
    if rec is None:
        raise SystemExit("outputs/tables/recoding.csv not found; run "
                         "`DATASET=real python src/paper.py` first")

    rows = []
    for crit in sorted(w.criterion.unique()):
        mats = {m: d[(d.criterion == crit) & (d.model == m)].pivot_table(
            index="case_num", columns="reader", values="score", aggfunc="first")
            for m in ("Model A", "Model B")}
        centred = centre_by_reader(mats)
        a_cent = alpha_centred(centred)

        n = len(next(iter(centred.values())))
        boots = []
        for _ in range(nb):
            idx = rng.integers(0, n, n)
            try:
                boots.append(alpha_centred({m: v.iloc[idx] for m, v in centred.items()}))
            except Exception:
                pass
        boots = np.asarray([b for b in boots if b == b])
        lo, hi = np.percentile(boots, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])

        r = rec[rec.criterion == crit].iloc[0]
        span = r.alpha_difference - r.alpha_absolute
        # 0 = lands exactly on the absolute value, 1 = exactly on the
        # difference value; undefined (nan) if the two are equal
        frac = (a_cent - r.alpha_absolute) / span if span else np.nan
        rows.append(dict(criterion=crit, alpha_absolute=r.alpha_absolute,
                         alpha_centred=round(a_cent, 4), ci_lo=round(lo, 4),
                         ci_hi=round(hi, 4), alpha_difference=r.alpha_difference,
                         frac_of_the_way_to_difference=round(frac, 3)))

    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    med_frac = t.frac_of_the_way_to_difference.median()
    print(f"\nmedian position between absolute (0) and difference (1): {med_frac:+.3f}")
    if abs(med_frac) < 0.25:
        verdict = "close to the ABSOLUTE value -- centring alone buys little"
    elif abs(med_frac - 1) < 0.25:
        verdict = "close to the DIFFERENCE value -- \u03b2r was doing real work"
    else:
        verdict = "between the two -- \u03b2r removal explains part but not most of the change"
    print(f"verdict: {verdict}")
    save(t, "reader_centred_alpha")


if __name__ == "__main__":
    main()