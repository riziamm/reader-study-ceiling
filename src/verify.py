"""Re-derive every headline number from the csvs and assert it matches
results.md. Exits non-zero on any mismatch. Run this last."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import stats as st
from utils import TABLES

TOL = 1e-3
fails = []


def check(name, got, want, tol=TOL):
    ok = abs(float(got) - float(want)) <= tol
    print(f"  {'pass' if ok else 'FAIL'}  {name:<44} {float(got):>12.6g} "
          f"vs {float(want):>12.6g}")
    if not ok:
        fails.append(f"{name}: recomputed {got:.6g}, stored {want:.6g}")


def main():
    p = TABLES / "_facts.json"
    if not p.exists():
        sys.exit(f"{p} not found; run src/paper.py first")
    F = json.loads(p.read_text())

    sat = pd.read_csv(TABLES / "saturation.csv")
    coef = pd.read_csv(TABLES / "coefficients_by_stratum.csv")
    vc = pd.read_csv(TABLES / "variance_components.csv")
    wit = pd.read_csv(TABLES / "within_case_deltas.csv")
    rec = pd.read_csv(TABLES / "recoding.csv")

    print("ratings")
    check("share at scale maximum",
          (sat.pct_at_max * sat.n).sum() / sat.n.sum(), F["pct_at_max"])
    check("median observed agreement", coef.p_o.median(), F["p_o"])
    check("median chance-expected agreement", coef.p_e.median(), F["p_e"])
    check("median over range-sensitive values",
          np.nanmedian(coef[st.SENSITIVE].to_numpy(float)), F["median_sensitive"])
    check("strata with negative Fleiss kappa", (coef.fleiss < 0).sum(),
          F["n_fleiss_negative"])
    check("between-case variance share", vc.pct_case.median(),
          F["between_case_share"])
    check("pooled within-case effect",
          (wit.mean_delta * wit.n_pairs).sum() / wit.n_pairs.sum(),
          F["within_case_pooled"])
    check("criteria excluding zero", wit.excludes_zero.sum(),
          F["n_criteria_excluding_zero"])

    print("\nrecoding")
    for r in rec.itertuples():
        check(f"{r.criterion[:20]} change", r.alpha_difference - r.alpha_absolute,
              r.change_alpha)
    n_up = int((rec.change_alpha > 0).sum())
    check("criteria higher on the difference", n_up, F["recode_n_higher"])
    if n_up < len(rec):
        print(f"  note  the recoding holds in {n_up} of {len(rec)} criteria, "
              f"not all of them; results.md must say so")

    lo = TABLES / "loro_difference.csv"
    if lo.exists():
        t = pd.read_csv(lo)
        check("worst leave-one-reader-out refit", t.n_criteria_higher.min(),
              F["loro_difference_worst"])

    print("\nno per-reader value in any published table")
    leaks = []
    for f in sorted(TABLES.glob("table*.csv")):
        bad = [c for c in pd.read_csv(f).columns
               if " " not in c.strip() and "reader" in c.lower()
               and c.lower() not in ("n_readers", "readers")]
        if bad:
            leaks.append(f"{f.name}: {bad}")
    for l in leaks:
        print(f"  FAIL  {l}")
    fails.extend(leaks)
    if not leaks:
        print("  pass  none found")

    if fails:
        print(f"\nFAIL: {len(fails)} mismatch(es)")
        for f in fails:
            print(f"  {f}")
        sys.exit(1)
    print("\nPASS: results.md agrees with every table")


if __name__ == "__main__":
    main()
