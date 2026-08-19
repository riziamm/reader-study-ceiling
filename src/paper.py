"""Build every table and figure the paper uses, from one pass over the data.

Each number is computed once here and written to results.md, so no quantity is
produced twice by two routes. verify.py re-derives them from the csvs and
asserts they match.

    python src/paper.py            ratings only (Tables 1, 2, C1, D-block; Figures 1, 2, C1, F1-F4)
    python src/paper.py --sim      also the simulation (Tables B1-B3, D1; Figure 3)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import analysis as an
import figures as fg
import stats as st
from utils import ROOT, TABLES, load_config, pairs, save, tidy


def ratings_block(cfg):
    d, w = tidy(cfg), pairs(cfg)
    sat, pct_max = an.saturation(cfg, d)
    coef, vc, med40, n_neg = an.coefficients(cfg, d)
    absolute, within, pooled, n_ex = an.absolute_within(cfg, d, w)
    rec = an.recoding(cfg, d, w, coef)
    lo_abs, lo_dif, flips, full_med, full_pos = an.loro(cfg, d, w, coef)
    loc = an.locality(cfg, w)
    worst, flip_n, pos_d, pos_p, ties = an.robustness(cfg, d, w)

    # Table 1
    t1 = pd.DataFrame([
        ("Judgements at the scale maximum", f"{pct_max:.1f}%",
         "The scale was saturated."),
        ("Median observed agreement", f"{coef.p_o.median():.3f}",
         "Readers agreed outright more often than not."),
        ("Median chance-expected agreement", f"{coef.p_e.median():.3f}",
         "The chance model expects more agreement than occurred."),
        (f"Median over all {coef[st.SENSITIVE].size} range-sensitive values",
         f"{med40:+.3f}", "Reads as no agreement beyond chance."),
        ("Fleiss kappa, strata with a negative value", f"{n_neg} of {len(coef)}",
         "Reads as systematic disagreement, which did not occur."),
        ("Between-case variance share", f"{vc.pct_case.median():.1f}%",
         "Almost no case-level signal for a variance ratio to use."),
        ("Within-case effect (pooled)", f"{pooled:+.4f}",
         "The same ratings show a consistent difference between systems."),
        ("Criteria whose interval excludes zero", f"{n_ex} of {len(within)}",
         "The effect is present in most criteria, not one outlier."),
    ], columns=["statistic", "value", "what it invites a reader to conclude"])
    save(t1, "table1_statistics")

    # Table 2, with a case-clustered bootstrap interval on each change and on
    # the median, computed by resampling cases and recomputing both alphas
    med_lo, med_hi = rec.attrs["median_change_ci"]
    t2 = rec[["criterion", "alpha_absolute", "alpha_difference", "change_alpha",
              "change_ci_lo", "change_ci_hi", "fleiss_absolute",
              "fleiss_difference", "change_fleiss"]].round(3).copy()
    t2["change_alpha [95% CI]"] = [
        f"{r.change_alpha:+.3f} [{r.change_ci_lo:+.3f}, {r.change_ci_hi:+.3f}]"
        for r in t2.itertuples()]
    t2 = t2.drop(columns=["change_ci_lo", "change_ci_hi"])
    t2.loc[len(t2)] = {"criterion": "Median",
                       "change_alpha": round(rec.change_alpha.median(), 3),
                       "change_fleiss": round(rec.change_fleiss.median(), 3),
                       "change_alpha [95% CI]":
                           f"{rec.change_alpha.median():+.3f} "
                           f"[{med_lo:+.3f}, {med_hi:+.3f}]"}
    save(t2, "table2_recoding")

    # Table C1
    save(coef[["criterion", "model"] + st.SENSITIVE + st.ROBUST + ["p_o", "p_e"]].round(5),
         "tableC1_all_coefficients")

    fg.fig1_paired(rec, coef)
    fg.fig2_absolute_within(absolute, within)
    fg.figC1_coefficients(coef.rename(columns={
        "alpha_ord": "krippendorff_ordinal", "alpha_int": "krippendorff_interval",
        "fleiss": "fleiss_kappa", "AC1": "gwet_ac1", "AC2": "gwet_ac2",
        "PABAK": "pabak", "p_o": "pct_observed_agreement",
        "p_e": "pct_expected_agreement"}))
    fg.figF1_distribution(d)
    fg.figF2_variance(vc)
    fg.figF3_deltas(w)
    fg.figF4_bland_altman(w)

    return dict(
        pct_at_max=pct_max, p_o=float(coef.p_o.median()), p_e=float(coef.p_e.median()),
        median_sensitive=med40, n_fleiss_negative=n_neg, n_strata=len(coef),
        between_case_share=float(vc.pct_case.median()), within_case_pooled=pooled,
        n_criteria_excluding_zero=n_ex, n_criteria=len(within),
        recode_n_higher=int((rec.change_alpha > 0).sum()),
        recode_median_change=float(rec.change_alpha.median()),
        recode_median_change_ci=list(rec.attrs["median_change_ci"]),
        recode_change_ci={r.criterion: [float(r.change_ci_lo), float(r.change_ci_hi)]
                          for r in rec.itertuples()},
        recode_p_e_difference=float(rec.p_e_difference.median()),
        loro_median_range=[float(lo_abs.median_sensitive.min()),
                           float(lo_abs.median_sensitive.max())],
        loro_positive_range=[int(lo_abs.n_strata_positive.min()),
                             int(lo_abs.n_strata_positive.max())],
        loro_full_median=full_med, loro_full_positive=full_pos,
        loro_difference_worst=int(lo_dif.n_criteria_higher.min()),
        loro_difference_change_range=[float(lo_dif.median_change.min()),
                                      float(lo_dif.median_change.max())],
        n_substantive_flips=int((~flips.marginal).sum()) if len(flips) else 0,
        max_stratum_shift=float((flips.after - flips.before).abs().max())
        if len(flips) else 0.0,
        locality=loc.set_index("kind").mean_delta.round(4).to_dict(),
        locality_p=float(loc.contrast_p.iloc[0]),
        loco_worst_pct=worst, position_delta=pos_d, position_p=pos_p,
        n_tie_conventions=len(ties), tie_max_p=float(ties.p.max()),
    )


def sim_block(cfg):
    import simulate as sim
    frames = [pd.read_csv(TABLES / f"sweep_{g}.csv")
              for g in ("main", "extension", "gradient")
              if (TABLES / f"sweep_{g}.csv").exists()]
    if not frames:
        print("no sweep tables; run src/simulate.py first")
        return {}
    s = pd.concat(frames, ignore_index=True)
    s["informative"] = s.informative.astype(bool)
    cells, summ, crit = sim.size_adjust()

    # Table B1 calibration, B2 critical values, B3 size and effect
    m = s[(s.arm == "power") & s.informative
          & s.true_gap.isin(sim.MATCH["gaps"])
          & s.n_readers.isin(sim.MATCH["readers"])
          & s.n_cases.isin(sim.MATCH["cases"])]
    b1, b3 = [], []
    for lab, cl in sim.LEVELS:
        mm, nul = m[m.ceiling_level == cl], s[(s.arm == "null_arm") & (s.ceiling_level == cl)]
        pn = summ[(summ.contrast == "paired - naive") & (summ.ceiling_level == cl)]
        if not len(mm) or not len(pn):
            continue
        pn = pn.iloc[0]
        b1.append(dict(level=lab, ceiling_level=cl,
                       pct_at_max=round(mm.realised_pct_max.mean(), 1),
                       between_case_pct=round(mm.realised_between_case_share.mean(), 2),
                       cells=len(mm)))
        b3.append(dict(level=lab, pct_at_max=round(mm.realised_pct_max.mean(), 1),
                       typeI_naive=round(nul.power_absolute_naive.mean(), 3),
                       typeI_conditioned=round(nul.power_absolute_conditioned.mean(), 3),
                       typeI_paired=round(nul.power_paired.mean(), 3), cells=len(mm),
                       adjusted=round(pn.adj_mean, 3),
                       adj_ci_lo=round(pn.adj_ci_lo, 3), adj_ci_hi=round(pn.adj_ci_hi, 3),
                       raw=round(pn.raw_mean, 3)))
    save(pd.DataFrame(b1), "tableB1_calibration")
    save(crit.round(4), "tableB2_critical_values")
    save(pd.DataFrame(b3), "tableB3_size_and_effect")

    # Table D1: across-level comparison on the size-adjusted difference
    from scipy import stats as sps
    from statsmodels.stats.multitest import multipletests
    c = cells[cells.informative].copy()
    c["adj_diff"] = c.adj_paired - c.adj_absolute_naive
    piv = c.pivot_table(index=["true_gap", "n_readers", "n_cases"],
                        columns="ceiling_level", values="adj_diff").dropna()
    rows, praw = [], []
    for a, b, la, lb in [(sim.LOW_CEILING, sim.MID_CEILING, "low", "mid"),
                         (sim.MID_CEILING, sim.ANCHOR_CEILING, "mid", "anchor"),
                         (sim.LOW_CEILING, sim.ANCHOR_CEILING, "low", "anchor")]:
        if a in piv.columns and b in piv.columns:
            p = float(sps.wilcoxon(piv[a], piv[b]).pvalue)
            rows.append(dict(comparison=f"{la} vs {lb}",
                             median_change=float(np.median(piv[b] - piv[a])), p=p))
            praw.append(p)
    d1 = pd.DataFrame(rows)
    if len(d1):
        d1["p_holm"] = multipletests(praw, method="holm")[1]
        d1["significant"] = d1.p_holm < 0.05
        save(d1.round(4), "tableD1_across_levels")

    fg.fig3_simulation(s, summ, sim.LEVELS, sim.ANCHOR_CEILING, sim.MATCH)
    return dict(sim_levels=b3, sim_n_matched_designs=int(len(piv)),
                sim_across_levels=d1.to_dict("records") if len(d1) else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="also build the simulation outputs")
    args = ap.parse_args()
    cfg = load_config()

    facts = ratings_block(cfg)
    if args.sim:
        facts.update(sim_block(cfg))
    (TABLES / "_facts.json").write_text(json.dumps(facts, indent=2, default=float))

    f = facts
    md = f"""# Results

Generated by `src/paper.py` on the **{cfg['dataset']}** dataset. Every number
below is computed once, here, and is the number the paper uses.

## The ratings

- {f['pct_at_max']:.1f}% of judgements at the scale maximum.
- Median observed agreement {f['p_o']:.3f}, median chance-expected
  {f['p_e']:.3f}. Fleiss kappa negative in {f['n_fleiss_negative']} of
  {f['n_strata']} strata; median over all range-sensitive values
  {f['median_sensitive']:+.4f}.
- Between-case variance share {f['between_case_share']:.2f}%.
- Pooled within-case difference {f['within_case_pooled']:+.4f};
  {f['n_criteria_excluding_zero']} of {f['n_criteria']} criteria exclude zero.

## Recoding (the headline)

Holding readers, session and judgements fixed and changing only the coding,
ordinal alpha is higher on the within-case difference in
**{f['recode_n_higher']} of {f['n_criteria']} criteria**, median change
{f['recode_median_change']:+.4f}
[{f['recode_median_change_ci'][0]:+.4f}, {f['recode_median_change_ci'][1]:+.4f}]
(case-clustered bootstrap, 44 clusters, {cfg['analysis']['n_boot']} replicates). Median chance-expected agreement is
{f['p_e']:.3f} on the absolute strata against
{f['recode_p_e_difference']:.3f} on the differences, so the two codings are
near-identical in concentration and concentration cannot explain the change.

This does not license the claim that the readers agreed: the coefficient on
the difference is itself small. It licenses the narrower claim that the
coefficient is not a fixed property of the panel.

Leaving out each reader in turn, the comparison stays higher in
{f['loro_difference_worst']} of {f['n_criteria']} criteria in the worst refit,
with median changes {f['loro_difference_change_range'][0]:+.4f} to
{f['loro_difference_change_range'][1]:+.4f}.

## Leave-one-reader-out on the absolute ratings

No single reader accounts for the collapse. It persists in every refit, with
medians {f['loro_median_range'][0]:+.4f} to {f['loro_median_range'][1]:+.4f}
against {f['loro_full_median']:+.4f} on the full panel, and the count of
strata with a positive median moving between
{f['loro_positive_range'][0]} and {f['loro_positive_range'][1]} of
{f['n_strata']}. Individual stratum values are unstable at this panel size,
shifting by up to {f['max_stratum_shift']:.3f} with
{f['n_substantive_flips']} sign changes outside the declared marginality band;
that instability is reported as a limitation of these coefficients at n=4.
The claim concerns the collapse, which is stable, not any stratum value.

## Robustness

- Leave-one-case-out: largest single-case influence
  {f['loco_worst_pct']:.1f}% of the pooled effect.
- Position: {f['position_delta']:+.4f}, p = {f['position_p']:.3g}.
- Tie handling: {f['n_tie_conventions']} conventions compared, largest
  p = {f['tie_max_p']:.4g}.
- Locality (pre-declared, triggered):
  {', '.join(f'{k} {v:+.3f}' for k, v in f['locality'].items())},
  p = {f['locality_p']:.4g}. Masks were never shown, so this confounds spatial
  locality with breadth of construct and is reported, not interpreted.
"""
    (ROOT / "results.md").write_text(md)
    print(f"\nwrote {ROOT / 'results.md'} and {TABLES / '_facts.json'}")


if __name__ == "__main__":
    main()
