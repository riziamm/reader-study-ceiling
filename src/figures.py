"""Figures used in the paper.

Figure 1  paired within-session comparison           fig1_paired
Figure 2  absolute and within-case views             fig2_absolute_within
Figure 3  simulation                                 fig3_simulation
Figure C1 why the coefficients inverted              figC1_coefficients
Figure F1 rating distribution                        figF1_distribution
Figure F2 variance decomposition                     figF2_variance
Figure F3 within-case paired differences             figF3_deltas
Figure F4 Bland-Altman                               figF4_bland_altman

Authored at 5.5in, the NeurIPS text width, so point sizes survive into print;
include at \\linewidth without further scaling. No in-figure titles: captions
do that. Every colour is paired with a marker and a dash pattern so the panels
read in greyscale. No per-reader value appears in any figure.
"""
from __future__ import annotations

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from utils import FIGURES

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.grid": True, "grid.alpha": .25,
    "axes.spines.top": False, "axes.spines.right": False,
})
LAB, TICK = 10.5, 8.5
OK = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
MODEL_COLORS = {"Model A": "#0072B2", "Model B": "#E69F00"}
MODEL_HATCH = {"Model A": "", "Model B": "///"}
SHORT = {"Absence of Artifacts": "Artifacts", "Anatomical Accuracy": "Anatomy",
         "Boundary Sharpness": "Boundary", "Overall Realism": "Realism"}
# two-line criterion labels; long names collide on a text-width figure
WRAP = {"Absence of Artifacts": "Absence of\nArtifacts",
        "Anatomical Accuracy": "Anatomical\nAccuracy",
        "Boundary Sharpness": "Boundary\nSharpness",
        "Overall Realism": "Overall\nRealism"}


def _save(name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plt.savefig(FIGURES / f"{name}.{ext}")
    plt.close("all")
    print(f"  wrote {(FIGURES / (name + '.pdf'))}")

def figF1_distribution(df):
    crits = sorted(df["criterion"].unique())
    models = ["Model A", "Model B"]
    fig, axes = plt.subplots(1, len(crits), figsize=(3.1 * len(crits), 3.6),
                             sharey=True)
    axes = np.atleast_1d(axes)
    greys = plt.get_cmap("cividis")(np.linspace(0.15, 0.95, 5))
    for ax, crit in zip(axes, crits):
        for i, m in enumerate(models):
            sub = df[(df.criterion == crit) & (df.model == m)]["score"]
            n = len(sub)
            left = 0
            for s in range(1, 6):
                frac = 100 * (sub == s).sum() / n if n else 0
                ax.barh(i, frac, left=left, color=greys[s - 1],
                        edgecolor="white", linewidth=.6)
                if frac > 7:
                    ax.text(left + frac / 2, i, str(s), ha="center", va="center",
                            fontsize=8,
                            color="white" if s <= 3 else "black")
                left += frac
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_title(crit, fontsize=9)
        ax.set_xlabel("% of judgements")
        ax.set_xlim(0, 100)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Distribution of 1-5 scores (numbers inside bars = score)", y=1.02)
    return _save("figF1_distribution")


def figF2_variance(vc):
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    comps = ["var_case", "var_reader", "var_residual"]
    labels = ["between-case", "between-reader", "residual"]
    colors = ["#0072B2", "#E69F00", "#999999"]
    x = np.arange(len(vc))
    bottom = np.zeros(len(vc))
    for c, lab, col in zip(comps, labels, colors):
        vals = vc[c].to_numpy(float)
        ax.bar(x, vals, bottom=bottom, label=lab, color=col,
               edgecolor="black", linewidth=.4)
        bottom += np.nan_to_num(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(vc["stratum"], rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("variance component")
    ax.set_title("Where the variance lives.\nReliability coefficients depend on the "
                 "between-case share; a ceiling shrinks it.", fontsize=10)
    ax.legend(fontsize=8)
    return _save("figF2_variance")


def fig2_absolute_within(abs_tab, within_tab):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    fig.subplots_adjust(wspace=0.30)
    # short criterion names, as in Figure 1: the full names collide at four
    # groups whatever the wrap, and the caption carries the mapping
    wrap = [SHORT.get(c, c) for c in abs_tab["criterion"]]

    ax = axes[0]
    x = np.arange(len(abs_tab))
    for i, m in enumerate(["Model A", "Model B"]):
        lo = (abs_tab[f"mean_{m}"] - abs_tab[f"ci_lo_{m}"]).to_numpy(float)
        hi = (abs_tab[f"ci_hi_{m}"] - abs_tab[f"mean_{m}"]).to_numpy(float)
        ax.bar(x + (i - .5) * .36, abs_tab[f"mean_{m}"], .36, yerr=[lo, hi],
               capsize=3, label=m, color=MODEL_COLORS[m], hatch=MODEL_HATCH[m],
               edgecolor="black", linewidth=.5)
    ax.set_xticks(x)
    ax.set_xticklabels(wrap, fontsize=TICK)
    ax.set_ylim(1, 5.4)
    ax.set_ylabel("mean absolute score", fontsize=LAB - 1)
    ax.tick_params(labelsize=TICK)
    ax.legend(fontsize=TICK - 1, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.16), frameon=False)

    ax = axes[1]
    for i, r in enumerate(within_tab.itertuples()):
        ax.errorbar(r.mean_delta, i,
                    xerr=[[r.mean_delta - r.ci_lo], [r.ci_hi - r.mean_delta]],
                    fmt="o", color="#0072B2", capsize=4, markersize=7)
    ax.axvline(0, color="black", ls="--", lw=1.2)
    ax.set_yticks(range(len(within_tab)))
    # labels on the right so they cannot run back over panel (a)
    ax.set_yticklabels([SHORT.get(c, c) for c in within_tab["criterion"]],
                       fontsize=TICK)
    ax.yaxis.tick_right()
    ax.invert_yaxis()
    ax.set_xlabel("within-case mean (Model A \u2212 Model B)", fontsize=LAB - 1)
    ax.tick_params(labelsize=TICK)

    for a, l in zip(axes, "ab"):
        a.text(0.01, 0.98, l, transform=a.transAxes, fontsize=11,
               fontweight="bold", va="top")
    return _save("fig2_absolute_within")


def figF3_deltas(w):
    crits = sorted(w["criterion"].unique())
    fig, axes = plt.subplots(1, len(crits), figsize=(3.1 * len(crits), 3.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, crits):
        d = w[w.criterion == c]["delta"]
        vals, cnts = np.unique(d, return_counts=True)
        ax.bar(vals, 100 * cnts / cnts.sum(), color="#0072B2",
               edgecolor="black", linewidth=.5, width=.8)
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.axvline(d.mean(), color="#D55E00", lw=2,
                   label=f"mean {d.mean():+.3f}")
        ax.set_title(c, fontsize=9)
        ax.set_xlabel("Model A − Model B")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("% of paired judgements")
    fig.suptitle("Within-case paired differences (0 = tie)", y=1.03)
    return _save("figF3_deltas")


def figF4_bland_altman(w):
    crits = sorted(w["criterion"].unique())
    fig, axes = plt.subplots(1, len(crits), figsize=(3.2 * len(crits), 3.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, c in zip(axes, crits):
        s = w[w.criterion == c]
        jx = s["mean_score"] + rng.normal(0, .06, len(s))
        jy = s["delta"] + rng.normal(0, .06, len(s))
        for r, mk in zip(sorted(s["reader"].unique()), ["o", "s", "^", "D", "v"]):
            k = s["reader"] == r
            ax.scatter(jx[k], jy[k], s=18, alpha=.65, marker=mk,
                       color=OK[int(r) % len(OK)], label=f"R{r}", linewidths=0)
        m, sd = s["delta"].mean(), s["delta"].std(ddof=1)
        ax.axhline(0, color="black", ls="--", lw=1)
        ax.axhline(m, color="#D55E00", lw=1.6)
        ax.axhline(m + 1.96 * sd, color="#D55E00", ls=":", lw=1.2)
        ax.axhline(m - 1.96 * sd, color="#D55E00", ls=":", lw=1.2)
        ax.set_title(c, fontsize=9)
        ax.set_xlabel("mean of the two scores")
    axes[0].set_ylabel("difference (A − B)")
    axes[-1].legend(fontsize=7, ncol=2)
    fig.suptitle("Bland-Altman by criterion (solid = mean bias, dotted = 95% limits "
                 "of agreement; points jittered)", y=1.04, fontsize=10)
    return _save("figF4_bland_altman")


def figC1_coefficients(tab):
    """
    Three panels.

    (a) Observed against chance-expected agreement, per stratum, with the
        identity line. Points below the line are strata where raters agreed
        LESS often than the chance model predicts from their own marginal
        distribution — which is the mechanism behind the collapse in (b).
    (b) Range-sensitive coefficients.
    (c) Range-robust coefficients, plus raw observed agreement drawn as
        context. Raw observed agreement is NOT chance-corrected and is NOT
        part of the reported robust median, which uses three coefficients.
    """
    sensitive = ["krippendorff_ordinal", "krippendorff_interval", "ICC2_1", "ICC3_1",
                 "fleiss_kappa"]
    robust = ["gwet_ac1", "gwet_ac2", "pabak"]
    context = "pct_observed_agreement"
    pretty = {"krippendorff_ordinal": "\u03b1\nord",
              "krippendorff_interval": "\u03b1\nint",
              "ICC2_1": "ICC\n(2,1)", "ICC3_1": "ICC\n(3,1)",
              "fleiss_kappa": "Fleiss\n\u03ba", "gwet_ac1": "Gwet\nAC1",
              "gwet_ac2": "Gwet\nAC2", "pabak": "PABAK",
              context: "raw obs.\nagr."}
    LAB, TICK = 10.0, 8.0

    # Six notional columns so row 2 can be split into equal halves: panel
    # (c) carries 3 coefficient groups x 8 strata plus the context markers
    # and needs as much width as (b).
    # Authored at the NeurIPS \textwidth of 5.5in so the point sizes below
    # survive into print unscaled. A figure authored wide and then shrunk
    # to fit renders its 8.5pt ticks at 6pt or less.
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(7.5, 2.9),
        gridspec_kw=dict(width_ratios=[1, 1.15, 1.15], wspace=0.42))

    # ---------------- (a) observed vs chance-expected ------------------
    if context in tab.columns and "pct_expected_agreement" in tab.columns:
        xx = tab["pct_expected_agreement"].to_numpy(float)
        yy = tab[context].to_numpy(float)
        both = np.concatenate([xx, yy])
        lo, hi = np.nanmin(both), np.nanmax(both)
        pad = max(0.05 * (hi - lo), 0.02)
        lim = (lo - pad, hi + pad)
        # identity line first, so it sits behind the points
        ax_a.plot(lim, lim, ls="--", lw=1.0, color="#999999", zorder=1)
        below = yy < xx
        ax_a.scatter(xx[below], yy[below], marker="o", s=46, zorder=3,
                     facecolors="#0072B2", edgecolors="black", linewidths=.7)
        ax_a.scatter(xx[~below], yy[~below], marker="o", s=46, zorder=3,
                     facecolors="none", edgecolors="black", linewidths=1.1)
        ax_a.set_xlim(*lim)
        ax_a.set_ylim(*lim)
        ax_a.set_box_aspect(1)
        ax_a.set_xlabel("chance-expected agreement", fontsize=LAB)
        ax_a.set_ylabel("observed agreement", fontsize=LAB)
        ax_a.tick_params(labelsize=TICK)
        # Deliberately no per-stratum labels: 8 points hugging the identity
        # line cannot be labelled without collision, and a crowded panel is
        # worse than an uncrowded one with the convention in the caption.

    # ---------------- (b) and (c) coefficient panels -------------------
    for ax, group in ((ax_b, sensitive), (ax_c, robust)):
        group = [g for g in group if g in tab.columns]
        w = 0.8 / max(len(tab), 1)
        for i, row in enumerate(tab.itertuples()):
            vals = [getattr(row, g) for g in group]
            ax.bar(np.arange(len(group)) + i * w - 0.4 + w / 2, vals, w,
                   label=f"{row.criterion[:14]} · {row.model[-1]}",
                   color=plt.get_cmap("cividis")(i / max(len(tab) - 1, 1)),
                   edgecolor="black", linewidth=.4)
        ax.set_xticks(range(len(group)))
        ax.set_xticklabels([pretty.get(g, g) for g in group], rotation=0,
                           ha="center", fontsize=TICK - 2.5)
        ax.axhline(0, color="black", lw=.8)
        ax.axhline(0.6, color="grey", ls=":", lw=1)
        ax.set_ylim(-0.35, 1.05)
        ax.tick_params(labelsize=TICK)
    ax_b.set_ylabel("coefficient value", fontsize=LAB)

    # raw observed agreement: markers, not bars, inside a shaded region so
    # its non-coefficient status is unmistakable without reading a caption
    if context in tab.columns:
        xc = len(robust)
        ax_c.axvspan(xc - 0.5, xc + 0.62, color="#000000", alpha=0.055, zorder=0)
        ax_c.scatter([xc] * len(tab), tab[context], marker="D", s=30,
                     facecolors="none", edgecolors="#333333", linewidths=1.1,
                     zorder=3)
        ax_c.set_xticks(list(range(len(robust))) + [xc])
        ax_c.set_xticklabels([pretty.get(g, g) for g in robust] + [pretty[context]],
                             rotation=0, ha="center", fontsize=TICK - 2.5)
        ax_c.set_xlim(-0.6, xc + 0.62)

    for ax, lab in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c")):
        ax.text(0.015, 0.985, lab, transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="top", ha="left")

    h, l = ax_b.get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.2, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.34), frameon=False)
    return _save("figC1_coefficients")


def fig1_paired(recode, coef):
    """(a) alpha on the absolute ratings against alpha on the within-case
    difference, joined per criterion. (b) observed against chance-expected
    agreement for both codings, with the identity line."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    fig.subplots_adjust(wspace=0.34)

    ax = axes[0]
    ax.axhline(0, color="black", lw=1.8, zorder=2)
    rec = recode.sort_values("criterion").reset_index(drop=True)
    for i, r in enumerate(rec.itertuples()):
        col, mk = OK[i % len(OK)], ["o", "s", "^", "D"][i % 4]
        ax.plot([0, 1], [r.alpha_absolute, r.alpha_difference], color=col, lw=1.5, zorder=3)
        ax.plot([0], [r.alpha_absolute], marker=mk, ms=6, mfc="white", mec=col,
                mew=1.6, zorder=4)
        ax.plot([1], [r.alpha_difference], marker=mk, ms=6, mfc=col, mec="black",
                mew=.7, zorder=4)
    # labels at the right end, nudged apart where two criteria land close
    span = float(rec[["alpha_absolute", "alpha_difference"]].to_numpy().ptp())
    gap = 0.055 * max(span, 1e-6)
    order = rec.alpha_difference.sort_values().index
    ypos = {}
    last = -np.inf
    for idx in order:
        y = max(float(rec.alpha_difference[idx]), last + gap)
        ypos[idx] = y
        last = y
    for i, idx in enumerate(rec.index):
        col = OK[i % len(OK)]
        ax.annotate(SHORT.get(rec.criterion[idx], rec.criterion[idx]),
                    xy=(1, ypos[idx]), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=TICK - 1.5, color=col)
    ax.set_xlim(-0.25, 1.75)
    ax.set_box_aspect(1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["absolute\nratings", "within-case\ndifference"], fontsize=TICK)
    ax.set_ylabel("Krippendorff \u03b1 (ordinal)", fontsize=LAB - 1)
    ax.tick_params(labelsize=TICK)
    ax.grid(axis="x", visible=False)
    ax.set_box_aspect(1)   # match panel b, which is forced square by its aspect

    ax = axes[1]
    both = np.concatenate([coef.p_e, coef.p_o, recode.p_e_difference,
                           recode.p_o_difference]).astype(float)
    lim = (np.nanmin(both) - .02, np.nanmax(both) + .02)
    ax.plot(lim, lim, ls="--", lw=1.0, color="#999999", zorder=1)
    ax.scatter(coef.p_e, coef.p_o, marker="o", s=30, facecolors="none",
               edgecolors="#D55E00", linewidths=1.2, zorder=3)
    ax.scatter(recode.p_e_difference, recode.p_o_difference, marker="^", s=42,
               facecolors="#0072B2", edgecolors="black", linewidths=.7, zorder=4)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_box_aspect(1)
    ax.set_xlabel("chance-expected agreement", fontsize=LAB - 1)
    ax.set_ylabel("observed agreement", fontsize=LAB - 1)
    ax.tick_params(labelsize=TICK)

    for a, l in zip(axes, "ab"):
        a.text(.02, .98, l, transform=a.transAxes, fontsize=11, fontweight="bold",
               va="top")
    fig.legend(handles=[
        Line2D([], [], ls="none", marker="o", mfc="none", mec="#555555",
               label="absolute ratings"),
        Line2D([], [], ls="none", marker="o", mfc="#555555", mec="black",
               label="within-case difference"),
        Line2D([], [], ls="none", marker="o", mfc="none", mec="#D55E00",
               label="(b) absolute strata"),
        Line2D([], [], ls="none", marker="^", mfc="#0072B2", mec="black",
               label="(b) differences")],
        fontsize=6.4, ncol=4, loc="lower center", bbox_to_anchor=(.5, -.14),
        frameon=False)
    _save("fig1_paired")


SIM_STYLE = {
    "absolute_naive": dict(c="#0072B2", m="o", ls="-", lab="naive absolute"),
    "absolute_conditioned": dict(c="#E69F00", m="s", ls="--",
                                 lab="case-conditioned absolute"),
    "paired": dict(c="#009E73", m="^", ls=":", lab="paired (sign contrast)"),
}
GAPM = {0.02: "v", 0.05: "o", 0.10: "s", 0.15: "D"}


def fig3_simulation(sweep, summary, levels, anchor, match):
    """(a) type I for the three analyses at three saturation levels,
    pre-adjustment. (b) power against panel size at the anchor, faceted by
    effect size, saturated cells shaded. (c) power difference between the
    paired contrast and naive absolute against realised saturation."""
    m = sweep[(sweep.arm == "power") & sweep.informative
              & sweep.true_gap.isin(match["gaps"])
              & sweep.n_readers.isin(match["readers"])
              & sweep.n_cases.isin(match["cases"])].copy()
    m["diff"] = m.power_paired - m.power_absolute_naive

    fig = plt.figure(figsize=(5.5, 7.0))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.05, 1.15], hspace=.95, wspace=.45)

    ax = fig.add_subplot(gs[0, :])
    nul = sweep[sweep.arm == "null_arm"]
    xs = np.arange(len(levels))
    for i, (name, sty) in enumerate(SIM_STYLE.items()):
        ys, es = [], []
        for _, cl in levels:
            s = nul[nul.ceiling_level == cl]
            ys.append(s[f"power_{name}"].mean())
            es.append(1.96 * np.sqrt(.05 * .95 / int(s.n_sims.iloc[0])))
        ax.errorbar(xs + (i - 1) * .17, ys, yerr=es, color=sty["c"], marker=sty["m"],
                    ls="none", ms=4.5, lw=1.2, capsize=2.5, mfc="white", mew=1.0)
    ax.axhline(.05, color="black", lw=1.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{lab}\n{cl:.2f}" for lab, cl in levels], fontsize=TICK)
    ax.set_ylabel("type I error rate", fontsize=LAB)
    ax.set_ylim(.032, .070); ax.set_xlim(-.5, len(levels) - .5)
    ax.tick_params(labelsize=TICK)

    d = sweep[(sweep.ceiling_level == anchor) & (sweep.n_cases == 44)]
    gaps = [g for g in sorted(d.true_gap.unique()) if g > 0][:4]
    for j, g in enumerate(gaps):
        ax = fig.add_subplot(gs[1, j])
        s = d[d.true_gap == g].sort_values("n_readers")
        for _, r in s[~s.informative.astype(bool)].iterrows():
            ax.axvspan(r.n_readers * .78, r.n_readers * 1.28, color="0.6",
                       alpha=.30, lw=0, zorder=0)
        for name, sty in SIM_STYLE.items():
            ax.plot(s.n_readers, s[f"power_{name}"], color=sty["c"], marker=sty["m"],
                    ls=sty["ls"], ms=3.0, lw=1.1, mfc="white", mew=.8)
        ax.axvline(4, color="black", lw=.9, ls="-.", alpha=.75)
        ax.axhline(1.0, color="0.35", lw=.7, ls=(0, (1, 2)))
        ax.set_xscale("log")
        t = sorted(s.n_readers.unique())
        ax.set_xticks(t); ax.set_xticklabels([int(v) for v in t], fontsize=TICK - 2)
        ax.minorticks_off(); ax.set_ylim(-.04, 1.08)
        ax.tick_params(labelsize=TICK - 1)
        if j:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("power", fontsize=LAB)
            ax.annotate("readers", xy=(2.35, -.30), xycoords="axes fraction",
                        ha="center", fontsize=LAB - 1)
        ax.text(.5, 1.04, f"gap {g:g}", transform=ax.transAxes, ha="center",
                fontsize=TICK - 1)

    ax = fig.add_subplot(gs[2, :])
    for g, mk in GAPM.items():
        q = m[m.true_gap == g]
        ax.scatter(q.realised_pct_max, q["diff"], marker=mk, s=20, facecolors="none",
                   edgecolors="#0072B2", linewidths=.8, zorder=2)
    pn = summary[summary.contrast == "paired - naive"]
    for lab, cl in levels:
        s, row = m[m.ceiling_level == cl], pn[pn.ceiling_level == cl]
        if not len(s) or not len(row):
            continue
        row = row.iloc[0]
        lo, hi = s.realised_pct_max.min(), s.realised_pct_max.max()
        pad = max((hi - lo) * .2, 1.6)
        ax.plot([lo - pad, hi + pad], [row.raw_mean] * 2, color="#D55E00", lw=2.2,
                ls="--", zorder=4)
        ax.errorbar([s.realised_pct_max.mean()], [row.adj_mean],
                    yerr=[[row.adj_mean - row.adj_ci_lo],
                          [row.adj_ci_hi - row.adj_mean]],
                    color="black", marker="D", ms=5, lw=1.4, capsize=3,
                    mfc="white", mew=1.2, zorder=5)
        ax.annotate(f"raw {row.raw_mean:+.3f}\nadj {row.adj_mean:+.3f}\n"
                    f"({int(row.n_cells)} cells)", xy=(hi + pad, row.raw_mean),
                    xytext=(3, 0), textcoords="offset points", fontsize=TICK - 2.5,
                    va="center", color="#333333")
    ax.axhline(0, color="black", lw=1.8, zorder=3)
    ax.axvline(73.1, color="black", lw=1.0, ls="-.", zorder=1)
    ax.text(73.1 - 1.6, ax.get_ylim()[0] * .92, "real study 73.1%", fontsize=TICK - 2,
            rotation=90, va="bottom")
    ax.set_xlabel("realised % of judgements at scale maximum", fontsize=LAB)
    ax.set_ylabel("power difference\n(paired \u2212 naive)", fontsize=LAB - .5)
    ax.set_xlim(-4, 100); ax.tick_params(labelsize=TICK)

    h = [Line2D([], [], color=v["c"], marker=v["m"], ls=v["ls"], mfc="white",
                label=v["lab"]) for v in SIM_STYLE.values()]
    h.append(plt.Rectangle((0, 0), 1, 1, fc="0.6", alpha=.30,
                           label="saturated: uninformative"))
    fig.legend(handles=h, fontsize=6.4, ncol=2, loc="lower center",
               bbox_to_anchor=(.5, .295), frameon=False)
    h2 = [Line2D([], [], ls="none", marker=mk, mfc="none", mec="#0072B2",
                 label=f"gap {g:g}") for g, mk in GAPM.items()]
    h2 += [Line2D([], [], color="#D55E00", ls="--", label="level mean (raw)"),
           Line2D([], [], color="black", marker="D", mfc="white", ls="none",
                  label="level mean (size-adjusted)")]
    fig.legend(handles=h2, fontsize=6.4, ncol=3, loc="lower center",
               bbox_to_anchor=(.5, -.055), frameon=False)
    for a, l in zip([fig.axes[0], fig.axes[1], fig.axes[-1]], "abc"):
        a.text(.012, .97, l, transform=a.transAxes, fontsize=11, fontweight="bold",
               va="top")
    _save("fig3_simulation")
