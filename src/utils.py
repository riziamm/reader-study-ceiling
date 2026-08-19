"""Config, score parsing, cluster bootstrap, table IO."""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"


def load_config(path=None):
    cfg = yaml.safe_load(open(path or ROOT / "config.yaml"))
    cfg["dataset"] = os.environ.get("DATASET", cfg["dataset"])
    if cfg["dataset"] not in ("synthetic", "real"):
        raise ValueError(cfg["dataset"])
    return cfg


def data_dir(cfg):
    return ROOT / "data" / cfg["dataset"]


_SCORE = re.compile(r"^\s*([1-9]\d*)\s*(?:[-–—:]\s*.*)?$")


def parse_score(raw, lo=1, hi=5):
    """Accepts 5, "5", "5 - good", "1 - Very Poor". Anything else is missing.
    Out-of-range values are never coerced into range."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan
    if isinstance(raw, (int, np.integer)):
        return float(raw) if lo <= int(raw) <= hi else np.nan
    s = str(raw).strip()
    if s == "" or s.lower() in {"nan", "na", "n/a", "none", "-"}:
        return np.nan
    m = _SCORE.match(s)
    if not m:
        return np.nan
    v = int(m.group(1))
    return float(v) if lo <= v <= hi else np.nan


def cluster_bootstrap(df, cluster, stat, n_boot=5000, ci=0.95, seed=0):
    """Resample whole clusters with replacement. Returns (point, lo, hi, boots)."""
    rng = np.random.default_rng(seed)
    keys = df[cluster].dropna().unique()
    if len(keys) < 2:
        return np.nan, np.nan, np.nan, np.array([])
    groups = {k: g for k, g in df.groupby(cluster, observed=True)}
    point = stat(df)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        try:
            boots.append(stat(pd.concat([groups[k] for k in pick], ignore_index=True)))
        except Exception:
            pass
    boots = np.asarray([b for b in boots if b == b], float)
    if boots.size < 10:
        return point, np.nan, np.nan, boots
    lo, hi = np.percentile(boots, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return point, lo, hi, boots


def save(df, name, index=False):
    """Write an aggregated table as csv plus a markdown preview."""
    TABLES.mkdir(parents=True, exist_ok=True)
    cols = {c.lower() for c in df.columns}
    if len(df) > 200 and len({"score", "reader", "case_num"} & cols) >= 3:
        raise RuntimeError(f"{name} looks like a per-row judgement dump")
    p = TABLES / f"{name}.csv"
    df.to_csv(p, index=index)
    (TABLES / f"{name}.md").write_text(df.to_markdown(index=index) + "\n")
    print(f"  wrote {p.relative_to(ROOT)}")
    return p


def tidy(cfg):
    """Long ratings: reader, case_num, criterion, position, model, score."""
    p = data_dir(cfg) / "ratings_tidy.csv"
    if not p.exists():
        raise SystemExit(f"{p} not found. Run src/ingest.py first.")
    d = pd.read_csv(p)
    return d.dropna(subset=["score"])


def pairs(cfg):
    """One row per reader x case x criterion with both models present.
    Rows missing either side are dropped, never imputed."""
    w = (tidy(cfg).pivot_table(index=["reader", "case_num", "criterion"],
                               columns="model", values="score", aggfunc="first")
         .reset_index())
    w.columns.name = None
    w = w.dropna(subset=["Model A", "Model B"]).copy()
    w["delta"] = w["Model A"] - w["Model B"]
    w["mean_score"] = (w["Model A"] + w["Model B"]) / 2
    return w
