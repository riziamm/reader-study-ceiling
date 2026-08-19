"""Wide MS Forms export + answer key -> tidy ratings.

Skip this if data/<dataset>/ratings_tidy.csv is already present; the released
ratings are supplied in that form.

Inputs in data/<dataset>/:
    results.csv     wide export, one row per reader
    answer_key.csv  case_filename, displayed_as_A, displayed_as_B
    model_key.csv   raw_token, label   (label is Model A or Model B)

The answer key joins to the export by row order: the export identifies cases
only by column suffix, and the key has no case-number column. Row count is
asserted against the design because a mismatch would silently mislabel every
model.
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from utils import data_dir, load_config, parse_score, save


def blocks(columns, rx, ):
    """{case_num: {(position, criterion): column}}"""
    out = {}
    for col in columns:
        m = re.match(rx, str(col).replace("\u00a0", " ").strip())
        if m:
            g = m.groupdict()
            cn = int(g["case_suffix"]) if g["case_suffix"] else 1
            crit = re.sub(r"\s+", " ", g["criterion"]).strip()
            out.setdefault(cn, {})[(g["position"], crit)] = col
    return out


def main():
    cfg = load_config()
    st, d = cfg["study"], data_dir(cfg)
    lo, hi = st["scale"]

    raw = pd.read_csv(d / "results.csv", dtype=str, keep_default_na=False,
                      encoding="utf-8-sig")
    pii = [c for c in st["drop_columns"] if c in raw.columns]
    print(f"{raw.shape[0]} readers x {raw.shape[1]} columns; dropping {pii}")

    blk = blocks(raw.columns, st["rating_regex"])
    want = {(c, p, k) for c in range(1, st["n_cases"] + 1) for p in "AB"
            for k in st["criteria"]}
    have = {(c, p, k) for c, v in blk.items() for (p, k) in v}
    absent = sorted(want - have)
    known = {(a["case_num"], a["position"], a["criterion"]) for a in st["known_absent"]}
    for a in absent:
        print(f"  column absent: case {a[0]:>3} {a[1]} {a[2]}"
              f"{'  (known form fault)' if a in known else '  UNEXPECTED'}")
    if set(absent) - known:
        sys.exit("unexpected absent columns; stopping rather than guessing")

    key = pd.read_csv(d / "answer_key.csv", encoding="utf-8-sig")
    key = key.loc[:, [c for c in key.columns if str(c).strip()]]
    key = key.dropna(subset=["case_filename"]).reset_index(drop=True)
    if len(key) != st["n_cases"]:
        sys.exit(f"answer key has {len(key)} rows, design says {st['n_cases']}; "
                 "the join is by row order so a mismatch mislabels every model")
    key["case_num"] = np.arange(1, len(key) + 1)

    mk = pd.read_csv(d / "model_key.csv")
    token = dict(zip(mk.raw_token, mk.label))
    if sorted(token.values()) != ["Model A", "Model B"]:
        sys.exit("model_key.csv labels must be exactly Model A and Model B")
    seen = set(key.displayed_as_A) | set(key.displayed_as_B)
    if seen != set(token):
        sys.exit(f"answer-key tokens {sorted(seen)} != key file {sorted(token)}")

    at = {}
    for _, r in key.iterrows():
        at[(r.case_num, "A")] = token[r.displayed_as_A]
        at[(r.case_num, "B")] = token[r.displayed_as_B]

    rows, bad = [], 0
    for _, row in raw.iterrows():
        reader = int(row[st["reader_id_column"]] if "reader_id_column" in st
                     else row.iloc[0])
        for cn, cells in blk.items():
            for (pos, crit), col in cells.items():
                s = parse_score(row[col], lo, hi)
                bad += int(s != s and str(row[col]).strip() != "")
                rows.append(dict(reader=reader, case_num=cn, criterion=crit,
                                 position=pos, model=at[(cn, pos)], score=s))
    long = pd.DataFrame(rows)

    scheduled = st["n_readers"] * st["n_cases"] * len(st["criteria"]) * 2
    expected = scheduled - st["n_readers"] * len(known)
    valid = int(long.score.notna().sum())
    npairs = int(long.pivot_table(index=["reader", "case_num", "criterion"],
                                  columns="model", values="score",
                                  aggfunc="first").dropna().shape[0])
    print(f"scheduled {scheduled}, minus {st['n_readers'] * len(known)} absent "
          f"= {expected} expected; {len(long)} present, {valid} valid, "
          f"{bad} unparseable")
    print(f"complete within-case pairs: {npairs}")
    if len(long) != expected:
        sys.exit("counts do not reconcile to the design")

    long.to_csv(d / "ratings_tidy.csv", index=False)
    print(f"wrote {(d / 'ratings_tidy.csv')}")
    save(pd.DataFrame([dict(scheduled=scheduled, expected=expected,
                            present=len(long), valid=valid, pairs=npairs,
                            unparseable=bad)]), "counts")


if __name__ == "__main__":
    main()
