"""Write a synthetic reader study in the shape of the real export.

Reproduces the real file's structure so that ingest.py exercises the same
path: wide Forms layout, PII columns, the two columns the form omitted, a
mixture of bare integers and "5 - Indistinguishable" text cells, and an
answer key with no case-number column.

The ratings come from the same generator as the simulation, at the anchor
ceiling, so the synthetic study saturates like the real one. It is not the
real study: use it to check the code runs, not to reproduce the numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from simulate import ANCHOR_CEILING, CRIT, simulate_long
from utils import ROOT, load_config

LABELS = {1: "Very Poor", 2: "", 3: "", 4: "", 5: "Indistinguishable"}
TOKENS = ("best_model", "baseline_model")


def cell(v, rng, text_fraction=0.2):
    if v != v:
        return ""
    v = int(v)
    if rng.random() < text_fraction and LABELS[v]:
        return f"{v} - {LABELS[v]}"
    return str(v)


def main():
    cfg = load_config()
    st = cfg["study"]
    rng = np.random.default_rng(cfg["seed"])
    out = ROOT / "data" / "synthetic"
    out.mkdir(parents=True, exist_ok=True)

    long, _ = simulate_long(ANCHOR_CEILING, 0.35, st["n_readers"], st["n_cases"],
                            seed=cfg["seed"])
    # Model A/B in the generator is the ground truth; hide it behind tokens the
    # answer key resolves, as in the real study
    a_is_best = rng.random(st["n_cases"]) < 0.5
    key = pd.DataFrame({
        "case_filename": [f"case_{i+1:03d}.jpg" for i in range(st["n_cases"])],
        "displayed_as_A": np.where(a_is_best, *TOKENS),
        "displayed_as_B": np.where(a_is_best, *TOKENS[::-1]),
    })
    key[""] = ""
    key.to_csv(out / "answer_key.csv", index=False)
    pd.DataFrame({"raw_token": TOKENS, "label": ["Model A", "Model B"]}) \
        .to_csv(out / "model_key.csv", index=False)

    absent = {(a["case_num"], a["position"], a["criterion"]) for a in st["known_absent"]}
    score = {(r.reader, r.case_num, r.criterion, r.model): r.score
             for r in long.itertuples()}

    header = ["ID", "Start time", "Completion time", "Email", "Name", "Full Name",
              "Email Address"]
    for c in range(1, st["n_cases"] + 1):
        sfx = "" if c == 1 else str(c)
        for pos in "AB":
            for crit in CRIT:
                if (c, pos, crit) not in absent:
                    header.append(f"{crit} -- Image {pos}{sfx}")
        header.append(f"Comments{sfx}")

    rows = []
    for r in range(1, st["n_readers"] + 1):
        row = {"ID": r, "Start time": f"09-02-26 9:0{r}",
               "Completion time": f"09-02-26 9:{40 + r}", "Email": "anonymous",
               "Name": "", "Full Name": r, "Email Address": f"{r}@synthetic.invalid"}
        for c in range(1, st["n_cases"] + 1):
            sfx = "" if c == 1 else str(c)
            at = {"A": "Model A" if a_is_best[c - 1] else "Model B",
                  "B": "Model B" if a_is_best[c - 1] else "Model A"}
            for pos in "AB":
                for crit in CRIT:
                    if (c, pos, crit) in absent:
                        continue
                    row[f"{crit} -- Image {pos}{sfx}"] = cell(
                        score.get((r, c, crit, at[pos]), np.nan), rng)
            row[f"Comments{sfx}"] = ""
        rows.append(row)

    pd.DataFrame(rows, columns=header).to_csv(out / "results.csv", index=False)
    print(f"wrote {out}/results.csv  ({len(header)} columns, "
          f"{st['n_readers']} readers)")
    print(f"wrote {out}/answer_key.csv, {out}/model_key.csv")


if __name__ == "__main__":
    main()
