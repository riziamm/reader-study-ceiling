# How Reliability Coefficients Fail Reader Studies at the Fidelity Ceiling

Code for the paper. Four radiologists rated two inpainting models on 44 cases
and four criteria; Judgements fell at the scale maximum, and
chance-corrected agreement went negative in six of eight strata. 
This reproduces every table and figure.

## Run it

```bash
pip install -r requirements.txt
make all                  # synthetic data, ratings analyses, tables, figures, verify
```

Roughly two minutes. `make all` ends with `PASS` if `results.md` agrees with
every table.

The simulation is separate as it needs compute:

```bash
make sim-quick            # 12-cell smoke subset, a few minutes
make sim WORKERS=32       # full grid, 162 cells, ~53 core-hours
make paper-sim            # adds Tables B1-B3, D1 and Figure 3
```

The sweep checkpoints per cell and per 25 replicates, so it resumes if
interrupted. Per-cell seeds derive from the cell index, so results are
identical regardless of worker count.

## Real data

`data/synthetic/` is generated and saturates like the real study, but is not
the real study. To run on the ratings:

```bash
# data/real/ratings_tidy.csv        released ratings, or
# data/real/{results.csv, answer_key.csv, model_key.csv}  raw export
DATASET=real make ratings paper verify
```

Reader identifiers in the released ratings are arbitrary labels ordered
independently of any reported quantity. No per-reader value is published
anywhere; `verify.py` asserts this.

## Layout

| file | what |
|---|---|
| `src/ingest.py` | wide Forms export + answer key to tidy ratings |
| `src/stats.py` | agreement coefficients, variance decomposition |
| `src/analysis.py` | saturation, coefficients, recoding, leave-one-out, robustness |
| `src/simulate.py` | generator, the three analyses, sweep, size adjustment |
| `src/figures.py` | the eight figures |
| `src/paper.py` | builds every table and figure, writes `results.md` |
| `src/verify.py` | re-derives the headline numbers and asserts they match |

Outputs land in `outputs/tables/` and `outputs/figures/`, and `results.md` is
the single source for numbers quoted in the paper.

## What maps to what

| paper | produced by |
|---|---|
| Table 1, Table 2, Table C1 | `paper.py` |
| Figure 1 paired recoding | `figures.fig1_paired` |
| Figure 2 two views | `figures.fig2_absolute_within` |
| Figure 3 simulation | `figures.fig3_simulation` (needs `make sim`) |
| Figure C1 coefficients by stratum | `figures.figC1_coefficients` |
| Figures F1-F4 | `figures.figF1_distribution` … `figF4_bland_altman` |
| Tables B1-B3, B2 critical values, D1 | `paper.py --sim` |
| Appendix D robustness | `analysis.locality`, `analysis.robustness`, `analysis.loro` |

## Notes that matter for reading the numbers

- **Reader is a fixed effect.** Four levels cannot identify a variance
  component, and the random-effect reading is the generalisation this design
  cannot support. Case carries the random intercept, and every bootstrap
  interval is clustered on the case.
- **Eight ratings are missing** because the form omitted one item for cases 22
  and 44: 1408 scheduled, 1400 valid, 696 complete within-case pairs. They are
  dropped, never imputed.
- **Leave-one-reader-out refits are ordered by their resulting value**, not by
  reader index. With four readers, a labelled row plus the full-sample value
  identifies that reader by subtraction.
- **`criterion_type` in `config.yaml` is a pre-declared classification**, fixed
  before results existed. The masks were never shown to readers, so the
  locality check confounds spatial locality with breadth of construct; it is
  reported because it was pre-declared, and not interpreted.
- **The generator's `CASE_SD = 0.18` is frozen.** It was fitted jointly with the
  ceiling levels against two measured properties of the ratings, the share at
  the scale maximum and the between-case variance share, before any power
  result existed.
- **The sweep's paired analysis discards ties**, whereas the study's own paired
  analysis uses the full ordinal difference. Ties cannot be counted as losses
  instead: with ties as losses P(A wins) is far from 0.5 under the null, so the
  tested hypothesis is false by construction.
