# Fidelity Ceilings Break Reliability Coefficients in Generative AI Evaluation

Code for the paper. Four radiologists rated two inpainting models on 44 cases
and four criteria; Judgements fell at the scale maximum, and
chance-corrected agreement went negative in six of eight strata. 
This reproduces every table and figure.

## Run

```bash
pip install -r requirements.txt
make all                  # synthetic data, ratings analyses, tables, figures, verify
```

Roughly two minutes. `make all` ends with `PASS` if `results.md` agrees with
every table.

The simulation is separate as it needs compute:

```bash
make sim-quick            # a few minutes
make sim WORKERS=32       # full grid, 162 cells, ~53 core-hours
make paper-sim            # Tables B1-B3, D1 and Figure 3
```

The sweep checkpoints per cell and per 25 replicates, so it can be resumed if
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

