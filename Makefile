PY := python3
SRC := src
DATASET ?= synthetic
WORKERS ?= 4

.PHONY: all synthetic ratings paper sim verify clean
all: synthetic ratings paper verify

synthetic:
	$(PY) $(SRC)/make_synthetic.py

ratings:
	DATASET=$(DATASET) $(PY) $(SRC)/ingest.py

paper:
	DATASET=$(DATASET) $(PY) $(SRC)/paper.py

# full sweep: 52 core-hours. --quick is a 12-cell smoke subset.
sim:
	$(PY) $(SRC)/simulate.py --workers $(WORKERS)

sim-quick:
	$(PY) $(SRC)/simulate.py --grid extension --quick --workers $(WORKERS)

paper-sim:
	DATASET=$(DATASET) $(PY) $(SRC)/paper.py --sim

verify:
	DATASET=$(DATASET) $(PY) $(SRC)/verify.py

clean:
	rm -rf outputs/tables/* outputs/figures/* results.md
