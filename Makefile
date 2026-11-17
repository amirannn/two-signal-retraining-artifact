# Thin wrapper around reproduce.py, which holds the commands.
# `make` is optional: `python reproduce.py <target>` does the same thing.

PY        ?= python3
ARTIFACTS ?= artifacts

.PHONY: all setup check-data tables sweep figure verify clean

all: ; $(PY) reproduce.py all
setup:
	$(PY) -m venv .venv
	./.venv/bin/python -m pip install --upgrade pip
	./.venv/bin/python -m pip install -r requirements.txt
	./.venv/bin/python -m pip install -e .
check-data: ; $(PY) reproduce.py check-data
tables:     ; $(PY) reproduce.py tables
sweep:      ; $(PY) reproduce.py sweep
figure:     ; $(PY) reproduce.py figure
verify:     ; $(PY) reproduce.py verify
clean:      ; rm -rf results figures/*.png
