"""Driver for the reproducibility targets (no dependencies beyond the stdlib).

Every command in the paper's reproduction story is defined here once; the
Makefile is a thin wrapper around the same targets.

    python reproduce.py check-data
    python reproduce.py tables     # Table I and Table II
    python reproduce.py sweep      # Section V-C reference-period sweep
    python reproduce.py figure     # Fig. 1
    python reproduce.py verify     # data-free check of the shipped artefacts
    python reproduce.py all

By default the interpreter is the project virtual environment when it exists
(``.venv\Scripts\python.exe`` on Windows, ``.venv/bin/python`` elsewhere),
otherwise the interpreter running this script.  Override with ``--python``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

KRONO = os.environ.get("KRONO", "data/KronoDroid_real_device.csv")
LAMDA = os.environ.get("LAMDA", "data/lamda")
RESULTS = os.environ.get("RESULTS", "results")
ARTIFACTS = os.environ.get("ARTIFACTS", "artifacts")
SEED = os.environ.get("SEED", "43")
BOOTSTRAP = os.environ.get("BOOTSTRAP", "42")
POLICIES = ["static", "periodic", "drift_aware", "psi_only", "two_signal"]


def _matrix(py: str, dataset: str, out: str, extra: list[str]) -> list[str]:
    data_flag = "--kronodroid-path" if dataset == "kronodroid" else "--lamda-path"
    data_path = KRONO if dataset == "kronodroid" else LAMDA
    return [
        py, "scripts/run_baseline_matrix.py",
        "--dataset", dataset, data_flag, data_path,
        "--seed", SEED, "--bootstrap-seed", BOOTSTRAP,
        *extra, "--out", out,
    ]


def targets(py: str) -> dict[str, list[list[str]]]:
    k, l = f"{RESULTS}/tables/krono", f"{RESULTS}/tables/lamda"
    s3k, s3l = f"{RESULTS}/sweep/rp3_krono", f"{RESULTS}/sweep/rp3_lamda"
    s10k, s10l = f"{RESULTS}/sweep/rp10_krono", f"{RESULTS}/sweep/rp10_lamda"
    return {
        "tables": [
            _matrix(py, "kronodroid", k, ["--model", "mlp", "--policies", *POLICIES]),
            _matrix(py, "lamda", l, ["--model", "mlp", "--policies", *POLICIES]),
            [py, "scripts/build_table1_source.py",
             "--kronodroid", k, "--lamda", l,
             "--json-out", f"{ARTIFACTS}/table1_source.json"],
            [py, "scripts/build_component_source.py",
             "--kronodroid", k, "--lamda", l,
             "--table1", f"{ARTIFACTS}/table1_source.json",
             "--json-out", f"{ARTIFACTS}/baseline_components.json"],
        ],
        "sweep": [
            _matrix(py, "kronodroid", s3k, ["--model", "mlp", "--reference-period", "3", "--policies", "static", "periodic", "two_signal"]),
            _matrix(py, "lamda", s3l, ["--model", "mlp", "--reference-period", "3", "--policies", "static", "periodic", "two_signal"]),
            _matrix(py, "kronodroid", s10k, ["--model", "mlp", "--reference-period", "10", "--policies", "static", "periodic", "two_signal"]),
            _matrix(py, "lamda", s10l, ["--model", "mlp", "--reference-period", "10", "--policies", "static", "periodic", "two_signal"]),
            [py, "scripts/build_reference_sweep.py",
             "--rp3", s3k, s3l, "--rp5", k, l, "--rp10", s10k, s10l,
             "--table1", f"{ARTIFACTS}/table1_source.json",
             "--json-out", f"{ARTIFACTS}/reference_period_mlp43.json"],
        ],
        "figure": [[py, "scripts/generate_algorithm_flowchart.py"]],
        "verify": [[py, "scripts/verify_numbers.py", "--artifacts", ARTIFACTS]],
    }


DATA_EXPECTED = Path("data/expected.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _data_path(rel: str) -> Path:
    """Map a path in data/expected.json onto the KRONO/LAMDA locations."""
    if rel.startswith("lamda/"):
        return Path(LAMDA) / rel[len("lamda/"):]
    return Path(KRONO).parent / rel


def check_data() -> None:
    if not DATA_EXPECTED.exists():
        sys.exit(f"{DATA_EXPECTED} is missing; it ships with the artifact")
    spec = json.loads(DATA_EXPECTED.read_text(encoding="utf-8"))
    rows, failures = [], []
    for entry in spec["files"]:
        rel, want_bytes, want_hash = entry["path"], entry["bytes"], entry["sha256"]
        required = entry.get("required", True)
        path = _data_path(rel)
        if not path.exists():
            rows.append((rel, "-", "MISSING" if required else "optional, absent"))
            if required:
                failures.append(f"{rel}: missing (expected {want_bytes:,} bytes)")
            continue
        size = path.stat().st_size
        digest = _sha256(path)
        if size != want_bytes or digest != want_hash:
            rows.append((rel, f"{size:,}", "MISMATCH"))
            failures.append(f"{rel}: sha256 {digest} ({size:,} bytes); "
                            f"expected {want_hash} ({want_bytes:,} bytes)")
        else:
            rows.append((rel, f"{size:,}", "OK"))

    width = max(len(r[0]) for r in rows)
    print(f"{'file'.ljust(width)}  {'bytes':>14}  status")
    for rel, size, status in rows:
        print(f"{rel.ljust(width)}  {size:>14}  {status}")

    if failures:
        print("\ndata check failed:")
        for line in failures:
            print("  -", line)
        print("\nfetch the files as described in data/README.md, then re-run "
              "`python reproduce.py check-data`")
        sys.exit(1)
    print(f"\n{sum(1 for r in rows if r[2] == 'OK')} file(s) verified "
          "against data/expected.json")


def run(py: str, name: str) -> None:
    if name in {"tables", "sweep"}:
        check_data()
    for cmd in targets(py)[name]:
        print("\n$ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", choices=["check-data", "tables", "sweep", "figure", "verify", "all"])
    parser.add_argument("--python", default=None, help="interpreter to use (default: .venv or current)")
    args = parser.parse_args(argv)

    if args.target == "check-data":
        check_data()
        print("data OK")
        return 0

    venv_py = [Path(".venv/Scripts/python.exe"), Path(".venv/bin/python")]
    py = args.python or next((str(p) for p in venv_py if p.exists()), sys.executable)
    order = ["tables", "sweep", "figure", "verify"] if args.target == "all" else [args.target]
    if args.target == "sweep":
        # The sweep's five-window anchor is the Table I artefact.
        order = ["tables", "sweep"]
    for name in order:
        print(f"\n=== {name} ===")
        run(py, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
