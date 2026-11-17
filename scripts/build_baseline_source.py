"""Aggregate the baseline-matrix runs into the single source of truth.

Reads ``baseline_summary.json`` from the KronoDroid, LAMDA and McNdroid
baseline-matrix runs and writes ``paper/baseline_source.json``: per-scenario
rows plus per-policy aggregates (median/mean savings, mean and worst accuracy
gap, number of settings inside the pre-specified 2% margin, number certified
by the one-sided bootstrap bound).  The paper's baseline table and prose read
this file, so no baseline number is ever typed by hand.

Example:
    python scripts/build_baseline_source.py \
      --kronodroid results/baseline_matrix_kronodroid \
      --lamda results/baseline_matrix_lamda \
      --mcndroid results/baseline_matrix_mcndroid \
      --json-out ../paper/baseline_source.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Policy order and human labels used by the paper.
POLICY_LABELS = {
    "two_signal": "Two-Signal (proposed)",
    "drift_aware": "Silhouette-only",
    "psi_only": "PSI-only",
    "ks_only": "KS-only",
    "adwin_only": "ADWIN",
    "kswin_only": "KSWIN",
    "static": "Static (no retraining)",
    "periodic": "Periodic (retrain always)",
}
# Rows of the paper's baseline table, in presentation order.
POLICY_ORDER = [
    "two_signal",
    "drift_aware",
    "psi_only",
    "ks_only",
    "adwin_only",
    "kswin_only",
]
# Aggregated as well, but not a row of the table: the operational reference
# points, kept so the paper can quote the static baseline without retyping it.
REFERENCE_POLICIES = ["static"]
AGGREGATED_POLICIES = POLICY_ORDER + REFERENCE_POLICIES


def _load(run_dirs: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for run in run_dirs:
        path = run / "baseline_summary.json"
        if not path.exists():
            raise SystemExit(f"missing baseline summary: {path}")
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit("no baseline rows loaded")
    return rows


def _aggregate(rows: list[dict], policy: str) -> dict:
    subset = [r for r in rows if r["policy"] == policy]
    if not subset:
        return {}
    savings = np.asarray([r["savings_pct"] for r in subset], dtype=float)
    gaps = np.asarray([r["gap_pct"] for r in subset], dtype=float)
    bounds = np.asarray([r["ci_lower_pct"] for r in subset], dtype=float)
    worst = subset[int(np.argmin(gaps))]
    return {
        "policy": policy,
        "label": POLICY_LABELS.get(policy, policy),
        "n_scenarios": len(subset),
        "median_savings_pct": round(float(np.median(savings)), 1),
        "mean_savings_pct": round(float(savings.mean()), 1),
        "min_savings_pct": round(float(savings.min()), 1),
        "max_savings_pct": round(float(savings.max()), 1),
        "median_gap_pct": round(float(np.median(gaps)), 2),
        "mean_gap_pct": round(float(gaps.mean()), 2),
        "worst_gap_pct": round(float(gaps.min()), 2),
        "worst_gap_scenario": worst["family"],
        "n_in_margin": int(sum(1 for r in subset if r["in_margin_point"])),
        "n_certified": int(sum(1 for r in subset if r["certified"])),
        "n_above_bound": int(sum(1 for r in subset if r["ci_lower_pct"] > -2.0)),
        "total_retrains": int(sum(r["retrains"] for r in subset)),
        "median_ci_lower_pct": round(float(np.median(bounds)), 2),
    }


def _per_dataset(rows: list[dict], policy: str) -> dict:
    out: dict[str, dict] = {}
    for dataset in ("kronodroid", "lamda", "mcndroid"):
        subset = [r for r in rows if r["policy"] == policy and r["dataset"] == dataset]
        if not subset:
            continue
        out[dataset] = {
            "n_scenarios": len(subset),
            "median_savings_pct": round(float(np.median([r["savings_pct"] for r in subset])), 1),
            "mean_gap_pct": round(float(np.mean([r["gap_pct"] for r in subset])), 2),
            "n_in_margin": int(sum(1 for r in subset if r["in_margin_point"])),
            "n_certified": int(sum(1 for r in subset if r["certified"])),
        }
    return out


def _check_against_table1(rows: list[dict], table1_path: Path) -> list[str]:
    """Verify the baseline run reproduces Table I for the proposed policy.

    The baseline matrix is only comparable to Table I if the run that produced
    it re-created the published proposed-policy numbers exactly (same protocol,
    same calibration).  Any mismatch is reported as an error rather than
    silently written into the artefact.
    """
    if not table1_path.exists():
        return [f"table1 source not found at {table1_path}; skipped cross-check"]
    table1 = json.loads(table1_path.read_text(encoding="utf-8"))["scenarios"]
    proposed = {
        (r["dataset"], r["family"].lower()): r for r in rows if r["policy"] == "two_signal"
    }
    problems: list[str] = []
    checked = 0
    for scenario in table1:
        key = (scenario["dataset"], scenario["family"].lower())
        row = proposed.get(key)
        if row is None:
            problems.append(f"scenario {key} missing from the baseline matrix")
            continue
        checked += 1
        if row["retrains"] != scenario["proposed_retrains"]:
            problems.append(
                f"{key}: retrains {row['retrains']} != Table I {scenario['proposed_retrains']}"
            )
        if abs(row["gap_pct"] - scenario["gap_pct"]) > 0.01:
            problems.append(
                f"{key}: gap {row['gap_pct']:+.2f} != Table I {scenario['gap_pct']:+.2f}"
            )
    print(f"cross-check against Table I: {checked} scenarios compared")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build paper/baseline_source.json")
    parser.add_argument(
        "--table1",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "paper" / "table1_source.json",
        help="Table I artefact used to cross-check the proposed-policy rows",
    )
    parser.add_argument("--kronodroid", type=Path, required=True)
    parser.add_argument("--lamda", type=Path, required=True)
    parser.add_argument("--mcndroid", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=Path("../paper/baseline_source.json"))
    args = parser.parse_args(argv)

    rows = _load([args.kronodroid, args.lamda, args.mcndroid])
    problems = _check_against_table1(rows, args.table1)
    if problems:
        for problem in problems:
            print(f"MISMATCH: {problem}")
        raise SystemExit(
            "baseline matrix does not reproduce Table I; refusing to write the artefact"
        )
    policies = [p for p in AGGREGATED_POLICIES if any(r["policy"] == p for r in rows)]
    table_policies = [p for p in policies if p in POLICY_ORDER]
    aggregate = {p: _aggregate(rows, p) for p in policies}
    per_dataset = {p: _per_dataset(rows, p) for p in policies}

    order = {name: index for index, name in enumerate(POLICY_ORDER)}
    payload = {
        "policies": aggregate,
        "per_dataset": per_dataset,
        "scenarios": sorted(
            rows,
            key=lambda r: (
                r["dataset"],
                r["family"],
                order.get(r["policy"], len(order)),
            ),
        ),
        "provenance": {
            "runs": [
                str(args.kronodroid).replace("\\", "/"),
                str(args.lamda).replace("\\", "/"),
                str(args.mcndroid).replace("\\", "/"),
            ],
            "unit": "pooled paired evaluation samples, 95% one-sided percentile",
            "margin_primary": 0.02,
            "script": "scripts/run_baseline_matrix.py",
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for policy in table_policies:
        agg = aggregate[policy]
        print(
            f"{agg['label']:<26} median savings {agg['median_savings_pct']:5.1f}%  "
            f"mean gap {agg['mean_gap_pct']:+6.2f}  worst {agg['worst_gap_pct']:+7.2f}  "
            f"in-margin {agg['n_in_margin']:2d}/{agg['n_scenarios']}  certified {agg['n_certified']:2d}"
        )
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
