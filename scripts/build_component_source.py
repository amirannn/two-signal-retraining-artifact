"""Aggregate the component-ablation runs into one artefact for the paper.

Reads the five-policy baseline-matrix runs (static, periodic, silhouette-only,
PSI-only, proposed) executed with the classifier and draw of the reported cell,
cross-checks the proposed rows against ``paper/table1_source.json``, and writes
``paper/baseline_components.json``.

Example:
    python scripts/build_component_source.py \
      --kronodroid results/baseline_mlp_s43_kronodroid \
      --lamda results/baseline_mlp_s43_lamda \
      --json-out ../paper/baseline_components.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

POLICIES = ("two_signal", "drift_aware", "psi_only", "static")
LABELS = {
    "static": "Static (no retraining)",
    "periodic": "Periodic (every window)",
    "psi_only": "PSI-only (screen alone)",
    "drift_aware": "Silhouette-only",
    "two_signal": "Two-signal (proposed)",
}


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
    worst = subset[int(np.argmin(gaps))]
    return {
        "policy": policy,
        "label": LABELS[policy],
        "n_scenarios": len(subset),
        "total_retrains": int(sum(r["retrains"] for r in subset)),
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
    }


def _check_against_table1(rows: list[dict], table1_path: Path) -> list[str]:
    table1 = json.loads(table1_path.read_text(encoding="utf-8"))["scenarios"]
    proposed = {
        (r["dataset"], r["family"].lower()): r
        for r in rows
        if r["policy"] == "two_signal"
    }
    problems: list[str] = []
    for scenario in table1:
        key = (scenario["dataset"], scenario["family"].lower())
        row = proposed.get(key)
        if row is None:
            problems.append(f"scenario {key} missing from the ablation run")
            continue
        if row["retrains"] != scenario["proposed_retrains"]:
            problems.append(
                f"{key}: retrains {row['retrains']} != Table I {scenario['proposed_retrains']}"
            )
        if abs(row["gap_pct"] - scenario["gap_pct"]) > 0.01:
            problems.append(
                f"{key}: gap {row['gap_pct']:+.2f} != Table I {scenario['gap_pct']:+.2f}"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build paper/baseline_components.json")
    parser.add_argument("--kronodroid", type=Path, required=True)
    parser.add_argument("--lamda", type=Path, required=True)
    parser.add_argument(
        "--table1",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "paper" / "table1_source.json",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "paper" / "baseline_components.json",
    )
    args = parser.parse_args(argv)

    rows = _load([args.kronodroid, args.lamda])
    problems = _check_against_table1(rows, args.table1)
    if problems:
        for problem in problems:
            print(f"MISMATCH: {problem}")
        raise SystemExit(
            "ablation run does not reproduce Table I; refusing to write the artefact"
        )

    scenarios = sorted(
        (r for r in rows if r["policy"] in POLICIES),
        key=lambda r: (r["dataset"], r["family"], POLICIES.index(r["policy"])),
    )
    n_streams = len({(r["dataset"], r["family"]) for r in scenarios})
    periodic_retrains = int(sum(r["n_windows"] for r in scenarios if r["policy"] == "static"))
    policies = {p: _aggregate(rows, p) for p in POLICIES}
    policies["periodic"] = {
        "policy": "periodic",
        "label": LABELS["periodic"],
        "n_scenarios": n_streams,
        "total_retrains": periodic_retrains,
        "median_savings_pct": 0.0,
        "mean_savings_pct": 0.0,
        "min_savings_pct": 0.0,
        "max_savings_pct": 0.0,
        "median_gap_pct": 0.0,
        "mean_gap_pct": 0.0,
        "worst_gap_pct": 0.0,
        "worst_gap_scenario": "—",
        "n_in_margin": n_streams,
        "n_certified": n_streams,
    }

    # Paired, per-stream comparison of the proposed policy against each rival,
    # so the prose can state counts of streams instead of only aggregates.
    index = {(r["dataset"], r["family"], r["policy"]): r for r in scenarios}
    streams = sorted({(r["dataset"], r["family"]) for r in scenarios})
    paired: dict[str, dict] = {}
    for rival in ("psi_only", "drift_aware", "static"):
        fewer = sum(
            1
            for s in streams
            if index[(*s, "two_signal")]["retrains"] < index[(*s, rival)]["retrains"]
        )
        better_gap = sum(
            1
            for s in streams
            if index[(*s, "two_signal")]["gap_pct"] > index[(*s, rival)]["gap_pct"]
        )
        paired[rival] = {"fewer_retrains_streams": fewer, "better_gap_streams": better_gap}

    payload = {
        "policies": policies,
        "paired": paired,
        "n_streams": n_streams,
        "scenarios": scenarios,
        "provenance": {
            "runs": [
                str(args.kronodroid).replace("\\", "/"),
                str(args.lamda).replace("\\", "/"),
            ],
            "unit": "pooled paired evaluation samples, 95% one-sided percentile",
            "margin_primary": 0.02,
            "cross_check": "two_signal reproduces Table I exactly",
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("static", "periodic", "psi_only", "drift_aware", "two_signal"):
        agg = policies[name]
        print(
            f"{agg['label']:<28} ops {agg['total_retrains']:4d}  "
            f"median savings {agg['median_savings_pct']:5.1f}%  mean gap {agg['mean_gap_pct']:+6.2f}  "
            f"in-margin {agg['n_in_margin']:2d}/{n_streams}  certified {agg['n_certified']:2d}"
        )
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
