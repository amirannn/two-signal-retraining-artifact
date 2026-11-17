"""Build the reference-period sensitivity artefact in the Table I cell.

The original ``reference_period_sweep.json`` was produced on the canonical
Random-Forest pipeline and its five-window anchor (229 retrains) does not match
the reported cell (256).  This script rebuilds the same summary from the
five-policy baseline-matrix runs executed with the classifier and draw of
Table I, for reference periods 3, 5 and 10, and cross-checks the five-window
row against ``paper/table1_source.json``.

Example:
    python scripts/build_reference_sweep.py \
      --rp3 results/sweep_mlp43_rp3_krono results/sweep_mlp43_rp3_lamda \
      --rp5 results/baseline_mlp_s43_kronodroid results/baseline_mlp_s43_lamda \
      --rp10 results/sweep_mlp43_rp10_krono results/sweep_mlp43_rp10_lamda \
      --json-out ../paper/reference_period_mlp43.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

PRE_SPECIFIED = 5


def _load(dirs: list[Path], policy: str) -> list[dict]:
    rows: list[dict] = []
    for d in dirs:
        path = d / "baseline_summary.json"
        if not path.exists():
            raise SystemExit(f"missing baseline summary: {path}")
        rows.extend(r for r in json.loads(path.read_text(encoding="utf-8")) if r["policy"] == policy)
    return rows


def _summarise(rows: list[dict]) -> dict:
    windows = sum(int(r["n_windows"]) for r in rows)
    retrains = sum(int(r["retrains"]) for r in rows)
    return {
        "streams": len(rows),
        "windows": windows,
        "retrains": retrains,
        "savings_pct": 100.0 * (1 - retrains / windows) if windows else 0.0,
        "in_margin": sum(1 for r in rows if r["in_margin_point"]),
        "certified_pooled": sum(1 for r in rows if r["certified"]),
        "median_gap_pct": statistics.median(float(r["gap_pct"]) for r in rows),
        "per_stream": {
            f"{r['dataset']}/{r['family'].lower()}": {
                "gap_pct": round(float(r["gap_pct"]), 2),
                "ci_lower_pct": round(float(r["ci_lower_pct"]), 2),
                "in_margin": bool(r["in_margin_point"]),
                "certified": bool(r["certified"]),
                "retrains": int(r["retrains"]),
            }
            for r in rows
        },
    }


def _check_against_table1(rows: list[dict], table1_path: Path) -> list[str]:
    table1 = json.loads(table1_path.read_text(encoding="utf-8"))["scenarios"]
    proposed = {(r["dataset"], r["family"].lower()): r for r in rows}
    problems: list[str] = []
    for scenario in table1:
        key = (scenario["dataset"], scenario["family"].lower())
        row = proposed.get(key)
        if row is None:
            problems.append(f"scenario {key} missing")
            continue
        if row["retrains"] != scenario["proposed_retrains"]:
            problems.append(f"{key}: retrains {row['retrains']} != Table I {scenario['proposed_retrains']}")
        if abs(row["gap_pct"] - scenario["gap_pct"]) > 0.01:
            problems.append(f"{key}: gap {row['gap_pct']:+.2f} != Table I {scenario['gap_pct']:+.2f}")
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build paper/reference_period_mlp43.json")
    p.add_argument("--rp3", nargs="+", type=Path, required=True)
    p.add_argument("--rp5", nargs="+", type=Path, required=True)
    p.add_argument("--rp10", nargs="+", type=Path, required=True)
    p.add_argument(
        "--table1",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "paper" / "table1_source.json",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "paper" / "reference_period_mlp43.json",
    )
    args = p.parse_args(argv)

    runs = {3: args.rp3, 5: args.rp5, 10: args.rp10}
    rows = {L: _load(dirs, "two_signal") for L, dirs in runs.items()}
    problems = _check_against_table1(rows[PRE_SPECIFIED], args.table1)
    if problems:
        for problem in problems:
            print(f"MISMATCH: {problem}")
        raise SystemExit("five-window run does not reproduce Table I; refusing to write")

    summary = {str(L): _summarise(rows[L]) for L in sorted(rows)}
    moves: dict[str, list[str]] = {}
    reference = summary[str(PRE_SPECIFIED)]["per_stream"]
    for key, row in reference.items():
        for L in sorted(rows):
            other = summary[str(L)]["per_stream"].get(key)
            if other is not None and other["in_margin"] != row["in_margin"]:
                moves.setdefault(str(L), []).append(key)

    payload = {
        "reference_periods": sorted(rows),
        "pre_specified": PRE_SPECIFIED,
        "summary": summary,
        "in_margin_moves_vs_pre_specified": moves,
        "provenance": {
            "cell": "Table I cell (MLP, seed 43)",
            "cross_check": "five-window run reproduces Table I exactly",
            "runs": {str(L): [str(d) for d in dirs] for L, dirs in runs.items()},
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for L in sorted(rows):
        s = summary[str(L)]
        print(
            f"rp={L:>3}: retrains {s['retrains']:4d} ({s['savings_pct']:.1f}% saved), "
            f"in-margin {s['in_margin']}/13, certified {s['certified_pooled']}/13, "
            f"median gap {s['median_gap_pct']:+.2f} pp"
        )
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
