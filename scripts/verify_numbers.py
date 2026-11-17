"""Check the shipped artefacts against the numbers printed in the paper.

The check is data-free: it reads ``artifacts/*.json`` and compares the headline
numbers with the constants below, which are copied from the manuscript
(Tables I and II and Section V-C).  Running it after ``make tables sweep``
therefore shows whether the reproduction landed on the published values.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

EXPECTED = {
    "table1": {
        "total_retrains": 256,
        "mean_gap_pct": -1.55,
        "in_margin": 10,
        "certified": 8,
        "streams": 13,
    },
    "components": {
        "static": 13,
        "periodic": 822,
        "psi_only": 548,
        "drift_aware": 318,
        "two_signal": 256,
        "two_signal_median_savings_pct": 78.9,
        "two_signal_mean_gap_pct": -1.55,
        "two_signal_in_margin": 10,
        "two_signal_certified": 8,
    },
    "sweep": {
        "3": {"retrains": 397, "certified": 10, "median_gap_pct": -0.30},
        "5": {"retrains": 256, "certified": 8, "median_gap_pct": -1.14},
        "10": {"retrains": 210, "certified": 7, "median_gap_pct": -1.17},
    },
}

failures: list[str] = []


def check(name: str, got, want, tol: float = 0.011) -> None:
    ok = (
        abs(float(got) - float(want)) <= tol
        if isinstance(want, float)
        else got == want
    )
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<42} got {got!s:<10} expected {want}")
    if not ok:
        failures.append(f"{name}: got {got}, expected {want}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    art = args.artifacts

    print("Table I")
    table1 = json.loads((art / "table1_source.json").read_text(encoding="utf-8"))
    scenarios = table1["scenarios"]
    total = sum(s["proposed_retrains"] for s in scenarios)
    mean_gap = sum(s["gap_pct"] for s in scenarios) / len(scenarios)
    check("streams", len(scenarios), EXPECTED["table1"]["streams"])
    check("total proposed retrains", total, EXPECTED["table1"]["total_retrains"])
    check("mean accuracy gap (pp)", round(mean_gap, 2), EXPECTED["table1"]["mean_gap_pct"])
    check(
        "streams inside the 2% margin",
        sum(1 for s in scenarios if s["in_margin_point"]["primary"]),
        EXPECTED["table1"]["in_margin"],
    )
    check(
        "streams certified",
        sum(1 for s in scenarios if s["certified"]["primary"]),
        EXPECTED["table1"]["certified"],
    )

    print("Table II (ablation)")
    policies = json.loads((art / "baseline_components.json").read_text(encoding="utf-8"))["policies"]
    for name in ("static", "periodic", "psi_only", "drift_aware", "two_signal"):
        check(f"{name} retrains", policies[name]["total_retrains"], EXPECTED["components"][name])
    two = policies["two_signal"]
    check("two_signal median savings (%)", round(two["median_savings_pct"], 1),
          EXPECTED["components"]["two_signal_median_savings_pct"])
    check("two_signal mean gap (pp)", round(two["mean_gap_pct"], 2),
          EXPECTED["components"]["two_signal_mean_gap_pct"])
    check("two_signal in margin", two["n_in_margin"], EXPECTED["components"]["two_signal_in_margin"])
    check("two_signal certified", two["n_certified"], EXPECTED["components"]["two_signal_certified"])
    if not math.isclose(two["total_retrains"], total, abs_tol=1e-9):
        failures.append("Table II two_signal row disagrees with Table I total")

    print("Section V-C reference-period sweep")
    sweep = json.loads((art / "reference_period_mlp43.json").read_text(encoding="utf-8"))
    for length, want in EXPECTED["sweep"].items():
        got = sweep["summary"][length]
        check(f"rp={length} retrains", got["retrains"], want["retrains"])
        check(f"rp={length} certified", got["certified_pooled"], want["certified"])
        check(f"rp={length} median gap (pp)", round(got["median_gap_pct"], 2),
              want["median_gap_pct"])

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
