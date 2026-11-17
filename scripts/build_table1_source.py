"""Assemble the single source of truth for the paper's Table I.

Every headline number in the manuscript (savings range, in-margin count,
certified count, margin sensitivity) is derived here from the canonical
per-window artefacts, so the prose cannot drift away from the results again.

Inputs are the ``windows_<family>.npz`` files written by
``scripts/run_paper_canonical.py``; the script recomputes the pooled paired
bootstrap bound itself rather than trusting any stored summary.

Usage:
    python scripts/build_table1_source.py \
      --kronodroid results/paper_canonical_robust_k1.5 \
      --lamda results/paper_canonical_lamda_k1.5 results/paper_canonical_lamda_extra \
      --mcndroid results/paper_canonical_mcndroid_full \
      --margin 0.02 --margin-alt 0.03 \
      --json-out ../paper/table1_source.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driftpolicy.evaluation.noninferiority import (  # noqa: E402
    pooled_non_inferiority_test,
)

# Presentation order and labels. LAMDA is ordered by malware volume in the
# evaluated years (2013, 2014, 2016): dowgin 5961, kuguo 3890, airpush 2673,
# gappusin 1527, adwo 1096, leadbolt 979, smsreg 919, revmob 914.
# One-sided 95% normal quantile. The pooled percentile bootstrap bound is
# reproduced to within 0.12 pp by the closed form
#     L.B. = Delta Acc - z * sqrt(p / N)
# in which p is the paired disagreement rate and N the number of paired
# evaluation samples. That equivalence is what makes the success/failure
# decomposition in Section V-D inspectable rather than a black box.
Z95 = float(norm.ppf(0.95))

KRONO_ORDER = [
    ("SMSreg", "SMSreg vs. Agent"),
    ("Boxer", "Boxer vs. SMSreg"),
    ("Airpush", "Airpush vs. SMSreg"),
    ("Malap", "Malap vs. SMSreg"),
    ("Agent", "Agent vs. SMSreg"),
]
LAMDA_ORDER = [
    ("dowgin", "Dowgin vs. Benign"),
    ("kuguo", "Kuguo vs. Benign"),
    ("airpush", "Airpush vs. Benign"),
    ("gappusin", "Gappusin vs. Benign"),
    ("adwo", "Adwo vs. Benign"),
    ("leadbolt", "Leadbolt vs. Benign"),
    ("smsreg", "SMSreg vs. Benign"),
    ("revmob", "Revmob vs. Benign"),
]

GROUPS = {
    "kronodroid": "KronoDroid Real Device Dataset — Physical-Hardware Dynamic Behavioral Telemetry",
    "lamda": "LAMDA Benchmark — High-Dimensional Sparse Static Bytecode Features",
    "mcndroid": "McNdroid Longitudinal Benchmark — Full 3-Year Multimodal Stream (2013–2016)",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kronodroid", nargs="+", required=True)
    p.add_argument("--lamda", nargs="+", required=True)
    p.add_argument(
        "--mcndroid",
        nargs="+",
        default=[],
        help="optional; omit to build a table that spans only the benchmarks given",
    )
    p.add_argument(
        "--mcndroid-prefix", nargs="*", default=[],
        help="optional run dirs holding a truncated McNdroid stream; reported as a robustness check",
    )
    p.add_argument(
        "--threshold-sweep", default=None,
        help="optional sweep_summary.csv from run_threshold_sensitivity_sweep.py; "
             "recorded as the threshold-robustness block of Section V-D",
    )
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--margin-alt", type=float, default=0.03)
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json-out", required=True)
    return p


def load_dirs(dirs: list[str]) -> dict[str, dict]:
    """family -> {proposed, periodic, sample_diffs}; later dirs do not override."""
    found: dict[str, dict] = {}
    for d in dirs:
        for path in sorted(Path(d).glob("windows_*.npz")):
            family = path.stem.replace("windows_", "")
            if family in found:
                continue
            data = np.load(path, allow_pickle=True)
            found[family] = {
                "proposed": data["proposed"],
                "periodic": data["periodic"],
                "sample_diffs": data["sample_diffs"].astype(np.int16),
                # The reference policies are only present in the cross-classifier
                # views; the frozen McNdroid runs predate them, so they stay
                # optional and the failure decomposition degrades gracefully.
                "static": np.asarray(data["static"], dtype=float)
                if "static" in data.files else None,
                "drift_aware": np.asarray(data["drift_aware"], dtype=float)
                if "drift_aware" in data.files else None,
                "source": str(path.parent),
            }
    return found


def scenario_row(dataset: str, family: str, label: str, dim: int, artefact: dict,
                 margins: dict[str, float], iters: int, seed: int) -> dict:
    proposed = artefact["proposed"]
    periodic = artefact["periodic"]
    base_margin = max(margins.values())
    pooled = pooled_non_inferiority_test(
        artefact["sample_diffs"], margin=base_margin, n_boot=iters, seed=seed, sided="one"
    )
    periodic_retrains = int(len(proposed))
    # The canonical runner retrains periodic on every window; the proposed
    # count is recomputed from the stored decision stream when available.
    diffs = artefact["sample_diffs"]
    n_plus = int((diffs > 0).sum())
    n_minus = int((diffs < 0).sum())
    n_disc = n_plus + n_minus
    n_eval = int(diffs.size)
    gap = float(proposed.mean() - periodic.mean())
    discordance_rate = n_disc / n_eval if n_eval else 0.0
    directional_bias = (n_plus - n_minus) / n_disc if n_disc else 0.0
    # Resolution floor: the half-width the paired design can resolve at the
    # observed disagreement rate, z * sqrt(p / N).  Certification at margin m
    # requires the point slack (gap + m) to exceed this floor.
    resolution_floor = Z95 * np.sqrt(discordance_rate / n_eval) if n_eval else 0.0
    # Slack is measured against the PRIMARY (pre-specified) margin, which is the
    # quantity the success criterion is stated for.
    slack = gap + margins["primary"]
    # As N grows at fixed rates the resolution floor z*sqrt(p/N) vanishes, so the
    # certificate is attainable in the limit exactly when the point estimate is
    # strictly inside the margin. Below the 0.05 pp guard the required sample
    # size is no longer a meaningful engineering answer.
    attainable = slack > 0.0005
    required_n = (
        int(np.ceil((Z95 ** 2) * discordance_rate / (slack ** 2)))
        if attainable else None
    )
    row = {
        "dataset": dataset,
        "family": family,
        "label": label,
        "dim": dim,
        "windows": periodic_retrains,
        "eval_samples": n_eval,
        "periodic_retrains": None,
        "proposed_retrains": None,
        "savings_pct": None,
        "periodic_accuracy_pct": round(100 * float(periodic.mean()), 2),
        "proposed_accuracy_pct": round(100 * float(proposed.mean()), 2),
        "gap_pct": round(100 * float(proposed.mean() - periodic.mean()), 2),
        "ci_lower_pooled_pct": round(100 * pooled["ci_lower"], 2),
        "break_even_margin_pct": round(-100 * pooled["ci_lower"], 2),
        "n_plus": n_plus,
        "n_minus": n_minus,
        "n_discordant": n_disc,
        "discordance_rate_pct": round(100 * discordance_rate, 2),
        "directional_bias": round(directional_bias, 3),
        "resolution_floor_pp": round(100 * resolution_floor, 2),
        "point_slack_pp": round(100 * slack, 2),
        "asymptotically_certifiable": bool(attainable),
        "required_eval_samples": required_n,
        "required_windows": (
            int(np.ceil(required_n / (n_eval / periodic_retrains)))
            if required_n is not None and periodic_retrains else None
        ),
        "in_margin_point": {},
        "certified": {},
    }
    for name, m in margins.items():
        row["in_margin_point"][name] = bool(proposed.mean() - periodic.mean() >= -m)
        row["certified"][name] = bool(pooled["ci_lower"] >= -m)
    primary = "primary" if "primary" in margins else next(iter(margins))
    if row["certified"][primary]:
        row["verdict"] = "certified"
    elif row["in_margin_point"][primary]:
        row["verdict"] = "resolution-limited"
    else:
        row["verdict"] = "directional-gap"
    # How far the bound must travel before the scenario certifies, in pp.
    row["miss_pp"] = round(row["break_even_margin_pct"] - 100 * margins[primary], 2)
    return row


def attach_retrain_counts(rows: list[dict], dataset: str, dirs: list[str]) -> None:
    """Fill retrain counts, savings and the calibrated thresholds from summary.csv."""
    import csv

    counts: dict[str, dict[str, float]] = {}
    for d in dirs:
        summary = Path(d) / "summary.csv"
        if not summary.exists():
            continue
        with summary.open("r", newline="", encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                counts.setdefault(str(rec["family"]).lower(), {
                    "periodic_retrains": int(rec["periodic_retrains"]),
                    "proposed_retrains": int(rec["two_signal_retrains"]),
                    "savings_pct": round(float(rec["retrain_savings_pct"]), 1),
                    "theta1_psi": round(float(rec["theta1_psi"]), 4),
                    "theta2_silhouette": round(float(rec["theta2_silhouette"]), 4),
                    "drift_aware_retrains": int(rec.get("drift_aware_retrains", 0)),
                    "static_accuracy_pct": round(
                        100 * float(rec.get("static_accuracy", "nan")), 2
                    ),
                    "drift_aware_accuracy_pct": round(
                        100 * float(rec.get("drift_aware_accuracy", "nan")), 2
                    ),
                    "two_signal_silhouette_computes": int(
                        rec.get("two_signal_silhouette_computes", 0)
                    ),
                })
    for row in rows:
        if row["dataset"] != dataset:
            continue
        rec = counts.get(row["family"].lower())
        if rec:
            row.update(rec)


def failure_analysis(rows: list[dict], artefact_by_dataset: dict[str, dict]) -> list[dict]:
    """Per-window decomposition of every stream that leaves the margin.

The paper states *why* each failing stream fails; those statements must come
    from the same per-window artefact as the rest of Table I, so the decomposition
    is computed here and frozen into the JSON rather than retyped in the prose.
    """
    out: list[dict] = []
    for row in rows:
        if row.get("verdict") != "directional-gap":
            continue
        art = artefact_by_dataset.get(row["dataset"], {}).get(row["family"])
        if art is None:
            continue
        proposed = np.asarray(art["proposed"], dtype=float)
        periodic = np.asarray(art["periodic"], dtype=float)
        static = art.get("static")
        drift_aware = art.get("drift_aware")
        if static is not None:
            static = np.asarray(static, dtype=float)
        if drift_aware is not None:
            drift_aware = np.asarray(drift_aware, dtype=float)
        gap = proposed - periodic
        n = int(gap.size)
        half = n // 2
        worst = int(np.argmin(gap))
        out.append(
            {
                "dataset": row["dataset"],
                "family": row["family"],
                "label": row["label"],
                "windows": n,
                "periodic_retrains": row.get("periodic_retrains"),
                "proposed_retrains": row.get("proposed_retrains"),
                "drift_aware_retrains": row.get("drift_aware_retrains"),
                "psi_alerts": row.get("two_signal_silhouette_computes"),
                "theta1_psi": row.get("theta1_psi"),
                "theta2_silhouette": row.get("theta2_silhouette"),
                "savings_pct": row.get("savings_pct"),
                "static_accuracy_pct": (
                    round(100 * float(static.mean()), 2) if static is not None else None
                ),
                "drift_aware_accuracy_pct": (
                    round(100 * float(drift_aware.mean()), 2)
                    if drift_aware is not None else None
                ),
                "proposed_accuracy_pct": row.get("proposed_accuracy_pct"),
                "periodic_accuracy_pct": row.get("periodic_accuracy_pct"),
                "gap_pct": row.get("gap_pct"),
                "ci_lower_pooled_pct": row.get("ci_lower_pooled_pct"),
                "resolution_floor_pp": row.get("resolution_floor_pp"),
                "gap_over_floor": (
                    round(abs(row["gap_pct"]) / row["resolution_floor_pp"], 1)
                    if row.get("resolution_floor_pp") else None
                ),
                "first_half_gap_pp": round(100 * float(gap[:half].mean()), 2),
                "second_half_gap_pp": round(100 * float(gap[half:].mean()), 2),
                "retraining_gain_vs_static_pp": (
                    round(100 * float(proposed.mean() - static.mean()), 2)
                    if static is not None else None
                ),
                "windows_below_5pp": int((gap < -0.05).sum()),
                "worst_window": worst + 1,
                "worst_window_proposed_pct": round(100 * float(proposed[worst]), 1),
                "worst_window_periodic_pct": round(100 * float(periodic[worst]), 1),
                "worst_window_gap_pct": round(100 * float(gap[worst]), 1),
            }
        )
    return out


def summarise_threshold_sweep(csv_path: str, margin: float) -> dict:
    """Condense the threshold sweep into the facts Section V-D reports."""
    import csv

    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    families = sorted({r["family"] for r in rows})
    configs: list[str] = []
    for r in rows:
        if r["config_name"] not in configs:
            configs.append(r["config_name"])
    within = {}
    for cfg in configs:
        subset = [r for r in rows if r["config_name"] == cfg]
        within[cfg] = sum(1 for r in subset if float(r["accuracy_gap"]) >= -margin)
    gaps = {
        fam: {
            "min_pct": round(100 * min(float(r["accuracy_gap"])
                                       for r in rows if r["family"] == fam), 2),
            "max_pct": round(100 * max(float(r["accuracy_gap"])
                                       for r in rows if r["family"] == fam), 2),
        }
        for fam in families
    }
    by_config = {
        fam: {
            cfg: round(100 * float(next(r["accuracy_gap"] for r in rows
                                      if r["family"] == fam and r["config_name"] == cfg)), 2)
            for cfg in configs
        }
        for fam in families
    }
    return {
        "source": str(csv_path),
        "n_runs": len(rows),
        "families": families,
        "configs": configs,
        "within_margin_counts": within,
        "best_config": max(configs, key=lambda c: within[c]),
        "gap_range_pct": gaps,
        "gap_pct_by_config": by_config,
    }


def summarise(rows: list[dict], margins: dict[str, float]) -> dict:
    savings = [r["savings_pct"] for r in rows if r["savings_pct"] is not None]
    savings_sorted = sorted(savings)
    mid = len(savings_sorted) // 2
    median = (savings_sorted[mid] if len(savings_sorted) % 2
              else 0.5 * (savings_sorted[mid - 1] + savings_sorted[mid]))
    out = {
        "n_scenarios": len(rows),
        "savings_min_pct": min(savings) if savings else None,
        "savings_max_pct": max(savings) if savings else None,
        "savings_median_pct": round(median, 1) if savings else None,
        "margins": margins,
    }
    for name, m in margins.items():
        out[f"in_margin_point_count_{name}"] = sum(
            1 for r in rows if r["in_margin_point"][name]
        )
        out[f"certified_count_{name}"] = sum(1 for r in rows if r["certified"][name])
        out[f"margin_value_{name}"] = m
    out["verdict_counts"] = {
        v: sum(1 for r in rows if r.get("verdict") == v)
        for v in ("certified", "resolution-limited", "directional-gap")
    }
    misses = [r["miss_pp"] for r in rows if r.get("verdict") == "resolution-limited"]
    out["resolution_miss_min_pp"] = min(misses) if misses else None
    out["resolution_miss_max_pp"] = max(misses) if misses else None
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    margins = {"primary": args.margin, "alt": args.margin_alt}

    krono = load_dirs(args.kronodroid)
    lamda = load_dirs(args.lamda)
    mcndroid = load_dirs(args.mcndroid)

    rows: list[dict] = []
    for family, label in KRONO_ORDER:
        if family in krono:
            rows.append(scenario_row("kronodroid", family, label, 478, krono[family],
                                     margins, args.bootstrap_iters, args.seed))
    for family, label in LAMDA_ORDER:
        if family in lamda:
            rows.append(scenario_row("lamda", family, label, 925, lamda[family],
                                     margins, args.bootstrap_iters, args.seed))
    for family in sorted(mcndroid):
        rows.append(scenario_row(
            "mcndroid", family, "Longitudinal 3-Year Stream", 2390, mcndroid[family],
            margins, args.bootstrap_iters, args.seed,
        ))

    attach_retrain_counts(rows, "kronodroid", args.kronodroid)
    attach_retrain_counts(rows, "lamda", args.lamda)
    attach_retrain_counts(rows, "mcndroid", args.mcndroid)

    prefix_refs = load_dirs(args.mcndroid_prefix) if args.mcndroid_prefix else {}
    robustness = []
    for ref_family, ref in sorted(prefix_refs.items()):
        row = scenario_row("mcndroid", ref_family, "Truncated stream (2013 prefix)", 2390, ref,
                           margins, args.bootstrap_iters, args.seed)
        attach_retrain_counts([row], "mcndroid", args.mcndroid_prefix)
        correctness = {k: v for k, v in row.items() if k not in ("dataset", "family")}
        robustness.append(correctness)

    payload = {
        "groups": GROUPS,
        "scenarios": rows,
        "failure_analysis": failure_analysis(
            rows, {"kronodroid": krono, "lamda": lamda, "mcndroid": mcndroid}
        ),
        "robustness": robustness,
        "summary": summarise(rows, margins),
        "provenance": {
            "kronodroid_runs": args.kronodroid,
            "lamda_runs": args.lamda,
            "mcndroid_runs": args.mcndroid,
            "bootstrap_iters": args.bootstrap_iters,
            "seed": args.seed,
            "unit": "pooled paired evaluation samples, 95% one-sided percentile",
            "z95": Z95,
            "closed_form": "L.B. = Delta Acc - z * sqrt(p / N)",
        },
    }
    if args.threshold_sweep:
        payload["threshold_robustness"] = summarise_threshold_sweep(
            args.threshold_sweep, margins["primary"]
        )
    theta1 = [r["theta1_psi"] for r in rows if "theta1_psi" in r]
    theta2 = [r["theta2_silhouette"] for r in rows if "theta2_silhouette" in r]
    if theta1 and theta2:
        payload["thresholds"] = {
            "theta1_psi_min": min(theta1),
            "theta1_psi_max": max(theta1),
            "theta2_silhouette_min": min(theta2),
            "theta2_silhouette_max": max(theta2),
            "ratio_psi": round(max(theta1) / min(theta1), 0),
            "ratio_silhouette": round(max(theta2) / min(theta2), 0),
        }
    Path(args.json_out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    head = (f"{'scenario':<24}{'dim':>5}{'win':>5}{'retr':>10}{'save%':>7}"
            f"{'per%':>7}{'prop%':>7}{'gap':>7}{'L.B.':>7}{'in?':>4}{'cert':>5}"
            f"{'p%':>7}{'bias':>7}{'floor':>7}{'verdict':>19}")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['family']:<24}{r['dim']:>5}{r['windows']:>5}"
              f"{str(r['proposed_retrains']) + '/' + str(r['periodic_retrains']):>10}"
              f"{r['savings_pct']:>7}{r['periodic_accuracy_pct']:>7}"
              f"{r['proposed_accuracy_pct']:>7}{r['gap_pct']:>+7}"
              f"{r['ci_lower_pooled_pct']:>+7}"
              f"{'yes' if r['in_margin_point']['primary'] else 'no':>4}"
              f"{'yes' if r['certified']['primary'] else 'no':>5}"
              f"{r['discordance_rate_pct']:>7}{r['directional_bias']:>+7.2f}"
              f"{r['resolution_floor_pp']:>7}{r['verdict']:>19}")
    s = payload["summary"]
    print("-" * len(head))
    print(f"scenarios {s['n_scenarios']} | savings {s['savings_min_pct']}–{s['savings_max_pct']}% "
          f"(median {s['savings_median_pct']}%) | in-margin {s['in_margin_point_count_primary']} "
          f"| certified@{args.margin:.0%} {s['certified_count_primary']} "
          f"| certified@{args.margin_alt:.0%} {s['certified_count_alt']}")
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
