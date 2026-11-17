"""Run the single-detector baseline matrix under the paper's exact protocol.

Every baseline is a retraining policy that acts on one label-free signal only,
calibrated with the same robust MAD rule (``median + c * 1.4826 * MAD``) the
paper uses for the proposed two-signal policy, on the same temporal batches,
the same reference period, the same Random-Forest classifier and the same
current-batch retraining rule:

* ``psi_only``  -- aggregate PSI at the proposed policy's own $\theta_1$ (the
  exact ablation of the screen).
* ``ks_only``   -- feature-wise two-sample KS (max over features) between the
  previous and the current window above its calibrated threshold.
* ``adwin_only``-- ADWIN (Bifet & Gavalda, 2007) on the per-sample novelty
  score stream, delta = 0.002.
* ``kswin_only``-- KSWIN (Raab et al., 2020) on the same stream, alpha = 0.01,
  window 100 / statistic 50.
* ``drift_aware``  -- Silhouette-only baseline (single geometric signal).
* ``two_signal``   -- the proposed policy.
* ``static`` / ``periodic`` -- the two operational reference points.

The three geometric / sequential baselines are calibrated by their own
standard parameters, not by a tuned constant: KSWIN is scale-free (KS
statistic), and ADWIN's confidence ``delta`` is the value used in the original
publication.  The two threshold-based single-signal baselines consume the same robust MAD
calibration as the proposed policy: PSI-only is run at the proposed policy's
own $\theta_1$, so it differs from the conjunction by the confirmation stage
alone, and KS-only is calibrated on the reference-period variation of its own
(adjacent-window) statistic.

Example:
    python scripts/run_baseline_matrix.py --dataset kronodroid \
      --out results/baseline_matrix_kronodroid
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driftpolicy import (  # noqa: E402
    ExperimentConfig,
    ModelConfig,
    SignalConfig,
    TrainingDataConfig,
    run_experiment,
)
from driftpolicy.data.kronodroid import (  # noqa: E402
    load_kronodroid_csv,
    prepare_kronodroid_experiment,
)
from driftpolicy.data.lamda import (  # noqa: E402
    load_lamda_dataset,
    prepare_lamda_experiment,
)
from driftpolicy.data.mcndroid import (  # noqa: E402
    load_mcndroid,
    prepare_mcndroid_experiment,
)
from driftpolicy.evaluation.noninferiority import (  # noqa: E402
    pooled_non_inferiority_test,
)
from driftpolicy.experiment import run_policy  # noqa: E402
from driftpolicy.policies import (  # noqa: E402
    DetectorPolicy,
    SequentialDetectorPolicy,
)
from driftpolicy.signals import SignalProvider  # noqa: E402

KRONO_FAMILIES = {
    "Airpush": ("Airpush/StopSMS", "SMSreg"),
    "SMSreg": ("SMSreg", "Agent"),
    "Malap": ("Malap", "SMSreg"),
    "Boxer": ("Boxer", "SMSreg"),
    "Agent": ("Agent", "SMSreg"),
}
LAMDA_FAMILIES = ["airpush", "dowgin", "kuguo", "leadbolt", "smsreg", "gappusin", "adwo", "revmob"]

# Policies produced directly by ``run_experiment`` (no extra detector needed).
CORE_POLICIES = ("static", "periodic", "drift_aware", "two_signal")
# Single-signal baselines that need their own threshold / detector instance.
EXTRA_SIGNALS = ("psi_only", "ks_only", "adwin_only", "kswin_only")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-detector baseline matrix")
    p.add_argument("--dataset", choices=["kronodroid", "lamda", "mcndroid"], required=True)
    p.add_argument("--kronodroid-path", default="data/KronoDroid_real_device.csv")
    p.add_argument("--lamda-path", default="data/lamda")
    p.add_argument("--mcndroid-path", default="data/mcndroid")
    p.add_argument("--families", nargs="+", default=None)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--threshold-method", choices=["percentile", "robust"], default="robust")
    p.add_argument("--threshold-percentile", type=float, default=0.90)
    p.add_argument("--robust-k", type=float, default=1.5)
    p.add_argument("--reference-period", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--psi-bins", type=int, default=10)
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", choices=["random", "mlp", "svm", "xgboost"], default="random")
    p.add_argument("--policies", nargs="+", choices=list(CORE_POLICIES + EXTRA_SIGNALS), default=None)
    p.add_argument("--out", required=True)
    return p


def _signal_config(args: argparse.Namespace) -> SignalConfig:
    return SignalConfig(
        reference_period=args.reference_period,
        psi_bins=args.psi_bins,
        silhouette_mode="previous",
        threshold_method=args.threshold_method,
        threshold_percentile=args.threshold_percentile,
        robust_k=args.robust_k,
        seed=args.seed,
    )


def _provider(batches, args: argparse.Namespace) -> SignalProvider:
    return SignalProvider(
        batches.batches,
        reference_period=args.reference_period,
        psi_bins=args.psi_bins,
        psi_agg="mean",
        n_clusters=2,
        minibatch_size=25,
        seed=args.seed,
    )


def _record(
    *,
    dataset: str,
    family: str,
    policy: str,
    result,
    periodic,
    other_train: np.ndarray,
    other_test: np.ndarray,
    config: ExperimentConfig,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Summarize one policy outcome against the periodic baseline."""
    sample_diffs = np.concatenate(
        [p.astype(np.int8) - q.astype(np.int8) for p, q in zip(result.predictions, periodic.predictions)]
    )
    ni = pooled_non_inferiority_test(
        sample_diffs,
        margin=config.non_inferiority_margin,
        n_boot=args.bootstrap_iters,
        seed=args.bootstrap_seed,
    )
    gap = float(result.mean_accuracy - periodic.mean_accuracy)
    savings = (
        100.0 * (periodic.n_retrains - result.n_retrains) / periodic.n_retrains
        if periodic.n_retrains
        else 0.0
    )
    return {
        "dataset": dataset,
        "family": family,
        "policy": policy,
        "n_windows": result.n_batches,
        "retrains": result.n_retrains,
        "savings_pct": round(savings, 1),
        "accuracy_pct": round(100 * result.mean_accuracy, 2),
        "gap_pct": round(100 * gap, 2),
        "ci_lower_pct": round(100 * ni["ci_lower"], 2),
        "in_margin_point": bool(gap >= -config.non_inferiority_margin),
        "certified": bool(ni["non_inferior"]),
        "n_discordant": int(np.count_nonzero(sample_diffs)),
        "n_eval_samples": int(sample_diffs.size),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.dataset == "kronodroid":
        data = load_kronodroid_csv(args.kronodroid_path)
        families = args.families or list(KRONO_FAMILIES)
        prepare = lambda fam: prepare_kronodroid_experiment(  # noqa: E731
            data,
            family=KRONO_FAMILIES[fam][0],
            other_family=KRONO_FAMILIES[fam][1],
            batch_size=args.batch_size,
            seed=args.seed,
            reference_period=args.reference_period,
        )
    elif args.dataset == "lamda":
        data = load_lamda_dataset(args.lamda_path)
        families = args.families or LAMDA_FAMILIES
        prepare = lambda fam: prepare_lamda_experiment(  # noqa: E731
            data,
            family=fam,
            other_family="benign",
            batch_size=args.batch_size,
            seed=args.seed,
            reference_period=args.reference_period,
        )
    else:
        data = load_mcndroid(args.mcndroid_path)
        families = args.families or ["longitudinal"]
        prepare = lambda fam: prepare_mcndroid_experiment(  # noqa: E731
            data,
            batch_size=args.batch_size,
            seed=args.seed,
            reference_period=args.reference_period,
            max_batches=args.max_batches,
        )

    manifest = {
        "dataset": args.dataset,
        "n_features": data.n_features,
        "batch_size": args.batch_size,
        "reference_period": args.reference_period,
        "psi_bins": args.psi_bins,
        "threshold_method": args.threshold_method,
        "threshold_percentile": args.threshold_percentile,
        "robust_k": args.robust_k,
        "non_inferiority_margin": 0.02,
        "bootstrap_iters": args.bootstrap_iters,
        "bootstrap_seed": args.bootstrap_seed,
        "seed": args.seed,
        "model": args.model,
        "families": families,
        "max_batches": args.max_batches,
        "baselines": {
            "psi_only": "aggregate PSI vs reference at the proposed policy's own theta1",
            "ks_only": "max feature-wise KS vs previous window, MAD-calibrated threshold",
            "adwin_only": "ADWIN(delta=0.002) on per-sample novelty scores",
            "kswin_only": "KSWIN(window=100, stat=50, alpha=0.01, 100 permutations)",
            "drift_aware": "Silhouette-only (theta2), MAD-calibrated",
            "two_signal": "proposed PSI AND Silhouette, MAD-calibrated",
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    for fam in families:
        started = time.perf_counter()
        batches, other_train, other_test = prepare(fam)
        config = ExperimentConfig(
            batch_size=args.batch_size,
            model=ModelConfig(name=args.model),
            signal=_signal_config(args),
            training_data=TrainingDataConfig(mode="current"),
            bootstrap_iters=args.bootstrap_iters,
            seed=args.seed,
        )
        core = run_experiment(batches, other_train, other_test, config, collect_predictions=True)
        periodic = core.policies["periodic"]

        wanted = set(args.policies) if args.policies else set(CORE_POLICIES + EXTRA_SIGNALS)
        for name in CORE_POLICIES:
            if name == "periodic" or name not in wanted:
                continue
            rows.append(
                _record(
                    dataset=args.dataset,
                    family=fam,
                    policy=name,
                    result=core.policies[name],
                    periodic=periodic,
                    other_train=other_train,
                    other_test=other_test,
                    config=config,
                    args=args,
                )
            )

        for name in EXTRA_SIGNALS:
            if name not in wanted:
                continue
            provider = _provider(batches, args)
            if name in ("psi_only", "ks_only"):
                kind = name.removesuffix("_only")
                if kind == "psi":
                    # Exact ablation of the proposed screen: the identity of the
                    # threshold matters here, so the PSI-only baseline uses the
                    # very theta1 the proposed policy was run with (the paper's
                    # leave-one-out reference calibration) instead of a second,
                    # differently-sampled calibration of the same statistic.
                    threshold = core.theta1
                else:
                    threshold = provider.calibrate_detector_threshold(
                        kind,
                        method=args.threshold_method,
                        percentile=args.threshold_percentile,
                        robust_k=args.robust_k,
                        default=0.1,
                    )
                policy = DetectorPolicy(
                    kind,
                    provider,
                    threshold,
                    0.0,
                    reference_period=args.reference_period,
                    silhouette_mode="previous",
                )
            else:
                kind = name.removesuffix("_only")
                policy = SequentialDetectorPolicy(
                    kind,
                    provider,
                    core.theta1,
                    core.theta2,
                    reference_period=args.reference_period,
                    silhouette_mode="previous",
                )
            result = run_policy(
                policy, batches, other_train, other_test, config, collect_predictions=True
            )
            rows.append(
                _record(
                    dataset=args.dataset,
                    family=fam,
                    policy=name,
                    result=result,
                    periodic=periodic,
                    other_train=other_train,
                    other_test=other_test,
                    config=config,
                    args=args,
                )
            )

        elapsed = time.perf_counter() - started
        family_rows = [r for r in rows if r["family"] == fam]
        print(
            f"[{fam}] {elapsed:.0f}s :: "
            + " | ".join(
                f"{r['policy']}={r['retrains']}/"
                f"{core.policies['periodic'].n_retrains}"
                f"({r['savings_pct']:.0f}%,{r['gap_pct']:+.2f})"
                for r in family_rows
            ),
            flush=True,
        )

    fields = list(rows[0]) if rows else []
    with (out / "baseline_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "baseline_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(rows)} rows to {out / 'baseline_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
