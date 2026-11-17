"""Run the primary protocol across the four reference classifier families.

The base study this work extends evaluates drift-aware retraining with several
learners, so the classifier is a replication axis rather than a fixed
constant.  This runner repeats the *primary* configuration of the paper -- same
windows, same reference period, same robust MAD calibration ($c$ from the
command line), same four policies, same evaluation unit -- once per classifier
family, and records for every run:

* accuracy, F1 and balanced accuracy of each of the four policies;
* retraining counts and savings against periodic retraining;
* the silhouette (confirmation-stage) computation counts;
* the paired per-sample outcomes of proposed vs periodic, so the
  non-inferiority certificate can be recomputed from the same unit of analysis
  as the reported accuracy.

Both benchmarks are supported: the five ordered KronoDroid family pairs of the
primary configuration, and the eight LAMDA families against the fixed
verified-benign pool.

Examples:
    python scripts/run_classifier_matrix.py --dataset kronodroid \\
      --kronodroid-path data/KronoDroid_real_device.csv \\
      --out results/classifier_matrix_kronodroid
    python scripts/run_classifier_matrix.py --dataset lamda \\
      --lamda-path data/lamda --out results/classifier_matrix_lamda
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
from driftpolicy.evaluation.noninferiority import (  # noqa: E402
    pooled_non_inferiority_test,
)

# The five ordered family pairs of the primary configuration.
KRONO_FAMILIES = {
    "Airpush": ("Airpush/StopSMS", "SMSreg"),
    "SMSreg": ("SMSreg", "Agent"),
    "Malap": ("Malap", "SMSreg"),
    "Boxer": ("Boxer", "SMSreg"),
    "Agent": ("Agent", "SMSreg"),
}
LAMDA_FAMILIES = [
    "airpush",
    "dowgin",
    "kuguo",
    "leadbolt",
    "smsreg",
    "gappusin",
    "adwo",
    "revmob",
]
MODELS = ("random", "mlp", "svm", "xgboost")
POLICIES = ("static", "periodic", "drift_aware", "two_signal")

FIELDS = [
    "dataset",
    "family",
    "other_family",
    "model",
    "seed",
    "policy",
    "n_features",
    "n_batches",
    "theta1_psi",
    "theta2_silhouette",
    "accuracy",
    "f1",
    "balanced_accuracy",
    "retrain_count",
    "retrain_savings_vs_periodic_percent",
    "silhouette_computes",
    "n_eval_samples",
    "periodic_accuracy",
    "periodic_f1",
    "periodic_balanced_accuracy",
    "pooled_accuracy_gap",
    "ci_lower_pooled",
    "non_inferior_pooled",
    "non_inferior_point",
    "run_seconds",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cross-classifier replication matrix")
    p.add_argument("--dataset", choices=["kronodroid", "lamda"], required=True)
    p.add_argument("--kronodroid-path", default="data/KronoDroid_real_device.csv")
    p.add_argument("--lamda-path", default="data/lamda")
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    p.add_argument("--families", nargs="+", default=None)
    p.add_argument("--threshold-method", choices=["percentile", "robust"], default="robust")
    p.add_argument("--threshold-percentile", type=float, default=0.90)
    p.add_argument("--robust-k", type=float, default=1.5)
    p.add_argument("--reference-period", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--psi-bins", type=int, default=10)
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--out", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.dataset == "kronodroid":
        data = load_kronodroid_csv(args.kronodroid_path)
        families = args.families or list(KRONO_FAMILIES)
        runs = [
            (fam, KRONO_FAMILIES[fam][0], KRONO_FAMILIES[fam][1]) for fam in families
        ]

        def prepare(family_source, other_source, seed):
            return prepare_kronodroid_experiment(
                data,
                family=family_source,
                other_family=other_source,
                batch_size=args.batch_size,
                seed=seed,
                reference_period=args.reference_period,
            )

    else:
        data = load_lamda_dataset(args.lamda_path)
        families = args.families or list(LAMDA_FAMILIES)
        runs = [(fam, fam, "benign") for fam in families]

        def prepare(family_source, other_source, seed):
            return prepare_lamda_experiment(
                data,
                family=family_source,
                other_family=other_source,
                batch_size=args.batch_size,
                seed=seed,
                reference_period=args.reference_period,
            )

    manifest = {
        "protocol": "primary configuration: four policies, same windows and reference period",
        "dataset": args.dataset,
        "n_features": data.n_features,
        "batch_size": args.batch_size,
        "reference_period": args.reference_period,
        "psi_bins": args.psi_bins,
        "threshold_method": args.threshold_method,
        "threshold_percentile": args.threshold_percentile,
        "robust_k": args.robust_k,
        "silhouette_mode": "previous",
        "training_data": "current-only",
        "decision_rule": "PSI > theta1 AND delta_s > theta2",
        "non_inferiority_margin": args.margin,
        "bootstrap_iters": args.bootstrap_iters,
        "models": args.models,
        "seeds": args.seeds,
        "scenarios": [r[0] for r in runs],
        "n_runs": len(runs) * len(args.models) * len(args.seeds),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_path = out / "summary.csv"
    rows: list[dict[str, object]] = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    completed = {
        (int(r["seed"]), r["family"], r["model"]) for r in rows if r["policy"] == "static"
    }

    started_all = time.perf_counter()
    for seed in args.seeds:
        for label, family_source, other_source in runs:
            batches, other_train, other_test = prepare(family_source, other_source, seed)
            for model_name in args.models:
                if (seed, label, model_name) in completed:
                    print(f"skip seed={seed} {label}/{model_name} (already done)", flush=True)
                    continue
                started = time.perf_counter()
                config = ExperimentConfig(
                    batch_size=args.batch_size,
                    model=ModelConfig(name=model_name),
                    signal=SignalConfig(
                        reference_period=args.reference_period,
                        psi_bins=args.psi_bins,
                        silhouette_mode="previous",
                        threshold_method=args.threshold_method,
                        threshold_percentile=args.threshold_percentile,
                        robust_k=args.robust_k,
                        seed=seed,
                    ),
                    training_data=TrainingDataConfig(mode="current"),
                    bootstrap_iters=args.bootstrap_iters,
                    seed=seed,
                )
                result = run_experiment(
                    batches, other_train, other_test, config, collect_predictions=True
                )
                periodic = result.policies["periodic"]
                two = result.policies["two_signal"]
                sample_diffs = np.concatenate(
                    [
                        p.astype(np.int8) - q.astype(np.int8)
                        for p, q in zip(two.predictions, periodic.predictions)
                    ]
                )
                ni_pooled = pooled_non_inferiority_test(
                    sample_diffs,
                    margin=config.non_inferiority_margin,
                    n_boot=args.bootstrap_iters,
                    seed=seed,
                    sided="one",
                )
                elapsed = time.perf_counter() - started
                np.savez(
                    out / f"windows_{args.dataset}_{label}_{model_name}_s{seed}.npz",
                    sample_diffs=sample_diffs,
                    window_sizes=np.asarray(
                        [len(p) for p in two.predictions], dtype=np.int32
                    ),
                    **{
                        name: np.asarray(result.policies[name].per_window("accuracy"))
                        for name in POLICIES
                    },
                )
                for policy_name in POLICIES:
                    policy = result.policies[policy_name]
                    rows.append(
                        {
                            "dataset": args.dataset,
                            "family": label,
                            "other_family": other_source,
                            "model": model_name,
                            "seed": seed,
                            "policy": policy_name,
                            "n_features": data.n_features,
                            "n_batches": result.n_batches,
                            "theta1_psi": round(result.theta1, 6),
                            "theta2_silhouette": round(result.theta2, 6),
                            "accuracy": round(policy.mean_accuracy, 4),
                            "f1": round(policy.mean_f1, 4),
                            "balanced_accuracy": round(policy.mean_balanced_accuracy, 4),
                            "retrain_count": policy.n_retrains,
                            "retrain_savings_vs_periodic_percent": round(
                                100.0
                                * (periodic.n_retrains - policy.n_retrains)
                                / periodic.n_retrains
                                if periodic.n_retrains
                                else 0.0,
                                1,
                            ),
                            "silhouette_computes": policy.n_silhouette_computes,
                            "n_eval_samples": int(sample_diffs.size),
                            "periodic_accuracy": round(periodic.mean_accuracy, 4),
                            "periodic_f1": round(periodic.mean_f1, 4),
                            "periodic_balanced_accuracy": round(
                                periodic.mean_balanced_accuracy, 4
                            ),
                            "pooled_accuracy_gap": round(ni_pooled["mean_diff"], 4),
                            "ci_lower_pooled": round(ni_pooled["ci_lower"], 4),
                            "non_inferior_pooled": ni_pooled["non_inferior"],
                            "non_inferior_point": bool(
                                two.mean_accuracy - periodic.mean_accuracy
                                >= -config.non_inferiority_margin
                            ),
                            "run_seconds": round(elapsed, 1),
                        }
                    )
                with csv_path.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                print(
                    f"seed={seed} {label}/{model_name}: "
                    f"acc(static/periodic/drift/two)="
                    f"{result.policies['static'].mean_accuracy:.3f}/"
                    f"{periodic.mean_accuracy:.3f}/"
                    f"{result.policies['drift_aware'].mean_accuracy:.3f}/"
                    f"{two.mean_accuracy:.3f} f1(two)={two.mean_f1:.3f} "
                    f"retrains={two.n_retrains}/{periodic.n_retrains} "
                    f"CIpool={ni_pooled['ci_lower']:+.4f} "
                    f"{'PASS' if ni_pooled['non_inferior'] else 'fail'} [{elapsed:.1f}s]",
                    flush=True,
                )

    (out / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "runs": manifest["n_runs"],
        "rows": len(rows),
        "wall_seconds": round(time.perf_counter() - started_all, 1),
        "manifest": manifest,
    }
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(rows)} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
