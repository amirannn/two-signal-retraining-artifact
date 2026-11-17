"""Experiment orchestration: run all policies over temporal batches."""

from __future__ import annotations

from dataclasses import dataclass, field

import time

import numpy as np

from .config import ExperimentConfig
from .data.batches import TemporalBatches
from .evaluation.metrics import compute_metrics
from .evaluation.noninferiority import non_inferiority_test
from .models import build_model
from .policies import (
    DriftAwarePolicy,
    PeriodicPolicy,
    Policy,
    StaticPolicy,
    TwoSignalPolicy,
)
from .signals import SignalProvider

POLICY_ORDER = ["static", "periodic", "drift_aware", "two_signal"]


@dataclass
class WindowRecord:
    """Everything logged for one time window (batch)."""

    batch: int
    retrain: bool
    model_id: int
    psi: float | None = None
    delta_s: float | None = None
    psi_alert: bool | None = None
    delta_alert: bool | None = None
    silhouette_computed: bool = False
    detector_name: str | None = None
    detector_value: float | None = None
    detector_alert: bool | None = None
    fusion_score: float | None = None
    emergency_trigger: bool = False
    accuracy: float = 0.0
    f1: float = 0.0
    balanced_accuracy: float = 0.0
    decision_seconds: float = 0.0
    retrain_seconds: float = 0.0
    inference_seconds: float = 0.0


@dataclass
class PolicyResult:
    """Outcome of running one policy over all batches."""

    name: str
    records: list[WindowRecord]
    n_models: int
    n_silhouette_computes: int
    silhouette_sample_count: int = 0
    # Per-window boolean correctness vectors, aligned per sample with every other
    # policy (all policies are scored on the identical window test sets).  Only
    # populated when the experiment is run with ``collect_predictions=True``;
    # used for the pooled sample-level bootstrap of the accuracy difference.
    predictions: list[np.ndarray] = field(default_factory=list)

    @property
    def n_batches(self) -> int:
        return len(self.records)

    @property
    def n_retrains(self) -> int:
        return sum(1 for r in self.records if r.retrain)

    @property
    def retrain_frequency(self) -> float:
        """Relative retraining frequency (retrains / windows)."""
        return self.n_retrains / self.n_batches if self.n_batches else 0.0

    @property
    def silhouette_compute_rate(self) -> float:
        """Silhouette computations per eligible post-reference window."""
        eligible = max(self.n_batches - 2, 0)
        return self.n_silhouette_computes / eligible if eligible else 0.0

    @property
    def silhouette_distance_pairs(self) -> int:
        """Number of pairwise distances implied by recorded silhouette calls."""
        return self.silhouette_sample_count * max(self.silhouette_sample_count - 1, 0) // 2

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean([r.accuracy for r in self.records]))

    @property
    def mean_f1(self) -> float:
        return float(np.mean([r.f1 for r in self.records]))

    @property
    def mean_balanced_accuracy(self) -> float:
        return float(np.mean([r.balanced_accuracy for r in self.records]))

    def per_window(self, key: str) -> list[float]:
        return [float(getattr(r, key)) for r in self.records]

    @property
    def total_decision_seconds(self) -> float:
        return float(sum(r.decision_seconds for r in self.records))

    @property
    def total_retrain_seconds(self) -> float:
        return float(sum(r.retrain_seconds for r in self.records))

    @property
    def total_inference_seconds(self) -> float:
        return float(sum(r.inference_seconds for r in self.records))

    @property
    def total_runtime_seconds(self) -> float:
        return self.total_decision_seconds + self.total_retrain_seconds + self.total_inference_seconds

    def cumulative_retrains(self) -> list[int]:
        total = 0
        out = []
        for r in self.records:
            if r.retrain:
                total += 1
            out.append(total)
        return out


@dataclass
class ExperimentResult:
    """Full result of a four-policy experiment."""

    config: ExperimentConfig
    theta1: float
    theta2: float
    policies: dict[str, PolicyResult] = field(default_factory=dict)
    n_batches: int = 0

    def non_inferiority(
        self,
        metric: str = "accuracy",
        proposed: str = "two_signal",
        baseline: str = "periodic",
    ) -> dict:
        """Paired non-inferiority of ``proposed`` vs ``baseline`` on ``metric``."""
        prop = self.policies[proposed].per_window(metric)
        base = self.policies[baseline].per_window(metric)
        return non_inferiority_test(
            prop,
            base,
            margin=self.config.non_inferiority_margin,
            n_boot=self.config.bootstrap_iters,
            seed=self.config.seed,
        )


def _make_provider(batches: TemporalBatches, config: ExperimentConfig) -> SignalProvider:
    sig = config.signal
    return SignalProvider(
        batches.batches,
        reference_period=sig.reference_period,
        psi_bins=sig.psi_bins,
        psi_agg=sig.psi_agg,
        n_clusters=sig.n_clusters,
        minibatch_size=sig.minibatch_size,
        seed=sig.seed,
    )


def _build_policies(
    batches: TemporalBatches,
    config: ExperimentConfig,
    theta1: float,
    theta2: float,
    drift_theta2: float | None = None,
) -> dict[str, Policy]:
    sig = config.signal
    policies: dict[str, Policy] = {}
    drift_theta2 = theta2 if drift_theta2 is None else drift_theta2
    for cls in (StaticPolicy, PeriodicPolicy, DriftAwarePolicy, TwoSignalPolicy):
        provider = _make_provider(batches, config)
        policy_theta2 = drift_theta2 if cls is DriftAwarePolicy else theta2
        policies[cls.name] = cls(
            provider,
            theta1,
            policy_theta2,
            reference_period=sig.reference_period,
            silhouette_mode=sig.silhouette_mode,
        )
    return policies


def _positive_training_data(
    batches: TemporalBatches, k: int, config: ExperimentConfig
) -> np.ndarray:
    """Return current plus optional historical positive-class training data."""
    current = batches.train_half(k)
    training = config.training_data
    if training.mode == "current" or k == 0:
        return current

    if training.mode == "rolling":
        first = max(0, k - training.rolling_batches + 1)
        return np.vstack([batches.train_half(i) for i in range(first, k + 1)])

    if training.mode == "reservoir":
        previous = np.vstack([batches.train_half(i) for i in range(k)])
        if len(previous) > training.reservoir_size:
            rng = np.random.default_rng(config.seed + k)
            selected = rng.choice(len(previous), size=training.reservoir_size, replace=False)
            previous = previous[np.sort(selected)]
        return np.vstack([previous, current])

    raise ValueError(f"unknown training data mode: {training.mode!r}")


def run_policy(
    policy: Policy,
    batches: TemporalBatches,
    other_train: np.ndarray,
    other_test: np.ndarray,
    config: ExperimentConfig,
    collect_predictions: bool = False,
) -> PolicyResult:
    """Run a single policy over all batches and log every window decision."""
    records: list[WindowRecord] = []
    predictions: list[np.ndarray] = []
    model = None
    model_id = 0
    for k in range(batches.n_batches):
        decision_started = time.perf_counter()
        retrain, info = policy.decide(k)
        decision_seconds = time.perf_counter() - decision_started
        retrain_seconds = 0.0
        if retrain:
            retrain_started = time.perf_counter()
            positive_train = _positive_training_data(batches, k, config)
            X_train = np.vstack([positive_train, other_train])
            y_train = np.array(
                [1] * len(positive_train) + [0] * len(other_train)
            )
            model = build_model(
                config.model.name, config.model.params, seed=config.seed
            )
            model.fit(X_train, y_train)
            retrain_seconds = time.perf_counter() - retrain_started
            model_id += 1

        X_test = np.vstack([batches.test_half(k), other_test])
        inference_started = time.perf_counter()
        y_true = np.array([1] * len(batches.test_half(k)) + [0] * len(other_test))
        y_pred = model.predict(X_test)
        inference_seconds = time.perf_counter() - inference_started
        metrics = compute_metrics(y_true, y_pred)
        if collect_predictions:
            predictions.append((y_pred == y_true).astype(np.int8))

        records.append(
            WindowRecord(
                batch=k,
                retrain=retrain,
                model_id=model_id,
                psi=info.psi,
                delta_s=info.delta_s,
                psi_alert=info.psi_alert,
                delta_alert=info.delta_alert,
                silhouette_computed=info.silhouette_computed,
                detector_name=info.detector_name,
                detector_value=info.detector_value,
                detector_alert=info.detector_alert,
                fusion_score=info.fusion_score,
                emergency_trigger=info.emergency_trigger,
                accuracy=metrics["accuracy"],
                f1=metrics["f1"],
                balanced_accuracy=metrics["balanced_accuracy"],
                decision_seconds=decision_seconds,
                retrain_seconds=retrain_seconds,
                inference_seconds=inference_seconds,
            )
        )
    return PolicyResult(
        name=policy.name,
        records=records,
        n_models=model_id,
        n_silhouette_computes=policy.provider.n_silhouette_computes,
        silhouette_sample_count=policy.provider.silhouette_sample_count,
        predictions=predictions,
    )


def run_experiment(
    batches: TemporalBatches,
    other_train: np.ndarray,
    other_test: np.ndarray,
    config: ExperimentConfig | None = None,
    collect_predictions: bool = False,
) -> ExperimentResult:
    """Run the four policies and return the full result.

    Thresholds are calibrated once from the reference period, then each policy
    runs with its own signal provider (so silhouette-computation counts are
    per-policy).
    """
    config = config or ExperimentConfig()
    sig = config.signal

    calibration = _make_provider(batches, config)
    theta1, theta2 = calibration.calibrate_thresholds(
        method=sig.threshold_method,
        percentile=sig.threshold_percentile,
        robust_k=sig.robust_k,
        default_psi=sig.default_psi_threshold,
        default_delta=sig.default_delta_threshold,
    )
    if sig.fixed_thresholds:
        theta2 = sig.reference_drift_threshold
    drift_theta2 = (
        sig.reference_drift_threshold
        if sig.use_reference_threshold_for_drift_aware
        else theta2
    )

    policies = _build_policies(batches, config, theta1, theta2, drift_theta2=drift_theta2)
    results: dict[str, PolicyResult] = {}
    for name in POLICY_ORDER:
        policy = policies[name]
        results[name] = run_policy(
            policy, batches, other_train, other_test, config,
            collect_predictions=collect_predictions,
        )

    return ExperimentResult(
        config=config,
        theta1=theta1,
        theta2=theta2,
        policies=results,
        n_batches=batches.n_batches,
    )