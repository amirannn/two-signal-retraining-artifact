"""Retraining-timing policies.

Four policies are evaluated side by side, following the proposal:

* ``static`` -- the model is trained once on the first batch and never retrained.
* ``periodic`` -- the model is retrained at every batch (maximum-cost baseline).
* ``drift_aware`` -- single-signal policy of the reference study: retrain when
  the silhouette change ΔS crosses theta2.
* ``two_signal`` -- the proposed policy: a cheap PSI gate (theta1) first; the
  expensive silhouette confirmation (theta2) is only computed when the PSI
  gate fires, and retraining happens only when *both* signals confirm.

No retraining decisions are made during the reference period (the first
``reference_period`` batches), which is used to calibrate the thresholds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .detectors import DETECTOR_FACTORIES, novelty_scores
from .signals import SignalProvider


@dataclass
class DecisionInfo:
    """Signals and alert flags associated with a window decision."""

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


class Policy(ABC):
    """Base class for a retraining-timing policy."""

    name = "policy"

    def __init__(
        self,
        provider: SignalProvider,
        theta1: float,
        theta2: float,
        reference_period: int = 5,
        silhouette_mode: str = "previous",
    ) -> None:
        self.provider = provider
        self.theta1 = theta1
        self.theta2 = theta2
        self.reference_period = reference_period
        self.silhouette_mode = silhouette_mode

    def _delta(self, k: int) -> float | None:
        if self.silhouette_mode == "reference":
            return self.provider.delta_s_reference(k)
        return self.provider.delta_s(k)

    @abstractmethod
    def decide(self, k: int) -> tuple[bool, DecisionInfo]:
        """Return (retrain?, signal info) for batch index ``k`` (0-based)."""
        raise NotImplementedError


class StaticPolicy(Policy):
    """Train once on the first batch; never retrain."""

    name = "static"

    def decide(self, k: int) -> tuple[bool, DecisionInfo]:
        return k == 0, DecisionInfo()


class PeriodicPolicy(Policy):
    """Retrain at every batch (maximum-cost baseline)."""

    name = "periodic"

    def decide(self, k: int) -> tuple[bool, DecisionInfo]:
        return True, DecisionInfo()


class DriftAwarePolicy(Policy):
    """Single-signal policy of the reference study (silhouette only)."""

    name = "drift_aware"

    def decide(self, k: int) -> tuple[bool, DecisionInfo]:
        if k == 0:
            return True, DecisionInfo()
        if k < self.reference_period:
            return False, DecisionInfo()
        delta = self._delta(k)
        info = DecisionInfo(delta_s=delta, silhouette_computed=True)
        info.delta_alert = delta is not None and delta > self.theta2
        return bool(info.delta_alert), info


class DetectorPolicy(Policy):
    """Single-detector baseline using a label-free SignalProvider detector."""

    def __init__(self, kind: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kind = kind
        self.name = f"{kind}_only"

    def decide(self, k: int) -> tuple[bool, DecisionInfo]:
        if k == 0:
            return True, DecisionInfo(detector_name=self.kind)
        if k < self.reference_period:
            return False, DecisionInfo(detector_name=self.kind)
        value = self.provider.detector_value(self.kind, k)
        alert = value is not None and value > self.theta1
        return bool(alert), DecisionInfo(
            detector_name=self.kind,
            detector_value=value,
            detector_alert=alert,
        )


class SequentialDetectorPolicy(Policy):
    """Standard sequential detector (ADWIN / KSWIN) driving retraining.

    The detector consumes the per-sample novelty score of every window with
    respect to the reference period, so it sees the same label-free
    information as the PSI screen and needs no labels.  Windows inside the
    reference period warm the detector up but cannot trigger retraining, which
    keeps the protocol identical to the PSI / Silhouette policies.
    """

    def __init__(self, kind: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if kind not in DETECTOR_FACTORIES:
            raise ValueError(f"unknown sequential detector: {kind!r}")
        self.kind = kind
        self.name = f"{kind}_only"
        self.detector = DETECTOR_FACTORIES[kind](self.provider.seed)
        self.reference_data = np.vstack(
            [
                self.provider.batches[i]
                for i in range(self.provider.reference_period)
            ]
        )

    def decide(self, k: int) -> tuple[bool, DecisionInfo]:
        if k == 0:
            return True, DecisionInfo(detector_name=self.kind)
        scores = novelty_scores(self.reference_data, self.provider.batches[k])
        fired = self.detector.update_many(scores)
        info = DecisionInfo(
            detector_name=self.kind,
            detector_value=float(np.mean(scores)),
            detector_alert=bool(fired),
        )
        if k < self.reference_period:
            # Warm-up: the detector learns the reference regime but cannot act.
            return False, info
        return bool(fired), info


class ScoredTwoSignalPolicy(Policy):
    """Two-signal policy with normalized evidence and a severe-PSI escape hatch."""

    name = "two_signal_scored"

    def __init__(self, *args, score_threshold: float = 1.0, emergency_multiplier: float = 2.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.score_threshold = score_threshold
        self.emergency_multiplier = emergency_multiplier

    def decide(self, k: int) -> tuple[bool, DecisionInfo]:
        if k == 0:
            return True, DecisionInfo()
        if k < self.reference_period:
            return False, DecisionInfo()

        psi = self.provider.psi(k)
        info = DecisionInfo(psi=psi, psi_alert=psi > self.theta1)
        emergency = psi > self.emergency_multiplier * self.theta1
        if emergency:
            info.emergency_trigger = True
            info.fusion_score = psi / max(self.theta1, 1e-12)
            return True, info
        if not info.psi_alert:
            return False, info

        delta = self._delta(k)
        info.delta_s = delta
        info.silhouette_computed = True
        info.delta_alert = delta is not None and delta > self.theta2
        if delta is not None:
            info.fusion_score = 0.5 * (
                psi / max(self.theta1, 1e-12)
                + delta / max(self.theta2, 1e-12)
            )
        retrain = info.fusion_score is not None and info.fusion_score >= self.score_threshold
        return bool(retrain), info


class EvidenceAccumulationPolicy(Policy):
    """Severity-aware policy that accumulates label-free drift evidence over time."""

    name = "evidence_cost_aware"

    def __init__(
        self,
        *args,
        evidence_decay: float = 0.8,
        action_threshold: float = 1.0,
        silhouette_weight: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.evidence_decay = evidence_decay
        self.action_threshold = action_threshold
        self.silhouette_weight = silhouette_weight
        self.evidence = 0.0

    def decide(self, k: int) -> tuple[bool, DecisionInfo]:
        if k == 0:
            self.evidence = 0.0
            return True, DecisionInfo(detector_name="evidence")
        if k < self.reference_period:
            return False, DecisionInfo(detector_name="evidence")

        psi = self.provider.psi(k)
        info = DecisionInfo(
            psi=psi,
            psi_alert=psi > self.theta1,
            detector_name="evidence",
        )
        distribution_evidence = max(0.0, psi / max(self.theta1, 1e-12) - 1.0)
        structural_evidence = 0.0
        if info.psi_alert:
            delta = self._delta(k)
            info.delta_s = delta
            info.silhouette_computed = True
            info.delta_alert = delta is not None and delta > self.theta2
            if delta is not None:
                structural_evidence = max(
                    0.0, delta / max(self.theta2, 1e-12) - 1.0
                )

        increment = distribution_evidence + self.silhouette_weight * structural_evidence
        self.evidence = self.evidence_decay * self.evidence + increment
        info.detector_value = self.evidence
        info.fusion_score = self.evidence
        info.detector_alert = self.evidence >= self.action_threshold
        if info.detector_alert:
            self.evidence = 0.0
            return True, info
        return False, info


class CalibratedSequentialPolicy(Policy):
    """Sequentially combine empirical signal evidence before retraining.

    PSI and silhouette are calibrated against reference-only natural variation;
    their empirical upper-tail p-values are accumulated with a CUSUM-like
    forgetting rule. The silhouette stage remains conditional on a PSI screen,
    and a minimum amount of post-reference data prevents acting on one noisy
    window. This is an experimental extension, not a change to the proposal.
    """

    name = "calibrated_sequential"

    def __init__(
        self,
        *args,
        screen_alpha: float = 0.5,
        evidence_decay: float = 0.8,
        evidence_threshold: float = 3.0,
        evidence_drift: float = 1.0,
        silhouette_weight: float = 1.0,
        min_new_batches: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.screen_alpha = screen_alpha
        self.evidence_decay = evidence_decay
        self.evidence_threshold = evidence_threshold
        self.evidence_drift = evidence_drift
        self.silhouette_weight = silhouette_weight
        self.min_new_batches = min_new_batches
        self.evidence = 0.0
        self.new_batches = 0

    def decide(self, k: int) -> tuple[bool, DecisionInfo]:
        if k == 0:
            self.evidence = 0.0
            self.new_batches = 0
            return True, DecisionInfo(detector_name="calibrated_sequential")
        if k < self.reference_period:
            return False, DecisionInfo(detector_name="calibrated_sequential")

        self.new_batches += 1
        psi = self.provider.psi(k)
        psi_p = self.provider.psi_p_value(k)
        info = DecisionInfo(
            psi=psi,
            psi_alert=psi_p <= self.screen_alpha,
            detector_name="calibrated_sequential",
        )
        # Evidence below this allowance is forgotten rather than accumulated.
        increment = max(0.0, -np.log(max(psi_p, 1e-12)) - self.evidence_drift)
        if info.psi_alert:
            delta_p = self.provider.delta_s_p_value(k)
            delta = self._delta(k)
            info.delta_s = delta
            info.silhouette_computed = True
            info.delta_alert = delta_p is not None and delta_p <= self.screen_alpha
            if delta_p is not None:
                increment += self.silhouette_weight * max(
                    0.0, -np.log(max(delta_p, 1e-12)) - self.evidence_drift
                )
        self.evidence = self.evidence_decay * self.evidence + increment
        info.detector_value = self.evidence
        info.fusion_score = self.evidence
        info.detector_alert = (
            self.evidence >= self.evidence_threshold
            and self.new_batches >= self.min_new_batches
        )
        if info.detector_alert:
            self.evidence = 0.0
            self.new_batches = 0
            return True, info
        return False, info


class TwoSignalPolicy(Policy):
    """Proposed two-stage policy: PSI gate first, silhouette confirmation second.

    The expensive clustering stage is skipped whenever the cheap PSI gate does
    not fire, so the number of silhouette computations is a measurable cost of
    the policy (see ``provider.n_silhouette_computes``).
    """

    name = "two_signal"

    def decide(self, k: int) -> tuple[bool, DecisionInfo]:
        if k == 0:
            return True, DecisionInfo()
        if k < self.reference_period:
            return False, DecisionInfo()

        psi = self.provider.psi(k)
        info = DecisionInfo(psi=psi, psi_alert=psi > self.theta1)
        if not info.psi_alert:
            # Cheap stage fails: skip the expensive clustering stage entirely.
            return False, info

        delta = self._delta(k)
        info.delta_s = delta
        info.silhouette_computed = True
        info.delta_alert = delta is not None and delta > self.theta2
        return bool(info.delta_alert), info