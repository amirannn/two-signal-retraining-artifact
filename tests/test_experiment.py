"""End-to-end experiment tests on synthetic data."""

from __future__ import annotations

from driftpolicy import ExperimentConfig, run_experiment
from driftpolicy.data.synthetic import make_synthetic_pair


def _default_small():
    batches, other_train, other_test = make_synthetic_pair(
        n_batches=25,
        batch_size=32,
        n_features=16,
        seed=42,
        drift_at=(8, 18),
        spike_at=None,
    )
    config = ExperimentConfig()
    config.signal.reference_period = 4
    return run_experiment(batches, other_train, other_test, config)


def test_experiment_runs_all_four_policies():
    result = _default_small()
    assert set(result.policies) == {"static", "periodic", "drift_aware", "two_signal"}
    for p in result.policies.values():
        assert p.n_batches == 25


def test_static_degrades_and_periodic_keeps_up():
    result = _default_small()
    static = result.policies["static"]
    periodic = result.policies["periodic"]
    assert periodic.mean_accuracy > static.mean_accuracy + 0.05
    assert periodic.n_retrains == 25


def test_two_signal_retrains_less_than_periodic_and_keeps_accuracy():
    result = _default_small()
    two = result.policies["two_signal"]
    periodic = result.policies["periodic"]
    assert two.n_retrains < periodic.n_retrains
    assert two.mean_accuracy > periodic.mean_accuracy - 0.05


def test_two_signal_uses_fewer_silhouette_computes_than_drift_aware():
    result = _default_small()
    two = result.policies["two_signal"]
    drift = result.policies["drift_aware"]
    assert two.n_silhouette_computes <= drift.n_silhouette_computes
    assert two.silhouette_distance_pairs <= drift.silhouette_distance_pairs
    assert two.silhouette_compute_rate <= drift.silhouette_compute_rate


def test_report_contains_computational_savings():
    from driftpolicy.report import render_markdown

    report = render_markdown(_default_small())
    assert "گزارش صرفه‌جویی محاسباتی" in report
    assert "دفعات اجرای سیلوئت" in report


def test_standalone_savings_exports(tmp_path):
    import csv
    import json
    from driftpolicy.report import write_savings_summary

    write_savings_summary(_default_small(), tmp_path)
    csv_path = tmp_path / "computational_savings.csv"
    json_path = tmp_path / "computational_savings.json"
    assert csv_path.exists()
    assert json_path.exists()
    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert {row["metric"] for row in rows} == {
        "silhouette_computations",
        "silhouette_compute_rate_per_window",
        "silhouette_distance_pairs_estimate",
    }
    assert len(payload["metrics"]) == len(rows)


def test_non_inferiority_verdict():
    result = _default_small()
    ni = result.non_inferiority("accuracy")
    assert ni["non_inferior"] is True


def test_window_records_contain_decision_log():
    result = _default_small()
    two = result.policies["two_signal"]
    for r in two.records:
        assert r.psi is not None or r.batch < 4
        assert r.psi_alert in (True, False, None)
    assert two.cumulative_retrains()[-1] == two.n_retrains


def test_spike_is_filtered_by_two_signal_but_not_drift_aware():
    """A transient spike should fire PSI but not the silhouette confirmation."""
    batches, other_train, other_test = make_synthetic_pair(
        n_batches=25,
        batch_size=32,
        n_features=16,
        seed=42,
        drift_at=(8, 18),
        spike_at=5,
        spike_features=4,
        spike_scale=3.0,
    )
    config = ExperimentConfig()
    config.signal.reference_period = 4
    result = run_experiment(batches, other_train, other_test, config)
    two = result.policies["two_signal"]
    drift = result.policies["drift_aware"]

    # The spike window (batch 5) should produce a PSI alert ...
    spike_record = two.records[5]
    assert spike_record.psi_alert is True
    # ... that the silhouette stage does not confirm.
    assert spike_record.delta_alert is False
    assert spike_record.retrain is False
    # And the two-signal policy should retrain no more than drift-aware.
    assert two.n_retrains <= drift.n_retrains