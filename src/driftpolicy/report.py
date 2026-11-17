"""Rendering of experiment results: Markdown report, CSV log, plot."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .experiment import POLICY_ORDER, ExperimentResult, PolicyResult


def render_markdown(result: ExperimentResult) -> str:
    """Render a human-readable Markdown report of the experiment."""
    cfg = result.config
    policies = result.policies
    two = policies["two_signal"]
    drift = policies["drift_aware"]

    lines: list[str] = []
    lines.append("# گزارش آزمایش: سیاست بازآموزی دوسیگناله (PSI + Silhouette)")
    lines.append("")
    lines.append("## پیکربندی")
    lines.append("")
    lines.append(f"- تعداد برش‌های زمانی: **{result.n_batches}**")
    lines.append(f"- اندازه برش: **{cfg.batch_size}**")
    lines.append(f"- دوره مرجع: **{cfg.signal.reference_period}** برش نخست")
    lines.append(f"- مدل مبنا: **{cfg.model.name}**")
    lines.append(f"- آستانه θ₁ (PSI): **{result.theta1:.4f}**")
    lines.append(f"- آستانه θ₂ (ΔS): **{result.theta2:.4f}**")
    lines.append("")
    lines.append("## خلاصه سیاست‌ها")
    lines.append("")
    lines.append("| سیاست | بازآموزی‌ها | بسامد بازآموزی | محاسبات سیلوئت | نرخ اجرای سیلوئت | Accuracy | F1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in ("static", "periodic", "drift_aware", "two_signal"):
        p = policies[name]
        lines.append(
            f"| {_policy_label(name)} | {p.n_retrains} | {p.retrain_frequency:.2f} "
            f"| {p.n_silhouette_computes} | {p.silhouette_compute_rate:.2f} "
            f"| {p.mean_accuracy:.3f} | {p.mean_f1:.3f} |"
        )
    lines.append("")

    lines.append("## آزمون عدم‌کاستی (سیاست پیشنهادی در برابر سیاست دوره‌ای)")
    lines.append("")
    lines.append(f"حاشیه عدم‌کاستی δ = {cfg.non_inferiority_margin:.0%}")
    lines.append("")
    lines.append("| سنجه | میانگین اختلاف | بازه اطمینان 95% | نتیجه |")
    lines.append("|---|---|---|---|")
    for metric in ("accuracy", "f1", "balanced_accuracy"):
        ni = result.non_inferiority(metric=metric)
        verdict = "عدم‌کاستی ✓" if ni["non_inferior"] else "عدم‌کاستی ✗"
        lines.append(
            f"| {metric} | {ni['mean_diff']:+.4f} | "
            f"[{ni['ci_lower']:+.4f}, {ni['ci_upper']:+.4f}] | {verdict} |"
        )
    lines.append("")

    lines.append("## تحلیل رفتار سیاست دوسیگنال")
    lines.append("")
    psi_alerts = sum(1 for r in two.records if r.psi_alert)
    confirmed = sum(1 for r in two.records if r.delta_alert)
    lines.append(f"- هشدارهای PSI: **{psi_alerts}**")
    lines.append(f"- هشدارهای تأییدشده با سیلوئت: **{confirmed}**")
    if psi_alerts:
        lines.append(f"- نرخ تأیید هشدارها: **{confirmed / psi_alerts:.2f}**")
    lines.append(f"- بازآموزی‌های نهایی: **{two.n_retrains}**")
    baseline_computes = max(drift.n_silhouette_computes, 1)
    saved_computes = drift.n_silhouette_computes - two.n_silhouette_computes
    saved_pairs = drift.silhouette_distance_pairs - two.silhouette_distance_pairs
    compute_saving = 100 * saved_computes / baseline_computes
    pair_saving = 100 * saved_pairs / max(drift.silhouette_distance_pairs, 1)
    lines.append("### گزارش صرفه‌جویی محاسباتی")
    lines.append("")
    lines.append("این مقایسه فقط هزینهٔ مرحلهٔ پرهزینهٔ سیلوئت را می‌سنجد؛ PSI و آموزش مدل جداگانه‌اند.")
    lines.append("")
    lines.append("| شاخص هزینه | Drift-aware | Two-signal | صرفه‌جویی |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| دفعات اجرای سیلوئت | {drift.n_silhouette_computes} | "
        f"{two.n_silhouette_computes} | {saved_computes} ({compute_saving:.1f}%) |"
    )
    lines.append(
        f"| نرخ اجرا به‌ازای پنجره | {drift.silhouette_compute_rate:.2f} | "
        f"{two.silhouette_compute_rate:.2f} | {drift.silhouette_compute_rate - two.silhouette_compute_rate:.2f} |"
    )
    lines.append(
        f"| زوج فاصلهٔ محاسبه‌شده* | {drift.silhouette_distance_pairs:,} | "
        f"{two.silhouette_distance_pairs:,} | {saved_pairs:,} ({pair_saving:.1f}%) |"
    )
    lines.append("")
    lines.append(
        f"**نتیجه:** سیاست دوسیگنال {compute_saving:.1f}% اجرای سیلوئت کمتری دارد "
        f"({drift.n_silhouette_computes} در برابر {two.n_silhouette_computes}) و "
        f"{pair_saving:.1f}% زوج فاصلهٔ کمتری را وارد مرحلهٔ سیلوئت می‌کند."
    )
    lines.append("")
    lines.append("\u002a هر اجرای سیلوئت روی دو پنجره با اندازهٔ نمونهٔ ثبت‌شده انجام شده و زوج‌های فاصله به‌عنوان تقریب هزینهٔ فاصله‌محور گزارش شده‌اند.")
    lines.append("")

    lines.append("## گزارش تصمیم‌گیری پنجره‌به‌پنجره (سیاست دوسیگنال)")
    lines.append("")
    lines.extend(_window_table(two, include_psi=True))
    lines.append("")

    lines.append("## گزارش تصمیم‌گیری پنجره‌به‌پنجره (سیاست رانش‌آگاه)")
    lines.append("")
    lines.extend(_window_table(drift, include_psi=False))
    lines.append("")

    lines.append("## توضیح‌پذیری: دلایل اقدام / عدم اقدام")
    lines.append("")
    for r in two.records:
        if r.batch < cfg.signal.reference_period:
            continue
        if r.retrain:
            reason = "هر دو سیگنال (PSI و ΔS) از آستانه عبور کردند → بازآموزی"
        elif r.psi_alert:
            reason = "PSI هشدار داد اما تغییر ساختاری (ΔS) تأیید نشد → عدم بازآموزی"
        else:
            reason = "PSI از آستانه عبور نکرد → عدم بازآموزی"
        lines.append(f"- پنجره {r.batch + 1}: {reason}")
    return "\n".join(lines) + "\n"


def _window_table(policy: PolicyResult, include_psi: bool) -> list[str]:
    """Render the per-window decision table for one policy.

    For the two-signal policy (``include_psi=True``) the table shows PSI and
    its alert flag; for the drift-aware policy only the silhouette ΔS signal
    is shown (it is the single signal that drives the decision).
    """
    header = "| پنجره | ΔS | هشدار ΔS | بازآموزی | تجمعی |"
    sep = "|---|---|---|---|---|"
    if include_psi:
        header = "| پنجره | PSI | هشدار PSI | ΔS | هشدار ΔS | بازآموزی | تجمعی |"
        sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    cum = 0
    for r in policy.records:
        if r.retrain:
            cum += 1
        ds_txt = f"{r.delta_s:.4f}" if r.delta_s is not None else "—"
        da = "✓" if r.delta_alert else ("✗" if r.delta_alert is False else "—")
        rt = "✓" if r.retrain else "—"
        if include_psi:
            psi_txt = f"{r.psi:.4f}" if r.psi is not None else "—"
            pa = "✓" if r.psi_alert else ("✗" if r.psi_alert is False else "—")
            lines.append(
                f"| {r.batch + 1} | {psi_txt} | {pa} | {ds_txt} | {da} | {rt} | {cum} |"
            )
        else:
            lines.append(f"| {r.batch + 1} | {ds_txt} | {da} | {rt} | {cum} |")
    return lines


def write_csv(result: ExperimentResult, path: str | Path) -> None:
    """Write per-window records of all policies to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "policy",
        "window",
        "retrain",
        "model_id",
        "psi",
        "delta_s",
        "psi_alert",
        "delta_alert",
        "silhouette_computed",
        "accuracy",
        "f1",
        "balanced_accuracy",
        "cumulative_retrains",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for name in POLICY_ORDER:
            p = result.policies[name]
            cum = 0
            for r in p.records:
                if r.retrain:
                    cum += 1
                writer.writerow(
                    {
                        "policy": name,
                        "window": r.batch + 1,
                        "retrain": int(r.retrain),
                        "model_id": r.model_id,
                        "psi": "" if r.psi is None else f"{r.psi:.6f}",
                        "delta_s": "" if r.delta_s is None else f"{r.delta_s:.6f}",
                        "psi_alert": "" if r.psi_alert is None else int(r.psi_alert),
                        "delta_alert": "" if r.delta_alert is None else int(r.delta_alert),
                        "silhouette_computed": int(r.silhouette_computed),
                        "accuracy": f"{r.accuracy:.6f}",
                        "f1": f"{r.f1:.6f}",
                        "balanced_accuracy": f"{r.balanced_accuracy:.6f}",
                        "cumulative_retrains": cum,
                    }
                )


def _computational_savings_rows(result: ExperimentResult) -> list[dict[str, object]]:
    """Return machine-readable computational-savings metrics."""
    baseline = result.policies["drift_aware"]
    proposed = result.policies["two_signal"]
    baseline_computes = baseline.n_silhouette_computes
    proposed_computes = proposed.n_silhouette_computes
    baseline_pairs = baseline.silhouette_distance_pairs
    proposed_pairs = proposed.silhouette_distance_pairs

    def saving(current: int, reference: int) -> tuple[int, float]:
        amount = reference - current
        percent = 100.0 * amount / reference if reference else 0.0
        return amount, percent

    compute_amount, compute_percent = saving(proposed_computes, baseline_computes)
    pair_amount, pair_percent = saving(proposed_pairs, baseline_pairs)
    return [
        {
            "metric": "silhouette_computations",
            "baseline_policy": "drift_aware",
            "proposed_policy": "two_signal",
            "baseline_value": baseline_computes,
            "proposed_value": proposed_computes,
            "savings_value": compute_amount,
            "savings_percent": compute_percent,
        },
        {
            "metric": "silhouette_compute_rate_per_window",
            "baseline_policy": "drift_aware",
            "proposed_policy": "two_signal",
            "baseline_value": baseline.silhouette_compute_rate,
            "proposed_value": proposed.silhouette_compute_rate,
            "savings_value": baseline.silhouette_compute_rate - proposed.silhouette_compute_rate,
            "savings_percent": 100.0 * (baseline.silhouette_compute_rate - proposed.silhouette_compute_rate) / baseline.silhouette_compute_rate if baseline.silhouette_compute_rate else 0.0,
        },
        {
            "metric": "silhouette_distance_pairs_estimate",
            "baseline_policy": "drift_aware",
            "proposed_policy": "two_signal",
            "baseline_value": baseline_pairs,
            "proposed_value": proposed_pairs,
            "savings_value": pair_amount,
            "savings_percent": pair_percent,
        },
    ]


def write_savings_summary(result: ExperimentResult, out_dir: str | Path) -> None:
    """Write standalone computational-savings CSV and JSON summaries."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _computational_savings_rows(result)
    fields = [
        "metric", "baseline_policy", "proposed_policy", "baseline_value",
        "proposed_value", "savings_value", "savings_percent",
    ]
    with open(out_dir / "computational_savings.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "comparison": {
            "baseline_policy": "drift_aware",
            "proposed_policy": "two_signal",
            "n_batches": result.n_batches,
            "reference_period": result.config.signal.reference_period,
        },
        "metrics": rows,
    }
    (out_dir / "computational_savings.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def plot_results(result: ExperimentResult, path: str | Path) -> None:
    """Plot per-window accuracy for all policies plus retraining markers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = np.arange(result.n_batches)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"static": "#d62728", "periodic": "#1f77b4", "drift_aware": "#ff7f0e", "two_signal": "#2ca02c"}
    for name in POLICY_ORDER:
        p = result.policies[name]
        ax.plot(xs, p.per_window("accuracy"), marker="o", ms=3, label=_policy_label(name), color=colors[name])
        retrain_at = [r.batch for r in p.records if r.retrain]
        if retrain_at:
            ax.scatter(retrain_at, [p.records[r].accuracy for r in retrain_at], marker="^", s=60, color=colors[name], zorder=5)
    ax.axvspan(0, result.config.signal.reference_period - 0.5, color="gray", alpha=0.15, label="reference period")
    ax.set_xlabel("زمان (پنجره)")
    ax.set_ylabel("Accuracy")
    ax.set_title("دقت هر سیاست در طول زمان (مثلث‌ها = بازآموزی)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _policy_label(name: str) -> str:
    return {
        "static": "Static (ایستا)",
        "periodic": "Periodic (دوره‌ای)",
        "drift_aware": "Drift-aware (رانش‌آگاه)",
        "two_signal": "Two-signal (پیشنهادی)",
    }[name]