from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import beta

from camkit_ai.config import ConfidenceIntervalConfig


@dataclass
class EstimateCI:
    point: float
    lower: float | None
    upper: float | None
    method: str
    samples: int | None = None
    skipped: int | None = None
    numerator: int | None = None
    denominator: int | None = None

    def as_record(self, metric: str, **extra: object) -> dict[str, object]:
        record = {
            "metric": metric,
            "point": self.point,
            "ci_lower": self.lower,
            "ci_upper": self.upper,
            "ci_method": self.method,
            "ci_samples": self.samples,
            "ci_skipped": self.skipped,
            "numerator": self.numerator,
            "denominator": self.denominator,
        }
        record.update(extra)
        return record


def _nan_estimate(method: str, point: float = float("nan")) -> EstimateCI:
    return EstimateCI(point=point, lower=None, upper=None, method=method)


def format_estimate(estimate: EstimateCI, digits: int = 3) -> str:
    finite_parts = [estimate.point]
    if estimate.lower is not None:
        finite_parts.append(estimate.lower)
    if estimate.upper is not None:
        finite_parts.append(estimate.upper)
    if any(np.isinf(value) for value in finite_parts):
        return "not calculable"

    def fmt(value: float) -> str:
        return f"{value:.{digits}f}"

    if estimate.lower is None or estimate.upper is None or np.isnan(estimate.point):
        return fmt(estimate.point) if np.isfinite(estimate.point) else "NA"
    return (
        f"{fmt(estimate.point)} "
        f"({fmt(estimate.lower)} to {fmt(estimate.upper)})"
    )


def proportion_ci(
    successes: int,
    total: int,
    config: ConfidenceIntervalConfig,
) -> EstimateCI:
    if total <= 0:
        return _nan_estimate(method=f"proportion_{config.proportion_method}")

    point = successes / total
    alpha = 1.0 - config.confidence_level
    method = config.proportion_method.lower()

    if method == "wilson":
        z = 1.959963984540054
        denominator = 1.0 + (z * z) / total
        center = (point + (z * z) / (2.0 * total)) / denominator
        margin = (
            z
            * np.sqrt((point * (1.0 - point) + (z * z) / (4.0 * total)) / total)
            / denominator
        )
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
    elif method == "exact":
        lower = 0.0 if successes == 0 else beta.ppf(alpha / 2.0, successes, total - successes + 1)
        upper = 1.0 if successes == total else beta.ppf(
            1.0 - alpha / 2.0, successes + 1, total - successes
        )
    else:
        raise ValueError(f"Unsupported proportion CI method '{config.proportion_method}'.")

    return EstimateCI(
        point=point,
        lower=float(lower),
        upper=float(upper),
        method=f"proportion_{method}",
        numerator=successes,
        denominator=total,
    )


def likelihood_ratio_ci(
    *,
    sensitivity: EstimateCI,
    specificity: EstimateCI,
    metric: str,
    point: float,
) -> EstimateCI:
    if sensitivity.lower is None or sensitivity.upper is None:
        return _nan_estimate("lr_exact_propagated", point=point)
    if specificity.lower is None or specificity.upper is None:
        return _nan_estimate("lr_exact_propagated", point=point)

    if metric == "lr_plus":
        lower_denominator = 1.0 - specificity.lower
        upper_denominator = 1.0 - specificity.upper
        lower = sensitivity.lower / lower_denominator if lower_denominator > 0.0 else float("inf")
        upper = sensitivity.upper / upper_denominator if upper_denominator > 0.0 else float("inf")
    elif metric == "lr_minus":
        lower_numerator = max(0.0, 1.0 - sensitivity.upper)
        upper_numerator = max(0.0, 1.0 - sensitivity.lower)
        lower = lower_numerator / specificity.upper if specificity.upper > 0.0 else float("inf")
        upper = upper_numerator / specificity.lower if specificity.lower > 0.0 else float("inf")
    else:
        raise ValueError(f"Unsupported likelihood-ratio metric '{metric}'.")

    return EstimateCI(
        point=float(point),
        lower=float(lower),
        upper=float(upper),
        method="lr_exact_propagated",
    )


MetricFn = Callable[[np.ndarray, np.ndarray], float | None]


def _bootstrap_indices(
    y_true: np.ndarray,
    rng: np.random.Generator,
    stratified: bool,
) -> np.ndarray:
    if not stratified:
        return rng.integers(0, len(y_true), size=len(y_true))

    indices: list[np.ndarray] = []
    for label in np.unique(y_true):
        label_idx = np.flatnonzero(y_true == label)
        sampled = rng.choice(label_idx, size=len(label_idx), replace=True)
        indices.append(sampled)
    combined = np.concatenate(indices)
    rng.shuffle(combined)
    return combined


def bootstrap_metric_bundle(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_functions: dict[str, MetricFn],
    config: ConfidenceIntervalConfig,
    requires_both_classes: bool = True,
) -> dict[str, EstimateCI]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    point_estimates = {
        name: func(y_true, y_score) for name, func in metric_functions.items()
    }
    samples: dict[str, list[float]] = {name: [] for name in metric_functions}
    skipped = 0
    rng = np.random.default_rng(config.random_state)

    for _ in range(config.bootstrap_iterations):
        sample_idx = _bootstrap_indices(y_true, rng, config.bootstrap_stratified)
        y_boot = y_true[sample_idx]
        if requires_both_classes and len(np.unique(y_boot)) < 2:
            skipped += 1
            continue
        score_boot = y_score[sample_idx]
        valid_sample = False
        for name, func in metric_functions.items():
            value = func(y_boot, score_boot)
            if value is None or not np.isfinite(value):
                continue
            samples[name].append(float(value))
            valid_sample = True
        if not valid_sample:
            skipped += 1

    alpha = 1.0 - config.confidence_level
    results: dict[str, EstimateCI] = {}
    for name, point in point_estimates.items():
        values = samples[name]
        if not values:
            results[name] = EstimateCI(
                point=float("nan") if point is None else float(point),
                lower=None,
                upper=None,
                method="bootstrap",
                samples=0,
                skipped=skipped,
            )
            continue
        lower = float(np.quantile(values, alpha / 2.0))
        upper = float(np.quantile(values, 1.0 - alpha / 2.0))
        results[name] = EstimateCI(
            point=float("nan") if point is None else float(point),
            lower=lower,
            upper=upper,
            method="bootstrap",
            samples=len(values),
            skipped=skipped,
        )
    return results


def bootstrap_paired_difference(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    metric_functions: dict[str, MetricFn],
    config: ConfidenceIntervalConfig,
    requires_both_classes: bool = True,
) -> dict[str, EstimateCI]:
    """Confidence interval for ``metric(a) - metric(b)`` on shared resamples.

    Two separately computed intervals cannot answer whether two models differ.
    They discard the correlation induced by scoring the same patients, and
    overlapping intervals are routinely compatible with a difference that
    excludes zero — so comparing them by eye understates the evidence in both
    directions. Drawing one patient resample per iteration and evaluating both
    models on it keeps the pairing, which is what makes the difference
    interpretable.

    ``samples`` counts the usable resamples and ``skipped`` those discarded for
    having one outcome class or a non-finite metric, which matters at this event
    count: with 18 prospective events, average precision (AP) is fragile enough that the number
    discarded is part of reading the result.
    """
    y_true = np.asarray(y_true)
    y_score_a = np.asarray(y_score_a)
    y_score_b = np.asarray(y_score_b)
    if not (len(y_true) == len(y_score_a) == len(y_score_b)):
        raise ValueError(
            "Paired comparison needs one score per patient from each model."
        )

    point_differences = {
        name: func(y_true, y_score_a) - func(y_true, y_score_b)
        for name, func in metric_functions.items()
    }
    samples: dict[str, list[float]] = {name: [] for name in metric_functions}
    skipped = 0
    rng = np.random.default_rng(config.random_state)

    for _ in range(config.bootstrap_iterations):
        sample_idx = _bootstrap_indices(y_true, rng, config.bootstrap_stratified)
        y_boot = y_true[sample_idx]
        if requires_both_classes and len(np.unique(y_boot)) < 2:
            skipped += 1
            continue
        a_boot = y_score_a[sample_idx]
        b_boot = y_score_b[sample_idx]
        valid_sample = False
        for name, func in metric_functions.items():
            difference = func(y_boot, a_boot) - func(y_boot, b_boot)
            if difference is None or not np.isfinite(difference):
                continue
            samples[name].append(float(difference))
            valid_sample = True
        if not valid_sample:
            skipped += 1

    alpha = 1.0 - config.confidence_level
    results: dict[str, EstimateCI] = {}
    for name, point in point_differences.items():
        values = samples[name]
        if not values:
            results[name] = EstimateCI(
                point=float(point),
                lower=None,
                upper=None,
                method="paired_bootstrap",
                samples=0,
                skipped=skipped,
            )
            continue
        results[name] = EstimateCI(
            point=float(point),
            lower=float(np.quantile(values, alpha / 2.0)),
            upper=float(np.quantile(values, 1.0 - alpha / 2.0)),
            method="paired_bootstrap",
            samples=len(values),
            skipped=skipped,
        )
    return results
