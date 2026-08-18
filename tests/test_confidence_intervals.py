import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from camkit_ai.confidence_intervals import (
    EstimateCI,
    bootstrap_metric_bundle,
    bootstrap_paired_difference,
    format_estimate,
    proportion_ci,
)
from camkit_ai.config import ConfidenceIntervalConfig


def test_exact_proportion_ci_handles_perfect_rate() -> None:
    config = ConfidenceIntervalConfig(confidence_level=0.95, proportion_method="exact")
    estimate = proportion_ci(12, 12, config)
    assert estimate.point == 1.0
    assert estimate.lower is not None
    assert estimate.upper == 1.0
    assert estimate.lower < 1.0


def test_infinite_likelihood_ratio_is_displayed_as_not_calculable() -> None:
    estimate = EstimateCI(
        point=float("inf"),
        lower=1.5,
        upper=float("inf"),
        method="lr_exact_propagated",
    )

    assert format_estimate(estimate) == "not calculable"


def test_bootstrap_metric_bundle_returns_intervals() -> None:
    config = ConfidenceIntervalConfig(
        bootstrap_iterations=200,
        confidence_level=0.95,
        random_state=7,
        bootstrap_stratified=True,
    )
    y_true = [0, 0, 0, 1, 1, 1]
    y_prob = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    result = bootstrap_metric_bundle(
        y_true,
        y_prob,
        {"mean_score": lambda yt, yp: float(sum(yp) / len(yp))},
        config=config,
        requires_both_classes=False,
    )["mean_score"]
    assert result.lower is not None
    assert result.upper is not None
    assert result.samples is not None
    assert result.samples > 0



def test_paired_difference_is_zero_when_scores_are_identical() -> None:
    rng = np.random.default_rng(0)
    y_true = np.array([0, 1] * 25)
    score = rng.random(50)

    estimates = bootstrap_paired_difference(
        y_true,
        score,
        score,
        {"auroc": lambda yt, ys: roc_auc_score(yt, ys)},
        config=ConfidenceIntervalConfig(bootstrap_iterations=200, random_state=1),
    )

    # Identical scores must give an exactly zero difference on every resample,
    # not merely an interval that happens to contain zero.
    assert estimates["auroc"].point == 0.0
    assert estimates["auroc"].lower == 0.0
    assert estimates["auroc"].upper == 0.0
    assert estimates["auroc"].method == "paired_bootstrap"


def test_paired_difference_detects_a_clearly_better_model() -> None:
    y_true = np.array([0] * 30 + [1] * 30)
    perfect = np.concatenate([np.linspace(0.0, 0.4, 30), np.linspace(0.6, 1.0, 30)])
    useless = np.tile([0.5, 0.51], 30)

    estimates = bootstrap_paired_difference(
        y_true,
        perfect,
        useless,
        {"auroc": lambda yt, ys: roc_auc_score(yt, ys)},
        config=ConfidenceIntervalConfig(bootstrap_iterations=300, random_state=7),
    )

    assert estimates["auroc"].point > 0.4
    assert estimates["auroc"].lower > 0.0


def test_paired_difference_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="one score per patient"):
        bootstrap_paired_difference(
            np.array([0, 1, 0]),
            np.array([0.1, 0.2, 0.3]),
            np.array([0.1, 0.2]),
            {"auroc": lambda yt, ys: roc_auc_score(yt, ys)},
            config=ConfidenceIntervalConfig(bootstrap_iterations=10),
        )
