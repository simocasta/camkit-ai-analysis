"""Tests for comparing an unnormalised score with a predicted probability.

The original CamKIT score runs 0 to 12. Comparing it with CamKIT-AI's predicted
probability needs rank-based metrics only: Brier score and calibration are
undefined for a score outside [0, 1], and scikit-learn rejects it outright.
These tests pin the property that makes the comparison safe — the two AUCs are
invariant to any increasing transformation of the score, so no reported number
depends on whether the score is rescaled.
"""

from __future__ import annotations

import numpy as np
import pytest

from camkit_ai.metrics import discrimination_metrics, rank_discrimination_metrics


def _cohort() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    y_true = np.repeat([0, 1], [67, 18])
    # Injured patients score higher on average, but the bands overlap.
    score = np.where(y_true == 1, rng.integers(4, 13, 85), rng.integers(0, 9, 85))
    return y_true, score.astype(float)


def test_rank_metrics_accept_a_score_outside_the_unit_interval() -> None:
    y_true, score = _cohort()
    assert score.max() > 1.0

    metrics = rank_discrimination_metrics(y_true, score)

    assert np.isfinite(metrics["auprc"])
    assert np.isfinite(metrics["auroc"])


def test_full_metric_bundle_rejects_the_same_score() -> None:
    # Pinned so the routing cannot silently revert to the full metric bundle.
    y_true, score = _cohort()

    with pytest.raises(ValueError):
        discrimination_metrics(y_true, score)


def test_rank_metrics_are_invariant_to_rescaling() -> None:
    # Dividing the CamKIT score by 12 leaves both AUCs bit-identical.
    y_true, score = _cohort()

    raw = rank_discrimination_metrics(y_true, score)
    scaled = rank_discrimination_metrics(y_true, score / 12.0)

    assert raw["auprc"] == scaled["auprc"]
    assert raw["auroc"] == scaled["auroc"]


def test_rank_metrics_are_invariant_to_any_increasing_transform() -> None:
    y_true, score = _cohort()

    raw = rank_discrimination_metrics(y_true, score)
    shifted = rank_discrimination_metrics(y_true, 3.0 * score + 7.0)

    assert raw["auprc"] == pytest.approx(shifted["auprc"])
    assert raw["auroc"] == pytest.approx(shifted["auroc"])


def test_rank_metrics_agree_with_the_full_bundle_for_probabilities() -> None:
    # Where both are valid, they must give the same answer, so switching the
    # comparator over cannot move the CamKIT-AI estimates.
    rng = np.random.default_rng(3)
    y_true = np.repeat([0, 1], [67, 18])
    probabilities = rng.uniform(0.0, 1.0, 85)

    rank = rank_discrimination_metrics(y_true, probabilities)
    full = discrimination_metrics(y_true, probabilities)

    assert rank["auprc"] == pytest.approx(full["auprc"])
    assert rank["auroc"] == pytest.approx(full["auroc"])


def test_rank_metrics_return_nan_for_a_single_class() -> None:
    # Bootstrap resamples can be degenerate; the caller filters on NaN.
    metrics = rank_discrimination_metrics(np.zeros(10, dtype=int), np.arange(10.0))

    assert np.isnan(metrics["auprc"])
    assert np.isnan(metrics["auroc"])
