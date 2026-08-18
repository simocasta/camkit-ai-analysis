import numpy as np
import pandas as pd

from camkit_ai.config import ConfidenceIntervalConfig
from camkit_ai.shift import (
    _resample_to_prevalence,
    decompose_discrimination_shift,
    decompose_from_prediction_summary,
    load_split_predictions,
)


def _config() -> ConfidenceIntervalConfig:
    return ConfidenceIntervalConfig(bootstrap_iterations=200, random_state=7)


def test_resample_to_prevalence_hits_the_requested_class_mix() -> None:
    y_true = np.array([0] * 80 + [1] * 20)
    y_prob = np.linspace(0.0, 1.0, 100)
    rng = np.random.default_rng(0)

    y_boot, prob_boot = _resample_to_prevalence(y_true, y_prob, 0.40, 100, rng)

    assert len(y_boot) == 100
    assert len(prob_boot) == 100
    assert y_boot.sum() == 40


def test_resample_to_prevalence_keeps_within_class_scores() -> None:
    """Cases and controls are drawn separately, so class-conditional scores are preserved."""
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.concatenate([np.full(50, 0.1), np.full(50, 0.9)])
    rng = np.random.default_rng(1)

    y_boot, prob_boot = _resample_to_prevalence(y_true, y_prob, 0.25, 100, rng)

    assert set(prob_boot[y_boot == 0]) == {0.1}
    assert set(prob_boot[y_boot == 1]) == {0.9}


def test_decompose_attributes_nothing_to_prevalence_when_ranking_is_identical() -> None:
    """Same score-vs-outcome ranking in both cohorts leaves a residual gap near zero."""
    rng = np.random.default_rng(3)
    holdout_y = np.array([0] * 70 + [1] * 30)
    holdout_prob = np.where(holdout_y == 1, rng.uniform(0.6, 0.9, 100), rng.uniform(0.1, 0.4, 100))
    prospective_y = np.array([0] * 80 + [1] * 20)
    prospective_prob = np.where(
        prospective_y == 1, rng.uniform(0.6, 0.9, 100), rng.uniform(0.1, 0.4, 100)
    )

    result = decompose_discrimination_shift(
        holdout_y,
        holdout_prob,
        prospective_y,
        prospective_prob,
        _config(),
        model_label="Test.model",
    )

    auroc_residual = result[
        (result["metric"] == "auroc") & (result["quantity"] == "gap_residual")
    ].iloc[0]
    assert abs(auroc_residual["point"]) < 0.05
    assert auroc_residual["ci_lower"] < 0.0 < auroc_residual["ci_upper"]


def test_decompose_assigns_auprc_loss_to_prevalence_when_only_prevalence_changes() -> None:
    """A pure prevalence drop moves average precision (AP) but not AUC-ROC.

    The class-conditional score distributions overlap, because perfectly
    separated classes give AP 1.0 at every prevalence and the effect under
    test would vanish.
    """
    rng = np.random.default_rng(5)

    def cohort(n_pos: int, n_neg: int) -> tuple[np.ndarray, np.ndarray]:
        y = np.array([1] * n_pos + [0] * n_neg)
        prob = np.concatenate(
            [rng.uniform(0.30, 1.00, n_pos), rng.uniform(0.00, 0.70, n_neg)]
        )
        return y, prob

    holdout_y, holdout_prob = cohort(45, 55)
    prospective_y, prospective_prob = cohort(5, 95)

    result = decompose_discrimination_shift(
        holdout_y,
        holdout_prob,
        prospective_y,
        prospective_prob,
        _config(),
        model_label="Test.model",
    )

    auprc = result[result["metric"] == "auprc"].set_index("quantity")["point"]
    auroc = result[result["metric"] == "auroc"].set_index("quantity")["point"]

    assert auprc["gap_prevalence_attributable"] > 0.10
    assert abs(auroc["gap_prevalence_attributable"]) < 0.05


def test_decompose_gaps_are_internally_consistent() -> None:
    rng = np.random.default_rng(11)
    holdout_y = rng.integers(0, 2, 90)
    holdout_prob = rng.uniform(0.0, 1.0, 90)
    prospective_y = rng.integers(0, 2, 70)
    prospective_prob = rng.uniform(0.0, 1.0, 70)

    result = decompose_discrimination_shift(
        holdout_y,
        holdout_prob,
        prospective_y,
        prospective_prob,
        _config(),
        model_label="Test.model",
    )

    for metric in ("auprc", "auroc"):
        points = result[result["metric"] == metric].set_index("quantity")["point"]
        assert np.isclose(
            points["gap_total"],
            points["gap_prevalence_attributable"] + points["gap_residual"],
        )


def test_load_split_predictions_and_summary_roundtrip() -> None:
    frame = pd.DataFrame(
        {
            "model": ["Injury.top12"] * 8,
            "dataset": ["holdout"] * 4 + ["prospective"] * 4,
            "y_true": [0, 1, 0, 1, 0, 1, 0, 1],
            "y_probability": [0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.15, 0.75],
        }
    )

    y_true, y_prob = load_split_predictions(frame, "Injury.top12", "holdout")
    assert list(y_true) == [0, 1, 0, 1]
    assert list(y_prob) == [0.1, 0.8, 0.2, 0.7]

    decomposition = decompose_from_prediction_summary(
        frame,
        ConfidenceIntervalConfig(bootstrap_iterations=25, random_state=2),
        model_labels=["Injury.top12"],
    )
    assert set(decomposition["metric"]) == {"auprc", "auroc"}
    assert decomposition["model"].unique().tolist() == ["Injury.top12"]
