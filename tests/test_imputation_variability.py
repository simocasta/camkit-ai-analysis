import numpy as np
import pandas as pd
import pytest

from camkit_ai.imputation_variability import (
    assign_bands,
    collect_draws,
    format_draw_matrix,
    pool_predictions,
    pooled_basis_label,
    pooled_estimate_stability,
    predict_draw,
    summarise_band_instability,
    summarise_draw_metrics,
    summarise_patient_draws,
    summarise_variability,
)


class _JitteryModel:
    """Stands in for the AutoPrognosis pipeline.

    Rows flagged as incomplete pick up fresh noise from the global numpy stream
    on every call; the rest are held fixed. The flags are passed in rather than
    inferred from the frame, so these tests exercise the draw collection and the
    seeding rather than pandas' missing-value detection. That the real pipeline
    moves exactly the incomplete rows is an empirical finding about the saved
    artefact, established by comparing two runs, and no stub can stand in for it.
    """

    def __init__(
        self, base: list[float], incomplete: list[bool], scale: float = 0.02
    ) -> None:
        self.base = np.asarray(base, dtype=float)
        self.incomplete = np.asarray(incomplete, dtype=bool)
        self.scale = scale

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        noise = np.random.normal(0.0, self.scale, len(self.base))
        probability = np.clip(
            self.base + np.where(self.incomplete, noise, 0.0), 0.0, 1.0
        )
        return np.column_stack([1.0 - probability, probability])


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bmi": [np.nan, 24.0, np.nan, 31.0],
            "age": [20.0, 35.0, 41.0, 52.0],
            "Injury": [1, 0, 1, 0],
        }
    )


def test_predict_draw_is_reproducible_under_a_fixed_seed() -> None:
    frame = _frame()
    model = _JitteryModel([0.30, 0.50, 0.70, 0.20], [True, False, True, False])
    features = frame.drop(columns=["Injury"])

    first = predict_draw(model, features, seed=11)
    second = predict_draw(model, features, seed=11)
    different = predict_draw(model, features, seed=12)

    assert np.allclose(first, second)
    assert not np.allclose(first, different)


def test_collect_draws_varies_only_the_rows_the_model_varies() -> None:
    """Draw collection must add no variation of its own and lose none either."""
    frame = _frame()
    model = _JitteryModel([0.30, 0.50, 0.70, 0.20], [True, False, True, False])

    y_true, draws = collect_draws(model, frame, "Injury", n_draws=8, base_seed=3)

    assert draws.shape == (8, 4)
    assert y_true.tolist() == [1, 0, 1, 0]
    assert np.ptp(draws[:, 1]) == 0.0
    assert np.ptp(draws[:, 3]) == 0.0
    assert np.ptp(draws[:, 0]) > 0.0
    assert np.ptp(draws[:, 2]) > 0.0


def test_pool_predictions_is_reproducible_and_averages_the_draws() -> None:
    frame = _frame()
    model = _JitteryModel([0.30, 0.50, 0.70, 0.20], [True, False, True, False])
    features = frame.drop(columns=["Injury"])

    pooled = pool_predictions(model, features, n_draws=32, base_seed=5)
    repeated = pool_predictions(model, features, n_draws=32, base_seed=5)
    _, draws = collect_draws(model, frame, "Injury", n_draws=32, base_seed=5)

    assert np.allclose(pooled, repeated)
    assert np.allclose(pooled, draws.mean(axis=0))
    assert pooled[1] == pytest.approx(0.50)
    assert pooled[3] == pytest.approx(0.20)


def test_pool_predictions_is_more_stable_than_a_single_draw() -> None:
    """The point of pooling: the reported value stops depending on the draw."""
    frame = _frame()
    model = _JitteryModel([0.30, 0.50, 0.70, 0.20], [True, False, True, False])
    features = frame.drop(columns=["Injury"])

    singles = np.vstack([predict_draw(model, features, seed) for seed in (1, 2, 3, 4)])
    pooled = np.vstack(
        [pool_predictions(model, features, n_draws=64, base_seed=s) for s in (1, 200, 400, 600)]
    )

    assert np.ptp(pooled[:, 0]) < np.ptp(singles[:, 0])


def test_pool_predictions_rejects_an_empty_draw_count() -> None:
    model = _JitteryModel([0.30], [True])
    with pytest.raises(ValueError, match="at least 1"):
        pool_predictions(model, _frame().drop(columns=["Injury"]), n_draws=0, base_seed=1)


def test_pooled_basis_label_records_how_predictions_were_made() -> None:
    assert pooled_basis_label(None, None) == "single_draw_unseeded"
    assert pooled_basis_label(200, 42) == "pooled_200_draws_seed_42"


def test_assign_bands_matches_the_locked_band_definition() -> None:
    bands = assign_bands(np.array([0.10, 0.29, 0.50, 0.69, 0.95]), 0.29, 0.69)
    assert bands.tolist() == ["Green", "Amber", "Amber", "Red", "Red"]


def test_assign_bands_works_on_a_draw_matrix() -> None:
    draws = np.array([[0.10, 0.80], [0.40, 0.20]])
    assert assign_bands(draws, 0.29, 0.69).tolist() == [
        ["Green", "Red"],
        ["Amber", "Green"],
    ]


def test_summarise_draw_metrics_counts_missed_injuries_in_the_discharge_band() -> None:
    y_true = np.array([1, 0, 1, 0])
    draws = np.array([[0.10, 0.20, 0.80, 0.50]])

    metrics = summarise_draw_metrics(
        y_true, draws, lower=0.29, upper=0.69, dataset="holdout"
    ).iloc[0]

    assert metrics["n_green"] == 2
    assert metrics["n_amber"] == 1
    assert metrics["n_red"] == 1
    assert metrics["missed_injuries_green"] == 1
    assert metrics["red_ppv"] == pytest.approx(1.0)


def test_summarise_patient_draws_flags_only_patients_that_cross_a_cut_point() -> None:
    """Jitter that stays inside a band is not instability; crossing one is."""
    y_true = np.array([0, 1])
    draws = np.array(
        [
            [0.285, 0.50],
            [0.295, 0.52],
            [0.288, 0.48],
        ]
    )

    patients = summarise_patient_draws(
        y_true, draws, lower=0.29, upper=0.69, dataset="prospective"
    )

    assert patients.loc[0, "band_unstable"]
    assert not patients.loc[1, "band_unstable"]
    assert patients.loc[0, "fraction_green"] == pytest.approx(2 / 3)
    assert patients.loc[0, "fraction_amber"] == pytest.approx(1 / 3)
    assert patients.loc[1, "band_modal"] == "Amber"


def test_summarise_band_instability_separates_injuries_ever_discharged() -> None:
    patients = pd.DataFrame(
        {
            "dataset": ["prospective"] * 3,
            "y_true": [1, 0, 1],
            "probability_sd": [0.01, 0.00, 0.02],
            "probability_range": [0.05, 0.00, 0.09],
            "fraction_green": [0.4, 0.0, 0.0],
            "fraction_amber": [0.6, 1.0, 1.0],
            "fraction_red": [0.0, 0.0, 0.0],
            "band_unstable": [True, False, False],
        }
    )

    summary = summarise_band_instability(patients).iloc[0]

    assert summary["n_band_unstable"] == 1
    assert summary["n_band_unstable_with_injury"] == 1
    assert summary["n_injury_ever_green"] == 1
    assert summary["fraction_band_unstable"] == pytest.approx(1 / 3)


def test_summarise_variability_reports_a_spread_across_draws() -> None:
    draw_metrics = pd.DataFrame(
        {
            "dataset": ["holdout"] * 5,
            "draw": range(5),
            "auprc": [0.40, 0.45, 0.50, 0.55, 0.60],
        }
    )

    summary = summarise_variability(draw_metrics, confidence_level=0.95).iloc[0]

    assert summary["median"] == pytest.approx(0.50)
    assert summary["minimum"] == pytest.approx(0.40)
    assert summary["maximum"] == pytest.approx(0.60)
    assert summary["n_draws"] == 5


def test_summarise_variability_ignores_non_finite_draws() -> None:
    draw_metrics = pd.DataFrame(
        {
            "dataset": ["prospective"] * 3,
            "draw": range(3),
            "red_ppv": [np.nan, 0.80, 1.00],
        }
    )

    summary = summarise_variability(draw_metrics, confidence_level=0.95).iloc[0]

    assert summary["n_draws"] == 2
    assert summary["median"] == pytest.approx(0.90)


def test_format_draw_matrix_records_every_draw_with_its_seed() -> None:
    draws = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    y_true = np.array([0, 1])

    matrix = format_draw_matrix(y_true, draws, dataset="holdout", base_seed=42)

    assert len(matrix) == 6
    assert matrix["seed"].tolist() == [42, 42, 43, 43, 44, 44]
    assert matrix["row_id"].tolist() == [1, 2, 1, 2, 1, 2]
    assert matrix["probability"].tolist() == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert matrix["y_true"].tolist() == [0, 1, 0, 1, 0, 1]


def test_pooled_estimate_stability_recovers_the_pooled_mean() -> None:
    # Four draws whose running means are easy to check by hand.
    draws = np.array([[0.0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.6, 0.4]])
    matrix = format_draw_matrix(np.array([0, 1]), draws, dataset="holdout", base_seed=0)

    stability = pooled_estimate_stability(
        matrix,
        {"mean_of_first_patient": lambda yt, pooled: pooled[0]},
        draw_counts=[1, 2, 4, 99],
    )

    values = dict(zip(stability["n_draws"], stability["value"]))
    assert values[1] == pytest.approx(0.0)
    assert values[2] == pytest.approx(0.1)
    assert values[4] == pytest.approx(0.3)
    # A draw count beyond what was collected is skipped, not silently truncated.
    assert 99 not in values
