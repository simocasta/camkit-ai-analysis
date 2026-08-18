import pandas as pd

from camkit_ai.config import ConfidenceIntervalConfig, ThresholdConfig
from camkit_ai.thresholds import (
    bootstrap_threshold_stability,
    derive_thresholds_from_predictions,
    evaluate_locked_threshold_pair,
    evaluate_threshold_pair,
    select_safety_first_thresholds,
    summarize_threshold_stability,
)


def test_select_safety_first_thresholds_prefers_safe_lower_and_high_ppv_upper() -> None:
    sweep = pd.DataFrame(
        [
            {"threshold": 0.20, "npv": 0.94, "discharge_rate": 0.30, "lr_minus": 0.20, "ppv": 0.60, "lr_plus": 3.0, "net_benefit_lower": 0.20, "net_benefit_upper": 0.10},
            {"threshold": 0.29, "npv": 0.96, "discharge_rate": 0.52, "lr_minus": 0.12, "ppv": 0.70, "lr_plus": 4.0, "net_benefit_lower": 0.35, "net_benefit_upper": 0.11},
            {"threshold": 0.69, "npv": 0.80, "discharge_rate": 0.85, "lr_minus": 0.50, "ppv": 1.00, "lr_plus": 12.0, "net_benefit_lower": 0.05, "net_benefit_upper": 0.40},
            {"threshold": 0.75, "npv": 0.79, "discharge_rate": 0.88, "lr_minus": 0.55, "ppv": 0.95, "lr_plus": 11.0, "net_benefit_lower": 0.04, "net_benefit_upper": 0.32},
        ]
    )
    selected = select_safety_first_thresholds(sweep, ThresholdConfig())
    assert selected.lower_threshold == 0.29
    assert selected.upper_threshold == 0.69
    assert selected.feasible_pair is True


def test_locked_thresholds_are_evaluated_without_reselection() -> None:
    sweep = pd.DataFrame(
        [
            {
                "threshold": 0.20,
                "npv": 1.00,
                "discharge_rate": 0.40,
                "lr_minus": 0.00,
                "ppv": 0.50,
                "lr_plus": 2.0,
                "net_benefit_lower": 0.90,
                "net_benefit_upper": 0.10,
            },
            {
                "threshold": 0.29,
                "npv": 0.90,
                "discharge_rate": 0.30,
                "lr_minus": 0.25,
                "ppv": 0.60,
                "lr_plus": 3.0,
                "net_benefit_lower": 0.20,
                "net_benefit_upper": 0.20,
            },
            {
                "threshold": 0.60,
                "npv": 0.80,
                "discharge_rate": 0.70,
                "lr_minus": 0.50,
                "ppv": 1.00,
                "lr_plus": 20.0,
                "net_benefit_lower": 0.10,
                "net_benefit_upper": 0.90,
            },
            {
                "threshold": 0.69,
                "npv": 0.78,
                "discharge_rate": 0.80,
                "lr_minus": 0.60,
                "ppv": 0.70,
                "lr_plus": 4.0,
                "net_benefit_lower": 0.05,
                "net_benefit_upper": 0.30,
            },
        ]
    )
    config = ThresholdConfig(
        use_locked_thresholds=True,
        locked_lower_threshold=0.29,
        locked_upper_threshold=0.69,
    )

    selected = evaluate_locked_threshold_pair(sweep, config)

    assert selected.lower_threshold == 0.29
    assert selected.upper_threshold == 0.69
    assert selected.feasible_pair is False
    assert selected.warning is not None


def test_evaluate_threshold_pair_returns_band_metrics() -> None:
    config = ConfidenceIntervalConfig(
        bootstrap_iterations=100,
        random_state=11,
        bootstrap_stratified=True,
    )
    y_true = [0, 0, 0, 1, 1, 1, 0, 1]
    y_prob = [0.1, 0.2, 0.3, 0.35, 0.6, 0.8, 0.75, 0.9]
    records = evaluate_threshold_pair(
        y_true=y_true,
        y_prob=y_prob,
        lower_threshold=0.29,
        upper_threshold=0.69,
        config=config,
        dataset_name="holdout",
        model_name="Injury.top12",
    )
    metrics = {record["metric"] for record in records}
    assert "green_rate" in metrics
    assert "amber_rate" in metrics
    assert "red_rate" in metrics
    assert "npv" in metrics
    assert "ppv" in metrics


def test_lr_minus_ci_is_not_collapsed_when_no_false_negatives() -> None:
    config = ConfidenceIntervalConfig(
        bootstrap_iterations=10,
        confidence_level=0.95,
        proportion_method="exact",
    )
    y_true = [0] * 12 + [1] * 4
    y_prob = [0.1] * 5 + [0.4] * 7 + [0.8] * 4

    records = evaluate_threshold_pair(
        y_true=y_true,
        y_prob=y_prob,
        lower_threshold=0.3,
        upper_threshold=0.7,
        config=config,
        dataset_name="example",
        model_name="Injury.top12",
    )
    lr_minus = next(record for record in records if record["metric"] == "lr_minus")

    assert lr_minus["point"] == 0.0
    assert lr_minus["ci_method"] == "lr_exact_propagated"
    assert lr_minus["ci_lower"] == 0.0
    assert lr_minus["ci_upper"] > lr_minus["ci_lower"]


def test_derive_thresholds_from_predictions_returns_sweep_and_selection() -> None:
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_prob = [0.05, 0.08, 0.12, 0.20, 0.45, 0.58, 0.80, 0.92]
    sweep, selected = derive_thresholds_from_predictions(
        y_true,
        y_prob,
        ThresholdConfig(step=0.05, min_discharge_rate=0.20),
    )
    assert not sweep.empty
    assert 0.0 <= selected.lower_threshold < selected.upper_threshold <= 1.0
    assert selected.lower_metrics["threshold"] == selected.lower_threshold


def test_bootstrap_threshold_stability_summarizes_feasible_rate() -> None:
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_prob = [0.02, 0.05, 0.08, 0.15, 0.22, 0.42, 0.55, 0.72, 0.85, 0.95]
    threshold_config = ThresholdConfig(
        step=0.05,
        min_discharge_rate=0.20,
        min_ppv=0.80,
        bootstrap_threshold_iterations=25,
    )
    _, selected = derive_thresholds_from_predictions(y_true, y_prob, threshold_config)
    stability = bootstrap_threshold_stability(
        y_true,
        y_prob,
        threshold_config,
        ConfidenceIntervalConfig(bootstrap_stratified=True, random_state=7),
    )
    summary = summarize_threshold_stability(
        stability,
        selected=selected,
        target="Injury",
        variant="top12",
        selection_source="training_oof",
    )
    assert len(stability) == 25
    assert summary.loc[0, "n_bootstrap"] == 25
    assert 0.0 <= summary.loc[0, "feasible_pair_rate"] <= 1.0
