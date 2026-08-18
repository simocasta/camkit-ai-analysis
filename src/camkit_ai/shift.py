"""Decomposition of the retrospective-to-prospective discrimination gap.

The prospective cohort differs from the internal hold-out set in outcome
prevalence (25.8% vs 21.2%) as well as in respondent type, calendar period and
case-mix. Falling prevalence mechanically depresses average precision (AP), so a raw
hold-out-to-prospective comparison cannot attribute the observed drop to the
change in data source.

This module resamples each cohort to the other's prevalence while leaving the
within-class score distributions untouched, which separates the part of the gap
explained by prevalence alone from the residual. AUC-ROC is reported alongside
AP because it does not depend on class balance: a residual AUC-ROC gap
cannot be a prevalence artefact, although it can still reflect spectrum or
severity shift rather than respondent type.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from camkit_ai.config import ConfidenceIntervalConfig, ProjectConfig

METRIC_FUNCTIONS = {
    "auprc": average_precision_score,
    "auroc": roc_auc_score,
}

STANDARDISED_METHOD = "bootstrap_prevalence_standardised"
OBSERVED_METHOD = "bootstrap_stratified"


def _stratified_resample(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample preserving the cohort's own prevalence."""
    indices: list[np.ndarray] = []
    for label in np.unique(y_true):
        label_idx = np.flatnonzero(y_true == label)
        indices.append(rng.choice(label_idx, size=len(label_idx), replace=True))
    sampled = np.concatenate(indices)
    return y_true[sampled], y_prob[sampled]


def _resample_to_prevalence(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_prevalence: float,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample to a fixed prevalence, keeping within-class scores unchanged.

    Cases and controls are drawn separately with replacement, so the predicted
    probability distribution within each outcome class is preserved and only the
    class mix changes.
    """
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Prevalence standardisation requires both outcome classes.")

    n_pos = int(round(size * target_prevalence))
    n_pos = min(max(n_pos, 1), size - 1)
    n_neg = size - n_pos

    sampled_pos = rng.choice(pos_idx, size=n_pos, replace=True)
    sampled_neg = rng.choice(neg_idx, size=n_neg, replace=True)
    sampled = np.concatenate([sampled_pos, sampled_neg])
    return y_true[sampled], y_prob[sampled]


def _score_all(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {name: float("nan") for name in METRIC_FUNCTIONS}
    return {
        name: float(func(y_true, y_prob)) for name, func in METRIC_FUNCTIONS.items()
    }


def _interval(values: list[float], confidence_level: float) -> tuple[float, float, float]:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return float("nan"), float("nan"), float("nan")
    alpha = 1.0 - confidence_level
    return (
        float(np.median(finite)),
        float(np.quantile(finite, alpha / 2.0)),
        float(np.quantile(finite, 1.0 - alpha / 2.0)),
    )


def decompose_discrimination_shift(
    holdout_y: np.ndarray,
    holdout_prob: np.ndarray,
    prospective_y: np.ndarray,
    prospective_prob: np.ndarray,
    config: ConfidenceIntervalConfig,
    *,
    model_label: str,
) -> pd.DataFrame:
    """Split the hold-out to prospective discrimination gap into prevalence and residual parts.

    Returns one row per (metric, quantity) with a point estimate and a
    percentile interval. Observed quantities use the full-sample estimate as the
    point; standardised quantities have no single-sample analogue and use the
    bootstrap median, flagged by ``ci_method``.
    """
    holdout_y = np.asarray(holdout_y).astype(int)
    holdout_prob = np.asarray(holdout_prob, dtype=float)
    prospective_y = np.asarray(prospective_y).astype(int)
    prospective_prob = np.asarray(prospective_prob, dtype=float)

    holdout_prevalence = float(np.mean(holdout_y))
    prospective_prevalence = float(np.mean(prospective_y))
    holdout_n = len(holdout_y)
    prospective_n = len(prospective_y)

    holdout_observed = _score_all(holdout_y, holdout_prob)
    prospective_observed = _score_all(prospective_y, prospective_prob)

    samples: dict[str, list[float]] = {
        f"{prefix}_{metric}": []
        for metric in METRIC_FUNCTIONS
        for prefix in (
            "holdout_observed",
            "holdout_standardised",
            "prospective_observed",
            "prospective_standardised",
            "gap_total",
            "gap_prevalence",
            "gap_residual",
        )
    }

    rng = np.random.default_rng(config.random_state)
    for _ in range(config.bootstrap_iterations):
        h_obs = _score_all(*_stratified_resample(holdout_y, holdout_prob, rng))
        h_std = _score_all(
            *_resample_to_prevalence(
                holdout_y, holdout_prob, prospective_prevalence, holdout_n, rng
            )
        )
        p_obs = _score_all(*_stratified_resample(prospective_y, prospective_prob, rng))
        p_std = _score_all(
            *_resample_to_prevalence(
                prospective_y, prospective_prob, holdout_prevalence, prospective_n, rng
            )
        )
        for metric in METRIC_FUNCTIONS:
            samples[f"holdout_observed_{metric}"].append(h_obs[metric])
            samples[f"holdout_standardised_{metric}"].append(h_std[metric])
            samples[f"prospective_observed_{metric}"].append(p_obs[metric])
            samples[f"prospective_standardised_{metric}"].append(p_std[metric])
            samples[f"gap_total_{metric}"].append(h_obs[metric] - p_obs[metric])
            samples[f"gap_prevalence_{metric}"].append(h_obs[metric] - h_std[metric])
            samples[f"gap_residual_{metric}"].append(h_std[metric] - p_obs[metric])

    rows: list[dict[str, object]] = []
    for metric in METRIC_FUNCTIONS:
        standardised_holdout_median, _, _ = _interval(
            samples[f"holdout_standardised_{metric}"], config.confidence_level
        )
        standardised_prospective_median, _, _ = _interval(
            samples[f"prospective_standardised_{metric}"], config.confidence_level
        )

        quantities = {
            "holdout_observed": (holdout_observed[metric], OBSERVED_METHOD, holdout_prevalence),
            "holdout_standardised_to_prospective": (
                standardised_holdout_median,
                STANDARDISED_METHOD,
                prospective_prevalence,
            ),
            "prospective_observed": (
                prospective_observed[metric],
                OBSERVED_METHOD,
                prospective_prevalence,
            ),
            "prospective_standardised_to_holdout": (
                standardised_prospective_median,
                STANDARDISED_METHOD,
                holdout_prevalence,
            ),
            "gap_total": (
                holdout_observed[metric] - prospective_observed[metric],
                OBSERVED_METHOD,
                None,
            ),
            "gap_prevalence_attributable": (
                holdout_observed[metric] - standardised_holdout_median,
                STANDARDISED_METHOD,
                None,
            ),
            "gap_residual": (
                standardised_holdout_median - prospective_observed[metric],
                STANDARDISED_METHOD,
                None,
            ),
        }
        sample_keys = {
            "holdout_observed": "holdout_observed",
            "holdout_standardised_to_prospective": "holdout_standardised",
            "prospective_observed": "prospective_observed",
            "prospective_standardised_to_holdout": "prospective_standardised",
            "gap_total": "gap_total",
            "gap_prevalence_attributable": "gap_prevalence",
            "gap_residual": "gap_residual",
        }

        for quantity, (point, method, prevalence) in quantities.items():
            values = samples[f"{sample_keys[quantity]}_{metric}"]
            _, lower, upper = _interval(values, config.confidence_level)
            rows.append(
                {
                    "model": model_label,
                    "metric": metric,
                    "quantity": quantity,
                    "point": point,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "ci_method": method,
                    "ci_samples": len([v for v in values if np.isfinite(v)]),
                    "target_prevalence": prevalence,
                    "holdout_n": holdout_n,
                    "prospective_n": prospective_n,
                    "holdout_prevalence": holdout_prevalence,
                    "prospective_prevalence": prospective_prevalence,
                }
            )

    return pd.DataFrame(rows)


def load_split_predictions(
    predictions: pd.DataFrame,
    model_label: str,
    dataset: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Pull one model/dataset pair out of a saved prediction summary."""
    subset = predictions[
        (predictions["model"] == model_label) & (predictions["dataset"] == dataset)
    ]
    if subset.empty:
        raise ValueError(f"No predictions found for model '{model_label}' on '{dataset}'.")
    return (
        subset["y_true"].to_numpy(dtype=int),
        subset["y_probability"].to_numpy(dtype=float),
    )


def decompose_from_prediction_summary(
    predictions: pd.DataFrame,
    config: ConfidenceIntervalConfig,
    *,
    model_labels: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model_label in model_labels:
        holdout_y, holdout_prob = load_split_predictions(predictions, model_label, "holdout")
        prospective_y, prospective_prob = load_split_predictions(
            predictions, model_label, "prospective"
        )
        frames.append(
            decompose_discrimination_shift(
                holdout_y,
                holdout_prob,
                prospective_y,
                prospective_prob,
                config,
                model_label=model_label,
            )
        )
    return pd.concat(frames, ignore_index=True)


def format_decomposition_markdown(decomposition: pd.DataFrame) -> str:
    """Render the decomposition as a manuscript-ready markdown table."""
    lines = [
        "# Discrimination shift decomposition (hold-out to prospective)",
        "",
        "`gap_prevalence_attributable` is the part of the hold-out to prospective gap",
        "explained by the fall in outcome prevalence alone; `gap_residual` is what",
        "remains after standardising the hold-out set to the prospective prevalence.",
        "AUC-ROC does not depend on class balance, so a residual AUC-ROC gap cannot be",
        "a prevalence artefact, although it may still reflect spectrum or case-mix",
        "shift rather than respondent type.",
        "",
    ]
    for model_label, model_frame in decomposition.groupby("model", sort=False):
        lines.append(f"## {model_label}")
        lines.append("")
        lines.append("| Metric | Quantity | Estimate (95% CI) |")
        lines.append("| --- | --- | --- |")
        for _, row in model_frame.iterrows():
            estimate = (
                f"{row['point']:.3f} ({row['ci_lower']:.3f} to {row['ci_upper']:.3f})"
            )
            lines.append(f"| {row['metric']} | {row['quantity']} | {estimate} |")
        lines.append("")
    return "\n".join(lines)


def run_shift_analysis(
    config: ProjectConfig,
    *,
    model_labels: list[str] | None = None,
    predictions_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Run the decomposition from saved predictions and write the outputs.

    This reads the stored prediction summary rather than reloading AutoPrognosis
    artefacts, so it runs without the legacy training environment.
    """
    labels = model_labels or ["Injury.top12", "Injury.full"]
    source = predictions_path or (
        config.paths.output_root / "manuscript" / "prediction_summary.csv"
    )
    if not source.exists():
        raise FileNotFoundError(
            f"Prediction summary not found at {source}. Run 'report-manuscript' first."
        )

    predictions = pd.read_csv(source)
    decomposition = decompose_from_prediction_summary(
        predictions,
        config.confidence_intervals,
        model_labels=labels,
    )

    output_root = output_dir or (config.paths.output_root / "analysis")
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "decomposition": output_root / "discrimination_shift_decomposition.csv",
        "decomposition_markdown": output_root / "discrimination_shift_decomposition.md",
    }
    decomposition.to_csv(paths["decomposition"], index=False)
    paths["decomposition_markdown"].write_text(
        format_decomposition_markdown(decomposition),
        encoding="utf-8",
    )
    return decomposition, paths
