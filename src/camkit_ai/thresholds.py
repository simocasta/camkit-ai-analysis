from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from camkit_ai.config import ConfidenceIntervalConfig, ThresholdConfig
from camkit_ai.confidence_intervals import EstimateCI, likelihood_ratio_ci, proportion_ci
from camkit_ai.metrics import classification_metrics_at_threshold


def calculate_net_benefit(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    threshold: float,
    mode: str,
) -> float:
    total = tp + fp + fn + tn
    if total == 0:
        return 0.0
    pt = min(max(float(threshold), 1e-12), 1.0 - 1e-12)
    if mode == "lower":
        weight = (1.0 - pt) / pt
        return (tn / total) - (fn / total) * weight
    weight = pt / (1.0 - pt)
    return (tp / total) - (fp / total) * weight


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    step: float,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    thresholds = np.arange(0.0, 1.0, step)
    for threshold in thresholds:
        metrics = classification_metrics_at_threshold(y_true, y_prob, threshold)
        tp = metrics["true_positive"]
        fp = metrics["false_positive"]
        fn = metrics["false_negative"]
        tn = metrics["true_negative"]
        discharge_rate = metrics["predicted_negative"] / len(y_true)
        referral_rate = metrics["predicted_positive"] / len(y_true)
        rows.append(
            {
                "threshold": threshold,
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "npv": metrics["npv"],
                "ppv": metrics["ppv"],
                "lr_plus": metrics["lr_plus"],
                "lr_minus": metrics["lr_minus"],
                "missed_cases": fn,
                "unnecessary_referrals": fp,
                "true_positives": tp,
                "true_negatives": tn,
                "total_positives": metrics["total_positive"],
                "total_negatives": metrics["total_negative"],
                "discharge_rate": discharge_rate,
                "referral_rate": referral_rate,
                "net_benefit_lower": calculate_net_benefit(tp, fp, fn, tn, threshold, "lower"),
                "net_benefit_upper": calculate_net_benefit(tp, fp, fn, tn, threshold, "upper"),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class SelectedThresholds:
    lower_threshold: float
    upper_threshold: float
    lower_metrics: dict[str, float]
    upper_metrics: dict[str, float]
    threshold_gap: float
    warning: str | None = None
    lower_feasible: bool = True
    upper_feasible: bool = True
    feasible_pair: bool = True


def evaluate_locked_threshold_pair(
    sweep: pd.DataFrame,
    config: ThresholdConfig,
) -> SelectedThresholds:
    """Evaluate, but do not re-optimise, the locked action thresholds.

    The cut-points 0.29 and 0.69 were fixed on the development-stage hold-out
    analysis. Later analyses must retain them so that changes caused by the
    reproducible prediction basis are not conflated with post hoc threshold
    re-selection.  The feasibility flags are therefore descriptive diagnostics,
    not a reason to move either threshold.
    """

    lower_threshold = float(config.locked_lower_threshold)
    upper_threshold = float(config.locked_upper_threshold)
    if not 0.0 <= lower_threshold < upper_threshold <= 1.0:
        raise ValueError(
            "Locked thresholds must satisfy "
            "0 <= locked_lower_threshold < locked_upper_threshold <= 1."
        )

    def row_at(threshold: float) -> pd.Series:
        distances = (sweep["threshold"].astype(float) - threshold).abs()
        row_index = distances.idxmin()
        tolerance = max(float(config.step) / 2.0, 1e-12)
        if float(distances.loc[row_index]) > tolerance:
            raise ValueError(
                f"Threshold sweep does not contain the locked threshold {threshold:.3f}."
            )
        return sweep.loc[row_index]

    lower_row = row_at(lower_threshold)
    upper_row = row_at(upper_threshold)
    lower_feasible = bool(
        float(lower_row["npv"]) >= config.min_npv
        and float(lower_row["discharge_rate"]) >= config.min_discharge_rate
        and float(lower_row["lr_minus"]) < config.target_lr_minus
    )
    upper_feasible = bool(
        float(upper_row["ppv"]) >= config.min_ppv
        or float(upper_row["lr_plus"]) > config.target_lr_plus
    )
    gap = upper_threshold - lower_threshold
    feasible_pair = lower_feasible and upper_feasible and gap >= config.min_gap

    warnings: list[str] = []
    if not lower_feasible:
        warnings.append("The locked lower threshold does not meet all configured criteria.")
    if not upper_feasible:
        warnings.append("The locked upper threshold does not meet either configured criterion.")
    if gap < config.min_gap:
        warnings.append(
            f"Threshold gap {gap:.3f} is below the configured minimum {config.min_gap:.3f}."
        )

    return SelectedThresholds(
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        lower_metrics=lower_row.to_dict(),
        upper_metrics=upper_row.to_dict(),
        threshold_gap=gap,
        warning=" ".join(warnings) or None,
        lower_feasible=lower_feasible,
        upper_feasible=upper_feasible,
        feasible_pair=feasible_pair,
    )


def select_safety_first_thresholds(
    sweep: pd.DataFrame,
    config: ThresholdConfig,
) -> SelectedThresholds:
    if config.selection_basis != "point":
        raise ValueError(
            "Only point-estimate threshold selection is implemented for the primary "
            "analysis. Use confidence intervals as diagnostics."
        )

    safe_lower = sweep[
        (sweep["npv"] >= config.min_npv)
        & (sweep["discharge_rate"] >= config.min_discharge_rate)
        & (sweep["lr_minus"] < config.target_lr_minus)
    ]
    lower_feasible = not safe_lower.empty
    if not safe_lower.empty:
        best_lower = safe_lower[safe_lower["net_benefit_lower"] == safe_lower["net_benefit_lower"].max()]
        lower_row = best_lower.iloc[-1]
    else:
        finite_lr = sweep[np.isfinite(sweep["lr_minus"])]
        lower_row = finite_lr.loc[finite_lr["lr_minus"].idxmin()] if not finite_lr.empty else sweep.loc[sweep["npv"].idxmax()]

    candidates = sweep[sweep["threshold"] >= float(lower_row["threshold"]) + config.min_gap]
    if candidates.empty:
        candidates = sweep[sweep["threshold"] > float(lower_row["threshold"])]
    if candidates.empty:
        candidates = sweep.loc[[lower_row.name]]

    good_upper = candidates[
        (candidates["ppv"] >= config.min_ppv) | (candidates["lr_plus"] > config.target_lr_plus)
    ]
    upper_feasible = not good_upper.empty
    if not good_upper.empty:
        best_upper = good_upper[good_upper["net_benefit_upper"] == good_upper["net_benefit_upper"].max()]
        upper_row = best_upper.iloc[0]
    else:
        upper_row = candidates.loc[candidates["net_benefit_upper"].idxmax()]

    gap = float(upper_row["threshold"] - lower_row["threshold"])
    warning = None
    feasible_pair = lower_feasible and upper_feasible and gap >= config.min_gap
    if gap < config.min_gap:
        warning = f"Threshold gap {gap:.3f} is below the configured minimum {config.min_gap:.3f}."

    return SelectedThresholds(
        lower_threshold=float(lower_row["threshold"]),
        upper_threshold=float(upper_row["threshold"]),
        lower_metrics=lower_row.to_dict(),
        upper_metrics=upper_row.to_dict(),
        threshold_gap=gap,
        warning=warning,
        lower_feasible=lower_feasible,
        upper_feasible=upper_feasible,
        feasible_pair=feasible_pair,
    )


def derive_thresholds_from_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config: ThresholdConfig,
) -> tuple[pd.DataFrame, SelectedThresholds]:
    sweep = threshold_sweep(
        np.asarray(y_true).astype(int),
        np.asarray(y_prob, dtype=float),
        step=config.step,
    )
    selected = select_safety_first_thresholds(sweep, config)
    return sweep, selected


def _threshold_bootstrap_indices(
    y_true: np.ndarray,
    rng: np.random.Generator,
    stratified: bool,
) -> np.ndarray:
    y_true = np.asarray(y_true)
    if not stratified:
        return rng.integers(0, len(y_true), size=len(y_true))

    indices: list[np.ndarray] = []
    for label in np.unique(y_true):
        label_idx = np.flatnonzero(y_true == label)
        indices.append(rng.choice(label_idx, size=len(label_idx), replace=True))
    sampled = np.concatenate(indices)
    rng.shuffle(sampled)
    return sampled


def bootstrap_threshold_stability(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_config: ThresholdConfig,
    ci_config: ConfidenceIntervalConfig,
) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(ci_config.random_state)
    rows: list[dict[str, object]] = []

    for bootstrap_id in range(1, int(threshold_config.bootstrap_threshold_iterations) + 1):
        sample_idx = _threshold_bootstrap_indices(
            y_true,
            rng,
            ci_config.bootstrap_stratified,
        )
        y_boot = y_true[sample_idx]
        p_boot = y_prob[sample_idx]
        if len(np.unique(y_boot)) < 2:
            rows.append(
                {
                    "bootstrap_id": bootstrap_id,
                    "lower_threshold": np.nan,
                    "upper_threshold": np.nan,
                    "threshold_gap": np.nan,
                    "lower_feasible": False,
                    "upper_feasible": False,
                    "feasible_pair": False,
                    "warning": "Bootstrap sample had fewer than two outcome classes.",
                }
            )
            continue

        sweep, selected = derive_thresholds_from_predictions(
            y_boot,
            p_boot,
            threshold_config,
        )
        rows.append(
            {
                "bootstrap_id": bootstrap_id,
                "lower_threshold": selected.lower_threshold,
                "upper_threshold": selected.upper_threshold,
                "threshold_gap": selected.threshold_gap,
                "lower_feasible": selected.lower_feasible,
                "upper_feasible": selected.upper_feasible,
                "feasible_pair": selected.feasible_pair,
                "warning": selected.warning,
                "lower_npv": selected.lower_metrics.get("npv"),
                "lower_lr_minus": selected.lower_metrics.get("lr_minus"),
                "lower_discharge_rate": selected.lower_metrics.get("discharge_rate"),
                "upper_ppv": selected.upper_metrics.get("ppv"),
                "upper_lr_plus": selected.upper_metrics.get("lr_plus"),
                "upper_referral_rate": selected.upper_metrics.get("referral_rate"),
            }
        )

    return pd.DataFrame(rows)


def summarize_threshold_stability(
    stability: pd.DataFrame,
    *,
    selected: SelectedThresholds,
    target: str,
    variant: str,
    selection_source: str,
) -> pd.DataFrame:
    feasible = stability[stability["feasible_pair"].astype(bool)].copy()

    def q(column: str, probability: float) -> float:
        if feasible.empty:
            return float("nan")
        values = feasible[column].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            return float("nan")
        return float(np.quantile(values, probability))

    total = len(stability)
    feasible_count = int(stability["feasible_pair"].sum()) if total else 0
    row = {
        "target": target,
        "variant": variant,
        "selection_source": selection_source,
        "n_bootstrap": total,
        "feasible_pair_count": feasible_count,
        "feasible_pair_rate": feasible_count / total if total else float("nan"),
        "lower_selected": selected.lower_threshold,
        "lower_median": q("lower_threshold", 0.50),
        "lower_iqr_low": q("lower_threshold", 0.25),
        "lower_iqr_high": q("lower_threshold", 0.75),
        "lower_ci_low": q("lower_threshold", 0.025),
        "lower_ci_high": q("lower_threshold", 0.975),
        "upper_selected": selected.upper_threshold,
        "upper_median": q("upper_threshold", 0.50),
        "upper_iqr_low": q("upper_threshold", 0.25),
        "upper_iqr_high": q("upper_threshold", 0.75),
        "upper_ci_low": q("upper_threshold", 0.025),
        "upper_ci_high": q("upper_threshold", 0.975),
    }
    return pd.DataFrame([row])


def _lr_exact_bundle(
    metrics: dict[str, float],
    config: ConfidenceIntervalConfig,
) -> dict[str, EstimateCI]:
    sensitivity = proportion_ci(
        int(metrics["true_positive"]),
        int(metrics["total_positive"]),
        config,
    )
    specificity = proportion_ci(
        int(metrics["true_negative"]),
        int(metrics["total_negative"]),
        config,
    )
    return {
        "lr_plus": likelihood_ratio_ci(
            sensitivity=sensitivity,
            specificity=specificity,
            metric="lr_plus",
            point=float(metrics["lr_plus"]),
        ),
        "lr_minus": likelihood_ratio_ci(
            sensitivity=sensitivity,
            specificity=specificity,
            metric="lr_minus",
            point=float(metrics["lr_minus"]),
        ),
    }


def evaluate_threshold_pair(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    lower_threshold: float,
    upper_threshold: float,
    config: ConfidenceIntervalConfig,
    dataset_name: str,
    model_name: str,
) -> list[dict[str, object]]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    lower = classification_metrics_at_threshold(y_true, y_prob, lower_threshold)
    upper = classification_metrics_at_threshold(y_true, y_prob, upper_threshold)
    green_mask = y_prob < lower_threshold
    amber_mask = (y_prob >= lower_threshold) & (y_prob < upper_threshold)
    red_mask = y_prob >= upper_threshold

    lr_lower = _lr_exact_bundle(lower, config)
    lr_upper = _lr_exact_bundle(upper, config)

    estimates = [
        proportion_ci(int(green_mask.sum()), len(y_true), config).as_record(
            "green_rate",
            dataset=dataset_name,
            model=model_name,
            threshold_role="three_way",
        ),
        proportion_ci(int(amber_mask.sum()), len(y_true), config).as_record(
            "amber_rate",
            dataset=dataset_name,
            model=model_name,
            threshold_role="three_way",
        ),
        proportion_ci(int(red_mask.sum()), len(y_true), config).as_record(
            "red_rate",
            dataset=dataset_name,
            model=model_name,
            threshold_role="three_way",
        ),
        proportion_ci(lower["true_negative"], lower["predicted_negative"], config).as_record(
            "npv",
            dataset=dataset_name,
            model=model_name,
            threshold_role="lower",
            threshold=lower_threshold,
        ),
        proportion_ci(lower["true_positive"], lower["total_positive"], config).as_record(
            "sensitivity",
            dataset=dataset_name,
            model=model_name,
            threshold_role="lower",
            threshold=lower_threshold,
        ),
        proportion_ci(upper["true_positive"], upper["predicted_positive"], config).as_record(
            "ppv",
            dataset=dataset_name,
            model=model_name,
            threshold_role="upper",
            threshold=upper_threshold,
        ),
        proportion_ci(upper["true_negative"], upper["total_negative"], config).as_record(
            "specificity",
            dataset=dataset_name,
            model=model_name,
            threshold_role="upper",
            threshold=upper_threshold,
        ),
        lr_lower["lr_minus"].as_record(
            "lr_minus",
            dataset=dataset_name,
            model=model_name,
            threshold_role="lower",
            threshold=lower_threshold,
        ),
        lr_upper["lr_plus"].as_record(
            "lr_plus",
            dataset=dataset_name,
            model=model_name,
            threshold_role="upper",
            threshold=upper_threshold,
        ),
    ]

    counts = [
        {
            "metric": "missed_injuries_green",
            "point": int(np.sum(green_mask & (y_true == 1))),
            "ci_lower": None,
            "ci_upper": None,
            "ci_method": "count",
            "ci_samples": None,
            "ci_skipped": None,
            "numerator": int(np.sum(green_mask & (y_true == 1))),
            "denominator": int(green_mask.sum()),
            "dataset": dataset_name,
            "model": model_name,
            "threshold_role": "lower",
            "threshold": lower_threshold,
        },
        {
            "metric": "unnecessary_referrals_red",
            "point": int(np.sum(red_mask & (y_true == 0))),
            "ci_lower": None,
            "ci_upper": None,
            "ci_method": "count",
            "ci_samples": None,
            "ci_skipped": None,
            "numerator": int(np.sum(red_mask & (y_true == 0))),
            "denominator": int(red_mask.sum()),
            "dataset": dataset_name,
            "model": model_name,
            "threshold_role": "upper",
            "threshold": upper_threshold,
        },
    ]
    return estimates + counts
