"""Frozen post hoc analyses for the transportability evaluation.

The functions in this module never load or score a fitted model.  They operate
only on the patient-level prediction files, inference-integrity rows and draw
matrix written by the audited 400-draw evaluation.  That separation is
deliberate: a sensitivity analysis must not silently become a repaired model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from camkit_ai.confidence_intervals import EstimateCI, format_estimate, proportion_ci
from camkit_ai.config import ProjectConfig
from camkit_ai.imputation_variability import assign_bands
from camkit_ai.metrics import calibration_slope_intercept, discrimination_metrics, safe_logit


@dataclass(frozen=True)
class CapacityMatchedResult:
    policies: pd.DataFrame
    overlap: pd.DataFrame
    curve: pd.DataFrame


@dataclass(frozen=True)
class CompleteRecordResult:
    metrics: pd.DataFrame
    band_counts: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _boolean_series(values: pd.Series, *, label: str) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    mapped = values.astype(str).str.strip().str.casefold().map(
        {"true": True, "1": True, "false": False, "0": False}
    )
    if mapped.isna().any():
        raise ValueError(f"{label} contains values that are not boolean.")
    return mapped.astype(bool)


def _prediction_provenance(
    frame: pd.DataFrame, *, label: str
) -> tuple[str, str, str]:
    """Return the one coherent frozen prediction provenance carried by a frame."""

    _require_columns(
        frame,
        {"prediction_basis", "prediction_source", "prediction_context"},
        label,
    )
    values: list[str] = []
    for column in ("prediction_basis", "prediction_source", "prediction_context"):
        unique = frame[column].dropna().astype(str).str.strip().unique()
        if len(unique) != 1 or not unique[0]:
            raise ValueError(f"{label} does not have one coherent {column} value.")
        values.append(str(unique[0]))
    return values[0], values[1], values[2]


def _policy_row(
    *,
    policy_id: str,
    label: str,
    tool: str,
    selection_rule: str,
    selected: np.ndarray,
    y_true: np.ndarray,
    config: ProjectConfig,
    post_hoc: bool,
    prediction_basis: str,
    prediction_source: str,
    prediction_context: str,
) -> dict[str, object]:
    selected = np.asarray(selected, dtype=bool)
    y_true = np.asarray(y_true, dtype=int)
    n = int(len(y_true))
    events = int(y_true.sum())
    referrals = int(selected.sum())
    captured = int(y_true[selected].sum())
    referral_rate = proportion_ci(referrals, n, config.confidence_intervals)
    capture = proportion_ci(captured, events, config.confidence_intervals)
    ppv = proportion_ci(captured, referrals, config.confidence_intervals)
    return {
        "policy_id": policy_id,
        "policy_label": label,
        "tool": tool,
        "selection_rule": selection_rule,
        "post_hoc": bool(post_hoc),
        "n": n,
        "events": events,
        "referrals": referrals,
        "injuries_captured": captured,
        "referral_rate": referral_rate.point,
        "referral_rate_ci_lower": referral_rate.lower,
        "referral_rate_ci_upper": referral_rate.upper,
        "referral_rate_formatted": format_estimate(referral_rate),
        "injury_capture": capture.point,
        "injury_capture_ci_lower": capture.lower,
        "injury_capture_ci_upper": capture.upper,
        "injury_capture_formatted": format_estimate(capture),
        "ppv": ppv.point,
        "ppv_ci_lower": ppv.lower,
        "ppv_ci_upper": ppv.upper,
        "ppv_formatted": format_estimate(ppv),
        "referrals_per_injury": referrals / captured if captured else np.nan,
        "prediction_basis": prediction_basis,
        "prediction_source": prediction_source,
        "prediction_context": prediction_context if tool == "CamKIT-AI" else None,
    }


def capacity_matched_analysis(
    patient_predictions: pd.DataFrame,
    config: ProjectConfig,
    *,
    capacity: int | None = None,
) -> CapacityMatchedResult:
    """Compare CamKIT-AI with CamKIT at CamKIT's observed referral capacity.

    CamKIT-AI patients are ranked by their already-frozen mean probabilities.
    Ties are rejected rather than resolved post hoc because the cumulative
    capacity curve reports every possible rank cut-point.
    """

    required = {
        "row_id",
        "Injury",
        "camkit_ai_probability",
        "camkit_ai_triage_band",
        "camkit_triage_band",
        "prediction_basis",
        "prediction_source",
        "prediction_context",
        "camkit_score_source",
    }
    _require_columns(patient_predictions, required, "patient predictions")
    frame = patient_predictions.copy()
    if frame["row_id"].duplicated().any():
        raise ValueError("Patient predictions contain duplicate row identifiers.")
    if len(frame) != 85 or int(frame["Injury"].sum()) != 18:
        raise ValueError("Capacity matching requires the frozen 85-patient/18-event cohort.")
    basis, ai_source, context = _prediction_provenance(
        frame, label="CamKIT-AI patient predictions"
    )
    camkit_sources = frame["camkit_score_source"].dropna().astype(str).str.strip().unique()
    if len(camkit_sources) != 1 or not camkit_sources[0]:
        raise ValueError("CamKIT patient predictions do not have one coherent source.")
    camkit_source = str(camkit_sources[0])

    y_true = frame["Injury"].to_numpy(dtype=int)
    ai_locked = frame["camkit_ai_triage_band"].astype(str).eq("Red").to_numpy()
    camkit_high = frame["camkit_triage_band"].astype(str).eq("Red").to_numpy()
    observed_capacity = int(camkit_high.sum())
    if capacity is None:
        capacity = observed_capacity
    if capacity != observed_capacity:
        raise ValueError(
            f"Requested capacity {capacity} does not equal CamKIT's observed {observed_capacity}."
        )
    if not 0 < capacity < len(frame):
        raise ValueError("Capacity must lie strictly inside the cohort size.")

    ranked = frame.sort_values(
        ["camkit_ai_probability", "row_id"], ascending=[False, True]
    ).reset_index(drop=True)
    ranked_probability = ranked["camkit_ai_probability"].to_numpy(dtype=float)
    adjacent_gaps = ranked_probability[:-1] - ranked_probability[1:]
    if np.any(adjacent_gaps <= 1e-12):
        raise ValueError(
            "CamKIT-AI has a probability tie in the prospective ranking; "
            "an every-capacity post hoc curve cannot use an arbitrary tie-break."
        )
    boundary = float(ranked.loc[capacity - 1, "camkit_ai_probability"])
    next_probability = float(ranked.loc[capacity, "camkit_ai_probability"])
    boundary_tie = bool(np.isclose(boundary, next_probability, rtol=0.0, atol=1e-12))
    if boundary_tie:
        raise ValueError(
            "CamKIT-AI has a probability tie at the matched-capacity boundary; "
            "a post hoc tie-break is not permitted."
        )
    selected_ids = set(ranked.iloc[:capacity]["row_id"].tolist())
    ai_matched = frame["row_id"].isin(selected_ids).to_numpy()

    policies = pd.DataFrame(
        [
            _policy_row(
                policy_id="camkit_ai_locked",
                label="CamKIT-AI locked p>=0.69",
                tool="CamKIT-AI",
                selection_rule="p >= 0.69",
                selected=ai_locked,
                y_true=y_true,
                config=config,
                post_hoc=False,
                prediction_basis=basis,
                prediction_source=ai_source,
                prediction_context=context,
            ),
            _policy_row(
                policy_id="camkit_ai_top_41",
                label="CamKIT-AI top 41 (post hoc)",
                tool="CamKIT-AI",
                selection_rule="41 highest frozen mean probabilities",
                selected=ai_matched,
                y_true=y_true,
                config=config,
                post_hoc=True,
                prediction_basis=basis,
                prediction_source=ai_source,
                prediction_context=context,
            ),
            _policy_row(
                policy_id="original_camkit_high",
                label="Original CamKIT High",
                tool="Original CamKIT",
                selection_rule="score 7-12",
                selected=camkit_high,
                y_true=y_true,
                config=config,
                post_hoc=False,
                prediction_basis=basis,
                prediction_source=camkit_source,
                prediction_context=context,
            ),
        ]
    )

    injury = y_true == 1
    overlap = pd.DataFrame(
        [
            {
                "capacity": capacity,
                "ai_boundary_probability": boundary,
                "ai_next_probability": next_probability,
                "boundary_tie": boundary_tie,
                "shared_referrals": int(np.sum(ai_matched & camkit_high)),
                "ai_only_referrals": int(np.sum(ai_matched & ~camkit_high)),
                "camkit_only_referrals": int(np.sum(~ai_matched & camkit_high)),
                "shared_injuries": int(np.sum(injury & ai_matched & camkit_high)),
                "ai_only_injuries": int(np.sum(injury & ai_matched & ~camkit_high)),
                "camkit_only_injuries": int(np.sum(injury & ~ai_matched & camkit_high)),
                "neither_injuries": int(np.sum(injury & ~ai_matched & ~camkit_high)),
                "prediction_basis": basis,
                "prediction_source": (
                    f"camkit_ai:{ai_source};camkit:{camkit_source}"
                ),
                "prediction_context": context,
            }
        ]
    )
    ranked_outcome = ranked["Injury"].to_numpy(dtype=int)
    cumulative_injuries = np.concatenate(
        [np.array([0], dtype=int), np.cumsum(ranked_outcome, dtype=int)]
    )
    referrals = np.arange(len(ranked) + 1, dtype=int)
    curve = pd.DataFrame(
        {
            "referrals": referrals,
            "injuries_captured": cumulative_injuries,
            "referral_rate": referrals / len(ranked),
            "injury_capture": cumulative_injuries / int(ranked_outcome.sum()),
            "rank_boundary_probability": np.concatenate(
                [np.array([np.nan]), ranked_probability]
            ),
            "marginal_injury": np.concatenate(
                [np.array([0], dtype=int), ranked_outcome]
            ),
            "n": len(ranked),
            "events": int(ranked_outcome.sum()),
            "post_hoc": True,
            "selection_rule": "top k frozen mean CamKIT-AI probabilities",
            "prediction_basis": basis,
            "prediction_source": ai_source,
            "prediction_context": context,
        }
    )
    return CapacityMatchedResult(policies=policies, overlap=overlap, curve=curve)


def _stratified_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sampled = []
    for label in np.unique(y_true):
        indices = np.flatnonzero(y_true == label)
        sampled.append(rng.choice(indices, size=len(indices), replace=True))
    combined = np.concatenate(sampled)
    rng.shuffle(combined)
    return combined


def _metric_estimates(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    config: ProjectConfig,
) -> dict[str, EstimateCI]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        point = discrimination_metrics(y_true, y_probability)
    samples: dict[str, list[float]] = {
        name: []
        for name in (
            "auprc",
            "auroc",
            "brier",
            "calibration_slope",
            "calibration_intercept",
        )
    }
    rng = np.random.default_rng(config.confidence_intervals.random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        for _ in range(config.confidence_intervals.bootstrap_iterations):
            indices = _stratified_indices(y_true, rng)
            y_boot = y_true[indices]
            p_boot = y_probability[indices]
            slope, intercept = calibration_slope_intercept(y_boot, p_boot)
            values = {
                "auprc": average_precision_score(y_boot, p_boot),
                "auroc": roc_auc_score(y_boot, p_boot),
                "brier": brier_score_loss(y_boot, p_boot),
                "calibration_slope": slope,
                "calibration_intercept": intercept,
            }
            for name, value in values.items():
                if np.isfinite(value):
                    samples[name].append(float(value))

    alpha = 1.0 - config.confidence_intervals.confidence_level
    estimates: dict[str, EstimateCI] = {}
    for name, values in samples.items():
        estimates[name] = EstimateCI(
            point=float(point[name]),
            lower=float(np.quantile(values, alpha / 2.0)) if values else None,
            upper=float(np.quantile(values, 1.0 - alpha / 2.0)) if values else None,
            method="bootstrap",
            samples=len(values),
            skipped=config.confidence_intervals.bootstrap_iterations - len(values),
        )
    return estimates


def complete_record_sensitivity(
    predictions: pd.DataFrame,
    integrity_rows: Mapping[str, pd.DataFrame],
    draw_matrix: pd.DataFrame,
    config: ProjectConfig,
    *,
    lower_threshold: float = 0.29,
    upper_threshold: float = 0.69,
    tolerance: float = 1e-12,
) -> CompleteRecordResult:
    """Evaluate the deterministic subset with all 12 primary inputs observed."""

    _require_columns(
        predictions,
        {
            "row_id",
            "model",
            "dataset",
            "y_true",
            "y_probability",
            "prediction_basis",
            "prediction_source",
            "prediction_context",
        },
        "prediction summary",
    )
    _require_columns(
        draw_matrix,
        {"dataset", "draw", "row_id", "y_true", "probability"},
        "draw matrix",
    )
    metric_rows: list[dict[str, object]] = []
    band_rows: list[dict[str, object]] = []
    expected = {"holdout": (52, 16), "prospective": (75, 16)}
    for dataset in ("holdout", "prospective"):
        if dataset not in integrity_rows:
            raise ValueError(f"No inference-integrity rows supplied for {dataset}.")
        integrity = integrity_rows[dataset].copy()
        _require_columns(integrity, {"row_id", "has_missing"}, f"{dataset} integrity rows")
        if integrity["row_id"].duplicated().any():
            raise ValueError(f"{dataset} integrity rows contain duplicate row identifiers.")
        integrity["has_missing"] = _boolean_series(
            integrity["has_missing"], label=f"{dataset} has_missing"
        )
        patient = predictions[
            (predictions["model"].astype(str) == "Injury.top12")
            & (predictions["dataset"].astype(str) == dataset)
        ].copy()
        patient = patient.merge(
            integrity[["row_id", "has_missing"]], on="row_id", how="left", validate="one_to_one"
        )
        if patient["has_missing"].isna().any():
            raise ValueError(f"{dataset} predictions could not all be matched to integrity rows.")
        complete = patient.loc[~patient["has_missing"].astype(bool)].sort_values("row_id")
        basis, source, context = _prediction_provenance(
            complete, label=f"{dataset} complete-record predictions"
        )
        expected_n, expected_events = expected[dataset]
        if len(complete) != expected_n or int(complete["y_true"].sum()) != expected_events:
            raise ValueError(
                f"{dataset} complete records do not match the frozen "
                f"{expected_n}-patient/{expected_events}-event subset."
            )

        matrix = draw_matrix[
            (draw_matrix["dataset"].astype(str) == dataset)
            & (draw_matrix["row_id"].isin(complete["row_id"]))
        ].copy()
        per_patient = matrix.groupby("row_id").agg(
            draws=("draw", "nunique"),
            probability_min=("probability", "min"),
            probability_max=("probability", "max"),
            outcome_values=("y_true", "nunique"),
        )
        if (
            len(per_patient) != expected_n
            or not per_patient["draws"].eq(400).all()
            or not per_patient["outcome_values"].eq(1).all()
        ):
            raise ValueError(f"{dataset} complete records do not have a complete 400-draw grid.")
        ranges = per_patient["probability_max"] - per_patient["probability_min"]
        maximum_range = float(ranges.max())
        if maximum_range > tolerance:
            raise ValueError(
                f"{dataset} purported complete records move across frozen draws "
                f"(maximum range {maximum_range:.3g})."
            )

        y_true = complete["y_true"].to_numpy(dtype=int)
        probability = complete["y_probability"].to_numpy(dtype=float)
        estimates = _metric_estimates(y_true, probability, config)
        prevalence = proportion_ci(int(y_true.sum()), len(y_true), config.confidence_intervals)
        metric_rows.append(
            {
                "dataset": dataset,
                "cohort": "Internal hold-out" if dataset == "holdout" else "Prospective",
                "n": len(complete),
                "events": int(y_true.sum()),
                "prevalence": prevalence.point,
                "prevalence_formatted": format_estimate(prevalence),
                "ap_to_prevalence_ratio": (
                    estimates["auprc"].point / prevalence.point
                ),
                **{name: estimate.point for name, estimate in estimates.items()},
                **{
                    f"{name}_formatted": format_estimate(estimate)
                    for name, estimate in estimates.items()
                },
                **{
                    f"{name}_bootstrap_samples": estimate.samples
                    for name, estimate in estimates.items()
                },
                "maximum_probability_range_across_400_draws": maximum_range,
                "prediction_basis": basis,
                "prediction_source": source,
                "prediction_context": context,
                "selection_rule": "all 12 primary-model inputs observed",
            }
        )

        bands = assign_bands(probability, lower_threshold, upper_threshold)
        for band, label in (
            ("Green", "Discharge"),
            ("Amber", "Reassess"),
            ("Red", "MRI referral"),
        ):
            mask = bands == band
            band_rows.append(
                {
                    "dataset": dataset,
                    "cohort": "Internal hold-out" if dataset == "holdout" else "Prospective",
                    "triage_band": band,
                    "action_label": label,
                    "n": int(mask.sum()),
                    "injuries": int(y_true[mask].sum()),
                    "no_injury": int(mask.sum() - y_true[mask].sum()),
                    "prediction_basis": basis,
                    "prediction_source": source,
                    "prediction_context": context,
                }
            )
    return CompleteRecordResult(
        metrics=pd.DataFrame(metric_rows), band_counts=pd.DataFrame(band_rows)
    )


def performance_context_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report AP relative to the cohort prevalence (the AP no-skill baseline)."""

    rows: list[dict[str, object]] = []
    for dataset in ("holdout", "prospective"):
        cohort = predictions[
            (predictions["model"].astype(str) == "Injury.top12")
            & (predictions["dataset"].astype(str) == dataset)
        ]
        y_true = cohort["y_true"].to_numpy(dtype=int)
        probability = cohort["y_probability"].to_numpy(dtype=float)
        basis, source, context = _prediction_provenance(
            cohort, label=f"{dataset} performance-context predictions"
        )
        prevalence = float(np.mean(y_true))
        ap = float(average_precision_score(y_true, probability))
        rows.append(
            {
                "dataset": dataset,
                "n": len(cohort),
                "events": int(y_true.sum()),
                "prevalence_no_skill_ap": prevalence,
                "average_precision": ap,
                "ap_to_prevalence_ratio": ap / prevalence,
                "prediction_basis": basis,
                "prediction_source": source,
                "prediction_context": context,
            }
        )
    return pd.DataFrame(rows)


def calibration_curve_bootstrap(
    predictions: pd.DataFrame,
    config: ProjectConfig,
    *,
    grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Create model-based calibration curves and pointwise bootstrap bands."""

    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or not np.all((grid > 0.0) & (grid < 1.0)):
        raise ValueError("Calibration grid must be a one-dimensional vector inside (0, 1).")
    grid_logit = safe_logit(grid)
    rows: list[dict[str, object]] = []
    for dataset in ("holdout", "prospective"):
        cohort = predictions[
            (predictions["model"].astype(str) == "Injury.top12")
            & (predictions["dataset"].astype(str) == dataset)
        ].copy()
        y_true = cohort["y_true"].to_numpy(dtype=int)
        probability = cohort["y_probability"].to_numpy(dtype=float)
        basis, source, context = _prediction_provenance(
            cohort, label=f"{dataset} calibration predictions"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            slope, intercept = calibration_slope_intercept(y_true, probability)
        point_curve = expit(intercept + slope * grid_logit)
        curves: list[np.ndarray] = []
        rng = np.random.default_rng(config.confidence_intervals.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            for _ in range(config.confidence_intervals.bootstrap_iterations):
                indices = _stratified_indices(y_true, rng)
                boot_slope, boot_intercept = calibration_slope_intercept(
                    y_true[indices], probability[indices]
                )
                if np.isfinite(boot_slope) and np.isfinite(boot_intercept):
                    curves.append(expit(boot_intercept + boot_slope * grid_logit))
        if not curves:
            raise ValueError(f"No valid calibration bootstrap curves for {dataset}.")
        matrix = np.vstack(curves)
        alpha = 1.0 - config.confidence_intervals.confidence_level
        lower = np.quantile(matrix, alpha / 2.0, axis=0)
        upper = np.quantile(matrix, 1.0 - alpha / 2.0, axis=0)
        for index, predicted in enumerate(grid):
            rows.append(
                {
                    "dataset": dataset,
                    "predicted_probability": float(predicted),
                    "calibrated_probability": float(point_curve[index]),
                    "ci_lower": float(lower[index]),
                    "ci_upper": float(upper[index]),
                    "calibration_slope": float(slope),
                    "calibration_intercept": float(intercept),
                    "bootstrap_iterations": config.confidence_intervals.bootstrap_iterations,
                    "valid_bootstrap_curves": len(curves),
                    "prediction_basis": basis,
                    "prediction_source": source,
                    "prediction_context": context,
                }
            )
    return pd.DataFrame(rows)


def run_capacity_match(
    config: ProjectConfig,
    *,
    split: str = "prospective",
    predictions_path: Path | None = None,
    output_dir: Path | None = None,
    capacity: int | None = None,
) -> tuple[CapacityMatchedResult, dict[str, Path]]:
    """Run the capacity-matched comparison from saved patient predictions.

    This is the comparison the paper's conclusion rests on: CamKIT-AI ranked
    down to the referral volume CamKIT actually used. It loads no model and
    scores nothing, so it runs without the legacy training environment.
    """
    source = predictions_path or (
        config.paths.output_root
        / "analysis"
        / f"camkit_ai_vs_camkit_patient_predictions_{split}.csv"
    )
    if not source.exists():
        raise FileNotFoundError(
            f"Patient predictions not found at {source}. Run 'compare-models' first."
        )

    result = capacity_matched_analysis(
        pd.read_csv(source), config, capacity=capacity
    )

    output_root = output_dir or (config.paths.output_root / "analysis")
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "policies": output_root / f"capacity_matched_policies_{split}.csv",
        "overlap": output_root / f"capacity_matched_overlap_{split}.csv",
        "curve": output_root / f"capacity_cumulative_curve_{split}.csv",
    }
    result.policies.to_csv(paths["policies"], index=False)
    result.overlap.to_csv(paths["overlap"], index=False)
    result.curve.to_csv(paths["curve"], index=False)
    return result, paths
