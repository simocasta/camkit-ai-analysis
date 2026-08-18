from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import numpy as np
import pandas as pd

from camkit_ai.confidence_intervals import EstimateCI, bootstrap_metric_bundle, format_estimate, proportion_ci
from camkit_ai.config import ProjectConfig
from camkit_ai.data import load_processed_dataset
from camkit_ai.metrics import discrimination_metrics, positive_class_probabilities
from camkit_ai.model_io import legacy_model_path, load_legacy_model
from camkit_ai.pooling import pool_predictions, pooled_basis_label
from camkit_ai.oof import load_or_generate_oof_predictions
from camkit_ai.presets import MANUSCRIPT_MODEL_SPECS, subgroup_masks, validate_variant
from camkit_ai.thresholds import (
    bootstrap_threshold_stability,
    evaluate_locked_threshold_pair,
    evaluate_threshold_pair,
    select_safety_first_thresholds,
    summarize_threshold_stability,
    threshold_sweep,
)


@dataclass
class ModelEvaluationResult:
    discrimination: pd.DataFrame
    subgroups: pd.DataFrame
    thresholds: pd.DataFrame
    predictions: pd.DataFrame
    threshold_sweep: pd.DataFrame | None = None
    threshold_sweep_source: str | None = None
    selected_thresholds: dict[str, object] | None = None
    oof_predictions: pd.DataFrame | None = None
    threshold_stability: pd.DataFrame | None = None
    threshold_stability_summary: pd.DataFrame | None = None
    prediction_basis: str | None = None
    prediction_source: str = "scored_from_model"
    prediction_context: str | None = None


#: Written into ``prediction_source`` when probabilities are read from the frozen
#: draw matrix rather than produced by calling the model. Labelling those rows
#: "scored_from_model" would misdescribe how the numbers were obtained.
FROZEN_MATRIX_SOURCE = "frozen_draw_matrix_mean"


def _required_probabilities(
    precomputed: Mapping[str, Mapping[str, np.ndarray]] | None,
    target: str,
    variant: str,
) -> Mapping[str, np.ndarray] | None:
    """Look up precomputed probabilities, refusing to fall back to scoring.

    A missing key would otherwise silently score the model instead, producing a
    table labelled with the fixed-cohort basis whose numbers came from a fresh
    set of draws. Failing here keeps the label and the computation together.
    """
    if precomputed is None:
        return None
    key = f"{target}.{variant}"
    if key not in precomputed:
        raise ValueError(
            f"Precomputed probabilities were supplied but none for {key}; "
            "refusing to fall back to scoring the model."
        )
    return precomputed[key]


def _stamp_context(frame: pd.DataFrame, context: str | None) -> pd.DataFrame:
    """Record what was in the scoring call, when the caller declared one.

    Left absent unless a context is supplied, so analyses that predate the
    fixed-batch estimand keep exactly the columns they had.
    """
    if context is not None and not frame.empty:
        frame["prediction_context"] = context
    return frame


def _estimate_rows(
    estimates: dict[str, EstimateCI],
    *,
    model: str,
    target: str,
    variant: str,
    dataset: str,
    section: str,
    subgroup: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric, estimate in estimates.items():
        rows.append(
            estimate.as_record(
                metric,
                model=model,
                target=target,
                variant=variant,
                dataset=dataset,
                section=section,
                subgroup=subgroup,
            )
        )
    return rows


def _overall_rows(
    model_label: str,
    target: str,
    variant: str,
    dataset: str,
    y_true: np.ndarray,
    config: ProjectConfig,
) -> list[dict[str, object]]:
    prevalence = proportion_ci(
        int(np.sum(y_true)),
        int(len(y_true)),
        config=config.confidence_intervals,
    )
    return [
        {
            "metric": "n",
            "point": int(len(y_true)),
            "ci_lower": None,
            "ci_upper": None,
            "ci_method": "count",
            "ci_samples": None,
            "ci_skipped": None,
            "numerator": int(len(y_true)),
            "denominator": int(len(y_true)),
            "model": model_label,
            "target": target,
            "variant": variant,
            "dataset": dataset,
            "section": "summary",
            "subgroup": None,
        },
        {
            "metric": "events",
            "point": int(np.sum(y_true)),
            "ci_lower": None,
            "ci_upper": None,
            "ci_method": "count",
            "ci_samples": None,
            "ci_skipped": None,
            "numerator": int(np.sum(y_true)),
            "denominator": int(len(y_true)),
            "model": model_label,
            "target": target,
            "variant": variant,
            "dataset": dataset,
            "section": "summary",
            "subgroup": None,
        },
        prevalence.as_record(
            "prevalence",
            model=model_label,
            target=target,
            variant=variant,
            dataset=dataset,
            section="summary",
            subgroup=None,
        ),
    ]


def _predict_probabilities(
    model,
    features: pd.DataFrame,
    *,
    pooled_draws: int | None = None,
    pool_seed: int = 0,
) -> np.ndarray:
    """Score a feature frame, optionally pooling over imputation draws.

    The saved pipeline imputes stochastically at inference, so a single call
    returns one draw from a distribution of possible outputs. Passing
    pooled_draws averages over a fixed seed sequence instead, which is both
    reproducible and independent of which draw happened to be taken.
    """
    if pooled_draws is None:
        return positive_class_probabilities(model.predict_proba(features))
    return pool_predictions(model, features, n_draws=pooled_draws, base_seed=pool_seed)


def _discrimination_estimates(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config: ProjectConfig,
) -> dict[str, EstimateCI]:
    point = discrimination_metrics(y_true, y_prob)
    bundle = bootstrap_metric_bundle(
        y_true,
        y_prob,
        {
            "auprc": lambda yt, yp: discrimination_metrics(yt, yp)["auprc"],
            "auroc": lambda yt, yp: discrimination_metrics(yt, yp)["auroc"],
            "brier": lambda yt, yp: discrimination_metrics(yt, yp)["brier"],
            "calibration_slope": lambda yt, yp: discrimination_metrics(yt, yp)["calibration_slope"],
            "calibration_intercept": lambda yt, yp: discrimination_metrics(yt, yp)["calibration_intercept"],
        },
        config=config.confidence_intervals,
        requires_both_classes=True,
    )
    for metric, point_value in point.items():
        if metric in {"n", "events", "prevalence"}:
            continue
        if metric not in bundle:
            continue
        if np.isfinite(point_value):
            bundle[metric].point = float(point_value)
    return bundle


def _evaluate_split(
    model,
    frame: pd.DataFrame,
    target: str,
    variant: str,
    dataset: str,
    config: ProjectConfig,
    y_prob: np.ndarray | None = None,
    pooled_draws: int | None = None,
    pool_seed: int = 0,
) -> pd.DataFrame:
    features = frame.drop(columns=[target])
    y_true = frame[target].to_numpy()
    if y_prob is None:
        y_prob = _predict_probabilities(
            model, features, pooled_draws=pooled_draws, pool_seed=pool_seed
        )
    model_label = f"{target}.{variant}"
    rows = _overall_rows(model_label, target, variant, dataset, y_true, config)
    rows.extend(
        _estimate_rows(
            _discrimination_estimates(y_true, y_prob, config),
            model=model_label,
            target=target,
            variant=variant,
            dataset=dataset,
            section="discrimination",
        )
    )
    frame_out = pd.DataFrame(rows)
    frame_out["formatted"] = frame_out.apply(
        lambda row: format_estimate(
            EstimateCI(
                point=float(row["point"]) if pd.notna(row["point"]) else float("nan"),
                lower=float(row["ci_lower"]) if pd.notna(row["ci_lower"]) else None,
                upper=float(row["ci_upper"]) if pd.notna(row["ci_upper"]) else None,
                method=str(row["ci_method"]),
            )
        ),
        axis=1,
    )
    return frame_out


def _evaluate_subgroups(
    base_frame: pd.DataFrame,
    full_split_probabilities: np.ndarray,
    target: str,
    variant: str,
    dataset: str,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Describe subgroups by subsetting the already frozen split predictions.

    Rescoring each subgroup would consume the stochastic imputer's random
    stream under a different batch composition and could assign a patient a
    probability different from the definitive full-cohort prediction table.
    Reuse makes every subgroup estimate a literal subset of that table.
    """

    full_split_probabilities = np.asarray(full_split_probabilities, dtype=float)
    if len(base_frame) != len(full_split_probabilities):
        raise ValueError(
            f"{dataset}: subgroup frame and definitive prediction vector differ in length."
        )
    rows: list[dict[str, object]] = []
    for subgroup, mask in subgroup_masks(base_frame).items():
        subgroup_frame = base_frame.loc[mask, [target]].copy()
        if subgroup_frame.empty:
            continue
        y_true = subgroup_frame[target].to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        y_prob = full_split_probabilities[np.asarray(mask, dtype=bool)]
        estimates = bootstrap_metric_bundle(
            y_true,
            y_prob,
            {
                "auprc": lambda yt, yp: discrimination_metrics(yt, yp)["auprc"],
                "auroc": lambda yt, yp: discrimination_metrics(yt, yp)["auroc"],
            },
            config=config.confidence_intervals,
            requires_both_classes=True,
        )
        model_label = f"{target}.{variant}"
        rows.append(
            {
                "metric": "n",
                "point": int(len(subgroup_frame)),
                "ci_lower": None,
                "ci_upper": None,
                "ci_method": "count",
                "ci_samples": None,
                "ci_skipped": None,
                "numerator": int(len(subgroup_frame)),
                "denominator": int(len(subgroup_frame)),
                "model": model_label,
                "target": target,
                "variant": variant,
                "dataset": dataset,
                "section": "subgroup",
                "subgroup": subgroup,
            }
        )
        rows.append(
            {
                "metric": "events",
                "point": int(np.sum(y_true)),
                "ci_lower": None,
                "ci_upper": None,
                "ci_method": "count",
                "ci_samples": None,
                "ci_skipped": None,
                "numerator": int(np.sum(y_true)),
                "denominator": int(len(subgroup_frame)),
                "model": model_label,
                "target": target,
                "variant": variant,
                "dataset": dataset,
                "section": "subgroup",
                "subgroup": subgroup,
            }
        )
        rows.extend(
            _estimate_rows(
                estimates,
                model=model_label,
                target=target,
                variant=variant,
                dataset=dataset,
                section="subgroup",
                subgroup=subgroup,
            )
        )
    subgroup_df = pd.DataFrame(rows)
    if subgroup_df.empty:
        return subgroup_df
    subgroup_df["formatted"] = subgroup_df.apply(
        lambda row: format_estimate(
            EstimateCI(
                point=float(row["point"]) if pd.notna(row["point"]) else float("nan"),
                lower=float(row["ci_lower"]) if pd.notna(row["ci_lower"]) else None,
                upper=float(row["ci_upper"]) if pd.notna(row["ci_upper"]) else None,
                method=str(row["ci_method"]),
            )
        ),
        axis=1,
    )
    return subgroup_df


def evaluate_model(
    config: ProjectConfig,
    target: str,
    variant: str,
    *,
    include_subgroups: bool = False,
    include_thresholds: bool = False,
    pooled_draws: int | None = None,
    pool_seed: int | None = None,
    basis_label: str | None = None,
    prediction_context: str | None = None,
    precomputed_probabilities: Mapping[str, np.ndarray] | None = None,
) -> ModelEvaluationResult:
    validate_variant(target, variant)

    # Supplying probabilities skips inference altogether. The analysis reads its
    # primary estimates straight out of the frozen draw matrix, so re-running 400
    # inference draws to arrive at numbers that must equal the saved mean anyway
    # would cost hours and prove nothing the matrix does not already record.
    model = None
    if precomputed_probabilities is None:
        model = load_legacy_model(legacy_model_path(config, target, variant))

    seed = config.confidence_intervals.random_state if pool_seed is None else pool_seed
    # basis_label lets the analysis declare the fixed-cohort estimand, whose name
    # carries the whole seed range. The computation is identical either way, so
    # the override may only rename a pooled basis, never a single-draw one.
    if basis_label is not None and pooled_draws is None:
        raise ValueError("A basis label override requires pooled draws.")
    basis = basis_label or pooled_basis_label(
        pooled_draws, seed if pooled_draws is not None else None
    )
    context = prediction_context
    source = (
        FROZEN_MATRIX_SOURCE
        if precomputed_probabilities is not None
        else "scored_from_model"
    )

    discrimination_frames: list[pd.DataFrame] = []
    subgroup_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, object]] = []
    sweep_df: pd.DataFrame | None = None
    sweep_source: str | None = None
    selected_thresholds: dict[str, object] | None = None
    oof_predictions: pd.DataFrame | None = None
    threshold_stability: pd.DataFrame | None = None
    threshold_stability_summary: pd.DataFrame | None = None

    split_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name in ("holdout", "prospective"):
        eval_frame = load_processed_dataset(config, target, split_name, variant)
        y_true = eval_frame[target].to_numpy()
        if precomputed_probabilities is not None:
            if split_name not in precomputed_probabilities:
                raise ValueError(
                    f"{target}.{variant}: no precomputed probabilities for {split_name}."
                )
            y_prob = np.asarray(precomputed_probabilities[split_name], dtype=float)
            if len(y_prob) != len(eval_frame):
                raise ValueError(
                    f"{target}.{variant} {split_name}: {len(y_prob)} precomputed "
                    f"probabilities for {len(eval_frame)} records."
                )
        else:
            y_prob = _predict_probabilities(
                model,
                eval_frame.drop(columns=[target]),
                pooled_draws=pooled_draws,
                pool_seed=seed,
            )
        prediction_frames.append(
            _stamp_context(
                pd.DataFrame(
                    {
                        "row_id": range(1, len(eval_frame) + 1),
                        "model": f"{target}.{variant}",
                        "target": target,
                        "variant": variant,
                        "dataset": split_name,
                        "y_true": y_true.astype(int),
                        "y_probability": y_prob.astype(float),
                        "prediction_basis": basis,
                        "prediction_source": source,
                    }
                ),
                context,
            )
        )
        discrimination_frames.append(
            _evaluate_split(
                model,
                eval_frame,
                target,
                variant,
                split_name,
                config,
                y_prob=y_prob,
            )
        )
        discrimination_frames[-1]["prediction_basis"] = basis
        discrimination_frames[-1]["prediction_source"] = source
        _stamp_context(discrimination_frames[-1], context)
        split_predictions[split_name] = (y_true, y_prob)

        if include_subgroups:
            subgroup_base = load_processed_dataset(config, target, split_name, "full")
            subgroup_outcome = subgroup_base[target].to_numpy()
            if not np.array_equal(subgroup_outcome, y_true):
                raise ValueError(
                    f"{split_name}: full and {variant} processed datasets have "
                    "different outcome ordering; subgroup reuse would be invalid."
                )
            subgroup_result = _evaluate_subgroups(
                subgroup_base,
                y_prob,
                target,
                variant,
                split_name,
                config,
            )
            if not subgroup_result.empty:
                subgroup_result["prediction_basis"] = basis
                subgroup_result["prediction_source"] = source
                _stamp_context(subgroup_result, context)
                subgroup_frames.append(subgroup_result)

    if include_thresholds:
        if target != "Injury" or variant != "top12":
            raise ValueError("Threshold analysis is only implemented for the Injury top12 model.")
        if config.thresholds.selection_source == "holdout":
            selection_y, selection_prob = split_predictions["holdout"]
            sweep_df = threshold_sweep(
                selection_y,
                selection_prob,
                step=config.thresholds.step,
            )
            sweep_source = "holdout"
        elif config.thresholds.selection_source == "training_oof":
            oof_predictions = load_or_generate_oof_predictions(config, target, variant)
            selection_y = oof_predictions["y_true"].to_numpy(dtype=int)
            selection_prob = oof_predictions["y_probability_oof"].to_numpy(dtype=float)
            sweep_df = threshold_sweep(
                selection_y,
                selection_prob,
                step=config.thresholds.step,
            )
            sweep_source = "training_oof"
            prediction_frames.append(
                _stamp_context(
                    pd.DataFrame(
                        {
                            "row_id": oof_predictions["row_id"].astype(int),
                            "model": f"{target}.{variant}",
                            "target": target,
                            "variant": variant,
                            "dataset": "training_oof",
                            "y_true": selection_y.astype(int),
                            "y_probability": selection_prob.astype(float),
                            "prediction_basis": "training_oof_repeated_cv",
                            "prediction_source": "generated_repeated_cv",
                        }
                    ),
                    # Cross-validated predictions are not cohort-batch scored, so
                    # they never carry the fixed-cohort context even when one is
                    # declared for the evaluation splits.
                    None,
                )
            )
        else:
            raise ValueError(
                f"Unsupported threshold selection source: {config.thresholds.selection_source}"
            )

        if config.thresholds.use_locked_thresholds:
            selected = evaluate_locked_threshold_pair(sweep_df, config.thresholds)
            threshold_status = "historical_locked"
        else:
            selected = select_safety_first_thresholds(sweep_df, config.thresholds)
            threshold_status = "selected_on_current_prediction_basis"

        selected_thresholds = {
            "lower_threshold": selected.lower_threshold,
            "upper_threshold": selected.upper_threshold,
            "threshold_gap": selected.threshold_gap,
            "selection_source": sweep_source,
            "selection_basis": config.thresholds.selection_basis,
            "threshold_status": threshold_status,
            "prediction_basis": (
                basis if sweep_source == "holdout" else "training_oof_repeated_cv"
            ),
            "prediction_source": source,
            "lower_feasible": selected.lower_feasible,
            "upper_feasible": selected.upper_feasible,
            "feasible_pair": selected.feasible_pair,
        }
        if context is not None and sweep_source == "holdout":
            selected_thresholds["prediction_context"] = context
        if oof_predictions is not None:
            selected_thresholds["oof_n_splits"] = config.thresholds.oof_n_splits
            selected_thresholds["oof_n_repeats"] = config.thresholds.oof_n_repeats

        model_label = f"{target}.{variant}"
        if oof_predictions is not None:
            threshold_rows.extend(
                evaluate_threshold_pair(
                    selection_y,
                    selection_prob,
                    selected.lower_threshold,
                    selected.upper_threshold,
                    config.confidence_intervals,
                    "training_oof",
                    model_label,
                )
            )
        for split_name, (y_true, y_prob) in split_predictions.items():
            threshold_rows.extend(
                evaluate_threshold_pair(
                    y_true,
                    y_prob,
                    selected.lower_threshold,
                    selected.upper_threshold,
                    config.confidence_intervals,
                    split_name,
                    model_label,
                )
            )
        # Run for whichever set the thresholds were actually selected on. The
        # manuscript selects on the hold-out set, so gating this on training_oof
        # left the selected cut-points with no stability estimate.
        if config.thresholds.bootstrap_thresholds and sweep_source is not None:
            threshold_stability = bootstrap_threshold_stability(
                selection_y,
                selection_prob,
                config.thresholds,
                config.confidence_intervals,
            )
            threshold_stability_summary = summarize_threshold_stability(
                threshold_stability,
                selected=selected,
                target=target,
                variant=variant,
                selection_source=sweep_source,
            )

    discrimination = pd.concat(discrimination_frames, ignore_index=True)
    subgroups = (
        pd.concat(subgroup_frames, ignore_index=True) if subgroup_frames else pd.DataFrame()
    )
    thresholds = pd.DataFrame(threshold_rows)
    if not thresholds.empty:
        thresholds["target"] = target
        thresholds["variant"] = variant
        thresholds["section"] = "threshold"
        thresholds["subgroup"] = None
        thresholds["prediction_basis"] = np.where(
            thresholds["dataset"].eq("training_oof"),
            "training_oof_repeated_cv",
            basis,
        )
        thresholds["prediction_source"] = source
        if context is not None:
            thresholds["prediction_context"] = np.where(
                thresholds["dataset"].eq("training_oof"),
                None,
                context,
            )
        thresholds["formatted"] = thresholds.apply(
            lambda row: format_estimate(
                EstimateCI(
                    point=float(row["point"]) if pd.notna(row["point"]) else float("nan"),
                    lower=float(row["ci_lower"]) if pd.notna(row["ci_lower"]) else None,
                    upper=float(row["ci_upper"]) if pd.notna(row["ci_upper"]) else None,
                    method=str(row["ci_method"]),
                )
            ),
            axis=1,
        )

    return ModelEvaluationResult(
        discrimination=discrimination,
        subgroups=subgroups,
        thresholds=thresholds,
        predictions=pd.concat(prediction_frames, ignore_index=True),
        threshold_sweep=sweep_df,
        threshold_sweep_source=sweep_source,
        selected_thresholds=selected_thresholds,
        oof_predictions=oof_predictions,
        threshold_stability=threshold_stability,
        threshold_stability_summary=threshold_stability_summary,
        prediction_basis=basis,
        prediction_source=source,
        prediction_context=context,
    )


def _save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_model_evaluation(
    result: ModelEvaluationResult,
    output_dir: Path,
) -> None:
    _save_dataframe(result.discrimination, output_dir / "discrimination.csv")
    _save_dataframe(result.subgroups, output_dir / "subgroups.csv")
    _save_dataframe(result.thresholds, output_dir / "thresholds.csv")
    _save_dataframe(result.predictions, output_dir / "predictions.csv")
    if result.threshold_sweep is not None:
        source = result.threshold_sweep_source or "holdout"
        sweep = result.threshold_sweep.copy()
        sweep["prediction_basis"] = (
            result.prediction_basis
            if source == "holdout"
            else "training_oof_repeated_cv"
        )
        sweep["prediction_source"] = result.prediction_source
        if source == "holdout":
            _stamp_context(sweep, result.prediction_context)
        _save_dataframe(sweep, output_dir / f"threshold_sweep_{source}.csv")
    if result.selected_thresholds is not None:
        pd.DataFrame([result.selected_thresholds]).to_csv(
            output_dir / "selected_thresholds.csv",
            index=False,
        )
    if result.oof_predictions is not None:
        _save_dataframe(result.oof_predictions, output_dir / "oof_predictions.csv")
    if result.threshold_stability is not None:
        stability_source = result.threshold_sweep_source or "training_oof"
        stability = result.threshold_stability.copy()
        stability["prediction_basis"] = (
            result.prediction_basis
            if stability_source == "holdout"
            else "training_oof_repeated_cv"
        )
        stability["prediction_source"] = result.prediction_source
        if stability_source == "holdout":
            _stamp_context(stability, result.prediction_context)
        stability["stability_estimand"] = "reoptimised_thresholds_in_bootstrap_samples"
        _save_dataframe(
            stability,
            output_dir / f"threshold_stability_{stability_source}.csv",
        )
    if result.threshold_stability_summary is not None:
        stability_summary = result.threshold_stability_summary.copy()
        stability_summary["prediction_basis"] = (
            result.prediction_basis
            if (result.threshold_sweep_source or "training_oof") == "holdout"
            else "training_oof_repeated_cv"
        )
        stability_summary["prediction_source"] = result.prediction_source
        if (result.threshold_sweep_source or "training_oof") == "holdout":
            _stamp_context(stability_summary, result.prediction_context)
        stability_summary["stability_estimand"] = (
            "reoptimised_thresholds_in_bootstrap_samples"
        )
        _save_dataframe(
            stability_summary,
            output_dir / "threshold_stability_summary.csv",
        )


def report_manuscript(
    config: ProjectConfig,
    *,
    pooled_draws: int | None = None,
    pool_seed: int | None = None,
    basis_label: str | None = None,
    prediction_context: str | None = None,
    model_specs: Sequence[tuple[str, str]] | None = None,
    precomputed_probabilities: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    include_subgroups: bool = True,
) -> dict[str, Path]:
    """Evaluate the manuscript models and write the reporting tables.

    ``model_specs`` narrows the set evaluated; the analysis reports only the
    primary 12-feature model on the definitive basis and leaves the secondary
    models on their historical single-call estimates rather than spending 400
    inference draws each on models the manuscript does not report.
    ``precomputed_probabilities`` is keyed by ``"{target}.{variant}"`` then by
    split, and skips inference for those models entirely.
    """
    output_dir = config.paths.output_root / "manuscript"
    output_dir.mkdir(parents=True, exist_ok=True)

    discrimination_frames: list[pd.DataFrame] = []
    subgroup_frames: list[pd.DataFrame] = []
    threshold_frames: list[pd.DataFrame] = []
    threshold_sweeps: list[pd.DataFrame] = []
    selected_thresholds: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    oof_prediction_frames: list[pd.DataFrame] = []
    threshold_stability_frames: list[pd.DataFrame] = []
    threshold_stability_summary_frames: list[pd.DataFrame] = []

    for target, variant in model_specs or MANUSCRIPT_MODEL_SPECS:
        result = evaluate_model(
            config,
            target,
            variant,
            include_subgroups=include_subgroups and target == "Injury",
            include_thresholds=(target == "Injury" and variant == "top12"),
            pooled_draws=pooled_draws,
            pool_seed=pool_seed,
            basis_label=basis_label,
            prediction_context=prediction_context,
            precomputed_probabilities=_required_probabilities(
                precomputed_probabilities, target, variant
            ),
        )
        discrimination_frames.append(result.discrimination)
        prediction_frames.append(result.predictions)
        if not result.subgroups.empty:
            subgroup_frames.append(result.subgroups)
        if not result.thresholds.empty:
            threshold_frames.append(result.thresholds)
        if result.threshold_sweep is not None:
            sweep = result.threshold_sweep.copy()
            sweep["target"] = target
            sweep["variant"] = variant
            sweep["selection_source"] = result.threshold_sweep_source
            sweep["prediction_basis"] = (
                result.prediction_basis
                if result.threshold_sweep_source == "holdout"
                else "training_oof_repeated_cv"
            )
            sweep["prediction_source"] = result.prediction_source
            if result.threshold_sweep_source == "holdout":
                _stamp_context(sweep, result.prediction_context)
            threshold_sweeps.append(sweep)
        if result.selected_thresholds is not None:
            selected = pd.DataFrame([result.selected_thresholds])
            selected["target"] = target
            selected["variant"] = variant
            selected_thresholds.append(selected)
        if result.oof_predictions is not None:
            oof_prediction_frames.append(result.oof_predictions.copy())
        if result.threshold_stability is not None:
            stability = result.threshold_stability.copy()
            stability["target"] = target
            stability["variant"] = variant
            stability["prediction_basis"] = (
                result.prediction_basis
                if result.threshold_sweep_source == "holdout"
                else "training_oof_repeated_cv"
            )
            stability["prediction_source"] = result.prediction_source
            if result.threshold_sweep_source == "holdout":
                _stamp_context(stability, result.prediction_context)
            stability["stability_estimand"] = (
                "reoptimised_thresholds_in_bootstrap_samples"
            )
            threshold_stability_frames.append(stability)
        if result.threshold_stability_summary is not None:
            stability_summary = result.threshold_stability_summary.copy()
            stability_summary["prediction_basis"] = (
                result.prediction_basis
                if result.threshold_sweep_source == "holdout"
                else "training_oof_repeated_cv"
            )
            stability_summary["prediction_source"] = result.prediction_source
            if result.threshold_sweep_source == "holdout":
                _stamp_context(stability_summary, result.prediction_context)
            stability_summary["stability_estimand"] = (
                "reoptimised_thresholds_in_bootstrap_samples"
            )
            threshold_stability_summary_frames.append(stability_summary)

    threshold_sweep_source = config.thresholds.selection_source
    outputs = {
        "discrimination": output_dir / "discrimination_summary.csv",
        "subgroups": output_dir / "subgroup_summary.csv",
        "thresholds": output_dir / "threshold_summary.csv",
        "threshold_sweep": output_dir / f"threshold_sweep_{threshold_sweep_source}.csv",
        "selected_thresholds": output_dir / "selected_thresholds.csv",
        "predictions": output_dir / "prediction_summary.csv",
        "oof_predictions": output_dir / "oof_prediction_summary.csv",
        "threshold_stability": output_dir
        / f"threshold_stability_{threshold_sweep_source}.csv",
        "threshold_stability_summary": output_dir / "threshold_stability_summary.csv",
    }

    _save_dataframe(pd.concat(discrimination_frames, ignore_index=True), outputs["discrimination"])
    if subgroup_frames:
        _save_dataframe(pd.concat(subgroup_frames, ignore_index=True), outputs["subgroups"])
    if threshold_frames:
        _save_dataframe(pd.concat(threshold_frames, ignore_index=True), outputs["thresholds"])
    if threshold_sweeps:
        _save_dataframe(pd.concat(threshold_sweeps, ignore_index=True), outputs["threshold_sweep"])
    if selected_thresholds:
        _save_dataframe(pd.concat(selected_thresholds, ignore_index=True), outputs["selected_thresholds"])
    if prediction_frames:
        _save_dataframe(pd.concat(prediction_frames, ignore_index=True), outputs["predictions"])
    if oof_prediction_frames:
        _save_dataframe(pd.concat(oof_prediction_frames, ignore_index=True), outputs["oof_predictions"])
    if threshold_stability_frames:
        _save_dataframe(
            pd.concat(threshold_stability_frames, ignore_index=True),
            outputs["threshold_stability"],
        )
    if threshold_stability_summary_frames:
        _save_dataframe(
            pd.concat(threshold_stability_summary_frames, ignore_index=True),
            outputs["threshold_stability_summary"],
        )

    return outputs
