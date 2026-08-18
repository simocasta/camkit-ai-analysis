from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from camkit_ai.camkit_score import (
    CamkitScoreResult,
    TRIAGE_BAND_ORDER,
    evaluate_camkit_score,
    save_camkit_score_outputs,
)
from camkit_ai.confidence_intervals import (
    EstimateCI,
    bootstrap_metric_bundle,
    bootstrap_paired_difference,
    format_estimate,
    proportion_ci,
)
from camkit_ai.config import ProjectConfig
from camkit_ai.data import load_processed_dataset
from camkit_ai.metrics import rank_discrimination_metrics
from camkit_ai.model_io import legacy_model_path, load_legacy_model
from camkit_ai.pooling import predict_with_basis


@dataclass(frozen=True)
class PredictionProvenance:
    """Where a set of CamKIT-AI probabilities came from.

    ``basis`` names the estimand — a single draw or a pooled average over a
    known seed sequence. ``source`` records whether the numbers were reused from
    the frozen manuscript run or scored afresh. Both travel with every output,
    because a comparison silently computed on a different basis than the
    manuscript is exactly the failure the reproducibility audit exists to catch.
    """

    basis: str
    source: str
    context: str | None = None


@dataclass(frozen=True)
class ModelComparisonResult:
    patient_predictions: pd.DataFrame
    summary: pd.DataFrame
    band_counts: pd.DataFrame
    band_agreement: pd.DataFrame
    paired_differences: pd.DataFrame
    long_metrics: pd.DataFrame
    camkit_published_count_check: pd.DataFrame
    camkit_published_counts_match: bool
    provenance: PredictionProvenance


def _frozen_prediction_basis(frozen: pd.DataFrame) -> str:
    """Read the basis recorded alongside frozen predictions.

    Prediction files written before pooling existed carry no basis column. They
    are labelled rather than guessed at, since assuming they are pooled is the
    mistake that would put a single-draw number in the manuscript.
    """
    if "prediction_basis" not in frozen.columns:
        return "unrecorded_predates_pooling"
    values = pd.unique(frozen["prediction_basis"].dropna())
    if len(values) != 1:
        raise ValueError(
            "Frozen CamKIT-AI predictions mix prediction bases "
            f"({sorted(values)}); the comparison would be incoherent."
        )
    return str(values[0])


def _frozen_prediction_source(frozen: pd.DataFrame) -> str:
    """Read how the frozen predictions were obtained, when they say."""
    if "prediction_source" not in frozen.columns:
        return "frozen_manuscript_predictions"
    values = pd.unique(frozen["prediction_source"].dropna())
    if len(values) != 1:
        raise ValueError(
            "Frozen CamKIT-AI predictions mix prediction sources "
            f"({sorted(values)}); the comparison would be incoherent."
        )
    return str(values[0])


def _predict_camkit_ai_probabilities(
    config: ProjectConfig,
    split: str,
    *,
    pooled_draws: int | None = None,
    pool_seed: int | None = None,
    basis_label: str | None = None,
    prediction_context: str | None = None,
) -> tuple[pd.DataFrame, PredictionProvenance]:
    """Return locked CamKIT-AI probabilities for one split and their provenance.

    Frozen predictions from the manuscript run are preferred only when their
    recorded basis matches the basis requested by the caller. This keeps the
    principal descriptive comparison on exactly the manuscript predictions
    without allowing a stale pooled file to override ``--no-pooling`` or a new
    draw count. When no matching frozen file is available, the model is scored
    directly.
    """
    frame = load_processed_dataset(config, "Injury", split, "top12")
    frozen_predictions = config.paths.output_root / "manuscript" / "prediction_summary.csv"
    if frozen_predictions.exists():
        predictions = pd.read_csv(frozen_predictions)
        mask = (
            (predictions["model"] == "Injury.top12")
            & (predictions["target"] == "Injury")
            & (predictions["variant"] == "top12")
            & (predictions["dataset"] == split)
        )
        frozen = predictions.loc[mask].copy()
        if not frozen.empty:
            frozen_basis = _frozen_prediction_basis(frozen)
            seed = config.confidence_intervals.random_state if pool_seed is None else pool_seed
            requested_basis = basis_label or (
                "single_draw_unseeded"
                if pooled_draws is None
                else f"pooled_{pooled_draws}_draws_seed_{seed}"
            )
            if frozen_basis != requested_basis:
                frozen = pd.DataFrame()
        if not frozen.empty:
            if len(frozen) != len(frame):
                raise ValueError(
                    "Frozen CamKIT-AI prediction count does not match the processed "
                    f"{split} frame."
                )
            y_true = frame["Injury"].astype(int).to_numpy()
            if not np.array_equal(frozen["y_true"].astype(int).to_numpy(), y_true):
                raise ValueError(
                    "Frozen CamKIT-AI predictions do not match the processed outcome "
                    f"ordering for {split}."
                )
            return (
                pd.DataFrame(
                    {
                        "row_id": range(1, len(frame) + 1),
                        "Injury": y_true,
                        "camkit_ai_probability": frozen["y_probability"]
                        .astype(float)
                        .to_numpy(),
                    }
                ),
                PredictionProvenance(
                    basis=frozen_basis,
                    # Carry the upstream source rather than flattening it to
                    # "frozen_manuscript_predictions": if those predictions were
                    # read from the frozen draw matrix, the comparator's numbers
                    # were too, and the table should say so.
                    source=_frozen_prediction_source(frozen),
                    context=prediction_context,
                ),
            )

    model = load_legacy_model(legacy_model_path(config, "Injury", "top12"))
    features = frame.drop(columns=["Injury"])
    seed = config.confidence_intervals.random_state if pool_seed is None else pool_seed
    probabilities, basis = predict_with_basis(
        model, features, pooled_draws=pooled_draws, pool_seed=seed
    )
    if basis_label is not None:
        if pooled_draws is None:
            raise ValueError("A basis label override requires pooled draws.")
        basis = basis_label
    return (
        pd.DataFrame(
            {
                "row_id": range(1, len(frame) + 1),
                "Injury": frame["Injury"].astype(int).to_numpy(),
                "camkit_ai_probability": probabilities,
            }
        ),
        PredictionProvenance(
            basis=basis,
            source="scored_from_model",
            context=prediction_context,
        ),
    )


def load_locked_thresholds(
    config: ProjectConfig,
    lower_threshold: float | None,
    upper_threshold: float | None,
) -> tuple[float, float]:
    if lower_threshold is not None and upper_threshold is not None:
        return float(lower_threshold), float(upper_threshold)

    candidates = [
        config.paths.output_root / "manuscript" / "selected_thresholds.csv",
        config.paths.output_root
        / "models"
        / "Injury.top12"
        / "selected_thresholds.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        row = frame.iloc[0]
        lower = float(row["lower_threshold"]) if lower_threshold is None else lower_threshold
        upper = float(row["upper_threshold"]) if upper_threshold is None else upper_threshold
        return lower, upper

    return (
        0.29 if lower_threshold is None else float(lower_threshold),
        0.69 if upper_threshold is None else float(upper_threshold),
    )


def assign_camkit_ai_triage_band(
    probability: pd.Series,
    lower_threshold: float,
    upper_threshold: float,
) -> pd.Series:
    conditions = [
        probability < lower_threshold,
        (probability >= lower_threshold) & (probability < upper_threshold),
        probability >= upper_threshold,
    ]
    labels = np.select(conditions, TRIAGE_BAND_ORDER, default="Unknown")
    return pd.Series(labels, index=probability.index, name="camkit_ai_triage_band")


def _format_ci(estimate: EstimateCI) -> str:
    return format_estimate(estimate, digits=3)


def _auc_estimates(
    y_true: np.ndarray,
    risk_score: np.ndarray,
    config: ProjectConfig,
) -> dict[str, EstimateCI]:
    # Rank metrics only, for the same reason as the paired differences: this is
    # called with the CamKIT score as well as with predicted probabilities, and
    # only the two AUCs are read out. Computing the full bundle would make the
    # function depend on the caller having rescaled the score first.
    point = rank_discrimination_metrics(y_true, risk_score)
    estimates = bootstrap_metric_bundle(
        y_true,
        risk_score,
        {
            "auc_prc": lambda yt, ys: rank_discrimination_metrics(yt, ys)["auprc"],
            "auc_roc": lambda yt, ys: rank_discrimination_metrics(yt, ys)["auroc"],
        },
        config=config.confidence_intervals,
        requires_both_classes=True,
    )
    estimates["auc_prc"].point = float(point["auprc"])
    estimates["auc_roc"].point = float(point["auroc"])
    return estimates


def _wide_ci(prefix: str, estimate: EstimateCI) -> dict[str, object]:
    return {
        prefix: estimate.point,
        f"{prefix}_ci_lower": estimate.lower,
        f"{prefix}_ci_upper": estimate.upper,
        f"{prefix}_formatted": _format_ci(estimate),
    }


def _append_metric_record(
    records: list[dict[str, object]],
    *,
    model: str,
    dataset: str,
    metric: str,
    estimate: EstimateCI,
    section: str,
) -> None:
    records.append(
        estimate.as_record(
            metric,
            model=model,
            dataset=dataset,
            section=section,
            formatted=_format_ci(estimate),
        )
    )


def _evaluate_triage_model(
    *,
    y_true: np.ndarray,
    risk_score: np.ndarray,
    triage_band: pd.Series,
    model_name: str,
    dataset: str,
    config: ProjectConfig,
) -> tuple[dict[str, object], pd.DataFrame, list[dict[str, object]]]:
    y_true = np.asarray(y_true).astype(int)
    risk_score = np.asarray(risk_score, dtype=float)
    triage_band = pd.Series(triage_band).astype(str)
    n = int(len(y_true))
    events = int(np.sum(y_true))

    green_mask = triage_band == "Green"
    amber_mask = triage_band == "Amber"
    red_mask = triage_band == "Red"

    green_n = int(green_mask.sum())
    amber_n = int(amber_mask.sum())
    red_n = int(red_mask.sum())
    green_injuries = int(np.sum(green_mask & (y_true == 1)))
    green_no_injuries = int(np.sum(green_mask & (y_true == 0)))
    red_injuries = int(np.sum(red_mask & (y_true == 1)))
    red_no_injuries = int(np.sum(red_mask & (y_true == 0)))
    non_green_injuries = int(np.sum((~green_mask) & (y_true == 1)))
    non_red_no_injuries = int(np.sum((~red_mask) & (y_true == 0)))

    auc = _auc_estimates(y_true, risk_score, config)
    prevalence = proportion_ci(events, n, config.confidence_intervals)
    green_rate = proportion_ci(green_n, n, config.confidence_intervals)
    amber_rate = proportion_ci(amber_n, n, config.confidence_intervals)
    red_rate = proportion_ci(red_n, n, config.confidence_intervals)
    green_npv = proportion_ci(green_no_injuries, green_n, config.confidence_intervals)
    green_sensitivity = proportion_ci(
        non_green_injuries,
        events,
        config.confidence_intervals,
    )
    red_ppv = proportion_ci(red_injuries, red_n, config.confidence_intervals)
    red_specificity = proportion_ci(
        non_red_no_injuries,
        int(np.sum(y_true == 0)),
        config.confidence_intervals,
    )

    row: dict[str, object] = {
        "model": model_name,
        "dataset": dataset,
        "n": n,
        "events": events,
        "green_n": green_n,
        "green_injuries": green_injuries,
        "amber_n": amber_n,
        "red_n": red_n,
        "red_injuries": red_injuries,
        "red_false_positives": red_no_injuries,
    }
    row.update(_wide_ci("prevalence", prevalence))
    row.update(_wide_ci("auc_prc", auc["auc_prc"]))
    row.update(_wide_ci("auc_roc", auc["auc_roc"]))
    row.update(_wide_ci("green_rate", green_rate))
    row.update(_wide_ci("green_npv", green_npv))
    row.update(_wide_ci("green_sensitivity", green_sensitivity))
    row.update(_wide_ci("amber_rate", amber_rate))
    row.update(_wide_ci("red_rate", red_rate))
    row.update(_wide_ci("red_ppv", red_ppv))
    row.update(_wide_ci("red_specificity", red_specificity))

    band_rows: list[dict[str, object]] = []
    for band in TRIAGE_BAND_ORDER:
        mask = triage_band == band
        injuries = int(np.sum(mask & (y_true == 1)))
        no_injuries = int(np.sum(mask & (y_true == 0)))
        band_rows.append(
            {
                "model": model_name,
                "dataset": dataset,
                "triage_band": band,
                "injury": injuries,
                "no_injury": no_injuries,
                "total": injuries + no_injuries,
            }
        )

    records: list[dict[str, object]] = []
    for metric, estimate in {
        "prevalence": prevalence,
        "auc_prc": auc["auc_prc"],
        "auc_roc": auc["auc_roc"],
        "green_rate": green_rate,
        "green_npv": green_npv,
        "green_sensitivity": green_sensitivity,
        "amber_rate": amber_rate,
        "red_rate": red_rate,
        "red_ppv": red_ppv,
        "red_specificity": red_specificity,
    }.items():
        _append_metric_record(
            records,
            model=model_name,
            dataset=dataset,
            metric=metric,
            estimate=estimate,
            section="comparison",
        )
    for metric, point, denominator in (
        ("green_injuries", green_injuries, green_n),
        ("red_injuries", red_injuries, red_n),
        ("red_false_positives", red_no_injuries, red_n),
    ):
        records.append(
            {
                "metric": metric,
                "point": point,
                "ci_lower": None,
                "ci_upper": None,
                "ci_method": "count",
                "ci_samples": None,
                "ci_skipped": None,
                "numerator": point,
                "denominator": denominator,
                "model": model_name,
                "dataset": dataset,
                "section": "comparison",
                "formatted": str(point),
            }
        )

    return row, pd.DataFrame(band_rows), records


def band_agreement_cross_tab(
    patient_predictions: pd.DataFrame,
    *,
    split: str,
) -> pd.DataFrame:
    """Joint distribution of the two tools' triage bands, injuries per cell.

    The band count tables give each tool's margins, which cannot show whether
    the tools disagree about the *same* patients: identical margins are
    compatible with perfect agreement and with none at all. The cell counts are
    what support a claim about incremental value, and the per-cell injury counts
    are what say whether the disagreements are the clinically costly ones.
    """
    rows: list[dict[str, object]] = []
    for ai_band in TRIAGE_BAND_ORDER:
        for camkit_band in TRIAGE_BAND_ORDER:
            mask = (patient_predictions["camkit_ai_triage_band"] == ai_band) & (
                patient_predictions["camkit_triage_band"] == camkit_band
            )
            cell = patient_predictions.loc[mask]
            injuries = int(cell["Injury"].sum())
            rows.append(
                {
                    "dataset": split,
                    "camkit_ai_triage_band": ai_band,
                    "camkit_triage_band": camkit_band,
                    "agreement": "concordant" if ai_band == camkit_band else "discordant",
                    "n_patients": int(len(cell)),
                    "n_injury": injuries,
                    "n_no_injury": int(len(cell)) - injuries,
                }
            )
    return pd.DataFrame(rows)


def paired_discrimination_differences(
    y_true: np.ndarray,
    camkit_ai_score: np.ndarray,
    camkit_score: np.ndarray,
    config: ProjectConfig,
    *,
    dataset: str,
) -> pd.DataFrame:
    """Difference in discrimination between the two tools, with a paired CI.

    The direction is CamKIT-AI minus CamKIT, so a positive value means the
    machine learning model discriminates better and an interval containing zero
    means the data do not establish a difference either way. Both metrics are
    rank based, so the CamKIT score needs no rescaling to be comparable with a
    predicted probability.
    """
    estimates = bootstrap_paired_difference(
        np.asarray(y_true).astype(int),
        np.asarray(camkit_ai_score, dtype=float),
        np.asarray(camkit_score, dtype=float),
        {
            # Rank metrics only. Routing through discrimination_metrics would
            # also compute Brier score and a calibration fit, which are undefined
            # for the 0-12 CamKIT score and would reject it outright.
            "auc_prc_difference": lambda yt, ys: rank_discrimination_metrics(yt, ys)["auprc"],
            "auc_roc_difference": lambda yt, ys: rank_discrimination_metrics(yt, ys)["auroc"],
        },
        config=config.confidence_intervals,
        requires_both_classes=True,
    )

    records: list[dict[str, object]] = []
    for metric, estimate in estimates.items():
        excludes_zero = (
            estimate.lower is not None
            and estimate.upper is not None
            and not (estimate.lower <= 0.0 <= estimate.upper)
        )
        records.append(
            estimate.as_record(
                metric,
                model="CamKIT-AI minus Original CamKIT",
                dataset=dataset,
                section="paired_difference",
                formatted=_format_ci(estimate),
                excludes_zero=excludes_zero,
            )
        )
    return pd.DataFrame(records)


def compare_camkit_ai_with_camkit(
    config: ProjectConfig,
    *,
    split: str = "prospective",
    lower_threshold: float | None = None,
    upper_threshold: float | None = None,
    pooled_draws: int | None = None,
    pool_seed: int | None = None,
    basis_label: str | None = None,
    prediction_context: str | None = None,
) -> ModelComparisonResult:
    lower, upper = load_locked_thresholds(config, lower_threshold, upper_threshold)
    ai, provenance = _predict_camkit_ai_probabilities(
        config,
        split,
        pooled_draws=pooled_draws,
        pool_seed=pool_seed,
        basis_label=basis_label,
        prediction_context=prediction_context,
    )
    ai["camkit_ai_triage_band"] = assign_camkit_ai_triage_band(
        ai["camkit_ai_probability"],
        lower,
        upper,
    )

    camkit = evaluate_camkit_score(config, split)
    scored = camkit.scored_frame.copy()
    if not ai["Injury"].equals(scored["Injury"]):
        raise ValueError(
            "CamKIT-AI and CamKIT score frames have different outcome ordering; "
            "patient-level comparison would be invalid."
        )

    patient_predictions = ai.merge(
        scored.loc[:, ["row_id", "camkit_score", "camkit_band", "triage_band"]],
        on="row_id",
        how="inner",
    )
    patient_predictions = patient_predictions.rename(
        columns={"triage_band": "camkit_triage_band"}
    )
    patient_predictions["camkit_ai_lower_threshold"] = lower
    patient_predictions["camkit_ai_upper_threshold"] = upper
    patient_predictions["prediction_basis"] = provenance.basis
    patient_predictions["prediction_source"] = provenance.source
    if provenance.context is not None:
        patient_predictions["prediction_context"] = provenance.context
    patient_predictions["camkit_score_source"] = "reconstructed_original_camkit_score"

    rows: list[dict[str, object]] = []
    band_frames: list[pd.DataFrame] = []
    metric_records: list[dict[str, object]] = []

    ai_row, ai_bands, ai_records = _evaluate_triage_model(
        y_true=patient_predictions["Injury"].to_numpy(),
        risk_score=patient_predictions["camkit_ai_probability"].to_numpy(),
        triage_band=patient_predictions["camkit_ai_triage_band"],
        model_name="CamKIT-AI",
        dataset=split,
        config=config,
    )
    ai_row["threshold_definition"] = f"p < {lower:.2f}; p >= {upper:.2f}"
    rows.append(ai_row)
    band_frames.append(ai_bands)
    metric_records.extend(ai_records)

    camkit_row, camkit_bands, camkit_records = _evaluate_triage_model(
        y_true=patient_predictions["Injury"].to_numpy(),
        risk_score=patient_predictions["camkit_score"].to_numpy() / 12.0,
        triage_band=patient_predictions["camkit_triage_band"],
        model_name="Original CamKIT",
        dataset=split,
        config=config,
    )
    camkit_row["threshold_definition"] = "Low 0-3; Medium 4-6; High 7-12"
    rows.append(camkit_row)
    band_frames.append(camkit_bands)
    metric_records.extend(camkit_records)

    summary = pd.DataFrame(rows)
    band_counts = pd.concat(band_frames, ignore_index=True)
    long_metrics = pd.DataFrame(metric_records)

    # The basis belongs on every table, not just the patient file: these are
    # read independently and each one needs to say what it rests on.
    for frame in (summary, band_counts, long_metrics):
        frame["prediction_basis"] = provenance.basis
        frame["prediction_source"] = np.where(
            frame["model"].eq("Original CamKIT"),
            "reconstructed_original_camkit_score",
            provenance.source,
        )
        if provenance.context is not None:
            # Only the CamKIT-AI rows were cohort-batch scored; the reconstructed
            # score is deterministic and has no batch context to declare.
            frame["prediction_context"] = np.where(
                frame["model"].eq("Original CamKIT"),
                None,
                provenance.context,
            )

    band_agreement = band_agreement_cross_tab(patient_predictions, split=split)
    band_agreement["prediction_basis"] = provenance.basis
    band_agreement["prediction_source"] = (
        f"camkit_ai:{provenance.source};"
        "camkit:reconstructed_original_camkit_score"
    )
    if provenance.context is not None:
        band_agreement["prediction_context"] = provenance.context

    paired_differences = paired_discrimination_differences(
        patient_predictions["Injury"].to_numpy(),
        patient_predictions["camkit_ai_probability"].to_numpy(),
        patient_predictions["camkit_score"].to_numpy(dtype=float),
        config,
        dataset=split,
    )
    paired_differences["prediction_basis"] = provenance.basis
    paired_differences["prediction_source"] = (
        f"camkit_ai:{provenance.source};"
        "camkit:reconstructed_original_camkit_score"
    )
    if provenance.context is not None:
        paired_differences["prediction_context"] = provenance.context

    return ModelComparisonResult(
        patient_predictions=patient_predictions,
        summary=summary,
        band_counts=band_counts,
        band_agreement=band_agreement,
        paired_differences=paired_differences,
        long_metrics=long_metrics,
        camkit_published_count_check=camkit.published_count_check,
        camkit_published_counts_match=camkit.published_counts_match,
        provenance=provenance,
    )


def save_model_comparison_outputs(
    result: ModelComparisonResult,
    output_dir: Path,
    split: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "patient_predictions": output_dir
        / f"camkit_ai_vs_camkit_patient_predictions_{split}.csv",
        "summary": output_dir / f"camkit_ai_vs_camkit_{split}.csv",
        "band_counts": output_dir / f"camkit_ai_vs_camkit_band_table_{split}.csv",
        "band_agreement": output_dir / f"camkit_ai_vs_camkit_band_agreement_{split}.csv",
        "paired_differences": output_dir
        / f"camkit_ai_vs_camkit_paired_differences_{split}.csv",
        "long_metrics": output_dir
        / f"camkit_ai_vs_camkit_long_metrics_{split}.csv",
        "camkit_published_count_check": output_dir
        / f"camkit_published_count_check_{split}.csv",
        "summary_markdown": output_dir / f"camkit_ai_vs_camkit_summary_{split}.md",
    }
    result.patient_predictions.to_csv(paths["patient_predictions"], index=False)
    result.summary.to_csv(paths["summary"], index=False)
    result.band_counts.to_csv(paths["band_counts"], index=False)
    result.band_agreement.to_csv(paths["band_agreement"], index=False)
    result.paired_differences.to_csv(paths["paired_differences"], index=False)
    result.long_metrics.to_csv(paths["long_metrics"], index=False)
    result.camkit_published_count_check.to_csv(
        paths["camkit_published_count_check"],
        index=False,
    )

    def csv_block(frame: pd.DataFrame) -> str:
        return "```csv\n" + frame.to_csv(index=False).strip() + "\n```"

    markdown = [
        f"# CamKIT-AI vs Original CamKIT ({split})",
        "",
        (
            f"CamKIT-AI predictions: {result.provenance.basis} "
            f"({result.provenance.source})."
        ),
        "",
        "## Summary",
        "",
        csv_block(result.summary),
        "",
        "## Band Counts",
        "",
        csv_block(result.band_counts),
        "",
        "## Band Agreement",
        "",
        csv_block(result.band_agreement),
        "",
        "## Paired Discrimination Differences (CamKIT-AI minus CamKIT)",
        "",
        csv_block(result.paired_differences),
        "",
    ]
    paths["summary_markdown"].write_text("\n".join(markdown), encoding="utf-8")
    return paths


def run_camkit_score_analysis(
    config: ProjectConfig,
    *,
    split: str = "prospective",
    output_dir: Path | None = None,
) -> tuple[CamkitScoreResult, dict[str, Path]]:
    result = evaluate_camkit_score(config, split)
    output_root = output_dir or (config.paths.output_root / "analysis")
    paths = save_camkit_score_outputs(result, output_root, split)
    return result, paths


def run_model_comparison_analysis(
    config: ProjectConfig,
    *,
    split: str = "prospective",
    output_dir: Path | None = None,
    lower_threshold: float | None = None,
    upper_threshold: float | None = None,
    pooled_draws: int | None = None,
    pool_seed: int | None = None,
    basis_label: str | None = None,
    prediction_context: str | None = None,
) -> tuple[ModelComparisonResult, dict[str, Path]]:
    output_root = output_dir or (config.paths.output_root / "analysis")
    camkit_result = evaluate_camkit_score(config, split)
    save_camkit_score_outputs(camkit_result, output_root, split)
    result = compare_camkit_ai_with_camkit(
        config,
        split=split,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        pooled_draws=pooled_draws,
        pool_seed=pool_seed,
        basis_label=basis_label,
        prediction_context=prediction_context,
    )
    paths = save_model_comparison_outputs(result, output_root, split)
    return result, paths
