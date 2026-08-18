"""Quantify how much the locked model's output moves between imputation draws.

The saved AutoPrognosis pipeline imputes missing values at inference time, and
that imputation is stochastic. Across 400 seeded calls, probabilities varied
for exactly the rows carrying at least one missing primary-model input and for
no complete rows: 45/97 hold-out records (40 missing BMI, 12 missing pain score,
with overlap) and 10/85 prospective records (all missing BMI).

That matters here rather than being a curiosity, because the missingness regime
is one of the things that changed when data collection moved from clinician
entry to patient entry. Any statement about how the model behaves prospectively
is therefore a statement about a distribution of possible outputs, not about a
single number, and this module measures that distribution.

Two quantities are reported and they answer different questions:

- Metric variability across draws says how reproducible the reported
  discrimination figures are. Average precision (AP) is the sensitive one, because with 18
  prospective events a handful of rank swaps near the top of the ordering moves
  it substantially.
- Triage-band variability says whether the variability reaches the patient. A
  probability that jitters by 0.01 is irrelevant unless it crosses a cut-point,
  so this is the clinically meaningful measure and the one to lead with.

Each draw seeds the global generators before calling predict_proba, so an
individual draw is reproducible even though the sequence deliberately varies.
The imputation stage inside the artefact takes no seed argument, so seeding the
generators it reaches for is the only available handle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from camkit_ai.camkit_score import TRIAGE_BAND_ORDER
from camkit_ai.comparators import load_locked_thresholds
from camkit_ai.config import ProjectConfig
from camkit_ai.data import load_processed_dataset
from camkit_ai.metrics import discrimination_metrics
from camkit_ai.model_io import legacy_model_path, load_legacy_model
from camkit_ai.pooling import (
    collect_draws,
    pool_predictions,
    pooled_basis_label,
    predict_draw,
    seed_global_rngs,
)
from camkit_ai.presets import validate_variant

__all__ = [
    "collect_draws",
    "pool_predictions",
    "pooled_basis_label",
    "predict_draw",
    "seed_global_rngs",
    "assign_bands",
    "format_draw_matrix",
    "pooled_estimate_stability",
    "summarise_draw_metrics",
    "summarise_patient_draws",
    "summarise_variability",
    "summarise_band_instability",
    "format_variability_markdown",
    "run_imputation_variability",
    "DEFAULT_DRAWS",
    "SPLITS",
]

DEFAULT_DRAWS = 100
SPLITS = ("holdout", "prospective")
GREEN, AMBER, RED = TRIAGE_BAND_ORDER


def assign_bands(probability: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Map probabilities to triage bands, matching the locked band definition.

    Works elementwise on both a single draw and a full draw matrix. The band
    boundaries are the same as assign_camkit_ai_triage_band in comparators:
    discharge is a strict ``p < lower`` and referral is ``p >= upper``.
    """
    return np.select(
        [probability < lower, probability < upper],
        [GREEN, AMBER],
        default=RED,
    )


def summarise_draw_metrics(
    y_true: np.ndarray,
    draws: np.ndarray,
    *,
    lower: float,
    upper: float,
    dataset: str,
) -> pd.DataFrame:
    """One row per draw: discrimination plus the triage counts that follow."""
    rows: list[dict[str, object]] = []
    for index, probability in enumerate(draws):
        metrics = discrimination_metrics(y_true, probability)
        bands = assign_bands(probability, lower, upper)
        green = bands == GREEN
        red = bands == RED
        rows.append(
            {
                "dataset": dataset,
                "draw": index,
                "auprc": metrics["auprc"],
                "auroc": metrics["auroc"],
                "brier": metrics["brier"],
                "calibration_slope": metrics["calibration_slope"],
                "calibration_intercept": metrics["calibration_intercept"],
                "n_green": int(green.sum()),
                "n_amber": int((bands == AMBER).sum()),
                "n_red": int(red.sum()),
                "missed_injuries_green": int((green & (y_true == 1)).sum()),
                "green_npv": float(1.0 - y_true[green].mean()) if green.any() else np.nan,
                "red_ppv": float(y_true[red].mean()) if red.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarise_patient_draws(
    y_true: np.ndarray,
    draws: np.ndarray,
    *,
    lower: float,
    upper: float,
    dataset: str,
) -> pd.DataFrame:
    """One row per patient: how far their probability and their band move.

    ``band_unstable`` is the headline column. A patient is unstable when the
    same locked model, given the same record, does not always place them in the
    same triage band.
    """
    band_matrix = assign_bands(draws, lower, upper)
    fractions = {band: (band_matrix == band).mean(axis=0) for band in TRIAGE_BAND_ORDER}
    stacked = np.vstack([fractions[band] for band in TRIAGE_BAND_ORDER])
    modal_index = np.argmax(stacked, axis=0)
    modal_fraction = stacked.max(axis=0)
    probability_mean = draws.mean(axis=0)

    return pd.DataFrame(
        {
            "dataset": dataset,
            "row_id": range(1, draws.shape[1] + 1),
            "y_true": y_true,
            "probability_mean": probability_mean,
            "probability_sd": draws.std(axis=0, ddof=1),
            "probability_min": draws.min(axis=0),
            "probability_max": draws.max(axis=0),
            "probability_range": draws.max(axis=0) - draws.min(axis=0),
            "distance_to_nearest_threshold": np.minimum(
                np.abs(probability_mean - lower), np.abs(probability_mean - upper)
            ),
            "band_modal": np.array(TRIAGE_BAND_ORDER)[modal_index],
            "band_modal_fraction": modal_fraction,
            "fraction_green": fractions[GREEN],
            "fraction_amber": fractions[AMBER],
            "fraction_red": fractions[RED],
            "band_unstable": modal_fraction < 1.0,
        }
    )


def format_draw_matrix(
    y_true: np.ndarray,
    draws: np.ndarray,
    *,
    dataset: str,
    base_seed: int,
) -> pd.DataFrame:
    """Long-format record of every draw for every patient.

    One row per (patient, draw) with the seed that produced it, so any pooled
    estimate can be recomputed later at any number of draws without touching the
    model again.
    """
    n_draws, n_patients = draws.shape
    draw_index = np.repeat(np.arange(n_draws), n_patients)
    return pd.DataFrame(
        {
            "dataset": dataset,
            "draw": draw_index,
            "seed": base_seed + draw_index,
            "row_id": np.tile(np.arange(1, n_patients + 1), n_draws),
            "y_true": np.tile(y_true, n_draws),
            "probability": draws.reshape(-1),
        }
    )


def pooled_estimate_stability(
    draw_matrix: pd.DataFrame,
    metric_functions: dict[str, "object"],
    *,
    draw_counts: list[int],
) -> pd.DataFrame:
    """Recompute pooled metrics at several draw counts from a saved matrix.

    Pooling with more draws reduces Monte Carlo error in the pooled prediction
    vector, so a metric that is still moving as m grows has not converged. This
    matters more than it sounds: the predictions themselves settle quickly while
    AP does not, because at this event count a change of a thousandth in a
    probability can reorder patients near the top of the ranking.
    """
    rows: list[dict[str, object]] = []
    for dataset, group in draw_matrix.groupby("dataset", sort=False):
        wide = group.pivot_table(
            index="draw", columns="row_id", values="probability", sort=True
        )
        y_true = (
            group.sort_values(["draw", "row_id"])
            .groupby("row_id", sort=True)["y_true"]
            .first()
            .to_numpy(dtype=int)
        )
        for n_draws in draw_counts:
            if n_draws > len(wide):
                continue
            pooled = wide.iloc[:n_draws].to_numpy().mean(axis=0)
            for name, func in metric_functions.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "n_draws": n_draws,
                        "metric": name,
                        "value": float(func(y_true, pooled)),
                    }
                )
    return pd.DataFrame(rows)


def _percentile_bounds(confidence_level: float) -> tuple[float, float]:
    tail = (1.0 - confidence_level) / 2.0 * 100.0
    return tail, 100.0 - tail


def summarise_variability(
    draw_metrics: pd.DataFrame,
    *,
    confidence_level: float,
) -> pd.DataFrame:
    """Collapse the per-draw table into a median and an imputation interval.

    The interval is a spread across imputation draws on fixed patients. It is
    not a sampling confidence interval and the two must be reported separately,
    since they answer different questions and do not combine by addition.
    """
    lower_pct, upper_pct = _percentile_bounds(confidence_level)
    metrics = [
        column
        for column in draw_metrics.columns
        if column not in {"dataset", "draw"}
    ]
    rows: list[dict[str, object]] = []
    for dataset, group in draw_metrics.groupby("dataset", sort=False):
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "n_draws": int(finite.size),
                    "median": float(np.median(finite)),
                    "mean": float(np.mean(finite)),
                    "sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                    "interval_lower": float(np.percentile(finite, lower_pct)),
                    "interval_upper": float(np.percentile(finite, upper_pct)),
                    "minimum": float(finite.min()),
                    "maximum": float(finite.max()),
                    "n_distinct": int(np.unique(finite).size),
                }
            )
    return pd.DataFrame(rows)


def summarise_band_instability(patients: pd.DataFrame) -> pd.DataFrame:
    """Per split, how many patients does the imputation noise actually reach."""
    rows: list[dict[str, object]] = []
    for dataset, group in patients.groupby("dataset", sort=False):
        unstable = group["band_unstable"]
        injuries = group["y_true"] == 1
        rows.append(
            {
                "dataset": dataset,
                "n_patients": int(len(group)),
                "n_band_unstable": int(unstable.sum()),
                "fraction_band_unstable": float(unstable.mean()),
                "n_band_unstable_with_injury": int((unstable & injuries).sum()),
                "n_ever_green": int((group["fraction_green"] > 0).sum()),
                "n_ever_red": int((group["fraction_red"] > 0).sum()),
                "n_injury_ever_green": int(((group["fraction_green"] > 0) & injuries).sum()),
                "max_probability_range": float(group["probability_range"].max()),
                "median_probability_sd": float(group["probability_sd"].median()),
            }
        )
    return pd.DataFrame(rows)


def format_variability_markdown(
    summary: pd.DataFrame,
    instability: pd.DataFrame,
    *,
    lower: float,
    upper: float,
    n_draws: int,
) -> str:
    lines = [
        "# Imputation variability of the locked CamKIT-AI model",
        "",
        (
            f"Each patient was scored {n_draws} times with the same locked artefact. "
            "Draws differ only in the stochastic imputation of missing values; the "
            "model, the thresholds and the records are fixed. Intervals below are "
            "percentiles across draws and are not sampling confidence intervals."
        ),
        "",
        f"Triage bands: discharge p < {lower:.2f}; MRI referral p >= {upper:.2f}.",
        "",
        "## Triage-band instability",
        "",
        "| Dataset | Patients | Band unstable | % | Unstable with injury | Injury ever discharged |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in instability.iterrows():
        lines.append(
            f"| {row['dataset']} | {int(row['n_patients'])} | "
            f"{int(row['n_band_unstable'])} | {row['fraction_band_unstable'] * 100:.1f}% | "
            f"{int(row['n_band_unstable_with_injury'])} | {int(row['n_injury_ever_green'])} |"
        )
    lines += ["", "## Metric variability across draws", ""]
    for dataset, group in summary.groupby("dataset", sort=False):
        lines += [
            f"### {dataset}",
            "",
            "| Metric | Median | Interval across draws | Range |",
            "| --- | --- | --- | --- |",
        ]
        for _, row in group.iterrows():
            lines.append(
                f"| {row['metric']} | {row['median']:.3f} | "
                f"{row['interval_lower']:.3f} to {row['interval_upper']:.3f} | "
                f"{row['minimum']:.3f} to {row['maximum']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def run_imputation_variability(
    config: ProjectConfig,
    target: str = "Injury",
    variant: str = "top12",
    *,
    n_draws: int = DEFAULT_DRAWS,
    base_seed: int | None = None,
    lower_threshold: float | None = None,
    upper_threshold: float | None = None,
    output_dir: Path | None = None,
    save_draw_matrix: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    """Score both splits repeatedly and write the variability outputs.

    This reloads the legacy AutoPrognosis artefact, so it needs the Python 3.10
    environment. No retraining happens and no threshold is re-derived; the
    locked model and the locked cut-points are held fixed throughout, which is
    what makes the spread attributable to imputation alone.

    ``save_draw_matrix`` additionally writes every draw for every patient. The
    summary tables keep only per-patient means and spreads, which is enough to
    describe the variability but not enough to ask how many draws pooling needs:
    that question requires re-pooling subsets of the draws, and without the raw
    matrix each new value of m costs another full scoring run.
    """
    validate_variant(target, variant)
    if n_draws < 2:
        raise ValueError("n_draws must be at least 2 to measure variability.")

    lower, upper = load_locked_thresholds(config, lower_threshold, upper_threshold)
    seed = config.confidence_intervals.random_state if base_seed is None else base_seed
    model = load_legacy_model(legacy_model_path(config, target, variant))

    draw_frames: list[pd.DataFrame] = []
    patient_frames: list[pd.DataFrame] = []
    matrix_frames: list[pd.DataFrame] = []
    for split in SPLITS:
        frame = load_processed_dataset(config, target, split, variant)
        y_true, draws = collect_draws(
            model, frame, target, n_draws=n_draws, base_seed=seed
        )
        draw_frames.append(
            summarise_draw_metrics(
                y_true, draws, lower=lower, upper=upper, dataset=split
            )
        )
        patient_frames.append(
            summarise_patient_draws(
                y_true, draws, lower=lower, upper=upper, dataset=split
            )
        )
        if save_draw_matrix:
            matrix_frames.append(
                format_draw_matrix(y_true, draws, dataset=split, base_seed=seed)
            )

    draw_metrics = pd.concat(draw_frames, ignore_index=True)
    patients = pd.concat(patient_frames, ignore_index=True)
    summary = summarise_variability(
        draw_metrics,
        confidence_level=config.confidence_intervals.confidence_level,
    )
    instability = summarise_band_instability(patients)

    frames = {
        "draws": draw_metrics,
        "patients": patients,
        "summary": summary,
        "instability": instability,
    }
    if matrix_frames:
        frames["draw_matrix"] = pd.concat(matrix_frames, ignore_index=True)

    output_root = output_dir or (config.paths.output_root / "analysis")
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "draws": output_root / "imputation_variability_draws.csv",
        "patients": output_root / "imputation_variability_patients.csv",
        "summary": output_root / "imputation_variability_summary.csv",
        "instability": output_root / "imputation_variability_instability.csv",
        "markdown": output_root / "imputation_variability.md",
    }
    if "draw_matrix" in frames:
        paths["draw_matrix"] = output_root / "imputation_variability_draw_matrix.csv"
    for name, frame in frames.items():
        frame.to_csv(paths[name], index=False)
    paths["markdown"].write_text(
        format_variability_markdown(
            summary, instability, lower=lower, upper=upper, n_draws=n_draws
        ),
        encoding="utf-8",
    )
    return frames, paths
