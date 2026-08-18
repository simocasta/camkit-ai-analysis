"""Small, auditable utilities for the post hoc manuscript analyses.

This module deliberately contains no model fitting.  It turns a saved matrix of
stochastic inference draws into a convergence decision and creates descriptive
cohort/missingness tables from the already prepared datasets.  Keeping these
operations pure makes the post hoc analysis easy to inspect and test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from camkit_ai.config import ProjectConfig
from camkit_ai.data import load_processed_dataset
from camkit_ai.imputation_variability import assign_bands
from camkit_ai.metrics import discrimination_metrics
from camkit_ai.presets import INJURY_TOP12_FEATURES


REQUIRED_DRAW_COLUMNS = {
    "dataset",
    "draw",
    "seed",
    "row_id",
    "y_true",
    "probability",
}


@dataclass(frozen=True)
class ConvergenceTolerances:
    """Clinical/reporting tolerances for choosing a Monte Carlo draw count."""

    max_auc_difference: float = 0.01
    max_probability_difference: float = 0.02
    max_band_changes: int = 0
    max_band_count_difference: int = 0


def validate_draw_matrix(draws: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_DRAW_COLUMNS - set(draws.columns))
    if missing:
        raise ValueError(f"Draw matrix is missing required columns: {missing}")
    if draws.empty:
        raise ValueError("Draw matrix is empty.")
    if draws[list(REQUIRED_DRAW_COLUMNS - {"dataset"})].isna().any().any():
        raise ValueError("Draw matrix contains missing identifiers, outcomes or probabilities.")
    if not draws["probability"].between(0.0, 1.0).all():
        raise ValueError("Draw-matrix probabilities must lie between zero and one.")

    for dataset, group in draws.groupby("dataset", sort=False):
        counts = group.groupby("draw")["row_id"].nunique()
        if counts.empty or counts.nunique() != 1:
            raise ValueError(f"{dataset}: draws do not contain a constant patient count.")
        expected_rows = set(group["row_id"].unique())
        for draw_id, draw_group in group.groupby("draw", sort=False):
            if set(draw_group["row_id"]) != expected_rows:
                raise ValueError(f"{dataset}, draw {draw_id}: patient rows are incomplete.")
            if draw_group["row_id"].duplicated().any():
                raise ValueError(f"{dataset}, draw {draw_id}: duplicate patient rows found.")
        outcomes_per_patient = group.groupby("row_id")["y_true"].nunique()
        if not outcomes_per_patient.eq(1).all():
            raise ValueError(f"{dataset}: outcomes change between draws.")


def _pooled_vector(group: pd.DataFrame, n_draws: int) -> tuple[np.ndarray, np.ndarray]:
    available = sorted(group["draw"].unique())
    if n_draws > len(available):
        raise ValueError(f"Requested {n_draws} draws but only {len(available)} are available.")
    selected = set(available[:n_draws])
    subset = group[group["draw"].isin(selected)]
    pooled = (
        subset.groupby("row_id", sort=True)["probability"]
        .mean()
        .to_numpy(dtype=float)
    )
    y_true = (
        subset.groupby("row_id", sort=True)["y_true"]
        .first()
        .to_numpy(dtype=int)
    )
    return y_true, pooled


def analyse_pooling_convergence(
    draws: pd.DataFrame,
    *,
    candidate_draws: Iterable[int],
    reference_draws: int,
    lower_threshold: float,
    upper_threshold: float,
    tolerances: ConvergenceTolerances,
) -> pd.DataFrame:
    """Compare candidate pooled estimates with the largest saved draw set."""

    validate_draw_matrix(draws)
    candidates = sorted({int(value) for value in candidate_draws})
    if any(value < 1 for value in candidates):
        raise ValueError("Candidate draw counts must all be positive.")
    if reference_draws < 2:
        raise ValueError("reference_draws must be at least two.")

    rows: list[dict[str, object]] = []
    for dataset, group in draws.groupby("dataset", sort=False):
        y_reference, p_reference = _pooled_vector(group, reference_draws)
        reference_metrics = discrimination_metrics(y_reference, p_reference)
        reference_bands = assign_bands(
            p_reference,
            lower_threshold,
            upper_threshold,
        )

        for n_draws in candidates:
            if n_draws > reference_draws:
                continue
            y_true, pooled = _pooled_vector(group, n_draws)
            if not np.array_equal(y_true, y_reference):
                raise ValueError(f"{dataset}: outcome ordering changed during pooling.")
            metrics = discrimination_metrics(y_true, pooled)
            bands = assign_bands(pooled, lower_threshold, upper_threshold)
            band_changes = int(np.sum(bands != reference_bands))
            candidate_counts = {
                band: int(np.sum(bands == band)) for band in ("Green", "Amber", "Red")
            }
            reference_counts = {
                band: int(np.sum(reference_bands == band))
                for band in ("Green", "Amber", "Red")
            }
            max_count_difference = max(
                abs(candidate_counts[band] - reference_counts[band])
                for band in candidate_counts
            )
            auprc_difference = abs(metrics["auprc"] - reference_metrics["auprc"])
            auroc_difference = abs(metrics["auroc"] - reference_metrics["auroc"])
            max_probability_difference = float(np.max(np.abs(pooled - p_reference)))
            converged = bool(
                auprc_difference <= tolerances.max_auc_difference
                and auroc_difference <= tolerances.max_auc_difference
                and max_probability_difference
                <= tolerances.max_probability_difference
                and band_changes <= tolerances.max_band_changes
                and max_count_difference <= tolerances.max_band_count_difference
            )
            rows.append(
                {
                    "dataset": dataset,
                    "candidate_draws": n_draws,
                    "reference_draws": reference_draws,
                    "candidate_prediction_basis": f"pooled_{n_draws}_draws",
                    "reference_prediction_basis": f"pooled_{reference_draws}_draws",
                    "candidate_auprc": metrics["auprc"],
                    "reference_auprc": reference_metrics["auprc"],
                    "absolute_auprc_difference": auprc_difference,
                    "candidate_auroc": metrics["auroc"],
                    "reference_auroc": reference_metrics["auroc"],
                    "absolute_auroc_difference": auroc_difference,
                    "maximum_patient_probability_difference": max_probability_difference,
                    "patients_changing_band": band_changes,
                    "maximum_band_count_difference": max_count_difference,
                    "candidate_green_n": candidate_counts["Green"],
                    "candidate_amber_n": candidate_counts["Amber"],
                    "candidate_red_n": candidate_counts["Red"],
                    "reference_green_n": reference_counts["Green"],
                    "reference_amber_n": reference_counts["Amber"],
                    "reference_red_n": reference_counts["Red"],
                    "meets_tolerances": converged,
                }
            )
    return pd.DataFrame(rows)


def choose_draw_count(
    convergence: pd.DataFrame,
    *,
    reference_draws: int,
) -> int:
    """Choose the smallest candidate meeting tolerances in every dataset."""

    required_datasets = set(convergence["dataset"].unique())
    for n_draws in sorted(convergence["candidate_draws"].unique()):
        rows = convergence[convergence["candidate_draws"] == n_draws]
        if set(rows["dataset"]) == required_datasets and rows["meets_tolerances"].all():
            return int(n_draws)
    return int(reference_draws)


def _median_iqr(series: pd.Series) -> str:
    observed = series.dropna().astype(float)
    if observed.empty:
        return "not available"
    return (
        f"{observed.median():.1f} "
        f"({observed.quantile(0.25):.1f}–{observed.quantile(0.75):.1f})"
    )


def _count_percent(series: pd.Series, value: int = 1) -> str:
    observed = series.dropna()
    if observed.empty:
        return "not available"
    count = int((observed == value).sum())
    return f"{count}/{len(series)} ({100.0 * count / len(series):.1f}%)"


def evaluation_cohorts(config: ProjectConfig) -> dict[str, pd.DataFrame]:
    train = load_processed_dataset(config, "Injury", "train", "full")
    holdout = load_processed_dataset(config, "Injury", "holdout", "full")
    prospective = load_processed_dataset(config, "Injury", "prospective", "full")
    return {
        "Retrospective overall": pd.concat([train, holdout], ignore_index=True),
        "Training": train,
        "Internal hold-out": holdout,
        "Prospective": prospective,
    }


def cohort_characteristics(config: ProjectConfig) -> pd.DataFrame:
    """Create the descriptive main cohort table without inferential testing."""

    cohorts = evaluation_cohorts(config)
    summaries: dict[str, dict[str, str]] = {}
    for label, frame in cohorts.items():
        events = int(frame["Injury"].sum())
        primary_inputs = frame.loc[:, INJURY_TOP12_FEATURES]
        summaries[label] = {
            "n": str(len(frame)),
            "age_median_iqr": _median_iqr(frame["age"]),
            "male_n_percent": _count_percent(frame["sex"], 1),
            "bmi_median_iqr_observed": _median_iqr(frame["bmi"]),
            "bmi_missing_n_percent": _count_percent(frame["bmi"].isna().astype(int), 1),
            "pain_scale_missing_n_percent": _count_percent(
                frame["pain_scale"].isna().astype(int), 1
            ),
            "any_primary_input_missing_n_percent": _count_percent(
                primary_inputs.isna().any(axis=1).astype(int), 1
            ),
            "injury_n_percent": f"{events}/{len(frame)} ({100.0 * events / len(frame):.1f}%)",
            "twisting_n_percent": _count_percent(frame["twisting"], 1),
            "hyperextension_n_percent": _count_percent(frame["hyperextension"], 1),
            "non_contact_n_percent": _count_percent(frame["contact_noncontact"], 1),
            "respondent_data_source": (
                "clinician-documented retrospective records"
                if label != "Prospective"
                else "patient-completed digital questionnaire"
            ),
            "recruitment_period": (
                "February–July 2023"
                if label != "Prospective"
                else "October 2023–March 2025"
            ),
        }

    characteristics = (
        ("Participants, n", "n"),
        ("Age, median (IQR), years", "age_median_iqr"),
        ("Male, n/N (%)", "male_n_percent"),
        ("BMI, median (IQR), observed values", "bmi_median_iqr_observed"),
        ("BMI missing, n/N (%)", "bmi_missing_n_percent"),
        ("Pain score missing, n/N (%)", "pain_scale_missing_n_percent"),
        (
            "At least one primary-model input missing, n/N (%)",
            "any_primary_input_missing_n_percent",
        ),
        ("Clinically significant STKI, n/N (%)", "injury_n_percent"),
        ("Twisting mechanism, n/N (%)", "twisting_n_percent"),
        ("Hyperextension mechanism, n/N (%)", "hyperextension_n_percent"),
        ("Non-contact mechanism, n/N (%)", "non_contact_n_percent"),
        ("Respondent/data source", "respondent_data_source"),
        ("Recruitment period", "recruitment_period"),
    )
    rows: list[dict[str, str]] = []
    for display, key in characteristics:
        row = {"Characteristic": display}
        row.update({label: summaries[label][key] for label in cohorts})
        rows.append(row)
    return pd.DataFrame(rows)


def primary_predictor_missingness(config: ProjectConfig) -> pd.DataFrame:
    """Report missingness for every input to the primary 12-feature model."""

    train = load_processed_dataset(config, "Injury", "train", "top12")
    holdout = load_processed_dataset(config, "Injury", "holdout", "top12")
    prospective = load_processed_dataset(config, "Injury", "prospective", "top12")
    cohorts = {
        "Retrospective overall": pd.concat([train, holdout], ignore_index=True),
        "Training": train,
        "Internal hold-out": holdout,
        "Prospective": prospective,
    }
    rows: list[dict[str, object]] = []
    for predictor in INJURY_TOP12_FEATURES:
        row: dict[str, object] = {"Predictor": predictor}
        for label, frame in cohorts.items():
            missing = int(frame[predictor].isna().sum())
            row[label] = (
                f"{missing}/{len(frame)} "
                f"({100.0 * missing / len(frame):.1f}%)"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def dataframe_to_markdown(frame: pd.DataFrame, *, digits: int = 3) -> str:
    """Render a DataFrame without requiring the optional tabulate package."""

    def display(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value).replace("|", "\\|")

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(display(row[column]) for column in frame.columns) + " |")
    return "\n".join(lines)
