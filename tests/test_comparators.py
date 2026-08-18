import numpy as np
import pandas as pd
import pytest

from camkit_ai.comparators import (
    _evaluate_triage_model,
    _frozen_prediction_basis,
    assign_camkit_ai_triage_band,
    band_agreement_cross_tab,
)
from camkit_ai.config import ConfidenceIntervalConfig, ProjectConfig, PathsConfig


def test_assign_camkit_ai_triage_band_uses_locked_thresholds() -> None:
    bands = assign_camkit_ai_triage_band(
        pd.Series([0.1, 0.29, 0.5, 0.69, 0.9]),
        lower_threshold=0.29,
        upper_threshold=0.69,
    )
    assert bands.tolist() == ["Green", "Amber", "Amber", "Red", "Red"]


def test_evaluate_triage_model_returns_expected_counts(tmp_path) -> None:
    config = ProjectConfig(
        paths=PathsConfig(
            retrospective_encoded=tmp_path / "retro.csv",
            prospective_encoded=tmp_path / "prospective.csv",
            legacy_workspace=tmp_path,
        ),
        confidence_intervals=ConfidenceIntervalConfig(bootstrap_iterations=20),
    )
    row, band_counts, records = _evaluate_triage_model(
        y_true=np.array([0, 0, 1, 1, 0, 1]),
        risk_score=np.array([0.1, 0.2, 0.4, 0.8, 0.7, 0.9]),
        triage_band=pd.Series(["Green", "Green", "Amber", "Red", "Red", "Red"]),
        model_name="Example",
        dataset="test",
        config=config,
    )
    assert row["green_n"] == 2
    assert row["green_injuries"] == 0
    assert row["red_n"] == 3
    assert row["red_injuries"] == 2
    assert set(band_counts["triage_band"]) == {"Green", "Amber", "Red"}
    assert {record["metric"] for record in records} >= {"green_npv", "red_ppv"}


def test_band_agreement_cross_tab_counts_cells_not_margins() -> None:
    # Margins alone cannot distinguish these two tools agreeing from these two
    # tools disagreeing about every patient, which is why the cross-tab exists.
    patient_predictions = pd.DataFrame(
        {
            "Injury": [0, 1, 1, 0, 1],
            "camkit_ai_triage_band": ["Green", "Amber", "Red", "Red", "Amber"],
            "camkit_triage_band": ["Green", "Red", "Red", "Amber", "Amber"],
        }
    )

    table = band_agreement_cross_tab(patient_predictions, split="prospective")

    assert len(table) == 9
    assert table["n_patients"].sum() == 5

    def cell(ai: str, camkit: str) -> pd.Series:
        mask = (table["camkit_ai_triage_band"] == ai) & (
            table["camkit_triage_band"] == camkit
        )
        return table.loc[mask].iloc[0]

    assert cell("Green", "Green")["n_patients"] == 1
    assert cell("Amber", "Red")["n_patients"] == 1
    assert cell("Amber", "Red")["n_injury"] == 1
    assert cell("Red", "Amber")["n_patients"] == 1
    assert cell("Red", "Amber")["n_injury"] == 0

    concordant = table.loc[table["agreement"] == "concordant", "n_patients"].sum()
    assert concordant == 3


def test_frozen_prediction_basis_flags_files_written_before_pooling() -> None:
    predates = pd.DataFrame({"y_probability": [0.1, 0.2]})
    assert _frozen_prediction_basis(predates) == "unrecorded_predates_pooling"

    pooled = pd.DataFrame(
        {
            "y_probability": [0.1, 0.2],
            "prediction_basis": ["pooled_50_draws_seed_42"] * 2,
        }
    )
    assert _frozen_prediction_basis(pooled) == "pooled_50_draws_seed_42"


def test_frozen_prediction_basis_rejects_mixed_bases() -> None:
    mixed = pd.DataFrame(
        {
            "y_probability": [0.1, 0.2],
            "prediction_basis": ["pooled_50_draws_seed_42", "single_draw_unseeded"],
        }
    )
    with pytest.raises(ValueError, match="mix prediction bases"):
        _frozen_prediction_basis(mixed)
