from __future__ import annotations

from pathlib import Path

import pandas as pd

from camkit_ai import evaluation
from camkit_ai.config import (
    ConfidenceIntervalConfig,
    PathsConfig,
    ProjectConfig,
    ThresholdConfig,
)
from camkit_ai.evaluation import ModelEvaluationResult, _evaluate_subgroups


def test_report_manuscript_preserves_holdout_threshold_source(monkeypatch, tmp_path) -> None:
    config = ProjectConfig(
        paths=PathsConfig(
            retrospective_encoded=tmp_path / "retro.csv",
            prospective_encoded=tmp_path / "prospective.csv",
            legacy_workspace=tmp_path / "workspace",
            output_root=tmp_path / "results",
        ),
        thresholds=ThresholdConfig(selection_source="holdout"),
    )

    def fake_evaluate_model(*args, **kwargs):
        return ModelEvaluationResult(
            discrimination=pd.DataFrame([{"metric": "n", "point": 1}]),
            subgroups=pd.DataFrame(),
            thresholds=pd.DataFrame(),
            predictions=pd.DataFrame([{"row_id": 1}]),
            threshold_sweep=pd.DataFrame([{"threshold": 0.29}]),
            threshold_sweep_source="holdout",
            selected_thresholds={
                "lower_threshold": 0.29,
                "upper_threshold": 0.69,
                "threshold_gap": 0.40,
                "selection_source": "holdout",
                "threshold_status": "historical_locked",
                "prediction_basis": "pooled_50_draws_seed_42",
            },
            prediction_basis="pooled_50_draws_seed_42",
        )

    monkeypatch.setattr(evaluation, "MANUSCRIPT_MODEL_SPECS", [("Injury", "top12")])
    monkeypatch.setattr(evaluation, "evaluate_model", fake_evaluate_model)

    outputs = evaluation.report_manuscript(config)
    selected = pd.read_csv(outputs["selected_thresholds"])
    sweep = pd.read_csv(outputs["threshold_sweep"])

    assert outputs["threshold_sweep"].name == "threshold_sweep_holdout.csv"
    assert selected.loc[0, "selection_source"] == "holdout"
    assert sweep.loc[0, "selection_source"] == "holdout"
    assert selected.loc[0, "threshold_status"] == "historical_locked"
    assert sweep.loc[0, "prediction_basis"] == "pooled_50_draws_seed_42"


def test_subgroups_reuse_definitive_full_split_predictions() -> None:
    frame = pd.DataFrame(
        {
            "sex": [0, 0, 1, 1, 0, 1],
            "age": [20, 40, 20, 40, 30, 18],
            "Injury": [0, 1, 1, 0, 0, 1],
        }
    )
    definitive = pd.Series([0.01, 0.91, 0.81, 0.11, 0.21, 0.71]).to_numpy()

    result = _evaluate_subgroups(
        frame,
        definitive,
        "Injury",
        "top12",
        "prospective",
        ProjectConfig(
            paths=PathsConfig(
                retrospective_encoded=Path("retro.csv"),
                prospective_encoded=Path("prospective.csv"),
                legacy_workspace=Path("workspace"),
            ),
            confidence_intervals=ConfidenceIntervalConfig(
                bootstrap_iterations=10,
                random_state=42,
            ),
        ),
    )

    female = result[result["subgroup"] == "female"]
    assert int(female[female["metric"] == "n"].iloc[0]["point"]) == 3
    assert int(female[female["metric"] == "events"].iloc[0]["point"]) == 1
