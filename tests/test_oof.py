import numpy as np
import pandas as pd

from camkit_ai.config import ProjectConfig, PathsConfig, ThresholdConfig
from camkit_ai.oof import generate_oof_predictions


class MeanProbabilityEstimator:
    def fit(self, features, labels):
        self.prevalence_ = float(np.mean(labels))
        return self

    def predict_proba(self, features):
        signal = np.asarray(features["x"], dtype=float)
        probabilities = np.clip(0.2 + 0.6 * signal + 0.2 * self.prevalence_, 0.01, 0.99)
        return np.column_stack([1.0 - probabilities, probabilities])


def test_generate_oof_predictions_preserves_rows_and_counts(monkeypatch, tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, 12),
            "Injury": [0, 1] * 6,
        }
    )

    def fake_load_processed_dataset(config, target, split, variant):
        assert target == "Injury"
        assert split == "train"
        assert variant == "top12"
        return frame.copy()

    monkeypatch.setattr("camkit_ai.oof.load_processed_dataset", fake_load_processed_dataset)
    config = ProjectConfig(
        paths=PathsConfig(
            retrospective_encoded=tmp_path / "retro.csv",
            prospective_encoded=tmp_path / "prospective.csv",
            legacy_workspace=tmp_path,
            output_root=tmp_path / "results",
        ),
        thresholds=ThresholdConfig(oof_n_splits=3, oof_n_repeats=2),
    )
    output_path = tmp_path / "oof_predictions.csv"

    oof = generate_oof_predictions(
        config,
        "Injury",
        "top12",
        model_template=MeanProbabilityEstimator(),
        output_path=output_path,
    )

    assert output_path.exists()
    assert oof["row_id"].tolist() == list(range(1, 13))
    assert oof["y_true"].tolist() == frame["Injury"].tolist()
    assert set(oof["n_oof_predictions"]) == {2}
    assert oof["y_probability_oof"].between(0.0, 1.0).all()
