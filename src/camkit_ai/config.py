from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


def _expand_path(value: str | Path, *, base_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


@dataclass
class PathsConfig:
    retrospective_encoded: Path
    prospective_encoded: Path
    legacy_workspace: Path
    processed_root: Path = Path("data/processed")
    output_root: Path = Path("results")
    prospective_labels: Path | None = None


@dataclass
class DatasetConfig:
    retrospective_name: str = "CamKIT_AI_retro_comb_enc"
    prospective_name: str = "External_set_enc"
    test_size: float = 0.3
    random_state: int = 42


@dataclass
class ThresholdConfig:
    min_npv: float = 0.95
    min_discharge_rate: float = 0.15
    min_ppv: float = 0.90
    min_gap: float = 0.20
    max_investigation_rate: float = 0.80
    target_lr_minus: float = 0.15
    target_lr_plus: float = 10.0
    step: float = 0.01
    selection_source: Literal["training_oof", "holdout"] = "training_oof"
    selection_basis: Literal["point"] = "point"
    oof_n_splits: int = 10
    oof_n_repeats: int = 5
    bootstrap_thresholds: bool = True
    bootstrap_threshold_iterations: int = 2000
    use_locked_thresholds: bool = False
    locked_lower_threshold: float = 0.29
    locked_upper_threshold: float = 0.69


@dataclass
class ConfidenceIntervalConfig:
    bootstrap_iterations: int = 2000
    confidence_level: float = 0.95
    random_state: int = 42
    bootstrap_stratified: bool = True
    proportion_method: str = "exact"


@dataclass
class TrainingConfig:
    metric: str = "aucprc"
    num_iter: int = 300
    num_study_iter: int = 10
    num_ensemble_iter: int = 25
    timeout: int = 360
    score_threshold: float = 0.45
    ensemble_size: int = 3
    n_folds_cv: int = 10
    sample_for_search: bool = True
    max_search_sample_size: int = 10000
    classifiers: list[str] = field(
        default_factory=lambda: ["xgboost", "lgbm", "random_forest", "catboost"]
    )
    imputers: list[str] = field(
        default_factory=lambda: [
            "gain",
            "ice",
            "mean",
            "median",
            "mice",
            "missforest",
            "most_frequent",
            "softimpute",
        ]
    )


@dataclass
class ProjectConfig:
    paths: PathsConfig
    data: DatasetConfig = field(default_factory=DatasetConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    confidence_intervals: ConfidenceIntervalConfig = field(
        default_factory=ConfidenceIntervalConfig
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _load_dataclass_section(data: dict[str, Any], key: str, cls: type[Any]) -> Any:
    section = data.get(key, {})
    return cls(**section)


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).expanduser().resolve()
    config_dir = config_path.parent
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    paths_payload = payload.get("paths", {})
    paths = PathsConfig(
        retrospective_encoded=_expand_path(
            paths_payload["retrospective_encoded"], base_dir=config_dir
        ),
        prospective_encoded=_expand_path(
            paths_payload["prospective_encoded"], base_dir=config_dir
        ),
        legacy_workspace=_expand_path(
            paths_payload.get("legacy_workspace", paths_payload.get("workspace", "workspace")),
            base_dir=config_dir,
        ),
        processed_root=_expand_path(
            paths_payload.get("processed_root", "data/processed"),
            base_dir=config_dir,
        ),
        output_root=_expand_path(
            paths_payload.get("output_root", paths_payload.get("results_root", "results")),
            base_dir=config_dir,
        ),
        prospective_labels=(
            _expand_path(paths_payload["prospective_labels"], base_dir=config_dir)
            if "prospective_labels" in paths_payload
            else None
        ),
    )

    return ProjectConfig(
        paths=paths,
        data=_load_dataclass_section(payload, "data", DatasetConfig),
        thresholds=_load_dataclass_section(payload, "thresholds", ThresholdConfig),
        confidence_intervals=_load_dataclass_section(
            payload, "confidence_intervals", ConfidenceIntervalConfig
        ),
        training=_load_dataclass_section(payload, "training", TrainingConfig),
    )
