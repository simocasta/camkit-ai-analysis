from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

from camkit_ai.config import ProjectConfig
from camkit_ai.presets import (
    EXTERNAL_COLUMN_RENAMES,
    FULL_FEATURE_COLUMNS,
    INJURY_TOP12_FEATURES,
    TARGETS,
    processed_dataset_path,
    validate_variant,
)


def _ensure_directories(root: Path) -> None:
    for split in ("train", "holdout", "prospective"):
        (root / split).mkdir(parents=True, exist_ok=True)


def load_retrospective_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(FULL_FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Retrospective dataset is missing required columns: {missing}")
    return frame.copy()


def load_prospective_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).rename(columns=EXTERNAL_COLUMN_RENAMES)
    frame["Cruciate"] = ((frame["ACL"] == 1) | (frame["PCL"] == 1)).astype(int)
    frame["Collateral"] = ((frame["MCL"] == 1) | (frame["LCL"] == 1)).astype(int)
    frame["Meniscus"] = (
        (frame["Medial meniscus"] == 1) | (frame["Lateral meniscus"] == 1)
    ).astype(int)
    missing = sorted(set(FULL_FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Prospective dataset is missing required columns: {missing}")
    return frame.copy()


def _target_frame(frame: pd.DataFrame, target: str, variant: str) -> pd.DataFrame:
    validate_variant(target, variant)
    features = INJURY_TOP12_FEATURES if variant == "top12" else FULL_FEATURE_COLUMNS
    return frame.loc[:, [*features, target]].copy()


def _legacy_prepared_path(
    data_root: Path,
    split: str,
    target: str,
    variant: str,
) -> Path | None:
    suffix = "_12core" if variant == "top12" else ""
    if split == "train":
        candidate = data_root / f"CamKIT_AI_retro_comb_enc_{target}{suffix}_train.csv"
    elif split == "holdout":
        candidate = data_root / "Validation" / "Test" / f"CamKIT_AI_retro_comb_enc_{target}{suffix}_test.csv"
    elif split == "prospective":
        candidate = data_root / "Validation" / "External" / f"External_set_enc_{target}{suffix}.csv"
    else:
        raise ValueError(f"Unsupported split '{split}'.")
    return candidate if candidate.exists() else None


def prepare_datasets(config: ProjectConfig) -> dict[str, str]:
    processed_root = config.paths.processed_root
    _ensure_directories(processed_root)
    legacy_data_root = config.paths.retrospective_encoded.parent

    retrospective = load_retrospective_dataset(config.paths.retrospective_encoded)
    prospective = load_prospective_dataset(config.paths.prospective_encoded)
    train_frame, holdout_frame = train_test_split(
        retrospective,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
    )

    manifest: dict[str, str] = {}
    for target in TARGETS:
        for split_name, split_frame in (
            ("train", train_frame),
            ("holdout", holdout_frame),
            ("prospective", prospective),
        ):
            full_path = processed_dataset_path(processed_root, split_name, target, "full")
            legacy_path = _legacy_prepared_path(legacy_data_root, split_name, target, "full")
            if legacy_path is not None:
                pd.read_csv(legacy_path).to_csv(full_path, index=False)
            else:
                _target_frame(split_frame, target, "full").to_csv(full_path, index=False)
            manifest[f"{split_name}.{target}.full"] = str(full_path)
        if target == "Injury":
            for split_name, split_frame in (
                ("train", train_frame),
                ("holdout", holdout_frame),
                ("prospective", prospective),
            ):
                top12_path = processed_dataset_path(processed_root, split_name, target, "top12")
                legacy_path = _legacy_prepared_path(legacy_data_root, split_name, target, "top12")
                if legacy_path is not None:
                    pd.read_csv(legacy_path).to_csv(top12_path, index=False)
                else:
                    _target_frame(split_frame, target, "top12").to_csv(top12_path, index=False)
                manifest[f"{split_name}.{target}.top12"] = str(top12_path)

    manifest_path = processed_root / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")
    return manifest


def load_processed_dataset(
    config: ProjectConfig,
    target: str,
    split: str,
    variant: str,
) -> pd.DataFrame:
    path = processed_dataset_path(config.paths.processed_root, split, target, variant)
    if not path.exists():
        prepare_datasets(config)
    return pd.read_csv(path)
