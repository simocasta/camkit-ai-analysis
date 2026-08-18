from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold

from camkit_ai.config import ProjectConfig
from camkit_ai.data import load_processed_dataset
from camkit_ai.metrics import positive_class_probabilities
from camkit_ai.model_io import legacy_model_path, load_legacy_model
from camkit_ai.presets import validate_variant


def oof_predictions_path(
    config: ProjectConfig,
    target: str,
    variant: str,
    *,
    output_dir: Path | None = None,
) -> Path:
    root = output_dir or (config.paths.output_root / "models" / f"{target}.{variant}")
    return root / "oof_predictions.csv"


def _patch_xgboost_compat(estimator):
    # Legacy AutoPrognosis pickles were saved with xgboost 2.1.x, before
    # XGBModel.__init__ declared parameters such as ``feature_weights``. In
    # xgboost >=3.0 those parameters are part of the introspected sklearn
    # signature, so ``get_params`` raises AttributeError on instances that
    # pre-date them. Backfill the current XGBModel defaults on every nested
    # XGBoost model so ``fit`` can proceed unchanged.
    try:
        import inspect
        from xgboost.sklearn import XGBModel
    except Exception:
        return estimator

    defaults = {
        name: param.default
        for name, param in inspect.signature(XGBModel.__init__).parameters.items()
        if name != "self" and param.default is not inspect.Parameter.empty
    }
    if not defaults:
        return estimator

    visited: set[int] = set()

    def walk(obj):
        if id(obj) in visited:
            return
        visited.add(id(obj))
        if isinstance(obj, XGBModel):
            for name, default in defaults.items():
                if not hasattr(obj, name):
                    try:
                        object.__setattr__(obj, name, default)
                    except Exception:
                        pass
        attrs = getattr(obj, "__dict__", None)
        if not attrs:
            return
        for value in list(attrs.values()):
            if isinstance(value, (list, tuple, set, frozenset)):
                for item in value:
                    if hasattr(item, "__dict__"):
                        walk(item)
            elif isinstance(value, dict):
                for item in value.values():
                    if hasattr(item, "__dict__"):
                        walk(item)
            elif hasattr(value, "__dict__"):
                walk(value)

    walk(estimator)
    return estimator


def _fresh_estimator(template):
    try:
        estimator = copy.deepcopy(template)
    except Exception:
        estimator = clone(template)
    return _patch_xgboost_compat(estimator)


def _validate_oof_predictions(
    frame: pd.DataFrame,
    *,
    expected_rows: int,
    n_repeats: int | None = None,
) -> pd.DataFrame:
    required = {"row_id", "y_true", "y_probability_oof", "n_oof_predictions"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"OOF predictions are missing required columns: {missing}")
    if len(frame) != expected_rows:
        raise ValueError(
            f"OOF prediction row count ({len(frame)}) does not match expected "
            f"training rows ({expected_rows})."
        )
    if frame["y_probability_oof"].isna().any():
        raise ValueError("OOF predictions contain missing probabilities.")
    if (frame["n_oof_predictions"] <= 0).any():
        raise ValueError("Every row must have at least one OOF prediction.")
    if n_repeats is not None and (frame["n_oof_predictions"] != n_repeats).any():
        raise ValueError(
            "OOF prediction counts do not match the configured repeat count. "
            "This usually indicates an incomplete repeated-CV run."
        )
    return frame


def load_oof_predictions(
    config: ProjectConfig,
    target: str,
    variant: str,
    *,
    path: Path | None = None,
    strict_repeats: bool = True,
) -> pd.DataFrame:
    validate_variant(target, variant)
    dataset = load_processed_dataset(config, target=target, split="train", variant=variant)
    oof_path = path or oof_predictions_path(config, target, variant)
    frame = pd.read_csv(oof_path)
    validated = _validate_oof_predictions(
        frame,
        expected_rows=len(dataset),
        n_repeats=config.thresholds.oof_n_repeats if strict_repeats else None,
    )
    expected_y = dataset[target].astype(int).to_numpy()
    if not np.array_equal(validated["y_true"].astype(int).to_numpy(), expected_y):
        raise ValueError("OOF predictions do not match the processed training outcome order.")
    return validated


def generate_oof_predictions(
    config: ProjectConfig,
    target: str,
    variant: str,
    *,
    model_template=None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    validate_variant(target, variant)
    dataset = load_processed_dataset(config, target=target, split="train", variant=variant)
    features = dataset.drop(columns=[target])
    labels = dataset[target].astype(int)

    n_splits = int(config.thresholds.oof_n_splits)
    n_repeats = int(config.thresholds.oof_n_repeats)
    if n_splits < 2:
        raise ValueError("OOF generation requires at least two folds.")
    if n_repeats < 1:
        raise ValueError("OOF generation requires at least one repeat.")

    min_class_count = int(labels.value_counts().min())
    if n_splits > min_class_count:
        raise ValueError(
            f"Configured oof_n_splits={n_splits} exceeds the smallest class count "
            f"({min_class_count})."
        )

    template = model_template
    if template is None:
        template = load_legacy_model(legacy_model_path(config, target, variant))

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=config.confidence_intervals.random_state,
    )
    probability_sum = np.zeros(len(dataset), dtype=float)
    prediction_count = np.zeros(len(dataset), dtype=int)

    for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(features, labels), start=1):
        seed = int(config.confidence_intervals.random_state + fold_id)
        random.seed(seed)
        np.random.seed(seed)
        estimator = _fresh_estimator(template)
        estimator.fit(features.iloc[train_idx], labels.iloc[train_idx])
        fold_prob = positive_class_probabilities(
            estimator.predict_proba(features.iloc[valid_idx])
        )
        probability_sum[valid_idx] += fold_prob
        prediction_count[valid_idx] += 1

    if np.any(prediction_count == 0):
        missing = np.flatnonzero(prediction_count == 0).tolist()
        raise RuntimeError(f"OOF generation missed training rows: {missing}")

    frame = pd.DataFrame(
        {
            "row_id": np.arange(1, len(dataset) + 1),
            "target": target,
            "variant": variant,
            "y_true": labels.to_numpy(dtype=int),
            "y_probability_oof": probability_sum / prediction_count,
            "n_oof_predictions": prediction_count,
            "oof_n_splits": n_splits,
            "oof_n_repeats": n_repeats,
        }
    )
    frame = _validate_oof_predictions(
        frame,
        expected_rows=len(dataset),
        n_repeats=n_repeats,
    )

    save_path = output_path or oof_predictions_path(config, target, variant)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(save_path, index=False)
    return frame


def load_or_generate_oof_predictions(
    config: ProjectConfig,
    target: str,
    variant: str,
) -> pd.DataFrame:
    path = oof_predictions_path(config, target, variant)
    if path.exists():
        return load_oof_predictions(config, target, variant, path=path)
    return generate_oof_predictions(config, target, variant, output_path=path)
