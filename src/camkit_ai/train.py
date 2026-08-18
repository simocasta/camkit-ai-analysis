from __future__ import annotations

from pathlib import Path

import pandas as pd

from camkit_ai.config import ProjectConfig
from camkit_ai.data import load_processed_dataset
from camkit_ai.presets import study_name, validate_variant


def train_study(
    config: ProjectConfig,
    target: str,
    variant: str,
) -> dict[str, object]:
    validate_variant(target, variant)
    try:
        from autoprognosis.plugins.preprocessors import Preprocessors
        from autoprognosis.studies.classifiers import ClassifierStudy
        from autoprognosis.utils.serialization import save_to_file
        from autoprognosis.utils.tester import evaluate_estimator
    except ImportError as exc:
        raise RuntimeError(
            "AutoPrognosis is required for training. Install the optional 'autop' dependency."
        ) from exc

    dataset = load_processed_dataset(config, target=target, split="train", variant=variant)
    study = ClassifierStudy(
        study_name=study_name(config.data.retrospective_name, target, variant),
        dataset=dataset,
        target=target,
        num_iter=config.training.num_iter,
        num_study_iter=config.training.num_study_iter,
        num_ensemble_iter=config.training.num_ensemble_iter,
        timeout=config.training.timeout,
        metric=config.training.metric,
        score_threshold=config.training.score_threshold,
        feature_scaling=Preprocessors(category="feature_scaling").list_available(),
        feature_selection=Preprocessors(category="dimensionality_reduction").list_available(),
        classifiers=config.training.classifiers,
        imputers=config.training.imputers,
        ensemble_size=config.training.ensemble_size,
        workspace=config.paths.legacy_workspace,
        sample_for_search=config.training.sample_for_search,
        max_search_sample_size=config.training.max_search_sample_size,
        n_folds_cv=config.training.n_folds_cv,
    )
    model = study.run()
    features = dataset.drop(columns=[target])
    labels = dataset[target]
    cv_score = evaluate_estimator(model, features, labels, n_folds=config.training.n_folds_cv)
    model.fit(features, labels)
    model_path = config.paths.legacy_workspace / study_name(config.data.retrospective_name, target, variant)
    model_path.mkdir(parents=True, exist_ok=True)
    save_to_file(model_path / f"model_{config.training.metric}.p", model)
    return {
        "study_name": study_name(config.data.retrospective_name, target, variant),
        "model_name": model.name(),
        "score": cv_score,
        "model_path": str(model_path / f"model_{config.training.metric}.p"),
    }
