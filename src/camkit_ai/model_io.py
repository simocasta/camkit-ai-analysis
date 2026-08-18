from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from camkit_ai.config import ProjectConfig
from camkit_ai.presets import study_name, validate_variant


def _import_autoprognosis_serialization():
    try:
        from autoprognosis.utils.serialization import load_model
    except ImportError as exc:
        raise RuntimeError(
            "AutoPrognosis is required for model loading. Install the optional "
            "'autop' dependency or use the existing local conda environment."
        ) from exc
    return load_model


@contextmanager
def _cpu_torch_load_patch(force_cpu: bool):
    if not force_cpu:
        yield
        return
    try:
        import torch
    except ImportError:
        yield
        return

    real_load = torch.load

    def cpu_load(*args, **kwargs):
        kwargs.setdefault("map_location", torch.device("cpu"))
        return real_load(*args, **kwargs)

    torch.load = cpu_load
    try:
        yield
    finally:
        torch.load = real_load


def legacy_model_path(config: ProjectConfig, target: str, variant: str) -> Path:
    validate_variant(target, variant)
    base_name = config.data.retrospective_name
    study = study_name(base_name, target, variant)
    metric_path = config.paths.legacy_workspace / study / f"model_{config.training.metric}.p"
    if metric_path.exists():
        return metric_path
    fallback = config.paths.legacy_workspace / study / "model.p"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not locate a saved model for {target} ({variant}).")


def load_legacy_model(path: Path, force_cpu: bool = True):
    load_model = _import_autoprognosis_serialization()
    with _cpu_torch_load_patch(force_cpu):
        try:
            return load_model(path.read_bytes())
        except TypeError as exc:
            if "code() argument 13 must be str, not int" in str(exc):
                raise RuntimeError(
                    "Legacy model deserialization failed because the artifact was likely "
                    "created under Python 3.10. Load it from the existing 'autop' "
                    "environment rather than Python 3.12."
                ) from exc
            raise
