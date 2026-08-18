"""Score the locked artefact reproducibly despite stochastic imputation.

The saved AutoPrognosis pipeline imputes missing values at inference time and
that imputation is stochastic, so ``predict_proba`` returns one draw from a
distribution rather than a fixed answer. Every module that scores the model
needs the same two operations — take a draw under a known seed, or average over
a fixed sequence of draws — so they live here rather than in any one caller.

The primitives are deliberately separate from the reporting built on top of
them in :mod:`camkit_ai.imputation_variability`, because ``comparators`` needs
to score the model and that module imports ``comparators``.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from camkit_ai.metrics import positive_class_probabilities


def seed_global_rngs(seed: int) -> None:
    """Seed every generator the saved pipeline might draw from.

    numpy's legacy global state and torch both matter: the neural imputers in
    AutoPrognosis sample through torch, while several classical imputers use
    numpy. Generators created internally with ``np.random.default_rng()`` are
    beyond reach, so a draw is reproducible only to the extent that the
    pipeline uses the global streams.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def predict_draw(model, features: pd.DataFrame, seed: int) -> np.ndarray:
    """Score one imputation draw under a known seed."""
    seed_global_rngs(seed)
    return positive_class_probabilities(model.predict_proba(features))


def draw_matrix(
    model,
    features: pd.DataFrame,
    *,
    n_draws: int,
    base_seed: int,
) -> np.ndarray:
    """Return an ``(n_draws, n_patients)`` matrix of predicted probabilities."""
    return np.vstack(
        [predict_draw(model, features, base_seed + index) for index in range(n_draws)]
    )


def collect_draws(
    model,
    frame: pd.DataFrame,
    target: str,
    *,
    n_draws: int,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return outcomes and an ``(n_draws, n_patients)`` matrix of probabilities."""
    features = frame.drop(columns=[target])
    y_true = frame[target].to_numpy(dtype=int)
    return y_true, draw_matrix(model, features, n_draws=n_draws, base_seed=base_seed)


def pool_predictions(
    model,
    features: pd.DataFrame,
    *,
    n_draws: int,
    base_seed: int,
) -> np.ndarray:
    """Average probabilities over a fixed sequence of inference draws.

    This is post hoc Monte Carlo integration over randomness in the saved
    inference pipeline. It is not Rubin's-rules multiple-imputation pooling and
    it is not identical to a single unseeded inference call.
    The fitted artefact is unchanged, but the averaged prediction is a
    stabilised inference specification that must be labelled as such.

    The seed sequence is fixed, so the Monte Carlo average is reproducible.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be at least 1 to pool predictions.")
    return draw_matrix(model, features, n_draws=n_draws, base_seed=base_seed).mean(axis=0)


def predict_with_basis(
    model,
    features: pd.DataFrame,
    *,
    pooled_draws: int | None,
    pool_seed: int,
) -> tuple[np.ndarray, str]:
    """Score a feature frame and say what the resulting numbers rest on.

    ``pooled_draws=None`` takes the unpooled path: one call to
    ``predict_proba`` with nothing seeded, which is not reproducible. Any
    other value averages over that many seeded draws. Keeping both behind one
    function means the basis label can never drift from the computation it
    describes.
    """
    if pooled_draws is None:
        probabilities = positive_class_probabilities(model.predict_proba(features))
        return probabilities, pooled_basis_label(None, None)
    probabilities = pool_predictions(
        model, features, n_draws=pooled_draws, base_seed=pool_seed
    )
    return probabilities, pooled_basis_label(pooled_draws, pool_seed)


def pooled_basis_label(n_draws: int | None, base_seed: int | None) -> str:
    """Describe the prediction basis so it can travel with the predictions.

    Every prediction written to disk carries this string. Without it there is no
    way to tell a pooled table from a single-draw one after the fact, which is
    the failure the reproducibility audit exists to catch.
    """
    if n_draws is None:
        return "single_draw_unseeded" if base_seed is None else f"single_draw_seed_{base_seed}"
    return f"pooled_{n_draws}_draws_seed_{base_seed}"


#: How the definitive estimates were produced. The basis label says how
#: many draws were averaged; this says what was in the scoring call. Both are
#: needed, because the diagnostics showed that predictions for records with
#: missing inputs depend on which other records were scored alongside them, so a
#: draw count alone does not identify the estimand.
FIXED_COHORT_CONTEXT = "offline_full_evaluation_cohort_scored_together"


def fixed_cohort_basis_label(n_draws: int, base_seed: int) -> str:
    """Name the definitive fixed-batch estimand, seed range included.

    ``pooled_basis_label`` names only the first seed, which was enough while the
    draw count was still being chosen. The definitive analysis averages a closed
    seed sequence, so the label carries both ends of it: a reader can tell from
    the string alone which draws are in the mean, and the audit can reject any
    table built from a different sequence.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be at least 1 to label a fixed-cohort mean.")
    return (
        f"fixed_cohort_batch_mean_{n_draws}_draws_"
        f"seed_{base_seed}_{base_seed + n_draws - 1}"
    )
