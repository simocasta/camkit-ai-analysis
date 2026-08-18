"""Tests for the pooling primitives and the provenance they carry.

The point of these is narrow: a stub model cannot show what the real
AutoPrognosis artefact does, so nothing here claims to. What they pin down is
that a seeded draw is repeatable, that pooling averages the sequence it says it
averages, and above all that the basis label never disagrees with the
computation that produced the numbers — a wrong label is how a single-draw
figure reaches the manuscript wearing a pooled name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from camkit_ai.pooling import (
    FIXED_COHORT_CONTEXT,
    collect_draws,
    fixed_cohort_basis_label,
    pool_predictions,
    pooled_basis_label,
    predict_draw,
    predict_with_basis,
)


class _SeedSensitiveModel:
    """Stand-in whose output depends on the global numpy seed.

    This mimics the one property that matters: the artefact consumes the global
    random stream when it imputes, so seeding it makes a draw repeatable.
    """

    def __init__(self, incomplete: list[bool], scale: float = 0.02) -> None:
        self.incomplete = np.asarray(incomplete, dtype=bool)
        self.scale = scale

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        base = np.linspace(0.2, 0.8, num=len(features))
        jitter = np.random.default_rng(np.random.randint(0, 2**31)).normal(
            scale=self.scale, size=len(features)
        )
        positive = base + np.where(self.incomplete[: len(features)], jitter, 0.0)
        return np.column_stack([1.0 - positive, positive])


def _frame(n_rows: int = 6) -> pd.DataFrame:
    return pd.DataFrame({"a": np.arange(n_rows, dtype=float)})


def test_predict_draw_is_repeatable_under_one_seed() -> None:
    model = _SeedSensitiveModel([True] * 6)
    features = _frame()

    first = predict_draw(model, features, 42)
    second = predict_draw(model, features, 42)

    np.testing.assert_array_equal(first, second)


def test_different_seeds_give_different_draws() -> None:
    model = _SeedSensitiveModel([True] * 6)
    features = _frame()

    assert not np.array_equal(
        predict_draw(model, features, 1), predict_draw(model, features, 2)
    )


def test_pooling_averages_the_named_seed_sequence() -> None:
    model = _SeedSensitiveModel([True] * 6)
    features = _frame()

    pooled = pool_predictions(model, features, n_draws=4, base_seed=7)
    expected = np.vstack([predict_draw(model, features, 7 + i) for i in range(4)]).mean(
        axis=0
    )

    np.testing.assert_allclose(pooled, expected)


def test_pooling_is_reproducible() -> None:
    model = _SeedSensitiveModel([True] * 6)
    features = _frame()

    np.testing.assert_array_equal(
        pool_predictions(model, features, n_draws=5, base_seed=42),
        pool_predictions(model, features, n_draws=5, base_seed=42),
    )


def test_pooling_rejects_a_draw_count_below_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        pool_predictions(_SeedSensitiveModel([True]), _frame(1), n_draws=0, base_seed=1)


def test_collect_draws_varies_only_the_rows_the_model_varies() -> None:
    incomplete = [True, False, True, False, False, True]
    model = _SeedSensitiveModel(incomplete)
    frame = _frame()
    frame["Injury"] = [0, 1, 0, 1, 0, 1]

    _, draws = collect_draws(model, frame, "Injury", n_draws=8, base_seed=3)
    moved = np.ptp(draws, axis=0) > 1e-12

    assert moved.tolist() == incomplete


def test_basis_label_distinguishes_pooled_from_single_draw() -> None:
    assert pooled_basis_label(50, 42) == "pooled_50_draws_seed_42"
    assert pooled_basis_label(None, None) == "single_draw_unseeded"
    assert pooled_basis_label(None, 42) == "single_draw_seed_42"


def test_fixed_cohort_label_names_the_whole_seed_range() -> None:
    # The definitive estimand. Both ends of the sequence appear, so a
    # table built from a different range cannot pass as this one.
    assert (
        fixed_cohort_basis_label(400, 42)
        == "fixed_cohort_batch_mean_400_draws_seed_42_441"
    )
    assert (
        fixed_cohort_basis_label(1, 42) == "fixed_cohort_batch_mean_1_draws_seed_42_42"
    )


def test_fixed_cohort_label_is_distinct_from_the_pooled_label() -> None:
    # These describe different estimands: the pooled label says only how many
    # draws, the fixed-cohort label also asserts what was in the scoring call.
    assert fixed_cohort_basis_label(400, 42) != pooled_basis_label(400, 42)


def test_fixed_cohort_label_rejects_an_empty_sequence() -> None:
    with pytest.raises(ValueError):
        fixed_cohort_basis_label(0, 42)


def test_fixed_cohort_context_states_that_scoring_was_cohort_batched() -> None:
    # The manuscript tables compare against this exact string.
    assert FIXED_COHORT_CONTEXT == "offline_full_evaluation_cohort_scored_together"


def test_predict_with_basis_labels_the_unpooled_path_honestly() -> None:
    model = _SeedSensitiveModel([True] * 6)

    _, basis = predict_with_basis(model, _frame(), pooled_draws=None, pool_seed=42)

    # The seed is deliberately ignored when not pooling: that path calls
    # predict_proba directly, so claiming a seed would misdescribe it.
    assert basis == "single_draw_unseeded"


def test_predict_with_basis_matches_pool_predictions() -> None:
    model = _SeedSensitiveModel([True] * 6)
    features = _frame()

    probabilities, basis = predict_with_basis(
        model, features, pooled_draws=3, pool_seed=11
    )

    assert basis == "pooled_3_draws_seed_11"
    np.testing.assert_allclose(
        probabilities, pool_predictions(model, features, n_draws=3, base_seed=11)
    )
