"""Pin the strict triage boundaries used throughout the analysis.

Discharge applies at ``p < 0.29``, not ``p <= 0.29``. The implemented rule was
already strict and the reported metrics were consistent with it, but nothing in
the test suite said so.

The exact-boundary cases are the ones that matter: a patient scoring precisely
0.29 must be reassessed rather than discharged, and one scoring precisely 0.69
must be referred.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from camkit_ai.comparators import assign_camkit_ai_triage_band

LOWER = 0.29
UPPER = 0.69


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, "Green"),
        (0.2899999, "Green"),
        # Exactly at the lower boundary: NOT discharge.
        (LOWER, "Amber"),
        (0.5, "Amber"),
        (0.6899999, "Amber"),
        # Exactly at the upper boundary: referral.
        (UPPER, "Red"),
        (1.0, "Red"),
    ],
)
def test_band_boundaries_are_strict_at_the_lower_cut_point(
    probability: float, expected: str
) -> None:
    band = assign_camkit_ai_triage_band(pd.Series([probability]), LOWER, UPPER)

    assert band.iloc[0] == expected


def test_discharge_is_strictly_below_the_lower_threshold() -> None:
    probabilities = pd.Series([0.28, LOWER, 0.30])

    bands = assign_camkit_ai_triage_band(probabilities, LOWER, UPPER)

    # One patient below 0.29 is discharged; the patient at exactly 0.29 is not.
    assert bands.tolist() == ["Green", "Amber", "Amber"]


def test_referral_includes_the_upper_threshold() -> None:
    probabilities = pd.Series([0.68, UPPER, 0.70])

    bands = assign_camkit_ai_triage_band(probabilities, LOWER, UPPER)

    assert bands.tolist() == ["Amber", "Red", "Red"]


def test_bands_partition_the_unit_interval() -> None:
    probabilities = pd.Series(np.linspace(0.0, 1.0, num=201))

    bands = assign_camkit_ai_triage_band(probabilities, LOWER, UPPER)

    assert set(bands) == {"Green", "Amber", "Red"}
    assert bands.notna().all()
    # Bands are monotone in probability: no patient with a higher probability
    # may fall in a lower-acuity band.
    order = {"Green": 0, "Amber": 1, "Red": 2}
    ranks = bands.map(order).to_numpy()
    assert (np.diff(ranks) >= 0).all()


def test_discharge_count_matches_a_strict_comparison() -> None:
    rng = np.random.default_rng(0)
    probabilities = pd.Series(rng.uniform(0.0, 1.0, size=500))

    bands = assign_camkit_ai_triage_band(probabilities, LOWER, UPPER)

    assert int((bands == "Green").sum()) == int((probabilities < LOWER).sum())
    assert int((bands == "Red").sum()) == int((probabilities >= UPPER).sum())
