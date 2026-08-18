"""Unit tests for the frozen post hoc analyses reported in the manuscript.

``capacity_matched_analysis`` produces the comparison the paper's conclusion
rests on (CamKIT-AI ranked to CamKIT's referral volume), so its selection,
tie-rejection and cumulative-capture behaviour are pinned here on a synthetic
cohort whose answers can be worked out by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from camkit_ai.config import ConfidenceIntervalConfig, PathsConfig, ProjectConfig
from camkit_ai.posthoc import capacity_matched_analysis, run_capacity_match


N_PATIENTS = 85
N_EVENTS = 18
CAPACITY = 41


def _config(tmp_path) -> ProjectConfig:
    return ProjectConfig(
        paths=PathsConfig(
            retrospective_encoded=tmp_path / "retro.csv",
            prospective_encoded=tmp_path / "prospective.csv",
            legacy_workspace=tmp_path,
            output_root=tmp_path / "results",
        ),
        confidence_intervals=ConfidenceIntervalConfig(bootstrap_iterations=20),
    )


def _cohort(
    *,
    injured_ranks: set[int],
    camkit_high_ranks: set[int],
    probabilities: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a frozen-shape cohort indexed by CamKIT-AI rank.

    Rank 0 is the highest CamKIT-AI probability.  ``injured_ranks`` and
    ``camkit_high_ranks`` are expressed in that rank order so a test can state
    the expected capture directly.
    """
    if probabilities is None:
        # Strictly decreasing and comfortably clear of the 1e-12 tie tolerance.
        probabilities = np.linspace(0.95, 0.05, N_PATIENTS)
    return pd.DataFrame(
        {
            "row_id": np.arange(1, N_PATIENTS + 1),
            "Injury": [1 if r in injured_ranks else 0 for r in range(N_PATIENTS)],
            "camkit_ai_probability": probabilities,
            "camkit_ai_triage_band": [
                "Red" if p >= 0.69 else "Amber" if p >= 0.29 else "Green"
                for p in probabilities
            ],
            "camkit_triage_band": [
                "Red" if r in camkit_high_ranks else "Amber"
                for r in range(N_PATIENTS)
            ],
            "prediction_basis": "frozen_mean_probability",
            "prediction_source": "test",
            "prediction_context": "complete_cohort",
            "camkit_score_source": "test",
        }
    )


def test_capacity_match_selects_the_top_k_and_counts_their_injuries(tmp_path) -> None:
    # 14 injuries inside CamKIT's 41-patient capacity, 4 below it.
    injured = set(range(14)) | {50, 60, 70, 80}
    frame = _cohort(injured_ranks=injured, camkit_high_ranks=set(range(CAPACITY)))

    result = capacity_matched_analysis(frame, _config(tmp_path))
    matched = result.policies.set_index("policy_id").loc["camkit_ai_top_41"]

    assert matched["referrals"] == CAPACITY
    assert matched["injuries_captured"] == 14
    assert matched["ppv"] == pytest.approx(14 / 41)
    assert matched["referrals_per_injury"] == pytest.approx(41 / 14, abs=5e-3)
    assert bool(matched["post_hoc"]) is True


def test_capacity_defaults_to_camkits_observed_referral_volume(tmp_path) -> None:
    frame = _cohort(
        injured_ranks=set(range(N_EVENTS)),
        camkit_high_ranks=set(range(30)),
    )

    result = capacity_matched_analysis(frame, _config(tmp_path))

    assert int(result.overlap.loc[0, "capacity"]) == 30
    assert result.policies.set_index("policy_id").loc["camkit_ai_top_41", "referrals"] == 30


def test_requested_capacity_must_equal_camkits_own(tmp_path) -> None:
    frame = _cohort(
        injured_ranks=set(range(N_EVENTS)),
        camkit_high_ranks=set(range(CAPACITY)),
    )

    with pytest.raises(ValueError, match="does not equal CamKIT's observed"):
        capacity_matched_analysis(frame, _config(tmp_path), capacity=40)


def test_probability_ties_are_rejected_rather_than_broken(tmp_path) -> None:
    probabilities = np.linspace(0.95, 0.05, N_PATIENTS)
    probabilities[10] = probabilities[11]  # an exact tie away from the boundary
    frame = _cohort(
        injured_ranks=set(range(N_EVENTS)),
        camkit_high_ranks=set(range(CAPACITY)),
        probabilities=probabilities,
    )

    with pytest.raises(ValueError, match="probability tie"):
        capacity_matched_analysis(frame, _config(tmp_path))


def test_overlap_partitions_referrals_and_injuries_without_double_counting(
    tmp_path,
) -> None:
    # CamKIT's 41 are shifted 5 ranks down, so 36 are shared with CamKIT-AI's top 41.
    injured = set(range(14)) | {50, 60, 70, 80}
    frame = _cohort(
        injured_ranks=injured,
        camkit_high_ranks=set(range(5, 5 + CAPACITY)),
    )

    overlap = capacity_matched_analysis(frame, _config(tmp_path)).overlap.loc[0]

    assert overlap["shared_referrals"] == 36
    assert overlap["ai_only_referrals"] == 5
    assert overlap["camkit_only_referrals"] == 5
    injury_partition = (
        overlap["shared_injuries"]
        + overlap["ai_only_injuries"]
        + overlap["camkit_only_injuries"]
        + overlap["neither_injuries"]
    )
    assert injury_partition == N_EVENTS


def test_cumulative_curve_covers_every_capacity_and_ends_at_full_capture(
    tmp_path,
) -> None:
    injured = set(range(14)) | {50, 60, 70, 80}
    frame = _cohort(injured_ranks=injured, camkit_high_ranks=set(range(CAPACITY)))

    curve = capacity_matched_analysis(frame, _config(tmp_path)).curve

    # One row per capacity from 0 to 85 inclusive.
    assert curve["referrals"].tolist() == list(range(N_PATIENTS + 1))
    assert curve.loc[0, "injuries_captured"] == 0
    assert curve["injuries_captured"].iloc[-1] == N_EVENTS
    assert curve["injuries_captured"].is_monotonic_increasing
    # Capture at the matched capacity agrees with the policy row.
    assert int(curve.loc[CAPACITY, "injuries_captured"]) == 14


def test_frozen_cohort_shape_is_enforced(tmp_path) -> None:
    frame = _cohort(
        injured_ranks=set(range(N_EVENTS)),
        camkit_high_ranks=set(range(CAPACITY)),
    ).iloc[:80]

    with pytest.raises(ValueError, match="frozen 85-patient/18-event cohort"):
        capacity_matched_analysis(frame, _config(tmp_path))


def test_run_capacity_match_reads_saved_predictions_and_writes_three_tables(
    tmp_path,
) -> None:
    config = _config(tmp_path)
    injured = set(range(14)) | {50, 60, 70, 80}
    source = tmp_path / "predictions.csv"
    _cohort(injured_ranks=injured, camkit_high_ranks=set(range(CAPACITY))).to_csv(
        source, index=False
    )

    result, paths = run_capacity_match(config, predictions_path=source)

    assert set(paths) == {"policies", "overlap", "curve"}
    assert all(path.is_file() for path in paths.values())
    # The driver must not alter the analysis it wraps.
    written = pd.read_csv(paths["policies"]).set_index("policy_id")
    assert written.loc["camkit_ai_top_41", "referrals"] == CAPACITY
    assert written.loc["camkit_ai_top_41", "injuries_captured"] == 14
    assert result.curve["referrals"].tolist() == list(range(N_PATIENTS + 1))


def test_run_capacity_match_names_the_missing_file_and_the_step_that_makes_it(
    tmp_path,
) -> None:
    with pytest.raises(FileNotFoundError, match="Run 'compare-models' first"):
        run_capacity_match(_config(tmp_path))
