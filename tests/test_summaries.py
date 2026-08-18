from __future__ import annotations

import pandas as pd

from camkit_ai.summaries import (
    ConvergenceTolerances,
    analyse_pooling_convergence,
    choose_draw_count,
    dataframe_to_markdown,
)


def _draw_frame() -> pd.DataFrame:
    rows = []
    outcomes = [0, 0, 1, 1]
    for dataset in ("holdout", "prospective"):
        for draw in range(4):
            probabilities = [0.10, 0.20 + draw * 0.001, 0.70, 0.90]
            for row_id, (outcome, probability) in enumerate(
                zip(outcomes, probabilities), start=1
            ):
                rows.append(
                    {
                        "dataset": dataset,
                        "draw": draw,
                        "seed": 42 + draw,
                        "row_id": row_id,
                        "y_true": outcome,
                        "probability": probability,
                    }
                )
    return pd.DataFrame(rows)


def test_convergence_chooses_smallest_candidate_passing_both_cohorts() -> None:
    convergence = analyse_pooling_convergence(
        _draw_frame(),
        candidate_draws=[2, 4],
        reference_draws=4,
        lower_threshold=0.29,
        upper_threshold=0.69,
        tolerances=ConvergenceTolerances(
            max_auc_difference=0.01,
            max_probability_difference=0.01,
            max_band_changes=0,
            max_band_count_difference=0,
        ),
    )

    assert choose_draw_count(convergence, reference_draws=4) == 2
    assert convergence[convergence["candidate_draws"] == 2][
        "meets_tolerances"
    ].all()


def test_dataframe_markdown_does_not_require_optional_tabulate() -> None:
    markdown = dataframe_to_markdown(
        pd.DataFrame([{"metric": "Average precision (AP)", "value": 0.5}])
    )

    assert "| metric | value |" in markdown
    assert "| Average precision (AP) | 0.500 |" in markdown
