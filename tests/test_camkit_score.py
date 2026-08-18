import pandas as pd

from camkit_ai.camkit_score import (
    CAMKIT_ITEM_COLUMNS,
    assign_camkit_band,
    calculate_camkit_score,
    camkit_band_counts,
    compare_to_published_prospective_counts,
    score_camkit_frame,
)


def _frame_with_scores(scores: list[int]) -> pd.DataFrame:
    scoring_columns = [column for column in CAMKIT_ITEM_COLUMNS if column != "weightbear"]
    rows = []
    for score in scores:
        row = {column: 0 for column in CAMKIT_ITEM_COLUMNS}
        row["weightbear"] = 1
        for column in scoring_columns[:score]:
            row[column] = 1
        if score == len(CAMKIT_ITEM_COLUMNS):
            row["weightbear"] = 0
        row["Injury"] = int(score >= 7)
        rows.append(row)
    return pd.DataFrame(rows)


def test_calculate_camkit_score_sums_twelve_binary_items() -> None:
    frame = _frame_with_scores([0, 3, 6, 7, 12])
    scores = calculate_camkit_score(frame)
    assert scores.tolist() == [0, 3, 6, 7, 12]


def test_calculate_camkit_score_inverts_weightbear_ability_column() -> None:
    frame = pd.DataFrame(
        [
            {**{column: 0 for column in CAMKIT_ITEM_COLUMNS}, "weightbear": 1},
            {**{column: 0 for column in CAMKIT_ITEM_COLUMNS}, "weightbear": 0},
        ]
    )
    scores = calculate_camkit_score(frame)
    assert scores.tolist() == [0, 1]


def test_assign_camkit_band_uses_original_thresholds() -> None:
    bands = assign_camkit_band(pd.Series([0, 3, 4, 6, 7, 12]))
    assert bands.tolist() == ["Low", "Low", "Medium", "Medium", "High", "High"]


def test_score_camkit_frame_adds_score_band_and_triage_band() -> None:
    scored = score_camkit_frame(_frame_with_scores([2, 5, 9]))
    assert scored["camkit_score"].tolist() == [2, 5, 9]
    assert scored["camkit_band"].tolist() == ["Low", "Medium", "High"]
    assert scored["triage_band"].tolist() == ["Green", "Amber", "Red"]


def test_camkit_band_counts_and_published_check() -> None:
    counts = camkit_band_counts(
        pd.Series([1, 0, 1, 0, 0, 0]),
        pd.Series(["High", "High", "Medium", "Medium", "Low", "Low"]),
    )
    high = counts[counts["camkit_band"] == "High"].iloc[0]
    assert high["injury"] == 1
    assert high["no_injury"] == 1

    check = compare_to_published_prospective_counts(
        pd.DataFrame(
            [
                {"camkit_band": "High", "injury": 15, "no_injury": 28, "total": 43},
                {"camkit_band": "Medium", "injury": 3, "no_injury": 26, "total": 29},
                {"camkit_band": "Low", "injury": 0, "no_injury": 13, "total": 13},
            ]
        )
    )
    assert check["matches_published"].all()
