from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from camkit_ai.config import ProjectConfig
from camkit_ai.data import load_processed_dataset


CAMKIT_ITEM_COLUMNS: tuple[str, ...] = (
    "h_injury",
    "activity_risk",
    "contact_noncontact",
    "swelling",
    "rapid_delayed",
    "weightbear",
    "reduced_rom",
    "twisting",
    "hyperextension",
    "instability",
    "popping",
    "locking",
)
CAMKIT_INVERTED_ITEM_COLUMNS: tuple[str, ...] = ("weightbear",)
PROSPECTIVE_RAW_SCORE_COLUMN = "camkit_Scores /12"
PROSPECTIVE_RAW_OUTCOME_COLUMN = "Diganostic outcome.1"
PROSPECTIVE_RAW_FALLBACK_OUTCOME_COLUMN = "Diganostic outcome"

CAMKIT_BAND_ORDER: tuple[str, ...] = ("Low", "Medium", "High")
TRIAGE_BAND_ORDER: tuple[str, ...] = ("Green", "Amber", "Red")
CAMKIT_TO_TRIAGE_BAND = {"Low": "Green", "Medium": "Amber", "High": "Red"}

PUBLISHED_PROSPECTIVE_BAND_COUNTS = pd.DataFrame(
    [
        {"camkit_band": "High", "injury": 15, "no_injury": 28, "total": 43},
        {"camkit_band": "Medium", "injury": 3, "no_injury": 26, "total": 29},
        {"camkit_band": "Low", "injury": 0, "no_injury": 13, "total": 13},
    ]
)


@dataclass(frozen=True)
class CamkitScoreResult:
    scored_frame: pd.DataFrame
    band_counts: pd.DataFrame
    published_count_check: pd.DataFrame
    published_counts_match: bool
    score_source: str
    reconstruction_check: pd.DataFrame | None = None


def validate_camkit_item_columns(frame: pd.DataFrame) -> None:
    missing = sorted(set(CAMKIT_ITEM_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Frame is missing CamKIT score columns: {missing}")

    invalid: dict[str, list[object]] = {}
    for column in CAMKIT_ITEM_COLUMNS:
        values = set(frame[column].dropna().unique().tolist())
        if not values.issubset({0, 1, 0.0, 1.0, False, True}):
            invalid[column] = sorted(values)
    if invalid:
        raise ValueError(f"CamKIT score columns must be binary 0/1 values: {invalid}")

    columns_with_missing = [
        column for column in CAMKIT_ITEM_COLUMNS if frame[column].isna().any()
    ]
    if columns_with_missing:
        raise ValueError(
            "CamKIT score requires complete binary inputs; missing values found in "
            f"{columns_with_missing}"
        )


def calculate_camkit_score(frame: pd.DataFrame) -> pd.Series:
    """Calculate the original 12-point CamKIT score from encoded binary columns."""
    validate_camkit_item_columns(frame)
    item_scores = frame.loc[:, CAMKIT_ITEM_COLUMNS].astype(int).copy()
    item_scores.loc[:, CAMKIT_INVERTED_ITEM_COLUMNS] = (
        1 - item_scores.loc[:, CAMKIT_INVERTED_ITEM_COLUMNS]
    )
    score = item_scores.sum(axis=1)
    score.name = "camkit_score"
    return score


def assign_camkit_band(score: pd.Series) -> pd.Series:
    bands = pd.cut(
        score,
        bins=[-1, 3, 6, 12],
        labels=CAMKIT_BAND_ORDER,
        ordered=True,
    )
    return pd.Series(bands.astype(str), index=score.index, name="camkit_band")


def camkit_triage_band(camkit_band: pd.Series) -> pd.Series:
    triage = camkit_band.map(CAMKIT_TO_TRIAGE_BAND)
    return pd.Series(triage, index=camkit_band.index, name="triage_band")


def score_camkit_frame(frame: pd.DataFrame, outcome_column: str = "Injury") -> pd.DataFrame:
    score = calculate_camkit_score(frame)
    camkit_band = assign_camkit_band(score)
    triage_band = camkit_triage_band(camkit_band)
    output = pd.DataFrame(
        {
            "row_id": range(1, len(frame) + 1),
            "camkit_score": score,
            "camkit_band": camkit_band,
            "triage_band": triage_band,
        }
    )
    if outcome_column in frame.columns:
        output[outcome_column] = frame[outcome_column].astype(int).to_numpy()
    return output


def _load_stored_prospective_camkit_score(
    config: ProjectConfig,
    processed_frame: pd.DataFrame,
    reconstructed: pd.DataFrame,
    outcome_column: str = "Injury",
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    path = config.paths.prospective_labels
    if path is None or not path.exists():
        return None

    raw = pd.read_excel(path)
    if len(raw) != len(processed_frame):
        raise ValueError(
            "Stored prospective CamKIT score row count does not match the processed "
            f"prospective frame: {len(raw)} vs {len(processed_frame)}."
        )
    if PROSPECTIVE_RAW_SCORE_COLUMN not in raw.columns:
        raise ValueError(
            f"Prospective labels file is missing '{PROSPECTIVE_RAW_SCORE_COLUMN}'."
        )

    raw_outcome_column = (
        PROSPECTIVE_RAW_OUTCOME_COLUMN
        if PROSPECTIVE_RAW_OUTCOME_COLUMN in raw.columns
        else PROSPECTIVE_RAW_FALLBACK_OUTCOME_COLUMN
    )
    if raw_outcome_column not in raw.columns:
        raise ValueError("Prospective labels file is missing the diagnostic outcome column.")

    raw_outcome = (
        raw[raw_outcome_column].astype(str).str.strip().str.casefold().ne("no injury").astype(int)
    )
    processed_outcome = processed_frame[outcome_column].astype(int).reset_index(drop=True)
    if not raw_outcome.reset_index(drop=True).equals(processed_outcome):
        raise ValueError(
            "Stored prospective CamKIT score outcomes do not match the processed "
            "prospective frame ordering."
        )

    stored_score = raw[PROSPECTIVE_RAW_SCORE_COLUMN].astype(int).reset_index(drop=True)
    stored_band = assign_camkit_band(stored_score)
    stored_triage_band = camkit_triage_band(stored_band)
    scored = pd.DataFrame(
        {
            "row_id": range(1, len(raw) + 1),
            "record_id": raw["Record ID"].to_numpy() if "Record ID" in raw.columns else None,
            "camkit_score": stored_score,
            "camkit_band": stored_band,
            "triage_band": stored_triage_band,
            outcome_column: processed_outcome.to_numpy(),
            "score_source": "stored_prospective_labels",
        }
    )

    reconstruction_check = pd.DataFrame(
        {
            "row_id": range(1, len(raw) + 1),
            "record_id": raw["Record ID"].to_numpy() if "Record ID" in raw.columns else None,
            "stored_camkit_score": stored_score,
            "reconstructed_camkit_score": reconstructed["camkit_score"].astype(int).to_numpy(),
            "stored_camkit_band": stored_band,
            "reconstructed_camkit_band": reconstructed["camkit_band"].to_numpy(),
            outcome_column: processed_outcome.to_numpy(),
        }
    )
    reconstruction_check["score_difference"] = (
        reconstruction_check["reconstructed_camkit_score"]
        - reconstruction_check["stored_camkit_score"]
    )
    reconstruction_check["band_matches"] = (
        reconstruction_check["stored_camkit_band"]
        == reconstruction_check["reconstructed_camkit_band"]
    )
    return scored, reconstruction_check


def camkit_band_counts(
    y_true: pd.Series,
    camkit_band: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    y_true = pd.Series(y_true).astype(int)
    camkit_band = pd.Series(camkit_band).astype(str)
    for band in ("High", "Medium", "Low"):
        mask = camkit_band == band
        injury = int((y_true[mask] == 1).sum())
        no_injury = int((y_true[mask] == 0).sum())
        rows.append(
            {
                "camkit_band": band,
                "triage_band": CAMKIT_TO_TRIAGE_BAND[band],
                "injury": injury,
                "no_injury": no_injury,
                "total": injury + no_injury,
            }
        )
    return pd.DataFrame(rows)


def compare_to_published_prospective_counts(counts: pd.DataFrame) -> pd.DataFrame:
    expected = PUBLISHED_PROSPECTIVE_BAND_COUNTS.rename(
        columns={
            "injury": "expected_injury",
            "no_injury": "expected_no_injury",
            "total": "expected_total",
        }
    )
    observed = counts.loc[:, ["camkit_band", "injury", "no_injury", "total"]].rename(
        columns={
            "injury": "observed_injury",
            "no_injury": "observed_no_injury",
            "total": "observed_total",
        }
    )
    merged = expected.merge(observed, on="camkit_band", how="left")
    for column in ("injury", "no_injury", "total"):
        merged[f"{column}_difference"] = (
            merged[f"observed_{column}"] - merged[f"expected_{column}"]
        )
    merged["matches_published"] = (
        (merged["injury_difference"] == 0)
        & (merged["no_injury_difference"] == 0)
        & (merged["total_difference"] == 0)
    )
    return merged


def evaluate_camkit_score(
    config: ProjectConfig,
    split: str = "prospective",
) -> CamkitScoreResult:
    frame = load_processed_dataset(config, "Injury", split, "full")
    reconstructed = score_camkit_frame(frame, outcome_column="Injury")
    stored = (
        _load_stored_prospective_camkit_score(config, frame, reconstructed)
        if split == "prospective"
        else None
    )
    if stored is None:
        scored = reconstructed
        reconstruction_check = None
        score_source = "processed_reconstruction"
    else:
        scored, reconstruction_check = stored
        score_source = "stored_prospective_labels"
    counts = camkit_band_counts(scored["Injury"], scored["camkit_band"])
    check = compare_to_published_prospective_counts(counts)
    return CamkitScoreResult(
        scored_frame=scored,
        band_counts=counts,
        published_count_check=check,
        published_counts_match=bool(check["matches_published"].all()),
        score_source=score_source,
        reconstruction_check=reconstruction_check,
    )


def save_camkit_score_outputs(
    result: CamkitScoreResult,
    output_dir: Path,
    split: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "scores": output_dir / f"camkit_score_{split}.csv",
        "band_counts": output_dir / f"camkit_band_counts_{split}.csv",
        "published_count_check": output_dir
        / f"camkit_published_count_check_{split}.csv",
    }
    if result.reconstruction_check is not None:
        paths["reconstruction_check"] = (
            output_dir / f"camkit_score_reconstruction_check_{split}.csv"
        )
    result.scored_frame.to_csv(paths["scores"], index=False)
    result.band_counts.to_csv(paths["band_counts"], index=False)
    result.published_count_check.to_csv(paths["published_count_check"], index=False)
    if result.reconstruction_check is not None:
        result.reconstruction_check.to_csv(paths["reconstruction_check"], index=False)
    return paths
