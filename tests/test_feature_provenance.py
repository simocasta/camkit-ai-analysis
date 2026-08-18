from __future__ import annotations

from pathlib import Path

from camkit_ai.camkit_score import CAMKIT_ITEM_COLUMNS
from camkit_ai.presets import FULL_FEATURE_COLUMNS, INJURY_TOP12_FEATURES


def _feature_rows() -> list[tuple[str, str, str]]:
    repo_root = Path(__file__).resolve().parents[1]
    table = repo_root / "docs" / "feature_provenance.md"
    rows = []
    for line in table.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
        rows.append((cells[0], cells[1], cells[2]))
    return rows


def test_feature_provenance_table_covers_full_and_top12_features() -> None:
    rows = _feature_rows()
    retrospective_features = [row[0] for row in rows]
    prospective_inputs = [row[1] for row in rows]
    top12_features = [row[0] for row in rows if row[2] == "Yes"]

    assert retrospective_features == FULL_FEATURE_COLUMNS
    assert prospective_inputs == FULL_FEATURE_COLUMNS
    assert set(top12_features) == set(INJURY_TOP12_FEATURES)


def test_camkit_ai_and_original_camkit_share_exactly_ten_inputs() -> None:
    ai_features = set(INJURY_TOP12_FEATURES)
    score_items = set(CAMKIT_ITEM_COLUMNS)

    assert ai_features & score_items == {
        "activity_risk",
        "hyperextension",
        "instability",
        "locking",
        "popping",
        "rapid_delayed",
        "reduced_rom",
        "swelling",
        "twisting",
        "weightbear",
    }
    assert ai_features - score_items == {"bmi", "pain_scale"}
    assert score_items - ai_features == {"h_injury", "contact_noncontact"}

