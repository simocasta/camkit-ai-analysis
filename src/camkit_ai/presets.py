from __future__ import annotations

from pathlib import Path

FULL_FEATURE_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "h_injury",
    "h_injury_c",
    "h_surgery",
    "h_surgery_c",
    "gjh",
    "activity_risk",
    "participation_level",
    "activity_train_comp",
    "surface",
    "footwear",
    "weather",
    "contact_noncontact",
    "pain_scale",
    "twisting",
    "hyperextension",
    "medial_lateral",
    "popping",
    "weightbear",
    "swelling",
    "rapid_delayed",
    "bruising",
    "reduced_rom",
    "locking",
    "instability",
]

INJURY_TOP12_FEATURES = [
    "swelling",
    "weightbear",
    "bmi",
    "reduced_rom",
    "twisting",
    "activity_risk",
    "pain_scale",
    "popping",
    "instability",
    "locking",
    "rapid_delayed",
    "hyperextension",
]

TARGETS = ["Injury", "ACL", "Cruciate", "Collateral", "Meniscus", "Surgery"]

MANUSCRIPT_MODEL_SPECS = [
    ("Injury", "full"),
    ("Injury", "top12"),
    ("ACL", "full"),
    ("Cruciate", "full"),
    ("Collateral", "full"),
    ("Meniscus", "full"),
    ("Surgery", "full"),
]

EXTERNAL_COLUMN_RENAMES = {
    "Medial Meniscus": "Medial meniscus",
    "Lateral Meniscus": "Lateral meniscus",
}


def validate_variant(target: str, variant: str) -> None:
    if variant not in {"full", "top12"}:
        raise ValueError(f"Unsupported variant '{variant}'. Expected 'full' or 'top12'.")
    if variant == "top12" and target != "Injury":
        raise ValueError("The top12 variant is only defined for the Injury target.")


def variant_suffix(variant: str) -> str:
    return "_12core" if variant == "top12" else ""


def features_for_variant(target: str, variant: str) -> list[str]:
    validate_variant(target, variant)
    return INJURY_TOP12_FEATURES if variant == "top12" else FULL_FEATURE_COLUMNS


def processed_dataset_path(processed_root: Path, split: str, target: str, variant: str) -> Path:
    validate_variant(target, variant)
    return processed_root / split / f"{target}.{variant}.csv"


def study_name(base_name: str, target: str, variant: str) -> str:
    validate_variant(target, variant)
    return f"{base_name}_{target}{variant_suffix(variant)}_train"


def subgroup_masks(frame) -> dict[str, object]:
    return {
        "female": frame["sex"] == 0,
        "male": frame["sex"] == 1,
        "older": frame["age"] > 25,
        "younger": frame["age"] <= 25,
    }
