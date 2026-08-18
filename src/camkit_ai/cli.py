from __future__ import annotations

import argparse
from pathlib import Path

from camkit_ai.comparators import (
    run_camkit_score_analysis,
    run_model_comparison_analysis,
)
from camkit_ai.config import load_config
from camkit_ai.data import prepare_datasets
from camkit_ai.evaluation import evaluate_model, report_manuscript, save_model_evaluation
from camkit_ai.imputation_variability import DEFAULT_DRAWS, run_imputation_variability
from camkit_ai.oof import generate_oof_predictions, oof_predictions_path
from camkit_ai.posthoc import run_capacity_match
from camkit_ai.shift import run_shift_analysis
from camkit_ai.train import train_study

# Pooled, seeded predictions are the default everywhere. The unpooled path is
# the one the paper reports as irreproducible, so it has to be asked for
# explicitly rather than arrived at by leaving a flag off.
DEFAULT_POOLED_DRAWS = 50
DEFAULT_POOL_SEED = 42


def _add_pooling_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pooled-draws",
        type=int,
        default=DEFAULT_POOLED_DRAWS,
        help=(
            "Average predictions over this many seeded imputation draws "
            f"(default {DEFAULT_POOLED_DRAWS})."
        ),
    )
    parser.add_argument(
        "--pool-seed",
        type=int,
        default=DEFAULT_POOL_SEED,
        help=f"Base seed for pooled draws (default {DEFAULT_POOL_SEED}).",
    )
    parser.add_argument(
        "--no-pooling",
        action="store_true",
        help=(
            "Take a single unseeded draw. This is the unpooled inference path "
            "the paper reports as irreproducible: use only to reproduce that "
            "behaviour, never for new results."
        ),
    )


def _resolve_pooling(args: argparse.Namespace) -> tuple[int | None, int | None]:
    """Turn the pooling flags into the (pooled_draws, pool_seed) pair."""
    if getattr(args, "no_pooling", False):
        return None, None
    return args.pooled_draws, args.pool_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CamKIT-AI pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Create processed train/holdout/prospective datasets.")
    prepare.add_argument("--config", required=True, help="Path to the YAML config file.")

    evaluate = subparsers.add_parser("evaluate-model", help="Evaluate one saved legacy model.")
    evaluate.add_argument("--config", required=True, help="Path to the YAML config file.")
    evaluate.add_argument("--target", required=True, help="Outcome target, for example Injury or ACL.")
    evaluate.add_argument("--variant", choices=["full", "top12"], default="full")
    evaluate.add_argument("--subgroups", action="store_true", help="Run subgroup evaluation.")
    evaluate.add_argument("--thresholds", action="store_true", help="Run threshold analysis.")
    evaluate.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=None,
        help="Override the bootstrap iteration count for this run.",
    )
    _add_pooling_arguments(evaluate)

    report = subparsers.add_parser("report-manuscript", help="Generate manuscript-facing evaluation tables.")
    report.add_argument("--config", required=True, help="Path to the YAML config file.")
    report.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=None,
        help="Override the bootstrap iteration count for this run.",
    )
    _add_pooling_arguments(report)

    train = subparsers.add_parser("train-study", help="Train a new AutoPrognosis study.")
    train.add_argument("--config", required=True, help="Path to the YAML config file.")
    train.add_argument("--target", required=True, help="Outcome target, for example Injury or ACL.")
    train.add_argument("--variant", choices=["full", "top12"], default="full")

    generate_oof = subparsers.add_parser(
        "generate-oof",
        help="Generate repeated-CV out-of-fold predictions for threshold derivation.",
    )
    generate_oof.add_argument("--config", required=True, help="Path to the YAML config file.")
    generate_oof.add_argument("--target", required=True, help="Outcome target, for example Injury.")
    generate_oof.add_argument("--variant", choices=["full", "top12"], default="full")

    compare_camkit = subparsers.add_parser(
        "compare-camkit",
        help="Recreate the original CamKIT score and band counts.",
    )
    compare_camkit.add_argument("--config", required=True, help="Path to the YAML config file.")
    compare_camkit.add_argument(
        "--split",
        choices=["train", "holdout", "prospective"],
        default="prospective",
        help="Processed split to score.",
    )
    compare_camkit.add_argument(
        "--output-dir",
        default=None,
        help="Directory for analysis outputs; defaults to <output_root>/analysis.",
    )
    compare_camkit.add_argument(
        "--require-published-counts",
        action="store_true",
        help="Exit with an error if prospective CamKIT band counts do not match the published feasibility paper.",
    )
    # No pooling flags: this subcommand recreates the original CamKIT score,
    # which is a deterministic function of the record and never calls the model.

    compare_models = subparsers.add_parser(
        "compare-models",
        help="Compare locked CamKIT-AI triage bands with the original CamKIT score.",
    )
    compare_models.add_argument("--config", required=True, help="Path to the YAML config file.")
    compare_models.add_argument(
        "--split",
        choices=["holdout", "prospective"],
        default="prospective",
        help="Processed split to compare.",
    )
    compare_models.add_argument(
        "--output-dir",
        default=None,
        help="Directory for analysis outputs; defaults to <output_root>/analysis.",
    )
    compare_models.add_argument(
        "--lower-threshold",
        type=float,
        default=None,
        help="Override the locked CamKIT-AI lower threshold.",
    )
    compare_models.add_argument(
        "--upper-threshold",
        type=float,
        default=None,
        help="Override the locked CamKIT-AI upper threshold.",
    )
    _add_pooling_arguments(compare_models)

    capacity = subparsers.add_parser(
        "capacity-match",
        help=(
            "Compare CamKIT-AI with the original CamKIT at CamKIT's observed "
            "referral capacity, from saved patient predictions."
        ),
    )
    capacity.add_argument("--config", required=True, help="Path to the YAML config file.")
    capacity.add_argument(
        "--split", default="prospective", help="Processed split to compare."
    )
    capacity.add_argument(
        "--predictions",
        default=None,
        help=(
            "Optional path to a patient-prediction CSV; defaults to "
            "<output_root>/analysis/camkit_ai_vs_camkit_patient_predictions_<split>.csv."
        ),
    )
    capacity.add_argument(
        "--capacity",
        type=int,
        default=None,
        help="Referral capacity; defaults to CamKIT's own observed referral count.",
    )
    capacity.add_argument(
        "--output-dir",
        default=None,
        help="Directory for analysis outputs; defaults to <output_root>/analysis.",
    )

    shift = subparsers.add_parser(
        "shift-analysis",
        help=(
            "Run the exploratory prevalence-standardisation analysis retained for "
            "audit purposes; it is not part of the primary analysis."
        ),
    )
    shift.add_argument("--config", required=True, help="Path to the YAML config file.")
    shift.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model labels to decompose; defaults to Injury.top12 and Injury.full.",
    )
    shift.add_argument(
        "--predictions",
        default=None,
        help=(
            "Optional path to a prediction summary CSV; "
            "defaults to <output_root>/manuscript/prediction_summary.csv."
        ),
    )
    shift.add_argument(
        "--output-dir",
        default=None,
        help="Directory for analysis outputs; defaults to <output_root>/analysis.",
    )
    shift.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=None,
        help="Override the bootstrap iteration count for this run.",
    )

    variability = subparsers.add_parser(
        "imputation-variability",
        help=(
            "Score the locked model repeatedly to measure how far its outputs and "
            "triage bands move between stochastic imputation draws."
        ),
    )
    variability.add_argument("--config", required=True, help="Path to the YAML config file.")
    variability.add_argument("--target", default="Injury", help="Outcome target.")
    variability.add_argument("--variant", choices=["full", "top12"], default="top12")
    variability.add_argument(
        "--draws",
        type=int,
        default=DEFAULT_DRAWS,
        help=f"Number of imputation draws per split (default {DEFAULT_DRAWS}).",
    )
    variability.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed; draw i uses seed + i. Defaults to the config random_state.",
    )
    variability.add_argument(
        "--lower-threshold",
        type=float,
        default=None,
        help="Override the locked CamKIT-AI lower threshold.",
    )
    variability.add_argument(
        "--upper-threshold",
        type=float,
        default=None,
        help="Override the locked CamKIT-AI upper threshold.",
    )
    variability.add_argument(
        "--output-dir",
        default=None,
        help="Directory for analysis outputs; defaults to <output_root>/analysis.",
    )
    variability.add_argument(
        "--save-draw-matrix",
        action="store_true",
        help=(
            "Also write every draw for every patient, so pooled estimates can "
            "be recomputed at any number of draws without re-scoring the model."
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    if hasattr(args, "bootstrap_iterations") and args.bootstrap_iterations is not None:
        config.confidence_intervals.bootstrap_iterations = args.bootstrap_iterations

    if args.command == "prepare-data":
        manifest = prepare_datasets(config)
        print(f"Prepared {len(manifest)} datasets under {config.paths.processed_root}")
        return

    if args.command == "evaluate-model":
        pooled_draws, pool_seed = _resolve_pooling(args)
        result = evaluate_model(
            config,
            args.target,
            args.variant,
            include_subgroups=args.subgroups,
            include_thresholds=args.thresholds,
            pooled_draws=pooled_draws,
            pool_seed=pool_seed,
        )
        output_dir = config.paths.output_root / "models" / f"{args.target}.{args.variant}"
        save_model_evaluation(result, output_dir)
        print(f"Wrote evaluation outputs to {output_dir}")
        return

    if args.command == "report-manuscript":
        pooled_draws, pool_seed = _resolve_pooling(args)
        outputs = report_manuscript(
            config,
            pooled_draws=pooled_draws,
            pool_seed=pool_seed,
        )
        for name, path in outputs.items():
            if path.exists():
                print(f"{name}: {path}")
        return

    if args.command == "train-study":
        result = train_study(config, args.target, args.variant)
        print(result)
        return

    if args.command == "generate-oof":
        output_path = oof_predictions_path(config, args.target, args.variant)
        frame = generate_oof_predictions(
            config,
            args.target,
            args.variant,
            output_path=output_path,
        )
        print(f"oof_predictions: {output_path}")
        print(f"rows: {len(frame)}")
        print(f"mean_predictions_per_row: {frame['n_oof_predictions'].mean():.1f}")
        return

    if args.command == "compare-camkit":
        result, paths = run_camkit_score_analysis(
            config,
            split=args.split,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
        print(f"published_counts_match: {result.published_counts_match}")
        if args.require_published_counts and not result.published_counts_match:
            raise SystemExit(
                "CamKIT score counts do not match the published prospective CamKIT table."
            )
        return

    if args.command == "compare-models":
        pooled_draws, pool_seed = _resolve_pooling(args)
        result, paths = run_model_comparison_analysis(
            config,
            split=args.split,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            lower_threshold=args.lower_threshold,
            upper_threshold=args.upper_threshold,
            pooled_draws=pooled_draws,
            pool_seed=pool_seed,
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
        print(f"prediction_basis: {result.provenance.basis}")
        print(f"prediction_source: {result.provenance.source}")
        return

    if args.command == "capacity-match":
        result, paths = run_capacity_match(
            config,
            split=args.split,
            predictions_path=Path(args.predictions) if args.predictions else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            capacity=args.capacity,
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
        matched = result.policies.set_index("policy_id")
        for policy_id in ("camkit_ai_top_41", "original_camkit_high"):
            if policy_id in matched.index:
                row = matched.loc[policy_id]
                print(
                    f"{policy_id}: {int(row['referrals'])} referrals, "
                    f"{int(row['injuries_captured'])} injuries captured"
                )
        return

    if args.command == "shift-analysis":
        _, paths = run_shift_analysis(
            config,
            model_labels=args.models,
            predictions_path=Path(args.predictions) if args.predictions else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
        return

    if args.command == "imputation-variability":
        frames, paths = run_imputation_variability(
            config,
            args.target,
            args.variant,
            n_draws=args.draws,
            base_seed=args.seed,
            lower_threshold=args.lower_threshold,
            upper_threshold=args.upper_threshold,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            save_draw_matrix=args.save_draw_matrix,
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
        for _, row in frames["instability"].iterrows():
            print(
                f"{row['dataset']}: {int(row['n_band_unstable'])} of "
                f"{int(row['n_patients'])} patients change triage band across draws"
            )
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
