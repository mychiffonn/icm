"""Main entry point to run ICM evaluations on TruthfulQA dataset.

Example usage:
    uv run src/main.py --few-shot-icm --zero-shot-base
"""

import argparse
import logging
import sys
from pathlib import Path

from src import BASE_MODEL, CHAT_MODEL
from src.eval import (
    evaluate_few_shot_golden,
    evaluate_few_shot_icm,
    evaluate_zero_shot_base,
    evaluate_zero_shot_chat,
)
from src.icm_search import ICMArgs
from src.utils import load_truthqa_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run ICM evaluations on TruthfulQA dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Evaluation selection
    eval_group = parser.add_argument_group("Evaluation Selection")
    eval_group.add_argument(
        "--all",
        action="store_true",
        help="Run all evaluations (default if no specific eval selected)",
    )
    eval_group.add_argument(
        "--zero-shot-base",
        action="store_true",
        help="Run zero-shot evaluation with base model",
    )
    eval_group.add_argument(
        "--zero-shot-chat",
        action="store_true",
        help="Run zero-shot evaluation with chat model",
    )
    eval_group.add_argument(
        "--few-shot-golden",
        action="store_true",
        help="Run few-shot evaluation with golden (ground truth) labels",
    )
    eval_group.add_argument(
        "--few-shot-icm",
        action="store_true",
        help="Run few-shot evaluation with ICM-predicted labels",
    )

    # Model settings
    model_group = parser.add_argument_group("Model Settings")
    model_group.add_argument(
        "--base-model",
        type=str,
        default=BASE_MODEL,
        help=f"Base model to use (default: {BASE_MODEL})",
    )
    model_group.add_argument(
        "--chat-model",
        type=str,
        default=CHAT_MODEL,
        help=f"Chat model to use (default: {CHAT_MODEL})",
    )

    # Few-shot settings
    few_shot_group = parser.add_argument_group("Few-Shot Settings")
    few_shot_group.add_argument(
        "--num-shots",
        type=int,
        default=30,
        help="Number of demonstration examples for few-shot golden (default: 30)",
    )

    # ICM settings
    icm_group = parser.add_argument_group("ICM Settings")
    icm_group.add_argument(
        "--icm-sample-size",
        type=int,
        default=8,
        help="Number of examples to sample for initial ICM dataset (default: 8)",
    )
    icm_group.add_argument(
        "--icm-max-iterations",
        type=int,
        default=100,
        help="Maximum number of ICM search iterations (default: 100)",
    )
    icm_group.add_argument(
        "--icm-initial-temp",
        type=float,
        default=10.0,
        help="Initial temperature for ICM simulated annealing (default: 10.0)",
    )
    icm_group.add_argument(
        "--icm-final-temp",
        type=float,
        default=0.01,
        help="Final temperature for ICM simulated annealing (default: 0.01)",
    )
    icm_group.add_argument(
        "--icm-cooling-rate",
        type=float,
        default=0.99,
        help="Cooling rate for ICM temperature schedule (default: 0.99)",
    )

    # Other settings
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Determine which evaluations to run
    run_all = args.all or not any(
        [
            args.zero_shot_base,
            args.zero_shot_chat,
            args.few_shot_golden,
            args.few_shot_icm,
        ]
    )

    evaluations = {
        "zero_shot_base": run_all or args.zero_shot_base,
        "zero_shot_chat": run_all or args.zero_shot_chat,
        "few_shot_golden": run_all or args.few_shot_golden,
        "few_shot_icm": run_all or args.few_shot_icm,
    }

    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Chat model: {args.chat_model}")
    logger.info(
        f"Evaluations to run: {', '.join([k for k, v in evaluations.items() if v])}"
    )

    # Load data
    logger.info("\nLoading TruthfulQA data...")
    train_data = load_truthqa_data(split="train")
    test_data = load_truthqa_data(split="test")
    logger.info(f"Loaded {len(train_data)} training examples")
    logger.info(f"Loaded {len(test_data)} test examples")

    show_progress = not args.no_progress
    results = {}

    # Run evaluations
    if evaluations["zero_shot_base"]:
        results["zero_shot_base"] = evaluate_zero_shot_base(
            test_data=test_data,
            model=args.base_model,
            show_progress=show_progress,
        )

    if evaluations["zero_shot_chat"]:
        results["zero_shot_chat"] = evaluate_zero_shot_chat(
            test_data=test_data,
            model=args.chat_model,
            show_progress=show_progress,
        )

    if evaluations["few_shot_golden"]:
        results["few_shot_golden"] = evaluate_few_shot_golden(
            test_data=test_data,
            train_data=train_data,
            num_shots=args.num_shots,
            model=args.base_model,
            show_progress=show_progress,
        )

    if evaluations["few_shot_icm"]:
        icm_args = ICMArgs(
            sample_size=args.icm_sample_size,
            max_iterations=args.icm_max_iterations,
            initial_temperature=args.icm_initial_temp,
            final_temperature=args.icm_final_temp,
            cooling_rate=args.icm_cooling_rate,
        )
        results["few_shot_icm"] = evaluate_few_shot_icm(
            test_data=test_data,
            train_data=train_data,
            icm_args=icm_args,
            model=args.base_model,
            show_progress=show_progress,
        )

    logger.info("EVALUATION SUMMARY")
    for eval_name, result in results.items():
        if result:
            acc = result["accuracy"]
            correct = result["correct"]
            total = result["total"]
            logger.info(f"{eval_name:20s}: {correct}/{total} = {acc:.2%}")


if __name__ == "__main__":
    main()
