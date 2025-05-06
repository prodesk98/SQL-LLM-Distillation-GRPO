import argparse
from control import DistillControl

from logging_ import setup_logger
setup_logger()
from loguru import logger


parser = argparse.ArgumentParser(
    prog="SQL-RL-Distillation-LLM",
    description="A framework for SQL RL Distillation with LLMs.",
    epilog="Developed by Protons.",
)

subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands like 'train', 'distill'")

# Train subparser
parser_train = subparsers.add_parser(
    "train",
    help="Train a model using the specified parameters.",
)
parser_train.add_argument(
    "--model",
    type=str,
    default="unsloth/gemma-3-1b-it",
    help="Model name for training. Default is 'unsloth/gemma-3-1b-it'.",
)
parser_train.add_argument(
    "--data-repo-id",
    type=str,
    required=True,
    help="Path to the training data file.",
)
#

# Distill subparser
parser_distill = subparsers.add_parser(
    "distill",
    help="Distill a model using the specified parameters.",
)
parser_distill.add_argument(
    "--model",
    type=str,
    default="gpt-4.1-nano",
    help="Model name for distillation. Default is 'gpt-4.1-nano'.",
)
parser_distill.add_argument(
    "--publish",
    action="store_true",
    help="Publish the distilled model to Hugging Face Hub. Default is False.",
)
parser_distill.add_argument(
    "--publish-repo-id",
    type=str,
    help="Path to the Hugging Face Hub repository for publishing.",
)
parser_distill.add_argument(
    "--private-repo",
    action="store_true",
    help="Make the published repository private. Default is True.",
)
parser_distill.add_argument(
    "--dataset-repo-id",
    type=str,
    default="gretelai/synthetic_text_to_sql",
    help="Path to the distillation dataset.",
)
parser_distill.add_argument(
    "--limit",
    type=int,
    default=100,
    help="Limit the number of samples to distill. Default is 100.",
)
#

args = parser.parse_args()

if args.command == "train":
    # Call the training function with the provided arguments
    print(f"Training model: {args.model}")
    # Add your training logic here
elif args.command == "distill":
    if args.publish and args.publish_repo_id is None:
        raise ValueError("publish_repo_id must be provided when publish is True")

    logger.info(
        f"Distilling model: {args.model} with dataset: {args.dataset_repo_id} and limit: {args.limit} samples."
    )

    distill = DistillControl(
        model=args.model,
        dataset_repo_id=args.dataset_repo_id,
        publish_repo_id=args.publish_repo_id,
        publish=args.publish,
        limit=args.limit,
    )
    distill.run()
else:
    parser.print_help()
    raise ValueError("Invalid command. Use 'train' or 'distill'.")
