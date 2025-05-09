import argparse

from logging_ import setup_logger
setup_logger()
from loguru import logger


parser = argparse.ArgumentParser(
    prog="uv run main.py",
    description=(
        "A modular framework for SQL data distillation with reasoning using Large Language Models (LLMs). "
        "Supports multiple backends (OpenAI, vLLM, Groq, Hugging Face) and integrates Chain of Thought prompting "
        "with online reinforcement learning (GRPO) for high-quality SQL generation."
    ),
    epilog="Developed by Protons · GitHub: https://github.com/prodesk98",
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
    default=None,
    help="Model name for distillation. Default is 'None'.",
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
parser_distill.add_argument(
    "--provider",
    type=str,
    default="OpenAI",
    choices=["OpenAI", "vLLM", "Groq", "HuggingFace"],
    help="Provider for the distillation process. Default is 'OpenAI'.",
)
parser_distill.add_argument(
    "--validate",
    action="store_true",
    help="Validate the SQL query. Default is False.",
)
parser_distill.add_argument(
    "--remove-no-valid",
    action="store_true",
    help="Remove invalid SQL queries from the dataset. Default is False.",
)
parser_distill.add_argument(
    "--batch-size",
    type=int,
    default=8,
    help="Batch size for the distillation process. Default is 8.",
)
parser_distill.add_argument(
    "--retries",
    type=int,
    default=3,
    help="Number of retries for the distillation process. Default is 3.",
)
parser_distill.add_argument(
    "--use-ray",
    action="store_true",
    help="Use Ray for distributed processing. Default is False.",
)
#

args = parser.parse_args()

if args.command == "train":
    # Call the training function with the provided arguments
    print(f"Training model: {args.model}")
    # Add your training logic here
elif args.command == "distill":
    from control import DistillControl
    if args.publish and args.publish_repo_id is None:
        raise ValueError("publish_repo_id must be provided when publish is True")
    if args.model is None and args.provider != "HuggingFace":
        raise ValueError("model must be provided when provider is not HuggingFace")

    logger.info(
        f"Distilling model: [{args.provider}] {args.model} with dataset: {args.dataset_repo_id} with {args.limit} samples."
    )

    distill = DistillControl(
        model=args.model,
        dataset_repo_id=args.dataset_repo_id,
        publish_repo_id=args.publish_repo_id,
        publish=args.publish,
        limit=args.limit,
        provider=args.provider,
        validate=args.validate,
        private_repo=args.private_repo,
        use_ray=args.use_ray,
    )
    distill.run()
else:
    parser.print_help()
    raise ValueError("Invalid command. Use 'train' or 'distill'.")
