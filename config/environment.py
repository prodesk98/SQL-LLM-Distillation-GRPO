from pathlib import Path
from typing import Any
from os import environ
from dotenv import load_dotenv
import yaml

load_dotenv()


def load_config(file_path: str) -> dict[str, Any]:
    """
    Load the configuration file.
    :param file_path: Path to the configuration file.
    :return: Configuration dictionary.
    """
    with open(file_path, 'r') as file:
        c = yaml.safe_load(file)
        file.close()
    return c

config: dict[str, Any] = load_config("%s/config.yaml" % Path(__file__).parent.parent)
config.update(environ.items())

BASE_URL: str = config.get("base_url", "https://api.openai.com/v1")
API_KEY: str | None = config.get("API_KEY")
HF_TOKEN: str | None = config.get("HF_TOKEN")
MAX_SEQ_LENGTH: int = int(config.get("max_seq_length", 8192))
GPU_MEMORY_UTILIZATION: float = float(config.get("gpu_memory_utilization", .9))
LORA_RANK: int = int(config.get("lora_rank", 16))
LORA_ALPHA: int = int(config.get("lora_alpha", 16))
LORA_DROPOUT: int = int(config.get("lora_dropout", 0))
MAX_PROMPT_LENGTH: int = int(config.get("max_prompt_length", 2048))
TEMPERATURE: float = float(config.get("temperature", .6))
TENSOR_PARALLEL_SIZE: int = int(config.get("tensor_parallel_size", 1))
CLIENT_REPLICAS: int = int(config.get("client_replicas", 1))
LEARNING_RATE: float = float(config.get("learning_rate", 5e-6))
ADAM_BETA1: float = float(config.get("adam_beta1", 0.9))
ADAM_BETA2: float = float(config.get("adam_beta2", 0.99))
WEIGHT_DECAY: float = float(config.get("weight_decay", 0.1))
WARMUP_RATIO: float = float(config.get("warmup_ratio", 0.1))
LR_SCHEDULER_TYPE: str = config.get("lr_scheduler_type", "cosine")
OPTIM: str = config.get("optim", "paged_adamw_8bit")
LOGGING_STEPS: int = int(config.get("logging_steps", 1))
PER_DEVICE_TRAIN_BATCH_SIZE: int = int(config.get("per_device_train_batch_size", 1))
NUM_GENERATIONS: int = int(config.get("num_generations", 4))
GRADIENT_ACCUMULATION_STEPS: int = int(config.get("gradient_accumulation_steps", 1))
MAX_COMPLETION_LENGTH: int = MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH if MAX_SEQ_LENGTH > MAX_PROMPT_LENGTH else MAX_PROMPT_LENGTH
MAX_STEPS: int = int(config.get("max_steps", 100))
SAVE_STEPS: int = int(config.get("save_steps", 100))
MAX_GRAD_NORM: float = float(config.get("max_grad_norm", 0.1))
REPORT_TO: str = config.get("report_to", "none")
OUTPUT_DIR: str = config.get("output_dir", "outputs")
NUM_TRAIN_SFT_EPOCHS: int = int(config.get("num_train_sft_epochs", 1))
NUM_TRAIN_GRPO_EPOCHS: int = int(config.get("num_train_grpo_epochs", -1))