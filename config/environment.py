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
MAX_SEQ_LENGTH: int = config.get("max_seq_length", 1024)
GPU_MEMORY_UTILIZATION: float = config.get("gpu_memory_utilization", .7)
LORA_RANK: int = config.get("lora_rank", 16)
LORA_ALPHA: int = config.get("lora_alpha", 16)
MAX_PROMPT_LENGTH: int = config.get("max_prompt_length", 512)
TEMPERATURE: float = config.get("temperature", .6)
TENSOR_PARALLEL_SIZE: int = config.get("tensor_parallel_size", 1)
CLIENT_REPLICAS: int = config.get("client_replicas", 1)
