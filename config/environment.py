from pathlib import Path
from typing import Any

import yaml


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

config = load_config("%s/config.yaml" % Path(__file__).parent.parent)

BASE_URL = config.get("base_url")
if BASE_URL is None:
    raise ValueError("base_url is not set in the configuration file.")
MODEL_NAME = config.get("model_name_or_path")
if MODEL_NAME is None:
    raise ValueError("model_name_or_path is not set in the configuration file.")
API_KEY = config.get("openai_api_key")
if API_KEY is None:
    raise ValueError("openai_api_key is not set in the configuration file.")
TEMPLATE = config.get("template")