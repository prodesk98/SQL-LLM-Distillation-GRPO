import re

from prompt_engineering.template import (
    SOLUTION_START, SOLUTION_END,
    REASONING_START, REASONING_END
)


def extract_sql(text: str) -> str | None:
    """
    Extracts the SQL using regex from the generated text.
    :param text:
    :return:
    """
    sql_match = re.search(rf"{SOLUTION_START}(.*?){SOLUTION_END}", text, re.DOTALL)
    if sql_match:
        return sql_match.group(1).strip()


def extract_think(text: str) -> str | None:
    """
    Extracts the think using regex from the generated text.
    :param text:
    :return:
    """
    think_match = re.search(rf"{REASONING_START}(.*?){REASONING_END}", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()


def extract_context(text: str) -> str | None:
    """
    Extracts the 'Context' block from the system prompt using regex.
    :param text: The full system prompt text.
    :return: The context string or None if not found.
    """
    match = re.search(r"Context:\s*(.+)", text, re.DOTALL)
    if match:
        return match.group(1).strip()


def get_system_prompt(prompts: list[dict[str, str]]) -> str | None:
    """
    Extracts the system prompt from the list of prompts.
    :param prompts:
    :return:
    """
    _prompts = [
        prompt["content"]
        for prompt in prompts
        if prompt["role"] == "system"
    ]
    if len(_prompts) == 0: return None
    return _prompts[0] if len(_prompts) > 0 else None
