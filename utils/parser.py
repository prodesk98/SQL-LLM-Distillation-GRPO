import re

from prompt_engineering.template import (
    SOLUTION_START, SOLUTION_END,
    REASONING_START, REASONING_END
)


def extract_sql(text: str | bytes) -> str | None:
    """
    Extracts the SQL using regex from the generated text.
    :param text:
    :return:
    """
    if not isinstance(text, str):
        text = text.decode()
    sql_match = re.search(rf"{SOLUTION_START}(.*?){SOLUTION_END}", text, re.DOTALL)
    if sql_match:
        return sql_match.group(1).strip()


def extract_think(text: str | bytes) -> str | None:
    """
    Extracts the think using regex from the generated text.
    :param text:
    :return:
    """
    if not isinstance(text, str):
        text = text.decode()
    think_match = re.search(rf"{REASONING_START}(.*?){REASONING_END}", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()


def extract_context(text: str | bytes) -> str | None:
    """
    Extracts the 'Context' block from the system prompt using regex.
    :param text: The full system prompt text.
    :return: The context string or None if not found.
    """
    if not isinstance(text, str):
        text = text.decode()
    match = re.search(r"Context:\s*(.*?)\s*Exceptions:", text, re.DOTALL)
    if match:
        return match.group(1).strip()
