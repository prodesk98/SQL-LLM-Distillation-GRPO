import re

from utils.constraints import (
    SOLUTION_START, SOLUTION_END,
    REASONING_START, REASONING_END
)


sql_match_format = re.compile(rf"{SOLUTION_START}(.*?){SOLUTION_END}", re.DOTALL)
reasoning_match_format = re.compile(rf"{REASONING_START}(.*?){REASONING_END}", re.DOTALL)


def extract_sql(text: str) -> str | None:
    """
    Extracts the SQL using regex from the generated text.
    :param text:
    :return:
    """
    sql_match = sql_match_format.search(text)
    if sql_match:
        return sql_match.group(1).strip()


def extract_think(text: str) -> str | None:
    """
    Extracts the think using regex from the generated text.
    :param text:
    :return:
    """
    think_match = reasoning_match_format.search(text)
    if think_match:
        return think_match.group(1).strip()

