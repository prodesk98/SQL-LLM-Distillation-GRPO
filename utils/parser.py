import re


def extract_sql(text: str) -> str | None:
    """
    Extracts the SQL using regex from the generated text.
    :param text:
    :return:
    """
    sql_match = re.search(r"<sql>(.*?)</sql>", text, re.DOTALL)
    if sql_match:
        return sql_match.group(1).strip()


def extract_think(text: str) -> str | None:
    """
    Extracts the think using regex from the generated text.
    :param text:
    :return:
    """
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
