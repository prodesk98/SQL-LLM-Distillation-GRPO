from datasets import DatasetDict, Dataset
from prompt_engineering import REASONING_SYSTEM_PROMPT_TEMPLATE
from utils.constraints import (
    REASONING_START, REASONING_END,
    SOLUTION_START, SOLUTION_END
)
from .parser import extract_sql, extract_think


def _reasoning_format(content: str) -> str:
    """
    Formatting prompt for exact match.
    :param content:
    :return:
    """
    __think = extract_think(content)
    __sql = extract_sql(content)
    return f"""{REASONING_START}\n{__think}\n{REASONING_END}\n{SOLUTION_START}\n{__sql}\n{SOLUTION_END}"""


def conversations_formatting(dataset: DatasetDict) -> Dataset | DatasetDict:
    """
    Format the conversations for the model.
    :param dataset:
    :return:
    """
    dataset = dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": REASONING_SYSTEM_PROMPT_TEMPLATE.format(context=x["sql_context"], instruction=x["sql_prompt"]).strip()},
            {"role": "assistant", "content": _reasoning_format(x["generation"])},
        ],
        "questions": x["sql_prompt"],
        "contexts": x["sql_context"],
        "answers": extract_sql(x["generation"]),
    })
    return dataset
