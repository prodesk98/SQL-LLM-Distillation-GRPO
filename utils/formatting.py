from datasets import DatasetDict, Dataset
from prompt_engineering import REASONING_SYSTEM_PROMPT_TEMPLATE
from utils.constraints import (
    REASONING_START, REASONING_END,
    SOLUTION_START, SOLUTION_END
)
from .parser import extract_sql, extract_think, extract_schema_tables


def correct_reasoning_format(content: str) -> str:
    """
    Formatting prompt for exact match.
    :param content:
    :return:
    """
    __think = extract_think(content)
    __sql = extract_sql(content)
    return f"""{REASONING_START}\n{__think}\n{REASONING_END}\n{SOLUTION_START}\n{__sql}\n{SOLUTION_END}"""


def conversations_grpo_format(dataset: DatasetDict) -> Dataset | DatasetDict:
    """
    Format the conversations for the model.
    :param dataset:
    :return:
    """
    dataset = dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": REASONING_SYSTEM_PROMPT_TEMPLATE.format(context=extract_schema_tables(x["sql_context"])).strip()},
            {"role": "user", "content": x["sql_prompt"]},
        ],
        "questions": x["sql_prompt"],
        "contexts": x["sql_context"],
        "answers": extract_sql(x["generation"]),
    }, remove_columns=[
        k for k in dataset.column_names if k not in ["questions", "contexts", "answers", "prompt"]
    ])
    return dataset


def conversations_supervised_fine_tuning_format(dataset: DatasetDict) -> Dataset | DatasetDict:
    """
    Format the conversations for supervised fine-tuning.
    :param dataset:
    :return:
    """
    dataset = dataset.map(lambda x: {
        "messages": [
            {"role": "system", "content": REASONING_SYSTEM_PROMPT_TEMPLATE.format(context=extract_schema_tables(x["sql_context"])).strip()},
            {"role": "user", "content": x["sql_prompt"]},
            {"role": "assistant", "content": correct_reasoning_format(x["generation"])},
        ],
    }, remove_columns=[
        k for k in dataset.column_names if k not in ["messages"]
    ])
    return dataset
