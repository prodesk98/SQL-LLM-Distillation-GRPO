from datasets import DatasetDict, Dataset
from prompt_engineering import REASONING_SYSTEM_PROMPT_TEMPLATE


def instruction_formatting(dataset: DatasetDict) -> Dataset | DatasetDict:
    """
    Format the instruction for the model.
    :param dataset:
    :return:
    """
    dataset = dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": REASONING_SYSTEM_PROMPT_TEMPLATE.format(
                context=x["sql_context"],
                exceptions="No exceptions (**SQL validated**)" if x["validation"] != "valid" else x["validation"],
            )},
            {"role": "user", "content": x["sql_prompt"]},
        ],
        "answer": x["generation"],
    })
    return dataset
