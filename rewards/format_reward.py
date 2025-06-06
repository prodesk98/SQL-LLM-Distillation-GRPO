import re
from utils.constraints import (
    REASONING_START, REASONING_END,
    SOLUTION_START, SOLUTION_END
)


reasoning_match_format = re.compile(
    rf"^{REASONING_START}\n.*?\n{REASONING_END}\n{SOLUTION_START}\n.*?\n{SOLUTION_END}$",
    re.DOTALL | re.MULTILINE
)

def match_format_exactly(completions: list[list[dict[str, str]]], **kwargs) -> list[float]: # noqa
    """
    Check if the completions match the expected format exactly.
    :param completions:
    :param kwargs:
    :return:
    """
    scores: list[float] = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        if reasoning_match_format.search(response) is not None: score += 1.1
        scores.append(score)
    return scores


def match_format_approximately(completions: list[list[dict[str, str]]], **kwargs) -> list[float]: # noqa
    """
    Check if the completions match the expected format approximately.
    :param completions:
    :param kwargs:
    :return:
    """
    scores: list[float] = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        score += 0.125 if response.count(f"{REASONING_START}\n")    == 1 else -1.0
        score += 0.125 if response.count(f"\n{REASONING_END}\n")    == 1 else -1.0
        score += 0.125 if response.count(f"\n{SOLUTION_START}\n")   == 1 else -1.0
        score += 0.125 if response.count(f"\n{SOLUTION_END}")       == 1 else -1.0
        scores.append(score)
    return scores
