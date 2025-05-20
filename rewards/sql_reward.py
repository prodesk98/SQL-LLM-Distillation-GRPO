from utils import validate_sql_query, extract_sql
from utils.constraints import COLORED_GREEN, COLORED_RESET, BOLD, COLORED_BLUE


def check_sql_reward(
    completions: list[list[dict[str, str]]],
    questions: list[str],
    answers: list[str],
    contexts: list[str],
    **kwargs # noqa
) -> list[float]:
    """
    Check the SQL reward for the given prompts and completions.
    :param completions:
    :param questions:
    :param answers:
    :param contexts:
    :param kwargs:
    :return:
    """
    responses = [completion[0]["content"] for completion in completions]
    pred_responses: list[str | None] = [extract_sql(r) for r in responses]

    print(
        '-'*20,
        f"{BOLD}Question:{COLORED_RESET} {questions[0]}{COLORED_RESET}",
        f"\n{BOLD}Response:{COLORED_RESET} \n{COLORED_GREEN}{responses[0]}{COLORED_RESET}",
        f"\n{BOLD}Real Answer:{COLORED_RESET} \n{COLORED_GREEN}{answers[0]}{COLORED_RESET}",
        f"\n{BOLD}Extracted:{COLORED_RESET} \n"
        f"{COLORED_BLUE}{pred_responses[0] if pred_responses[0] is not None else '-Invalid Format-'}{COLORED_RESET}",
    )

    scores: list[float] = []
    for pred, context, answer in zip(pred_responses, contexts, answers):
        score = 0
        if pred is None:
            scores.append(0) # SQL extraction failed / Not found / Limit Token
            continue

        # Check if the SQL query is valid
        is_valid, _ = validate_sql_query(pred, context)

        # Check if the SQL query is similar to the true SQL
        # Correct answer gets 3 points
        if is_valid:
            score += 1.2
            # Exact match
            if pred == answer: score += 1.2
            # Match if spaces are seen
            elif pred.strip() == answer.strip(): score += .7
        else: score -= 1.5 # Penalty for incorrect SQL
        scores.append(score)
    return scores
