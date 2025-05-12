from utils import validate_sql_query, extract_context
from .format_reward import reasoning_match_format


def check_sql_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Check the SQL reward for the given prompts and completions.
    :param prompts:
    :param completions:
    :param answer:
    :param kwargs:
    :return:
    """
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := reasoning_match_format.search(r)) is not None else None \
        for r in responses
    ]

    extracted_contexts = [
        extract_context(p).strip()
        for p in prompts
    ]

    scores: list[float] = []
    for guess, context, true_answer in zip(extracted_responses, extracted_contexts, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue

        # Check if the SQL query is valid
        is_valid, error_message = validate_sql_query(guess, context)

        # Check if the SQL query is similar to the true SQL
        # Correct answer gets 3 points
        if guess == true_answer and is_valid:
            score += 3.0
        # Match if spaces are seen
        elif guess.strip() == true_answer.strip() and is_valid:
            score += 1.5
        else:
            score -= 1.5 # Penalty for incorrect SQL
        scores.append(score)
    return scores
