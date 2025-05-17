from utils import validate_sql_query, extract_context, get_system_prompt
from utils.constraints import COLORED_GREEN, COLORED_RESET, COLORED_BLUE, BOLD
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
    q = prompts[0][-1]['content']
    system_prompt = get_system_prompt(prompts[0])
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses: list[str] = [
        guess.group(1)
        if (guess := reasoning_match_format.search(r)) is not None else None \
        for r in responses
    ]

    extracted_contexts: list[str] = [extract_context(get_system_prompt(p)) for p in prompts]

    print(
        '-'*20,
        f"{BOLD}Question:{COLORED_RESET} {q}{COLORED_RESET}",
        f"\n{BOLD}System Prompt:{COLORED_RESET} {COLORED_BLUE}{system_prompt}{COLORED_RESET}",
        f"\n{BOLD}Answer:{COLORED_RESET} \n{COLORED_GREEN}{answer[0]}{COLORED_RESET}",
        f"\n{BOLD}Response:{COLORED_RESET} \n{COLORED_GREEN}{responses[0]}{COLORED_RESET}",
        f"\n{BOLD}Extracted:{COLORED_RESET} \n{COLORED_GREEN}{extracted_responses[0] if extracted_responses[0] is not None else '-Invalid Format-'}{COLORED_RESET}",
    )

    scores: list[float] = []
    for guess, context, true_answer in zip(extracted_responses, extracted_contexts, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue

        # Check if the SQL query is valid
        is_valid, _ = validate_sql_query(guess, context)

        # Check if the SQL query is similar to the true SQL
        # Correct answer gets 3 points
        if is_valid:
            # Exact match
            if guess == true_answer:
                score += 3.0
            # Match if spaces are seen
            elif guess.strip() == true_answer.strip():
                score += 1.5
        else:
            score -= 1.5 # Penalty for incorrect SQL
        scores.append(score)
    return scores
