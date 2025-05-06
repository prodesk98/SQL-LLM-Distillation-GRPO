from typing import Optional

from distilabel.models import OpenAILLM
from distilabel.pipeline import RayPipeline, Pipeline
from distilabel.steps import StepResources
from distilabel.steps.tasks import TextGeneration

from config import BASE_URL, API_KEY, MAX_PROMPT_LENGTH, TEMPERATURE
from prompt_engineering import DISTILLATION_SYSTEM_PROMPT_TEMPLATE


def build_distilabel_pipeline(
    model: str,
    base_url: str = BASE_URL,
    api_key: Optional[str] = API_KEY,
    temperature: float = TEMPERATURE,
    top_p: Optional[float] = None,
    max_new_tokens: int = MAX_PROMPT_LENGTH,
    columns: Optional[list[str]] = None,
    mappings: Optional[dict[str, str]] = None,
    template: str = DISTILLATION_SYSTEM_PROMPT_TEMPLATE,
    num_generations: int = 1,
    input_batch_size: int = 64,
    client_replicas: int = 1,
    timeout: int = 900,
    retries: int = 3,
) -> RayPipeline:
    generation_kwargs = {"max_new_tokens": max_new_tokens, "temperature": temperature}
    if top_p is not None:
        generation_kwargs["top_p"] = top_p

    with Pipeline().ray() as pipeline:
        TextGeneration(
            llm=OpenAILLM(
                base_url=base_url,
                api_key=api_key,
                model=model,
                timeout=timeout,
                max_retries=retries,
                generation_kwargs=generation_kwargs,
            ),
            template=template,
            columns=columns,
            input_mappings=mappings,
            input_batch_size=input_batch_size,
            num_generations=num_generations,
            group_generations=True,
            resources=StepResources(replicas=client_replicas),
        )
    return pipeline

