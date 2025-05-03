from typing import Optional

from distilabel.models import OpenAILLM
from distilabel.pipeline import RayPipeline, Pipeline
from distilabel.steps import StepResources
from distilabel.steps.tasks import TextGeneration

from config import BASE_URL, MODEL_NAME, API_KEY, TEMPLATE


def build_distilabel_pipeline(
    model: str = MODEL_NAME,
    base_url: str = BASE_URL,
    api_key: str = API_KEY,
    temperature: float = 0.2,
    top_p: Optional[float] = None,
    max_new_tokens: int = 1024,
    columns: Optional[list[str]] = None,
    template: str = TEMPLATE,
    num_generations: int = 1,
    input_batch_size: int = 64,
    client_replicas: int = 1,
    timeout: int = 900,
    retries: int = 3,
) -> RayPipeline:
    if columns is None or len(columns) != 2:
        raise ValueError("columns must be a list of two strings: [instruction, context]")
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
            input_mappings={"instruction": columns[0], "context": columns[1]},
            input_batch_size=input_batch_size,
            num_generations=num_generations,
            group_generations=True,
            resources=StepResources(replicas=client_replicas),
        )
    return pipeline

