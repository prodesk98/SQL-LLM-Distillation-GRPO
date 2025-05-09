from typing import Optional, Literal

from distilabel.models import AsyncLLM, LLM
from distilabel.pipeline import Pipeline, RayPipeline
from distilabel.steps import StepResources
from distilabel.steps.tasks import TextGeneration

from config import (
    BASE_URL, API_KEY, MAX_PROMPT_LENGTH,
    TEMPERATURE, TENSOR_PARALLEL_SIZE, CLIENT_REPLICAS
)
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
    input_batch_size: int = 8,
    client_replicas: int = CLIENT_REPLICAS,
    timeout: int = 128,
    retries: int = 3,
    provider: Literal["OpenAI", "vLLM", "Groq", "HuggingFace"] = "OpenAI",
    use_ray: bool = False,
) -> Pipeline | RayPipeline:
    generation_kwargs = {"max_new_tokens": max_new_tokens, "temperature": temperature}
    if top_p is not None:
        generation_kwargs["top_p"] = top_p

    __llm: Optional[AsyncLLM | LLM] = None
    match provider:
        case "OpenAI":
            from distilabel.models import OpenAILLM

            __llm = OpenAILLM(
                base_url=base_url,
                api_key=api_key,
                model=model,
                timeout=timeout,
                max_retries=retries,
                generation_kwargs=generation_kwargs,
            )
        case "vLLM":
            from distilabel.models import vLLM

            __llm = vLLM(
                model=model,
                tokenizer=model,
                extra_kwargs={
                    "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
                    "max_model_len": max_new_tokens,
                },
                generation_kwargs=generation_kwargs,
            )
        case "HuggingFace":
            from distilabel.models.llms.huggingface import InferenceEndpointsLLM

            __llm = InferenceEndpointsLLM(
                base_url=base_url,
                api_key=api_key,
                generation_kwargs=generation_kwargs,
            )
        case "Groq":
            from distilabel.models import GroqLLM

            __llm = GroqLLM(
                model=model,
                api_key=api_key,
                max_retries=retries,
                timeout=timeout,
                generation_kwargs=generation_kwargs,
            )
        case _:
            raise ValueError(f"Unsupported provider: {provider}")

    with Pipeline() as pipeline:
        TextGeneration(
            llm=__llm,
            template=template,
            columns=columns,
            input_mappings=mappings,
            input_batch_size=input_batch_size,
            num_generations=num_generations,
            group_generations=True,
            resources=StepResources(replicas=client_replicas),
        )

    if use_ray:
        return pipeline.ray()
    return pipeline
