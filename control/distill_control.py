import re
from utils import extract_think, extract_sql
from typing import Optional, Literal
from config import HF_TOKEN

from distilabel.distiset import Distiset
from distilabel.pipeline import Pipeline

from distill import build_distilabel_pipeline
from datasets import load_dataset, DatasetDict, Dataset
from logging_ import logger
from utils import validate_sql_query


class DistillControl:
    def __init__(
        self,
        model: str,
        dataset_repo_id: str,
        publish_repo_id: Optional[str] = None,
        mappings: Optional[dict[str, str]] = None,
        answer_column: str = "sql",
        split: str = "train",
        limit: int = 5_000,
        publish: bool = False,
        private_repo: bool = True,
        batch_size: int = 8,
        retries: int = 3,
        provider: Literal[
            "OpenAI", "vLLM", "Groq", "HuggingFace",
        ] = "OpenAI",
        validate: bool = False,
        use_ray: bool = False,
    ):
        self.model = model
        self.dataset_repo_id = dataset_repo_id
        self.publish_repo_id = publish_repo_id
        self.provider = provider
        self.validate = validate
        self.publish = publish
        self.private_repo = private_repo
        self.answer_column = answer_column
        self.mappings = mappings
        self._distilabel_pipeline: Optional[Pipeline] = None
        self._dataset: Optional[DatasetDict | Dataset] = None
        self._distiset: Optional[Distiset] = None
        self._initialize(batch_size, retries, use_ray)  # Initialize the distillation pipeline
        self._load_dataset(split, limit)                # Load the dataset

    def _load_dataset(self, split: str = "train", limit: int = 5_000) -> DatasetDict | Dataset:
        if self.dataset_repo_id is None:
            raise ValueError("dataset_repo_id must be provided")
        self._dataset = load_dataset(self.dataset_repo_id, split=f"{split}[:{limit}]", token=HF_TOKEN)
        values_ = self.mappings.values()
        self._dataset = self._dataset.remove_columns(
            [
                col for col in self._dataset.column_names
                if col not in values_ and col != self.answer_column
            ]
        )
        return self._dataset

    def _initialize(self, batch_size: int, retries: int, use_ray: bool = False) -> None:
        if self.mappings is None:
            self.mappings = {
                "instruction": "sql_prompt",
                "context": "sql_context",
                "objective": "sql",
                "explanation": "sql_explanation",
            }
        self._distilabel_pipeline = build_distilabel_pipeline(
            self.model,
            columns=[k for k in self.mappings.keys()],
            mappings=self.mappings,
            provider=self.provider,
            input_batch_size=batch_size,
            retries=retries,
            use_ray=use_ray,
        )

    def run(self) -> None:
        try:
            self._distiset = self._distilabel_pipeline.run(dataset=self._dataset, use_cache=False)
            logger.info("Distillation completed successfully.")
        except Exception as e:
            logger.error(f"Error during distillation: {e}")

        if self.publish:
            self.push_to_hub()

    def _process(self, sample: dict) -> dict | None:
        """
        Process the sample to extract SQL and think components.
        :param sample:
        :return: dict
        """
        try:
            generation: str = next(iter(sample['generation']))
            generation = generation.replace("</think>\n\n", "</think>\n").strip()
            __think = extract_think(generation)
            __sql = " ".join(extract_sql(generation).replace("\n", " ").split())
            generation = re.sub(
                r"^<think>\n(.*?)\n</think>\n<sql>\n(.*?)\n</sql>",
                rf"<think>\n{__think.strip()}\n</think>\n<sql>\n{__sql.strip()}\n</sql>",
                generation,
                flags=re.DOTALL | re.MULTILINE,
            )
            generation = re.sub(r'(?s)(?<=</sql>)(.*?)(<sql>.*?</sql>)+', r'\1', generation, count=1)
            result = {"generation": generation}
            if self.validate:
                is_valid, _exception = validate_sql_query(__sql, sample['sql_context'])
                result.update({"validation": "valid" if is_valid else _exception})
            return result
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            return None

    def push_to_hub(self) -> None:
        if self._distiset is None:
            raise ValueError("Distillation must be run before publishing.")
        if self.publish_repo_id is None:
            raise ValueError("publish_repo_id must be provided to publish to Hugging Face Hub")
        if HF_TOKEN is None:
            raise ValueError("HF_TOKEN must be provided to publish to Hugging Face Hub")

        try:
            dataset: Dataset = self._distiset['default']['train']
            dataset = dataset.map(lambda x: self._process(x), remove_columns=['distilabel_metadata', 'model_name'])
            dataset.push_to_hub(
                repo_id=self.publish_repo_id,
                commit_message="Pushing distilled dataset to Hugging Face Hub",
                private=self.private_repo,
                token=HF_TOKEN,
            )
            logger.info(f"Distilled dataset published to Hugging Face Hub at {self.publish_repo_id}")
        except Exception as e:
            logger.error(f"Failed to publish distilled dataset: {e}")
