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
            "OpenAI", "vLLM", "Groq"
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

    def _validate_sql(self, sample: dict) -> dict:
        """
        Validate the SQL query in the sample.
        :param sample:
        :return:
        """
        sql_query = sample[self.answer_column]
        context = sample["sql_context"]
        is_valid, message = validate_sql_query(sql_query, context)
        sample["validation"] = "valid" if is_valid else message
        return sample

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
        if self.validate:
            self._dataset = self._dataset.map(
                self._validate_sql,
                batched=False,
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
            if self.publish_repo_id is None:
                raise ValueError("publish_repo_id must be provided when publish is True")
            if HF_TOKEN is None:
                raise ValueError("HF_TOKEN must be provided to publish to Hugging Face Hub")

            try:
                self._distiset.push_to_hub(
                    repo_id=self.publish_repo_id,
                    commit_message="Pushing distilled dataset to Hugging Face Hub",
                    private=self.private_repo,
                    generate_card=False,
                    token=HF_TOKEN,
                )
                logger.info(
                    f"Distilled dataset published to Hugging Face Hub at {self.publish_repo_id}"
                )
            except Exception as e:
                logger.error(f"Failed to publish distilled dataset: {e}")

    def publish(self, repo_id: str) -> None:
        if self._distiset is None:
            raise ValueError("Distillation must be run before publishing.")
        if HF_TOKEN is None:
            raise ValueError("HF_TOKEN must be provided to publish to Hugging Face Hub")

        try:
            self._distiset.push_to_hub(
                repo_id=repo_id,
                commit_message="Pushing distilled dataset to Hugging Face Hub",
                private=self.private_repo,
                generate_card=False,
                token=HF_TOKEN,
            )
            logger.info(f"Distilled dataset published to Hugging Face Hub at {repo_id}")
        except Exception as e:
            logger.error(f"Failed to publish distilled dataset: {e}")
