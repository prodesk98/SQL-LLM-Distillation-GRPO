from typing import Optional
from config import HF_TOKEN

from distilabel.distiset import Distiset
from distilabel.pipeline import RayPipeline

from distill import build_distilabel_pipeline
from datasets import load_dataset, DatasetDict, Dataset
from logging_ import logger


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
    ):
        self.model = model
        self.dataset_repo_id = dataset_repo_id
        self.publish_repo_id = publish_repo_id
        self.publish = publish
        self.private_repo = private_repo
        self.answer_column = answer_column
        self.mappings = mappings
        self._distilabel_pipeline: Optional[RayPipeline] = None
        self._dataset: Optional[DatasetDict | Dataset] = None
        self._distiset: Optional[Distiset] = None
        self._initialize()                  # Initialize the distillation pipeline
        self._load_dataset(split, limit)  # Load the dataset

    def _load_dataset(self, split: str = "train", limit: int = 5_000) -> DatasetDict | Dataset:
        if self.dataset_repo_id is None:
            raise ValueError("dataset_repo_id must be provided")
        self._dataset = load_dataset(self.dataset_repo_id, split=split, token=HF_TOKEN).select(range(limit))
        values_ = self.mappings.values()
        self._dataset = self._dataset.remove_columns(
            [
                col for col in self._dataset.column_names
                if col not in values_ and col != self.answer_column
            ]
        )
        return self._dataset

    def _initialize(self) -> None:
        if self.mappings is None:
            self.mappings = {
                "instruction": "sql_prompt",
                "context": "sql_context",
                "explanation": "sql_explanation",
            }
        self._distilabel_pipeline = build_distilabel_pipeline(
            self.model,
            columns=[k for k in self.mappings.keys()],
            mappings=self.mappings,
        )

    def run(self) -> None:
        self._distiset = self._distilabel_pipeline.run(dataset=self._dataset, use_cache=False)
        logger.info("Distillation completed successfully.")

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
