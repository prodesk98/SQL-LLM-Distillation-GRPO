from typing import Optional

from datasets import load_dataset

from utils import instruction_formatting

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
except ImportError:
    raise ImportError(
        "Unsloth is not installed. Please install it using 'pip install unsloth'."
    )
from trl import GRPOConfig, GRPOTrainer
from rewards import (
    check_sql_reward,
    match_format_exactly,
    match_format_approximately,
)

from config import (
    HF_TOKEN, MAX_SEQ_LENGTH,
    GPU_MEMORY_UTILIZATION, LORA_ALPHA,
    LORA_RANK,
)


class TrainerControl:
    def __init__(self, model: str, num_train_epochs: int = -1, dataset_repo_id: str = None):
        if dataset_repo_id is None:
            raise ValueError("dataset_repo_id must be provided")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model,
            max_seq_length = MAX_SEQ_LENGTH,
            max_lora_rank = LORA_RANK,
            load_in_4bit = True,
            token = HF_TOKEN,
            gpu_memory_utilization = GPU_MEMORY_UTILIZATION,
        )
        self.num_train_epochs = num_train_epochs
        self.training_args: Optional[GRPOConfig] = None
        self.trainer: Optional[GRPOTrainer] = None
        self.dataset = load_dataset(dataset_repo_id, split="train")
        self._initialize()

    def _initialize(self):
        dataset_prompt = instruction_formatting(self.dataset)
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = LORA_RANK,
            lora_alpha = LORA_ALPHA,
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
        )
        self.training_args = GRPOConfig(
            learning_rate = 5e-6,
            adam_beta1 = 0.9,
            adam_beta2 = 0.99,
            weight_decay = 0.1,
            warmup_ratio = 0.1,
            lr_scheduler_type = "cosine",
            optim = "paged_adamw_8bit",
            logging_steps = 1,
            bf16 = is_bfloat16_supported(),
            fp16 = not is_bfloat16_supported(),
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 1, # Increase to 4 for smoother training
            num_generations = 6, # Decrease if out of memory
            max_prompt_length = 256,
            max_completion_length = 200,
            num_train_epochs = self.num_train_epochs, # Set to 1 for a full training run
            max_steps = 100,
            save_steps = 250,
            max_grad_norm = 0.1,
            report_to = "none", # Can use Weights & Biases
            output_dir = "outputs",
        )
        self.trainer = GRPOTrainer(
            model = self.model,
            processing_class = self.tokenizer,
            reward_funcs = [ # type: ignore
                match_format_exactly,
                match_format_approximately,
                check_sql_reward,
            ],
            args = self.training_args,
            train_dataset = dataset_prompt,
        )

    def train(self):
        self.trainer.train()

    def evaluate(self):
        ...
