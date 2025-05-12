from typing import Optional

from datasets import load_dataset

from utils import instruction_formatting

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
except ImportError as e:
    raise ImportError(e) from e
from trl import GRPOConfig, GRPOTrainer
from rewards import (
    check_sql_reward,
    match_format_exactly,
    match_format_approximately,
)

from config import (
    HF_TOKEN, MAX_SEQ_LENGTH,
    GPU_MEMORY_UTILIZATION, LORA_ALPHA,
    LORA_RANK, MAX_PROMPT_LENGTH, LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, WEIGHT_DECAY, WARMUP_RATIO, LR_SCHEDULER_TYPE,
    OPTIM, LOGGING_STEPS, PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, NUM_GENERATIONS,
    MAX_COMPLETION_LENGTH, MAX_STEPS, SAVE_STEPS, MAX_GRAD_NORM, REPORT_TO, OUTPUT_DIR,
)


class TrainerControl:
    def __init__(
        self,
        model: str,
        num_train_epochs: int = -1,
        dataset_repo_id: str = None,
        use_vllm: bool = False,
        load_in_4bit: bool = True,
    ):
        if dataset_repo_id is None:
            raise ValueError("dataset_repo_id must be provided")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model,
            max_seq_length = MAX_SEQ_LENGTH,
            max_lora_rank = LORA_RANK,
            load_in_4bit = load_in_4bit,
            fast_inference = use_vllm,
            token = HF_TOKEN,
            gpu_memory_utilization = GPU_MEMORY_UTILIZATION,
        )
        self.num_train_epochs = num_train_epochs
        self.training_args: Optional[GRPOConfig] = None
        self.trainer: Optional[GRPOTrainer] = None
        self.dataset = load_dataset(dataset_repo_id, split="train")
        self._initialize(use_vllm)

    def _initialize(self, use_vllm: bool = False):
        dataset_prompt = instruction_formatting(self.dataset)
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = LORA_RANK,
            lora_alpha = LORA_ALPHA,
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
        )
        self.training_args = GRPOConfig(
            learning_rate = LEARNING_RATE,
            adam_beta1 = ADAM_BETA1,
            adam_beta2 = ADAM_BETA2,
            weight_decay = WEIGHT_DECAY,
            warmup_ratio = WARMUP_RATIO,
            lr_scheduler_type = LR_SCHEDULER_TYPE,
            optim = OPTIM,
            logging_steps = LOGGING_STEPS,
            bf16 = is_bfloat16_supported(),
            fp16 = not is_bfloat16_supported(),
            per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS, # Increase to 4 for smoother training
            num_generations = NUM_GENERATIONS, # Decrease if out of memory
            max_prompt_length = MAX_PROMPT_LENGTH,
            max_completion_length = MAX_COMPLETION_LENGTH,
            num_train_epochs = self.num_train_epochs, # Set to 1 for a full training run
            max_steps = MAX_STEPS,
            save_steps = SAVE_STEPS,
            max_grad_norm = MAX_GRAD_NORM,
            report_to = REPORT_TO, # Can use Weights & Biases
            output_dir = OUTPUT_DIR,
            use_vllm = use_vllm,
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
