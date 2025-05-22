from typing import Optional

from datasets import load_dataset

from utils import conversations_grpo_format, conversations_supervised_fine_tuning_format

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
except ImportError as e:
    raise ImportError(e) from e
from trl import GRPOConfig, GRPOTrainer, SFTTrainer, SFTConfig
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
    MAX_COMPLETION_LENGTH, MAX_STEPS, SAVE_STEPS, MAX_GRAD_NORM, REPORT_TO, OUTPUT_DIR, LORA_DROPOUT, TEMPERATURE,
    NUM_TRAIN_SFT_EPOCHS, NUM_TRAIN_GRPO_EPOCHS,
)


class TrainerControl:
    def __init__(
        self,
        model_name: str,
        dataset_repo_id: str = None,
        use_vllm: bool = True,
        load_in_4bit: bool = True,
        publish_repo_id: Optional[str] = None,
    ):
        if dataset_repo_id is None:
            raise ValueError("dataset_repo_id must be provided")
        self.publish_repo_id = publish_repo_id
        self.use_vllm = use_vllm
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_lora_rank = 64,
            max_seq_length = MAX_SEQ_LENGTH,
            load_in_4bit = load_in_4bit,
            fast_inference = use_vllm,
            token = HF_TOKEN,
            gpu_memory_utilization = GPU_MEMORY_UTILIZATION,
        )
        self.dataset = load_dataset(dataset_repo_id, split="train")
        self._initialize()

    def _initialize(self):
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = LORA_RANK,
            lora_alpha = LORA_ALPHA,
            lora_dropout = LORA_DROPOUT,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
            use_rslora = False,
        )

    def _sft_trainer(self):
        train_dataset = conversations_supervised_fine_tuning_format(self.dataset)
        train_dataset = train_dataset.map(
            lambda x: {"text": self.tokenizer.apply_chat_template(x["messages"], batched=True, tokenize=False)},
            remove_columns=["messages"])
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            args=SFTConfig(
                dataset_text_field="text",
                per_device_train_batch_size=8,
                gradient_accumulation_steps=1,
                warmup_steps=5,
                num_train_epochs=NUM_TRAIN_SFT_EPOCHS,
                learning_rate=2e-5,
                logging_steps=5,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                report_to="none",
            ),
        )
        trainer.train()

    def _grpo_trainer(self):
        train_dataset = conversations_grpo_format(self.dataset)
        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[  # type: ignore
                match_format_exactly,
                match_format_approximately,
                check_sql_reward,
            ],
            args=GRPOConfig(
                learning_rate=LEARNING_RATE,
                adam_beta1=ADAM_BETA1,
                adam_beta2=ADAM_BETA2,
                weight_decay=WEIGHT_DECAY,
                warmup_ratio=WARMUP_RATIO,
                lr_scheduler_type=LR_SCHEDULER_TYPE,
                optim=OPTIM,
                logging_steps=LOGGING_STEPS,
                bf16=is_bfloat16_supported(),
                fp16=not is_bfloat16_supported(),
                per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                num_generations=NUM_GENERATIONS,
                max_prompt_length=MAX_PROMPT_LENGTH,
                max_completion_length=MAX_COMPLETION_LENGTH,
                num_train_epochs=NUM_TRAIN_GRPO_EPOCHS,
                max_steps=MAX_STEPS,
                save_steps=SAVE_STEPS,
                max_grad_norm=MAX_GRAD_NORM,
                report_to=REPORT_TO,
                output_dir=OUTPUT_DIR,
                use_vllm=self.use_vllm,
                temperature=TEMPERATURE,
                seed=3407,
            ),
            train_dataset=train_dataset,
        )
        trainer.train()

    def train(self):
        self._sft_trainer()
        self._grpo_trainer()

    def publish(self):
        if self.publish_repo_id is None:
            raise ValueError("publish_repo_id must be provided")
        self.model.push_to_hub_merged(
            self.publish_repo_id,
            self.tokenizer,
            token = HF_TOKEN,
        )
