## Structured SQL Reasoning via CoT and LLM Distillation with GRPO Optimization

This project explores the generation of logically sound and executable SQL queries by combining Chain of Thought (CoT) prompting with LLM distillation and reinforcement learning, optimized using **GRPO (Group Relative Policy Optimization)**—a variant of PPO designed for complex reasoning tasks with improved memory efficiency.

Inspired by recent advances in mathematical reasoning (e.g., DeepSeekMath 7B), we adapt these principles to the SQL domain. CoT encourages step-by-step reasoning before final query generation (`<think>`...`</think>`), promoting interpretability and correctness. GRPO further refines the model through an online learning process that maximizes the advantage of preferred completions while preserving proximity to a reference policy. This is achieved through iterative training steps: completion generation, advantage calculation, KL divergence estimation, and policy update.

By distilling knowledge from a larger LLM into a smaller, reasoning-capable model, the approach yields SQL outputs that are both syntactically correct and grounded in transparent logic.

---

### Configuration

Update the `config.yaml` file to adjust core settings:

* `base_url`: API endpoint for sending model requests.
* `max_seq_length`: Total token limit (prompt + output).
* `max_prompt_length`: Maximum token length for input prompts.
* `temperature`: Controls output randomness (0 = deterministic, 1 = creative).
* `tensor_parallel_size`: Number of devices used for parallel execution (set to 1 for single GPU).


### Setup

**Virtual Environment:**
```bash
uv venv .venv --python 3.11 && source .venv/bin/activate && uv pip install --upgrade pip
```

**Variables:**

If you're using a cloud-based model, add your API key to the `API_KEY` field in the `.env` file.
To enable access to Hugging Face models, datasets and repositories, include your authentication token in the `HF_TOKEN` field.

To get started, copy the template:

```bash
cp template.env .env
cp template.config.yaml config.yaml
cp template.prod .train
```

### 🔧 `distill` Command – SQL-RL-Distillation-LLM

The `distill` command generates reasoning-enhanced SQL data using a specified LLM. It supports multiple providers (OpenAI, vLLM, Groq, Hugging Face) and optionally validates, filters, and publishes the results.

#### ✅ Common Usage:

```bash
uv run main.py distill \
  --model meta-llama/Llama-3.1-70B \
  --dataset-repo-id gretelai/synthetic_text_to_sql \
  --limit 25000 \
  --batch-size 64 \
  --provider vLLM \
  --validate \
  --publish \
  --publish-repo-id your_repo_id \
  --private-repo
```

---

### 📝 Parameter Breakdown:

| Option              | Description                                                                                  |
|---------------------|----------------------------------------------------------------------------------------------|
| `--model`           | Model to use for distillation (e.g., `meta-llama/Llama-3.1-70B`). Default: `'gpt-4.1-nano'`. |
| `--dataset-repo-id` | Source dataset repo on Hugging Face (e.g., `gretelai/synthetic_text_to_sql`).                |
| `--limit`           | Number of examples to distill. Default: `100`.                                               |
| `--provider`        | Backend used for generation. One of: `OpenAI`, `vLLM`, `Groq`, `HuggingFace`.                |
| `--batch-size`      | Number of examples processed per batch. Default: `8`.                                        |
| `--validate`        | Enables SQL validation during distillation.                                                  |
| `--remove-no-valid` | Discards examples with invalid SQL (use with `--validate`).                                  |
| `--publish`         | Publishes the distilled dataset to Hugging Face.                                             |
| `--publish-repo-id` | Target repo for publishing (e.g., `user/repo-name`).                                         |
| `--private-repo`    | Makes the published repo private (recommended for internal use).                             |
| `--retries`         | Number of retry attempts for failed completions. Default: `3`.                               |
| `--use-ray`         | Enables distributed processing via Ray (useful for large-scale distillation).                |


### 🛠️ `train` Command – SQL-RL-Training
The `train` command fine-tunes the distilled model using reinforcement learning. It supports various training configurations, including batch size, learning rate, and training steps.
#### ✅ Common Usage:

```bash
uv run main.py train \
  --model unsloth/Phi-4 \
  --dataset-repo-id proton98/sql-distill-gpt-4.1-nano-instruct-reasoning
```     