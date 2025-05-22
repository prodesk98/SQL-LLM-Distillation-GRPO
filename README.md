# 🧠 Structured SQL Reasoning with Chain-of-Thought and GRPO Distillation

This project focuses on generating **logical and executable SQL queries** by combining:

- 🧩 **Chain-of-Thought (CoT)** prompting for interpretable, step-by-step reasoning (`<think>...</think>`);
- 🔁 **LLM distillation** from larger to smaller models;
- 🎯 **GRPO (Group Relative Policy Optimization)** – a reinforcement learning strategy that improves reasoning efficiency while staying close to a reference policy.

Inspired by mathematical reasoning models like **DeepSeekMath**, this framework applies CoT to SQL generation and fine-tunes distilled models using GRPO to enhance both accuracy and interpretability.

---

## ⚙️ Configuration

Update `config.yaml` to control core parameters:

| Key                    | Description                                                |
|------------------------|------------------------------------------------------------|
| `base_url`             | Endpoint for API requests                                  |
| `max_seq_length`       | Total token length (prompt + completion)                   |
| `max_prompt_length`    | Max token length for input prompts                         |
| `temperature`          | Output randomness (0 = deterministic, 1 = more diverse)    |
| `tensor_parallel_size` | Devices for parallel inference (set to `1` for single GPU) |

---

## 🚀 Setup

**Create environment & install dependencies:**

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install --upgrade pip
````

**Configure environment:**

```bash
cp template.env .env
cp template.config.yaml config.yaml
cp template.prod .train
```

* Add your **API key** to `.env` (`API_KEY`)
* Add your **Hugging Face token** (`HF_TOKEN`) for dataset access

---

## 🧪 `distill` Command – SQL CoT Generation

Generates SQL examples with reasoning using your chosen LLM backend. Supports validation, filtering, and Hugging Face publishing.

### ✅ Example usage:

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

### 📝 Key Options:

| Flag                | Description                                          |
|---------------------|------------------------------------------------------|
| `--model`           | LLM for generation (e.g. `meta-llama/Llama-3.1-70B`) |
| `--dataset-repo-id` | Source dataset from Hugging Face                     |
| `--limit`           | Max examples to generate (default: `100`)            |
| `--provider`        | Backend: `OpenAI`, `vLLM`, `Groq`, `HuggingFace`     |
| `--validate`        | Validate SQL syntax/logical correctness              |
| `--remove-no-valid` | Remove invalid queries (requires `--validate`)       |
| `--publish`         | Upload final dataset to Hugging Face                 |
| `--publish-repo-id` | Target repo (e.g. `user/sql-dataset`)                |
| `--private-repo`    | Publish as private dataset                           |
| `--retries`         | Retry failed generations (default: `3`)              |
| `--use-ray`         | Enable parallel distillation with Ray (for scale)    |

---

## 🏋️ `train` Command – RL Fine-tuning with GRPO

Fine-tunes a distilled model using SQL-specific reinforcement learning with reasoning-based rewards.

### ✅ Example usage:

```bash
uv run main.py train \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --dataset-repo-id proton98/sql-distill-llama-3-1-70b-instruct-reasoning \
  --publish-repo-id sql-llama3.2-3b-it-reasoning
```

---

## 🧪 Test Prompt Format

Send model reasoning prompts using a structured message format:

```json
{
  "model": "proton98/sql-llama3.2-3b-it-reasoning",
  "messages": [
    {
      "role": "system",
      "content": "You are an expert in writing optimized SQL queries.\nThink about the problem and provide your working out.\nPlace it between <think> and </think>.\nThen, provide your solution between <sql></sql>"
    },
    {
      "role": "user",
      "content": "I need to know the number of elements within the omnium table that have id_pools equal to 10"
    }
  ]
}
```

### ✅ Output

```
<think>
To solve this problem, we need to count the number of rows in the omnium table where id_pools is equal to 10. We can use the COUNT() function to count the number of rows that meet this condition.

Since we are only interested in rows where id_pools is 10, we can use a WHERE clause to filter the rows. The WHERE clause allows us to specify conditions that the rows must meet in order to be included in the count.

In this case, the condition is simply id_pools = 10. We can use the COUNT() function to count the number of rows that meet this condition.

The COUNT() function returns the number of rows in the table that meet the specified condition. Since we are only counting rows where id_pools is 10, the COUNT() function will return the number of rows that have id_pools equal to 10.

We can use the COUNT() function in combination with the WHERE clause to count the number of rows that meet the specified condition. This will give us the number of elements within the omnium table that have id_pools equal to 10.
</think>
<sql>
SELECT COUNT(*) FROM omnium WHERE id_pools = 10;
</sql>
```

---

## 📬 Contact & Contributions

Feel free to open issues or pull requests if you’d like to contribute, fix bugs, or improve documentation.

---

## 📚 References

* DeepSeekMath: [github.com/deepseek-ai/DeepSeekMath](https://github.com/deepseek-ai/DeepSeekMath)
* GRPO Docs [https://huggingface.co/docs/trl/main/grpo_trainer](https://huggingface.co/docs/trl/main/grpo_trainer)
* Hugging Face: [huggingface.co](https://huggingface.co)
* Unsloth: [unsloth.ai](https://unsloth.ai)
* OpenR1: [github.com/huggingface/open-r1](https://github.com/huggingface/open-r1)