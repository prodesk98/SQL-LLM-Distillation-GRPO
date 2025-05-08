

### Distillation OpenAI
```bash
uv run main.py distill \
--limit 500 \
--publish \
--dataset-repo-id gretelai/synthetic_text_to_sql \
--publish-repo-id proton98/sql-distill-gpt-4.1-nano-reasoning \
--provider OpenAI \
--model gpt-4.1-nano \
--validate \
--private-repo \
--batch-size 8
```