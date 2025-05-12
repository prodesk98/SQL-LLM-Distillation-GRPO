#!/bin/bash

python main.py train \
--model ${MODEL} \
--dataset-repo-id ${DATASET_REPO_ID} \
--use-vllm