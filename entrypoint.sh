#!/bin/sh

python main.py train \
--model ${MODEL} \
--dataset-repo-id ${REPO_ID} \
--publish-repo-id ${PUBLISH_REPO_ID} \
--num-train-epochs 1