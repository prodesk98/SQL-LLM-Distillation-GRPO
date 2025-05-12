FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV TORCH_HOME=/root/.cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget build-essential python3.10 python3-pip python3.10-dev git ca-certificates curl kmod \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda create --name unsloth_env python=3.10 -y
SHELL ["/bin/bash", "-c"]
RUN echo "source activate unsloth_env" >> ~/.bashrc
ENV PATH /opt/conda/envs/unsloth_env/bin:$PATH
ENV CONDA_DEFAULT_ENV=unsloth_env

RUN conda install -n unsloth_env -y pytorch==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install trl peft xformers accelerate bitsandbytes duckdb loguru python-dotenv vllm

WORKDIR /trainer

COPY . .

CMD [ "python", "main.py", "train", "--model", "${MODEL}", "--dataset-repo-id", "${REPO_ID}", "--use-vllm" ]