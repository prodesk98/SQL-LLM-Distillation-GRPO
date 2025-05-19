FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV TORCH_HOME=/root/.cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl unzip build-essential python3.10 python3.10-dev python3.10-venv git ca-certificates pkg-config libssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN curl -Ls https://astral.sh/uv/install.sh | bash

ENV PATH="/root/.local/bin/:$PATH"

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN uv pip install --upgrade pip setuptools wheel \
 && uv pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN uv pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git" \
 && uv pip install vllm duckdb loguru python-dotenv jupyterlab

WORKDIR /jupyter
COPY . .

CMD [ "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser" ]