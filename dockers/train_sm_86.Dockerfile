FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

ENV TORCH_HOME=/root/.cache/torch
ENV PYTHON_VERSION=3.12.3

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl unzip build-essential \
    libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev \
    libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev \
    libbz2-dev libexpat1-dev liblzma-dev tk-dev uuid-dev \
    ca-certificates git

RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && make altinstall && \
    cd .. && rm -rf Python-${PYTHON_VERSION}*

RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.12 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.12 1

RUN curl -Ls https://astral.sh/uv/install.sh | bash
ENV PATH="/root/.local/bin/:$PATH"

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN uv pip install --upgrade pip setuptools wheel \
 && uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

RUN uv pip install "unsloth[cu124-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git" \
 && uv pip install vllm duckdb loguru python-dotenv

WORKDIR /trainer
COPY . .

RUN chmod +x ./entrypoint.sh
RUN ln -s /trainer/entrypoint.sh /usr/local/bin/trainer

ENTRYPOINT [ "trainer" ]
