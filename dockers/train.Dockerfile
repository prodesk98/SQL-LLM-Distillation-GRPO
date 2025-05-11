FROM python:3.11-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git curl build-essential libssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade uv && pip uninstall -y pip setuptools wheel

WORKDIR /trainer

COPY .. .