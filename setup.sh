#!/bin/bash

cp .env.template .env
source .venv/bin/activate
uv sync --locked --dev
pre-commit install
uv pip install -e .
