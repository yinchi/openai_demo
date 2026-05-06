#!/usr/bin/bash

# Get the HF_TOKEN and other values from the .env file
set -o allexport
source .env
set +o allexport

vllm serve google/gemma-4-26B-A4B-it \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32767 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 8192 \
  --trust-remote-code \
  --enforce-eager \
  --language-model-only \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4
