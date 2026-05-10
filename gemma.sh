#!/usr/bin/bash

set -euo pipefail

# Get the HF_TOKEN and other values from the .env file
set -o allexport
source .env
set +o allexport

exec vllm serve google/gemma-4-26B-A4B-it \
  --shutdown-timeout 30 \
  --distributed-executor-backend mp \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --max-model-len 24576 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16384 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \

