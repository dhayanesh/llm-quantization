#!/usr/bin/env bash

MODELS=(
  llama3_8b
  llama3_8b-INT8-W8A16
  llama3_8b-INT8-W8A8
  llama3_8b-INT4-W4A16
  llama3_8b-INT4-W4A16-AWQ
  llama3_8b-FP8-W8A8
  llama3_8b-INT8-W8A16-RTN
)

for m in "${MODELS[@]}"; do
  echo "=============================="
  echo "Benchmarking model: $m"
  echo "=============================="

  python benchmark.py \
    --model "$m" \
    --dtype float16 | tee "benchmark_${m}.json"

  echo
done
