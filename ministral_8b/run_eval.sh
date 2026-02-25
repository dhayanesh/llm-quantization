#!/bin/bash

MODELS=(
  ministral_8b
  ministral_8b-INT4-W4A16
)

TASKS="gsm8k,hellaswag,piqa,arc_easy"

for m in "${MODELS[@]}"; do
  echo "========================================"
  echo "Running 250-sample eval for $m"
  echo "========================================"

  lm_eval --model vllm \
    --model_args pretrained="./${m}",add_bos_token=true \
    --tasks ${TASKS} \
    --num_fewshot 5 \
    --limit 250 \
    --batch_size auto \
    --output_path "lm_eval_${m}_250.json"

  echo "Finished $m"
  echo
done
