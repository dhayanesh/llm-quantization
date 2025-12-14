MODELS=(
  llama3_8b
  llama3_8b-INT8-W8A16
  llama3_8b-INT8-W8A8
  llama3_8b-INT4-W4A16
  llama3_8b-INT4-W4A16-AWQ
  llama3_8b-FP8-W8A8
)

for m in "${MODELS[@]}"; do
  python lm_eval_benchmark.py \
    --model "$m" \
    --tasks hellaswag,piqa,arc_easy
done
