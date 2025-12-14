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
  echo "Running lm_eval for $m"

  lm_eval --model vllm \
    --model_args pretrained="./$m",add_bos_token=true \
    --tasks gsm8k,hellaswag,piqa,arc_easy \
    --num_fewshot 5 \
    --limit 250 \
    --batch_size auto \
    --output_path "lm_eval_${m}.json"
done
