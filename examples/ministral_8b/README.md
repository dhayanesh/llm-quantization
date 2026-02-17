# Ministral 8B

This example demonstrates downloading, quantizing, and evaluating **Mistral AI Ministral-8B-Instruct** using INT4 (GPTQ) and FP8 quantization, with lm_eval benchmarks.

## Files

| File | Description |
|------|-------------|
| `download_model.py` | Download Ministral-8B-Instruct from Hugging Face (`mistralai/Ministral-8B-Instruct-2410`) to `ministral_8b/` |
| `quantize_int4.py` | Quantize the model to INT4-W4A16 (GPTQ) → `ministral_8b-INT4-W4A16/` |
| `quantize_fp8.py` | Quantize the model to FP8-W8A8 (dynamic) → `ministral_8b-FP8-W8A8/` |
| `inference.py` | Run a short vLLM inference test (uses FP8 model by default) |
| `run_eval.sh` | Run lm_eval on baseline and INT4 models (250 samples, 5-shot) |


---

## lm_eval Results

Evaluations use **250 samples**, **5-shot**, tasks: `gsm8k`, `hellaswag`, `piqa`, `arc_easy`. Results below are from the JSON outputs in this folder.

### Summary Table

| Model | GSM8K (EM) | HellaSwag (acc_norm) | PIQA (acc_norm) | ARC-Easy (acc_norm) |
|-------|------------|------------------------|-----------------|---------------------|
| **ministral_8b** (baseline) | **81.6%** | 73.2% | **84.8%** | **86.8%** |
| **ministral_8b-INT4-W4A16** | 75.2–75.6% | **74.0%** | 84.4% | 86.0% |

### Baseline: ministral_8b

| Task | Metric | Value |
|------|--------|--------|
| arc_easy | acc | 0.832 |
| arc_easy | acc_norm | **0.868** |
| gsm8k | exact_match (strict / flexible) | **0.816** |
| hellaswag | acc | 0.568 |
| hellaswag | acc_norm | 0.732 |
| piqa | acc | 0.788 |
| piqa | acc_norm | **0.848** |

### Quantized: ministral_8b-INT4-W4A16

| Task | Metric | Value |
|------|--------|--------|
| arc_easy | acc | 0.812 |
| arc_easy | acc_norm | 0.860 |
| gsm8k | exact_match (strict) | 0.752 |
| gsm8k | exact_match (flexible) | 0.756 |
| hellaswag | acc | 0.572 |
| hellaswag | acc_norm | **0.740** |
| piqa | acc | 0.800 |
| piqa | acc_norm | 0.844 |

### Result files

- `lm_eval_ministral_8b_250_2026-02-14T22-35-12.683616.json` — baseline
- `lm_eval_ministral_8b-INT4-W4A16_250_2026-02-14T22-41-23.282639.json` — INT4-W4A16

INT4 retains most of the baseline accuracy; HellaSwag (acc_norm) is slightly higher for INT4 (74.0% vs 73.2%). GSM8K and ARC-Easy are somewhat lower for INT4.



# Infernece Runtime Performance Comparison

- GPU 0: NVIDIA A40 (46068 MiB total)

## Runtime Results

| Model | Startup (s) | Runtime (s) | Req/s | Completion tok/s | P50 latency (s) | P95 latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| Ministral-8B-Instruct-2410 | 47.26 | 11.45 | 5.59 | 478.86 | 2.860 | 2.891 |
| Ministral-8B-Instruct-2410-INT4-W4A16 | 43.18 | 4.12 | 15.53 | 1226.56 | 1.014 | 1.092 |

## GPU Memory Utilization

| Model | Model load mem (GiB) | Available KV cache (GiB) | Peak delta vs baseline (MiB) |
|---|---:|---:|---:|
| Ministral-8B-Instruct-2410 | 14.97 | 23.69 | 41824.19 |
| Ministral-8B-Instruct-2410-INT4-W4A16 | 5.36 | 33.29 | 41836.19 |
