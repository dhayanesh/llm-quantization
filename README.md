# LLM Quantization Benchmarking

This project provides tools for quantizing large language models (LLMs) using various quantization schemes and benchmarking their performance. It supports multiple quantization methods including FP8, INT8, and INT4 quantization with different weight and activation configurations.

## Features

- **Multiple Quantization Schemes**: Support for 6 different quantization configurations:
  - `FP8-W8A8`: FP8 dynamic quantization (weights and activations)
  - `INT8-W8A16`: INT8 weights with FP16 activations (GPTQ)
  - `INT8-W8A8`: INT8 weights and activations with SmoothQuant
  - `INT8-W8A16-RTN`: INT8 weights with FP16 activations (RTN - Round To Nearest)
  - `INT4-W4A16`: INT4 weights with FP16 activations (GPTQ)
  - `INT4-W4A16-AWQ`: INT4 weights with FP16 activations (AWQ - Activation-aware Weight Quantization)

- **Unified Compression Script**: Single script to apply any quantization scheme
- **Performance Benchmarking**: Measure model load time, memory usage, latency, and throughput
- **Evaluation**: Run standard NLP benchmarks (GSM8K, HellaSwag, PIQA, ARC-Easy) using `lm-eval`
- **Model Analysis**: Inspect quantized model checkpoints and runtime behavior

## Project Structure

```
.
├── compress.py                    # Unified compression script
├── download_model.py              # Download base model from HuggingFace
├── benchmark.py                   # Performance benchmarking script
├── inspect_model.py               # Model checkpoint and runtime inspection
├── vllm_test.py                   # Simple vLLM inference test
├── run_benchmark.sh               # Batch benchmarking script
├── run_benchmark_lm-eval.sh       # Batch evaluation script
├── run_inspection.sh              # Batch model inspection script
├── analysis_summary.md            # Detailed analysis of quantization results
├── examples/                      # Example scripts (legacy compression scripts)
│   ├── compress_*.py              # Individual compression examples
│   ├── vllm_pp.py                 # vLLM pipeline parallelism example
│   └── vllm_tp.py                 # vLLM tensor parallelism example
├── eval_results/                  # Evaluation results (JSON)
├── perf_results/                  # Performance benchmark results (JSON)
└── inspect_results/               # Model inspection results (JSON)
└── ministral_8b/                  # Benchmarks for ministral-8b
└── kv-quantization/               # Experiments on Quantized KV Cache
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (for model quantization and inference)
- PyTorch with CUDA support

### Dependencies

Install required packages:

```bash
pip install transformers torch
pip install llmcompressor  # For model quantization
pip install vllm           # For efficient inference
pip install datasets       # For calibration data
pip install safetensors    # For checkpoint inspection
pip install pynvml         # For GPU memory monitoring
pip install lm-eval        # For evaluation benchmarks
```

## Usage

### 1. Download Base Model

First, download the base model you want to quantize:

```bash
python download_model.py
```

This downloads `meta-llama/Meta-Llama-3-8B-Instruct` to `llama3_8b/` by default.

### 2. Compress Model

Use the unified `compress.py` script to apply quantization:

```bash
# Basic usage
python compress.py --compression_type FP8-W8A8

# With custom model
python compress.py --compression_type INT8-W8A8 --model_id llama3_8b

# With custom calibration parameters
python compress.py --compression_type INT4-W4A16 \
    --num_calibration_samples 1024 \
    --max_sequence_length 4096
```

**Supported compression types:**
- `FP8-W8A8`
- `INT8-W8A16`
- `INT8-W8A8`
- `INT8-W8A16-RTN`
- `INT4-W4A16`
- `INT4-W4A16-AWQ`

The compressed model will be saved to `{model_id}-{compression_type}/`.

### 3. Benchmark Performance

Benchmark a single model:

```bash
python benchmark.py --model llama3_8b-INT8-W8A8 --dtype float16
```

Or run benchmarks for all models:

```bash
bash run_benchmark.sh
```

The benchmark measures:
- Model load time
- GPU memory usage (model load, sequential, batch)
- Sequential latency (5 prompts)
- Batch latency (5 prompts in parallel)
- Throughput (tokens/second)

Results are saved to `perf_results/` directory.

### 4. Evaluate Model Quality

Run standard NLP benchmarks using `lm-eval`:

```bash
# Single model
lm_eval --model vllm \
    --model_args pretrained="./llama3_8b-INT8-W8A8",add_bos_token=true \
    --tasks gsm8k,hellaswag,piqa,arc_easy \
    --num_fewshot 5 \
    --limit 250 \
    --batch_size auto \
    --output_path "lm_eval_llama3_8b-INT8-W8A8.json"
```

Or run evaluations for all models:

```bash
bash run_benchmark_lm-eval.sh
```

Results are saved to `eval_results/` directory.

### 5. Inspect Quantized Model

Inspect a quantized model checkpoint:

```bash
# Inspect a single model
python inspect_model.py llama3_8b-INT8-W8A8

# Inspect multiple models
python inspect_model.py llama3_8b llama3_8b-INT8-W8A8 llama3_8b-INT4-W4A16

# Run inspection for all models
bash run_inspection.sh
```

This script:
- Inspects checkpoint metadata and configuration files
- Analyzes tensor dtypes in the checkpoint
- Checks runtime behavior with vLLM
- Provides hints about quantization format
- Saves inspection results to `inspect_results/` directory

### 6. Review Analysis Summary

After running benchmarks, evaluations, and inspections, review the comprehensive analysis:

```bash
# View the analysis summary
cat analysis_summary.md
```

The `analysis_summary.md` document provides:
- Detailed performance comparisons across all quantization methods
- Accuracy evaluation results on multiple benchmarks
- GPU-specific insights and hardware limitations
- Recommendations for different use cases
- Task-by-task performance breakdown

## Compression Methods Explained

### FP8-W8A8
- **Scheme**: FP8 dynamic quantization
- **Calibration**: Not required
- **Use case**: Fast quantization with minimal accuracy loss (requires FP8-capable hardware)

### INT8-W8A16
- **Scheme**: INT8 weights, FP16 activations (GPTQ)
- **Calibration**: Required (512 samples)
- **Use case**: Good balance between compression and accuracy

### INT8-W8A8
- **Scheme**: INT8 weights and activations with SmoothQuant
- **Calibration**: Required (512 samples)
- **Use case**: Maximum INT8 compression with activation quantization

### INT8-W8A16-RTN
- **Scheme**: INT8 weights, FP16 activations (Round To Nearest)
- **Calibration**: Required (but may be ignored)
- **Use case**: Simple post-training quantization

### INT4-W4A16
- **Scheme**: INT4 weights, FP16 activations (GPTQ)
- **Calibration**: Required (512 samples)
- **Use case**: High compression ratio with maintained accuracy

### INT4-W4A16-AWQ
- **Scheme**: INT4 weights, FP16 activations (AWQ)
- **Calibration**: Required (512 samples)
- **Use case**: Activation-aware quantization for better accuracy

## Results

The project includes example results in:
- `perf_results/`: Performance benchmarks (latency, throughput, memory)
- `eval_results/`: Model quality evaluations (GSM8K, HellaSwag, PIQA, ARC-Easy)
- `inspect_results/`: Model checkpoint inspection results (metadata, dtypes, runtime config)
- `analysis_summary.md`: Comprehensive analysis comparing all quantization methods

## Examples

The `examples/` directory contains:
- **Legacy compression scripts**: Individual `compress_*.py` files for each quantization type (now superseded by unified `compress.py`)
- **vLLM parallelism examples**: 
  - `vllm_pp.py`: Pipeline parallelism example
  - `vllm_tp.py`: Tensor parallelism example

These examples demonstrate advanced usage patterns and can be used as templates for custom implementations.

## Notes

- **Calibration Data**: Most quantization methods require calibration data. The default uses 512 samples from `HuggingFaceH4/ultrachat_200k`.
- **GPU Requirements**: FP8 quantization requires GPUs with compute capability 8.9+ (Ada/Hopper architecture).
- **Model Format**: Quantized models are saved in a format compatible with vLLM for efficient inference.
- **Memory**: Quantization significantly reduces model memory footprint, enabling deployment on smaller GPUs.

## Troubleshooting

### CUDA Out of Memory
- Reduce `num_calibration_samples` or `max_sequence_length`
- Use a smaller base model
- Ensure sufficient GPU memory for the quantization process

### FP8 Not Working
- Check GPU compute capability: `python -c "import torch; print(torch.cuda.get_device_capability(0))"`
- FP8 requires compute capability >= (8, 9)
- Fall back to INT8/INT4 quantization if FP8 is not supported

### Import Errors
- Ensure all dependencies are installed
- Check that `llmcompressor` and `vllm` are compatible versions
- Verify CUDA and PyTorch installation

## License

This project is for research and benchmarking purposes. Model usage is subject to the license of the base model (e.g., Meta Llama 3 license).

