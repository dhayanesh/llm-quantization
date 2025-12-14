import argparse
import time
import json
import threading

import torch
from vllm import LLM, SamplingParams
import pynvml


########################################
# ARGUMENTS
########################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


########################################
# PROMPTS
########################################
PROMPTS = [
    "Explain why GPUs are useful for deep learning.",
    "What is quantization in neural networks?",
    "Describe attention in transformer models.",
    "What are the tradeoffs of INT4 quantization?",
    "Explain pipeline parallelism in simple terms."
]


########################################
# NVML GPU UTILS (device-level, vLLM-safe)
########################################
def nvml_init():
    pynvml.nvmlInit()

def gpu_used_gb(device_index=0):
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1e9

def measure_peak_gpu_used(fn, device_index=0, interval=0.05):
    peak = 0.0
    stop = False

    def poll():
        nonlocal peak
        while not stop:
            peak = max(peak, gpu_used_gb(device_index))
            time.sleep(interval)

    t = threading.Thread(target=poll, daemon=True)
    t.start()
    try:
        fn()
    finally:
        stop = True
        t.join()

    return peak


########################################
# MAIN
########################################
def main():
    args = parse_args()
    nvml_init()

    MODEL_PATH = args.model
    DTYPE = args.dtype
    MAX_TOKENS = args.max_tokens
    GPU = args.gpu

    ########################################
    # MODEL LOAD
    ########################################
    base_mem = gpu_used_gb(GPU)
    start_load = time.time()

    llm = LLM(
        model=MODEL_PATH,
        dtype=DTYPE,
        tensor_parallel_size=1,
        disable_log_stats=True,
    )

    load_time = time.time() - start_load
    load_mem = gpu_used_gb(GPU) - base_mem

    print(f"\nModel load time: {load_time:.2f}s")
    print(f"GPU memory (model load): {load_mem:.2f} GB")

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
    )

    ########################################
    # DRY RUN (graph capture / compile)
    ########################################
    llm.generate(["Dry run prompt"], sampling)
    torch.cuda.synchronize()

    ########################################
    # WARMUP
    ########################################
    llm.generate(["Warmup prompt"], sampling)
    torch.cuda.synchronize()

    ########################################
    # SEQUENTIAL LATENCY
    ########################################
    def run_sequential():
        for p in PROMPTS:
            llm.generate([p], sampling)
        torch.cuda.synchronize()

    start = time.time()
    seq_peak = measure_peak_gpu_used(run_sequential, GPU)
    seq_time = time.time() - start

    ########################################
    # BATCH LATENCY
    ########################################
    def run_batch():
        nonlocal outs
        outs = llm.generate(PROMPTS, sampling)
        torch.cuda.synchronize()

    outs = None
    start = time.time()
    batch_peak = measure_peak_gpu_used(run_batch, GPU)
    batch_time = time.time() - start

    ########################################
    # TOKEN COUNT / THROUGHPUT
    ########################################
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    throughput = total_tokens / batch_time

    ########################################
    # RESULTS
    ########################################
    results = {
        "model": MODEL_PATH,
        "dtype": DTYPE,
        "load_time_sec": round(load_time, 3),
        "model_mem_gb": round(load_mem, 3),
        "seq_5_latency_sec": round(seq_time, 3),
        "batch_5_latency_sec": round(batch_time, 3),
        "throughput_tok_per_sec": round(throughput, 2),
        "seq_peak_mem_gb": round(seq_peak, 3),
        "batch_peak_mem_gb": round(batch_peak, 3),
    }

    print("\n===== BENCHMARK RESULTS =====")
    print(json.dumps(results, indent=2))


########################################
# ENTRY POINT
########################################
if __name__ == "__main__":
    main()
