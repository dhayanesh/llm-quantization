import argparse
import time
import torch
import json
import math
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

########################################
# ARGUMENTS
########################################
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--dtype", type=str, default="float16")
parser.add_argument("--max_tokens", type=int, default=128)
args = parser.parse_args()

MODEL_PATH = args.model
DTYPE = args.dtype
MAX_TOKENS = args.max_tokens
DEVICE = "cuda"

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
# UTILS
########################################
def cuda_peak():
    return torch.cuda.max_memory_allocated() / 1e9

########################################
# LOAD MODEL
########################################
torch.cuda.reset_peak_memory_stats()
start_load = time.time()

llm = LLM(
    model=MODEL_PATH,
    dtype=DTYPE,
    tensor_parallel_size=1,
)

load_time = time.time() - start_load
model_mem = cuda_peak()

print(f"\nModel load time: {load_time:.2f}s")
print(f"GPU memory (model load peak): {model_mem:.2f} GB")

sampling = SamplingParams(
    temperature=0.0,
    max_tokens=MAX_TOKENS,
)

########################################
# DRY RUN (ignored completely)
########################################
print("\nRunning dry inference (graph capture / compile)...")
_ = llm.generate(["Dry run prompt"], sampling)
torch.cuda.synchronize()

########################################
# WARMUP (still excluded from timing)
########################################
print("Running warmup inference...")
_ = llm.generate(["Warmup prompt"], sampling)
torch.cuda.synchronize()

########################################
# SEQUENTIAL LATENCY
########################################
torch.cuda.reset_peak_memory_stats()
start = time.time()

for p in PROMPTS:
    llm.generate([p], sampling)

torch.cuda.synchronize()
seq_time = time.time() - start
seq_mem = cuda_peak()

########################################
# BATCH LATENCY
########################################
torch.cuda.reset_peak_memory_stats()
start = time.time()

outputs = llm.generate(PROMPTS, sampling)

torch.cuda.synchronize()
batch_time = time.time() - start
batch_mem = cuda_peak()

########################################
# TOKEN COUNT / THROUGHPUT
########################################
total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
throughput = total_tokens / batch_time

########################################
# PERPLEXITY (WIKITEXT-2)
########################################
print("\nRunning perplexity eval (WikiText-2)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

eval_ds = load_dataset(
    "wikitext",
    "wikitext-2-raw-v1",
    split="test"
).select(range(128))

def compute_ppl():
    losses = []
    for ex in eval_ds:
        enc = tokenizer(
            ex["text"],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(DEVICE)

        with torch.no_grad():
            out = llm.model_runner.model(
                **enc,
                labels=enc["input_ids"]
            )
            losses.append(out.loss.item())
    return math.exp(sum(losses) / len(losses))

try:
    ppl = compute_ppl()
except Exception as e:
    ppl = None
    print("Perplexity eval failed:", e)

########################################
# RESULTS
########################################
results = {
    "model": MODEL_PATH,
    "dtype": DTYPE,
    "load_time_sec": load_time,
    "model_mem_gb": model_mem,
    "seq_5_latency_sec": seq_time,
    "batch_5_latency_sec": batch_time,
    "throughput_tok_per_sec": throughput,
    "seq_peak_mem_gb": seq_mem,
    "batch_peak_mem_gb": batch_mem,
    "perplexity_wikitext2": ppl,
}

print("\n===== BENCHMARK RESULTS =====")
print(json.dumps(results, indent=2))
