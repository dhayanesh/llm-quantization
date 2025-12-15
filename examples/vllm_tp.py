import gc
import time
import torch
from vllm import LLM, SamplingParams

num_gpus = torch.cuda.device_count()
tp_size = num_gpus if num_gpus > 1 else 1

start_load = time.time()
llm = LLM(
    model="llama3_8b",
    dtype="bfloat16",
)
print(f"Model loaded in {time.time() - start_load:.2f} seconds")

sampling_params = SamplingParams(
    temperature=0.1,
    max_tokens=256,
    top_p=1.0,
)

prompt = "Explain why GPUs are useful for deep learning."

_ = llm.generate([prompt], sampling_params)

start_gen = time.time()
for _ in range(5):
    outputs = llm.generate([prompt], sampling_params)
elapsed = time.time() - start_gen

print("\n=== Output ===")
print(outputs[0].outputs[0].text)
print(f"\nGeneration time: {elapsed:.2f} seconds")