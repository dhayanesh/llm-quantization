import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import evaluator

########################################
# ARGUMENTS
########################################
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument(
    "--tasks",
    type=str,
    default="hellaswag,piqa,arc_easy"
)
parser.add_argument("--batch_size", type=int, default=4)
args = parser.parse_args()

MODEL_PATH = args.model
TASKS = args.tasks.split(",")

########################################
# LOAD MODEL
########################################
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype="auto",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

########################################
# RUN EVALUATION
########################################
print(f"\nRunning lm_eval for model: {MODEL_PATH}")
print(f"Tasks: {TASKS}")

results = evaluator.simple_evaluate(
    model=model,
    tokenizer=tokenizer,
    tasks=TASKS,
    batch_size=args.batch_size,
    max_batch_size=args.batch_size,
    device="cuda",
)

########################################
# PRINT & SAVE
########################################
print("\n===== lm_eval RESULTS =====")
print(json.dumps(results["results"], indent=2))

out_file = f"lm_eval_{MODEL_PATH}.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved results to {out_file}")
