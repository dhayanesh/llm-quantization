#load model and quantize
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "llama3_8b"
SAVE_DIR = MODEL_ID + "-FP8-W8A8"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)

oneshot(
    model=model,
    recipe=recipe,
    output_dir=SAVE_DIR
)

# model.save_pretrained(SAVE_DIR, save_compressed=True)
# tokenizer.save_pretrained(SAVE_DIR)

print("Quantized model saved at:", SAVE_DIR)