import os
import torch
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_ID = "mistralai/Ministral-3-14B-Instruct-2512-BF16"
OUTPUT_DIR = "/home/runner/vllm_optimization/Ministral-3-14B-INT4-GPTQ"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQ_LEN = 4096


def patch_ministral3():
    # transformers 4.57.3
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.ministral.configuration_ministral import MinistralConfig
    CONFIG_MAPPING._extra_content["ministral3"] = MinistralConfig


def prepare_calibration_data(tokenizer, num_samples=256, max_length=4096):
    """Prepare calibration data from C4 dataset as a HF Dataset"""
    from datasets import load_dataset, Dataset
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

    input_ids_list = []
    attention_mask_list = []
    for example in ds:
        text = example["text"]
        if len(text) < 200:
            continue
        tokenized = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        if len(tokenized["input_ids"]) >= 512:
            input_ids_list.append(tokenized["input_ids"])
            attention_mask_list.append(tokenized["attention_mask"])
        if len(input_ids_list) >= num_samples:
            break

    avg_len = sum(len(ids) for ids in input_ids_list) / len(input_ids_list)
    print(f"  Collected {len(input_ids_list)} samples (avg {avg_len:.0f} tokens)")

    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
    })


def main():
    patch_ministral3()

    from transformers import Mistral3ForConditionalGeneration, AutoTokenizer
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor import oneshot

    model = Mistral3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # original config has sliding_window=None, but the Mistral3 forward pass
    # Setting it to max_position_embeddings makes it equivalent to full attention.
    max_pos = 262144  # from config.text_config.max_position_embeddings
    for cfg in [model.config, getattr(model.config, "text_config", None),
                getattr(model, "model", None) and getattr(model.model, "config", None)]:
        if cfg is not None:
            if not hasattr(cfg, "sliding_window") or cfg.sliding_window is None:
                cfg.sliding_window = max_pos


    calibration_data = prepare_calibration_data(tokenizer, NUM_CALIBRATION_SAMPLES, MAX_SEQ_LEN)

    ignore_patterns = [
        "lm_head",
        "re:vision_tower.*",
        "re:vision_model.*",
        "re:multi_modal_projector.*",
        "re:multimodal_projector.*",
        "re:model.embed_tokens.*",
        "re:model.norm.*",
    ]

    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=ignore_patterns,
    )

    print(f"\n Starting GPTQ quantization")

    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=calibration_data,
        recipe=recipe,
        output_dir=OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    print(f"\n Output: {OUTPUT_DIR}")
    total_size = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(OUTPUT_DIR) for f in files
    )
    print(f" Total size: {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
