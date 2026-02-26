import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)


MODEL_PATH = Path("ministral_8b_base")
SAVE_DIR = "ministral_8b_base-INT4-W4A16"
DATASET_PATH = "ultrachat_200k"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048


def load_patched_config():
    with (MODEL_PATH / "config.json").open("r", encoding="utf-8") as f:
        raw_cfg = json.load(f)

    text_cfg = raw_cfg.get("text_config", {})
    if text_cfg.get("model_type") == "ministral3":
        text_cfg["model_type"] = "mistral"
        raw_cfg["text_config"] = text_cfg

    raw_cfg.pop("quantization_config", None)

    model_type = raw_cfg["model_type"]
    config_kwargs = dict(raw_cfg)
    config_kwargs.pop("model_type")
    return AutoConfig.for_model(model_type, **config_kwargs)


def load_multimodal_model(config):
    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_PATH),
        config=config,
        device_map="auto",
        dtype="auto",
    )
    return model


def upcast_float8_if_needed(model):
    float8_types = (torch.float8_e4m3fn, torch.float8_e5m2)
    if any(
        param.dtype in float8_types
        for param in model.parameters()
    ):
        model = model.to(torch.bfloat16)
        print("Upcast float8 weights to bfloat16 before GPTQ")
    return model


def build_text_only_model(multimodal):
    with init_empty_weights():
        text_model = AutoModelForCausalLM.from_config(multimodal.config.text_config)
    text_model.model = multimodal.language_model
    text_model.lm_head = multimodal.lm_head
    text_model.config = multimodal.config.text_config
    text_model.config.architectures = ["MistralForCausalLM"]
    text_model.config._name_or_path = str(MODEL_PATH)
    print("Using text-only MistralForCausalLM view for quantization")
    return text_model


def load_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), use_fast=True)
    except Exception as err:
        print(f"AutoTokenizer failed, falling back to tokenizer.json ({err})")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(MODEL_PATH / "tokenizer.json")
        )
        with (MODEL_PATH / "tokenizer_config.json").open("r", encoding="utf-8") as f:
            tok_cfg = json.load(f)
        for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
            token_value = tok_cfg.get(key)
            if token_value is not None:
                setattr(tokenizer, key, token_value)
        chat_template_path = MODEL_PATH / "chat_template.jinja"
        if chat_template_path.exists():
            tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def to_text(row):
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return {"text": prompt}
    return {"text": str(row)}


def build_calibration_dataset(tokenizer):
    dataset = load_dataset(DATASET_PATH, split="train")
    sample_count = min(NUM_CALIBRATION_SAMPLES, len(dataset))
    dataset = dataset.shuffle(seed=42).select(range(sample_count))
    dataset = dataset.map(to_text)

    def tokenize_row(row):
        return tokenizer(
            row["text"],
            padding=False,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            add_special_tokens=False,
        )

    return dataset.map(tokenize_row, remove_columns=dataset.column_names)


def run_quantization(model, tokenizer, dataset):
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
    )
    oneshot(
        model=model,
        processor=tokenizer,
        recipe=recipe,
        dataset=dataset,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        save_compressed=True,
        output_dir=SAVE_DIR,
    )


def patch_output_config_for_vllm():
    out_cfg_path = Path(SAVE_DIR) / "config.json"
    with out_cfg_path.open("r", encoding="utf-8") as f:
        out_cfg = json.load(f)

    if isinstance(out_cfg.get("quantization_config"), dict):
        out_cfg["quantization_config"]["quant_method"] = "compressed-tensors"
    out_cfg["architectures"] = ["MistralForCausalLM"]

    with out_cfg_path.open("w", encoding="utf-8") as f:
        json.dump(out_cfg, f, indent=2, sort_keys=True)


def main():
    config = load_patched_config()
    multimodal = load_multimodal_model(config)
    multimodal = upcast_float8_if_needed(multimodal)
    model = build_text_only_model(multimodal)
    tokenizer = load_tokenizer()
    dataset = build_calibration_dataset(tokenizer)
    run_quantization(model, tokenizer, dataset)
    patch_output_config_for_vllm()
    print(f"Quantized model saved at: {SAVE_DIR}")

if __name__ == "__main__":
    main()
