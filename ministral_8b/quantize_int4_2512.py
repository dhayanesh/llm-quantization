import json
import shutil
from pathlib import Path

import torch
from accelerate import init_empty_weights
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)


MODEL_PATH = Path("Ministral-3-14B-Instruct-2512")
SAVE_DIR = "Ministral-3-14B-Instruct-2512-int4"
DATASET_PATH = "ultrachat_200k"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048


def load_mistral3_config():
    with (MODEL_PATH / "config.json").open("r", encoding="utf-8") as f:
        raw_cfg = json.load(f)

    text_cfg = raw_cfg.get("text_config", {})
    if text_cfg.get("model_type") == "ministral3":
        text_cfg["model_type"] = "mistral"
        raw_cfg["text_config"] = text_cfg

    raw_cfg.pop("quantization_config", None)

    model_type = raw_cfg.pop("model_type")
    return AutoConfig.for_model(model_type, **raw_cfg)


def load_multimodal_model():
    patched_cfg = load_mistral3_config()
    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_PATH),
        config=patched_cfg,
        device_map="auto",
        dtype="auto",
    )
    return model


def build_text_only_config(multimodal):
    raw_text_cfg = multimodal.config.text_config.to_dict()
    raw_text_cfg["model_type"] = "mistral"
    model_type = raw_text_cfg.pop("model_type")
    return AutoConfig.for_model(model_type, **raw_text_cfg)


def build_text_only_model(multimodal):
    text_config = build_text_only_config(multimodal)
    with init_empty_weights():
        text_model = AutoModelForCausalLM.from_config(text_config)

    text_model.model = multimodal.language_model
    text_model.lm_head = multimodal.lm_head
    text_model.config = text_config
    text_model.config.architectures = ["MistralForCausalLM"]
    text_model.config.model_type = "mistral"
    text_model.config._name_or_path = str(MODEL_PATH)
    text_model.tie_weights()
    print("Using text-only MistralForCausalLM view for GPTQ")
    return text_model


def _expand_block_scales(scale: torch.Tensor, out_features: int, in_features: int):
    if scale.ndim == 0:
        return scale
    if scale.ndim == 1:
        if scale.shape[0] == out_features:
            return scale[:, None]
        if scale.shape[0] == in_features:
            return scale[None, :]
        raise ValueError(
            f"Cannot broadcast 1D scale shape={tuple(scale.shape)} "
            f"to weight shape=({out_features}, {in_features})"
        )
    if scale.ndim != 2:
        raise ValueError(f"Expected 2D scale tensor, got shape={tuple(scale.shape)}")

    block_out = (out_features + scale.shape[0] - 1) // scale.shape[0]
    block_in = (in_features + scale.shape[1] - 1) // scale.shape[1]
    expanded = scale.repeat_interleave(block_out, dim=0).repeat_interleave(block_in, dim=1)
    return expanded[:out_features, :in_features]


def _load_weight_scales(model_path: Path, scale_keys):
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing checkpoint index file: {index_path}")

    with index_path.open("r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    keys_by_file = {}
    for key in scale_keys:
        filename = weight_map.get(key)
        if filename is not None:
            keys_by_file.setdefault(filename, []).append(key)

    scale_tensors = {}
    for filename, keys in keys_by_file.items():
        shard_path = model_path / filename
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            for key in keys:
                scale_tensors[key] = shard.get_tensor(key)

    return scale_tensors


def _resolve_scale_key(weight_map_keys: set[str], module_name: str):
    candidates = (
        f"language_model.model.{module_name}.weight_scale_inv",
        f"model.language_model.{module_name}.weight_scale_inv",
        f"{module_name}.weight_scale_inv",
    )
    for key in candidates:
        if key in weight_map_keys:
            return key
    return None


def dequantize_fp8_language_tower(multimodal):
    float8_types = (torch.float8_e4m3fn, torch.float8_e5m2)
    index_path = MODEL_PATH / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    weight_map_keys = set(weight_map.keys())

    linear_names = [
        name
        for name, module in multimodal.language_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    name_to_scale_key = {
        name: _resolve_scale_key(weight_map_keys, name)
        for name in linear_names
    }
    scale_keys = [key for key in name_to_scale_key.values() if key is not None]
    scale_tensors = _load_weight_scales(MODEL_PATH, scale_keys)

    converted = 0
    with torch.no_grad():
        for name, module in multimodal.language_model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if module.weight.dtype not in float8_types:
                continue

            scale_key = name_to_scale_key.get(name)
            weight_scale = scale_tensors.get(scale_key)
            if weight_scale is None:
                continue

            out_features, in_features = module.weight.shape
            weight_scale = _expand_block_scales(
                weight_scale.to(dtype=torch.float32, device=module.weight.device),
                out_features=out_features,
                in_features=in_features,
            )
            dequantized_weight = (module.weight.float() * weight_scale).to(torch.bfloat16)
            module.weight = torch.nn.Parameter(
                dequantized_weight, requires_grad=module.weight.requires_grad
            )
            converted += 1

    if converted == 0:
        raise RuntimeError("No FP8 linear layers were dequantized; checkpoint may be incompatible.")
    print(f"Dequantized {converted} FP8 language layers to bfloat16")


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
        for key in (
            "bos_token",
            "eos_token",
            "pad_token",
            "unk_token",
            "padding_side",
            "truncation_side",
            "model_max_length",
        ):
            value = tok_cfg.get(key)
            if value is not None:
                setattr(tokenizer, key, value)
        chat_template_path = MODEL_PATH / "chat_template.jinja"
        if chat_template_path.exists():
            tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def row_to_text(row):
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return {"text": prompt.strip()}

    messages = row.get("messages")
    if isinstance(messages, list):
        parts = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            role = msg.get("role")
            if isinstance(role, str) and role.strip():
                parts.append(f"{role.strip()}: {content.strip()}")
            else:
                parts.append(content.strip())
        if parts:
            return {"text": "\n".join(parts)}

    for value in row.values():
        if isinstance(value, str) and value.strip():
            return {"text": value.strip()}

    return {"text": json.dumps(row, ensure_ascii=True)}


def build_calibration_dataset(tokenizer):
    dataset = load_dataset(DATASET_PATH, split="train")
    sample_count = min(NUM_CALIBRATION_SAMPLES, len(dataset))
    dataset = dataset.shuffle(seed=42).select(range(sample_count))
    dataset = dataset.map(row_to_text, desc="Formatting calibration text")

    def tokenize_row(row):
        return tokenizer(
            row["text"],
            padding=False,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            add_special_tokens=True,
            return_token_type_ids=False,
        )

    return dataset.map(
        tokenize_row,
        remove_columns=dataset.column_names,
        desc="Tokenizing calibration text",
    )


def run_quantization(
    model,
    tokenizer,
    dataset,
):
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
        offload_hessians=True,
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

    out_cfg["architectures"] = ["MistralForCausalLM"]
    out_cfg["model_type"] = "mistral"

    # Remove multimodal-only fields in case they leak through during save.
    for key in (
        "text_config",
        "vision_config",
        "image_token_index",
        "multimodal_projector_bias",
        "projector_hidden_act",
        "spatial_merge_size",
        "vision_feature_layer",
    ):
        out_cfg.pop(key, None)

    if isinstance(out_cfg.get("quantization_config"), dict):
        out_cfg["quantization_config"]["quant_method"] = "compressed-tensors"

    with out_cfg_path.open("w", encoding="utf-8") as f:
        json.dump(out_cfg, f, indent=2, sort_keys=True)


def copy_runtime_assets():
    for filename in (
        "chat_template.jinja",
        "SYSTEM_PROMPT.txt",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tekken.json",
    ):
        src = MODEL_PATH / filename
        if src.exists():
            shutil.copy2(src, Path(SAVE_DIR) / filename)


def main():
    model = load_multimodal_model()
    dequantize_fp8_language_tower(model)
    text_model = build_text_only_model(model)
    tokenizer = load_tokenizer()
    dataset = build_calibration_dataset(tokenizer)
    run_quantization(
        model=text_model,
        tokenizer=tokenizer,
        dataset=dataset,
    )
    patch_output_config_for_vllm()
    copy_runtime_assets()
    print(f"Quantized model saved at: {SAVE_DIR}")


if __name__ == "__main__":
    main()
