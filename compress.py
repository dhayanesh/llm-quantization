#!/usr/bin/env python3
"""
Unified compression script that supports multiple quantization schemes.
Usage: python compress.py --compression_type <TYPE> [--model_id <MODEL_ID>]
"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier, GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier


def prepare_calibration_data(tokenizer, num_samples=512, max_seq_length=2048):
    """Prepare calibration dataset for quantization."""
    from datasets import load_dataset
    
    # Load and preprocess the dataset
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=42).select(range(num_samples))
    
    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
    ds = ds.map(preprocess)
    
    def tokenize(sample):
        return tokenizer(
            sample["text"], 
            padding=False, 
            max_length=max_seq_length, 
            truncation=True, 
            add_special_tokens=False
        )
    ds = ds.map(tokenize, remove_columns=ds.column_names)
    
    return ds


def get_compression_config(compression_type):
    """Get compression configuration based on type."""
    configs = {
        "FP8-W8A8": {
            "scheme": "FP8_DYNAMIC",
            "modifier": QuantizationModifier,
            "needs_calibration": False,
            "use_smoothquant": False,
        },
        "INT4-W4A16-AWQ": {
            "scheme": "W4A16_ASYM",
            "modifier": QuantizationModifier,
            "needs_calibration": True,
            "use_smoothquant": False,
        },
        "INT4-W4A16": {
            "scheme": "W4A16",
            "modifier": GPTQModifier,
            "needs_calibration": True,
            "use_smoothquant": False,
        },
        "INT8-W8A16-RTN": {
            "scheme": "W8A16",
            "modifier": QuantizationModifier,
            "needs_calibration": True,
            "use_smoothquant": False,
        },
        "INT8-W8A16": {
            "scheme": "W8A16",
            "modifier": GPTQModifier,
            "needs_calibration": True,
            "use_smoothquant": False,
        },
        "INT8-W8A8": {
            "scheme": "W8A8",
            "modifier": GPTQModifier,
            "needs_calibration": True,
            "use_smoothquant": True,
        },
    }
    
    if compression_type not in configs:
        raise ValueError(
            f"Unknown compression type: {compression_type}. "
            f"Supported types: {', '.join(configs.keys())}"
        )
    
    return configs[compression_type]


def compress_model(
    compression_type,
    model_id="llama3_8b",
    num_calibration_samples=512,
    max_sequence_length=2048,
):
    """Compress a model using the specified compression type."""
    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Get compression configuration
    config = get_compression_config(compression_type)
    save_dir = f"{model_id}-{compression_type}"
    
    print(f"Compression type: {compression_type}")
    print(f"Scheme: {config['scheme']}")
    print(f"Modifier: {config['modifier'].__name__}")
    print(f"Needs calibration: {config['needs_calibration']}")
    
    # Prepare calibration data if needed
    dataset = None
    if config["needs_calibration"]:
        print("Preparing calibration data...")
        dataset = prepare_calibration_data(
            tokenizer, 
            num_samples=num_calibration_samples,
            max_seq_length=max_sequence_length
        )
    
    # Create recipe
    modifier = config["modifier"](
        targets="Linear",
        scheme=config["scheme"],
        ignore=["lm_head"],
    )
    
    if config["use_smoothquant"]:
        recipe = [
            SmoothQuantModifier(smoothing_strength=0.8),
            modifier,
        ]
    else:
        recipe = modifier
    
    # Apply compression
    print(f"Applying compression...")
    oneshot_kwargs = {
        "model": model,
        "recipe": recipe,
        "output_dir": save_dir,
    }
    
    if config["needs_calibration"]:
        oneshot_kwargs.update({
            "dataset": dataset,
            "max_seq_length": max_sequence_length,
            "num_calibration_samples": num_calibration_samples,
        })
    
    oneshot(**oneshot_kwargs)
    
    print(f"Quantized model saved at: {save_dir}")
    return save_dir


def main():
    parser = argparse.ArgumentParser(
        description="Compress a language model using various quantization schemes"
    )
    parser.add_argument(
        "--compression_type",
        type=str,
        required=True,
        choices=[
            "FP8-W8A8",
            "INT4-W4A16-AWQ",
            "INT4-W4A16",
            "INT8-W8A16-RTN",
            "INT8-W8A16",
            "INT8-W8A8",
        ],
        help="Type of compression to apply",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="llama3_8b",
        help="Model ID to compress (default: llama3_8b)",
    )
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=512,
        help="Number of calibration samples (default: 512)",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    
    args = parser.parse_args()
    
    compress_model(
        compression_type=args.compression_type,
        model_id=args.model_id,
        num_calibration_samples=args.num_calibration_samples,
        max_sequence_length=args.max_sequence_length,
    )


if __name__ == "__main__":
    main()

