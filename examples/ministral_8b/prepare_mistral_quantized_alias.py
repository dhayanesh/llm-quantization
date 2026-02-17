#!/usr/bin/env python3
"""Generate a patched config for quantized Ministral models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def patch_config(config: dict) -> tuple[dict, list[str]]:
    updated = dict(config)
    changes: list[str] = []

    if updated.get("model_type") != "mistral":
        changes.append(f"model_type: {updated.get('model_type')!r} -> 'mistral'")
        updated["model_type"] = "mistral"

    if updated.get("architectures") != ["MistralForCausalLM"]:
        changes.append(
            f"architectures: {updated.get('architectures')!r} -> ['MistralForCausalLM']"
        )
        updated["architectures"] = ["MistralForCausalLM"]

    if "torch_dtype" not in updated and "dtype" in updated:
        changes.append("torch_dtype: added from dtype")
        updated["torch_dtype"] = updated["dtype"]

    return updated, changes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read <model_dir>/config.json and write a patched config file "
            "for vLLM Mistral resolution."
        )
    )
    parser.add_argument("model_dir", help="Directory containing config.json")
    parser.add_argument(
        "--output",
        default="config.mistral.json",
        help="Output config file path (default: ./config.mistral.json)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    source_config = model_dir / "config.json"
    output_path = Path(args.output).expanduser().resolve()

    if not source_config.exists():
        print(f"error: config not found: {source_config}", file=sys.stderr)
        return 2

    with source_config.open("r", encoding="utf-8") as f:
        config = json.load(f)

    patched, changes = patch_config(config)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(patched, f, indent=2)
        f.write("\n")

    print(f"patched config written: {output_path}")
    if changes:
        print("changes:")
        for change in changes:
            print(f"  - {change}")
    else:
        print("changes: none")

    print(f"to apply: cp {output_path} {source_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
