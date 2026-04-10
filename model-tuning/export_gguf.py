"""
Export fine-tuned LoRA adapter to GGUF for use with llama.cpp (Docker LLM service).

Two-step process:
  Step 1 — merge:  loads base model + LoRA adapter, merges weights, saves as HF model
  Step 2 — convert: converts the merged HF model to GGUF using llama.cpp tools

This script handles Step 1 automatically.
For Step 2 it prints the exact commands to run.

Usage:
    python export_gguf.py                          # uses granite-fish-lora (default)
    python export_gguf.py --adapter granite-fish-lora-fast
    python export_gguf.py --quant q8_0             # quantisation type (default: q4_k_m)

After running this script:
    1. Follow the printed GGUF conversion instructions.
    2. Copy the resulting GGUF to ../models/granite-fish.gguf
    3. Rebuild the llm Docker service:
         docker compose build llm
         docker compose up -d llm
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

load_dotenv()

MODEL_ID    = "ibm-granite/granite-vision-3.2-2b"
OUTPUT_BASE = Path("output")
MODELS_DIR  = Path("../models")   # where Dockerfile.llm looks for local GGUFs


def merge(adapter_name: str) -> Path:
    """Load base model + LoRA adapter, merge weights, save as HF model."""
    adapter_dir = OUTPUT_BASE / adapter_name
    merged_dir  = OUTPUT_BASE / f"{adapter_name}-merged"

    if not adapter_dir.exists():
        print(f"[ERROR] Adapter not found: {adapter_dir}")
        print(f"        Run train.py or train_fast.py first.")
        sys.exit(1)

    if merged_dir.exists():
        print(f"[INFO] Merged model already exists at {merged_dir}, skipping merge.")
        print("       Delete it first to force a re-merge.")
        return merged_dir

    print(f"[1/3] Loading base model from HuggingFace: {MODEL_ID}")
    print("      This downloads ~4 GB on first run (cached after that).")
    base = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="cpu",          # merge on CPU — no VRAM needed
        torch_dtype=torch.float16,
    )

    print(f"[2/3] Applying LoRA adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base, str(adapter_dir))

    print("[3/3] Merging weights (merge_and_unload)...")
    model = model.merge_and_unload()

    # Save only the language model backbone — NOT the full LlavaNextForConditionalGeneration.
    # The converter (convert_hf_to_gguf.py) handles GraniteForCausalLM cleanly.
    # The vision tower (SigLIP) and image_newline tensor are LLaVA-Next scaffolding that
    # the converter can't map, and they aren't needed — the mmproj GGUF covers the vision
    # side and is downloaded as-is from HuggingFace (LoRA never touched the vision tower).
    # The LLaVA-Next model structure for Granite Vision:
    #   LlavaNextForConditionalGeneration
    #     .model.language_model  → GraniteModel (backbone only, no LM head)
    #     .lm_head               → Linear (the language model head, at the top level)
    #
    # convert_hf_to_gguf.py needs GraniteForCausalLM (backbone + head together).
    # We assemble it here from the two parts before saving.
    from transformers import GraniteForCausalLM

    backbone  = model.model.language_model   # GraniteModel
    lm_head   = model.lm_head                # Linear

    print(f"      Backbone type : {type(backbone).__name__}")
    print(f"      LM head type  : {type(lm_head).__name__}  shape={tuple(lm_head.weight.shape)}")

    print("      Assembling GraniteForCausalLM from backbone + lm_head...")
    causal_lm = GraniteForCausalLM(backbone.config)
    causal_lm.model.load_state_dict(backbone.state_dict())
    causal_lm.lm_head.load_state_dict(lm_head.state_dict())

    print(f"      Saving GraniteForCausalLM to: {merged_dir}")
    causal_lm.save_pretrained(str(merged_dir))

    # Copy the tokenizer so the converter can read vocabulary/special tokens
    processor = AutoProcessor.from_pretrained(str(adapter_dir))
    processor.tokenizer.save_pretrained(str(merged_dir))

    print(f"\n[OK] Merged model saved: {merged_dir}")
    return merged_dir


def print_conversion_instructions(merged_dir: Path, quant: str):
    """Print the llama.cpp GGUF conversion commands."""
    abs_merged = merged_dir.resolve()
    abs_models = MODELS_DIR.resolve()

    print("\n" + "=" * 60)
    print("STEP 2 — convert merged HF model to GGUF")
    print("=" * 60)
    print("""
llama.cpp includes a Python converter. Run these commands once:

  # 1. Clone llama.cpp (shallow clone, only needs convert script)
  git clone https://github.com/ggerganov/llama.cpp --depth 1

  # 2. Install conversion dependencies
  pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

  # 3. Convert to GGUF (Q4_K_M = good quality/size tradeoff, ~1.5 GB)""")
    print(f"  python llama.cpp/convert_hf_to_gguf.py \\")
    print(f"      {abs_merged} \\")
    print(f"      --outtype {quant} \\")
    print(f"      --outfile granite-fish-finetuned.gguf")
    print(f"""
  # 4. Move the GGUF to the models/ directory at the project root
  mkdir -p {abs_models}
  mv granite-fish-finetuned.gguf {abs_models}/granite-vision.gguf

  # 5. Rebuild and restart the llm Docker service
  cd ..
  docker compose build llm
  docker compose up -d llm
""")
    print("NOTE: the mmproj file (vision encoder) stays as-is.")
    print("      LoRA only fine-tunes the language backbone, not the vision tower.")
    print("      Dockerfile.llm downloads the mmproj from HuggingFace automatically.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and export to GGUF.")
    parser.add_argument(
        "--adapter",
        default="granite-fish-lora",
        help="Adapter folder name inside output/ (default: granite-fish-lora)",
    )
    parser.add_argument(
        "--quant",
        default="q4_k_m",
        help="GGUF quantisation type (default: q4_k_m). Options: q4_k_m, q8_0, f16",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge the adapter, skip printing conversion instructions",
    )
    args = parser.parse_args()

    merged_dir = merge(args.adapter)

    if not args.merge_only:
        print_conversion_instructions(merged_dir, args.quant)


if __name__ == "__main__":
    main()
