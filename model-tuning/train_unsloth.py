"""
STEP 2 (Unsloth variant): Fine-tune Granite Vision 3.2-2B with QLoRA via Unsloth.

Unsloth handles the vision tower quantization correctly (avoids the SigLIP +
bitsandbytes dtype mismatch that crashes standard HuggingFace QLoRA loading).

Expected runtime: ~15-20 minutes on a 16 GB consumer GPU.

Usage:
    python train_unsloth.py
"""

import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
from PIL import Image
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTConfig, SFTTrainer

# ── Hyperparameters ────────────────────────────────────────────────────────────
MODEL_ID        = "ibm-granite/granite-vision-3.2-2b"
OUTPUT_DIR      = Path("output/granite-fish-unsloth")
TRAIN_FILE      = Path("dataset/train.jsonl")
NUM_EPOCHS      = 3
LEARNING_RATE   = 2e-4
BATCH_SIZE      = 2
GRAD_ACCUM      = 2       # effective batch = BATCH_SIZE * GRAD_ACCUM = 4

# ── LoRA config ────────────────────────────────────────────────────────────────
LORA_RANK       = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0
LORA_TARGETS    = ["q_proj", "v_proj", "k_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
# ───────────────────────────────────────────────────────────────────────────────

PROMPT = (
    "You are reviewing a frame from a low-visibility underwater monitoring camera "
    "to decide whether a fish is visible.\n\n"
    "Context:\n"
    "- The water is murky, noisy, low-contrast, and may contain haze, sediment, "
    "blur, shadows, and reflections.\n"
    "- Fish may appear from the side, head-on, tail-on, partially cropped, very "
    "blurry, or very close to the lens.\n"
    "- Ignore any green boxes, timestamps, labels, or overlays.\n\n"
    "Answer YES only if there is a plausible fish form present.\n"
    "Answer NO otherwise.\n\n"
    "Reply with exactly one word: YES or NO"
)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(records: list[dict]):
    """Build chat-formatted dataset with images for Unsloth vision training."""
    samples = []
    for r in records:
        image = Image.open(r["image"]).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": r["label"]}],
            },
        ]
        samples.append({"messages": messages, "images": [image]})
    return samples


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"{TRAIN_FILE} not found -- run prepare_dataset.py first.")

    train_records = load_jsonl(TRAIN_FILE)
    print(f"Train samples: {len(train_records)}")

    # ── Load model + tokenizer via Unsloth ─────────────────────────────────────
    print(f"Loading {MODEL_ID} via Unsloth (4-bit)...")
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        dtype=torch.float16,
    )

    # ── Apply LoRA via Unsloth ─────────────────────────────────────────────────
    model = FastVisionModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        finetune_vision_layers=False,     # freeze vision encoder
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )

    # ── Build dataset ──────────────────────────────────────────────────────────
    train_data = build_dataset(train_records)

    # ── Training config ────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        bf16=False,
        logging_steps=5,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
    )

    print("\nStarting Unsloth QLoRA training...")
    trainer.train()

    # ── Save LoRA adapter ─────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\nLoRA adapter saved to {OUTPUT_DIR}/")
    print("Next step: python inference.py --eval")


if __name__ == "__main__":
    main()
