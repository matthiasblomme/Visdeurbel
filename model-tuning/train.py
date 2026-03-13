"""
STEP 2: Fine-tune Granite Vision 3.2-2B with QLoRA.

Reads dataset/train.jsonl and dataset/val.jsonl (created by prepare_dataset.py),
applies 4-bit quantization + LoRA adapters, trains, and saves the adapter to
output/granite-fish-lora/.

Usage:
    python train.py

Requirements:
    - CUDA GPU with ~12 GB VRAM (for 4-bit / QLoRA)
    - pip install -r requirements.txt
"""

import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

# ── Hyperparameters (adjust as needed) ─────────────────────────────────────────
MODEL_ID        = "ibm-granite/granite-vision-3.2-2b"
OUTPUT_DIR      = Path("output/granite-fish-lora")
TRAIN_FILE      = Path("dataset/train.jsonl")
VAL_FILE        = Path("dataset/val.jsonl")
NUM_EPOCHS      = 3
LEARNING_RATE   = 2e-4
BATCH_SIZE      = 1       # keep at 1 for most consumer GPUs; increase if you have VRAM to spare
GRAD_ACCUM      = 4       # effective batch = BATCH_SIZE * GRAD_ACCUM
MAX_SEQ_LEN     = 512     # max tokens per sample

# ── LoRA config ─────────────────────────────────────────────────────────────────
# Applied only to the language backbone — vision encoder stays frozen.
# rank=16 is a good default: higher = more capacity but more VRAM/time.
LORA_RANK       = 16
LORA_ALPHA      = 32      # scaling factor; typically 2× rank
LORA_DROPOUT    = 0.05
LORA_TARGETS    = ["q_proj", "v_proj", "k_proj", "o_proj"]
# ───────────────────────────────────────────────────────────────────────────────

PROMPT = (
    "This is a frame from an underwater riverbed camera. "
    "Is there a fish visible in this image? Answer only yes or no."
)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class FishDataset(Dataset):
    """Wraps a list of {image, label} records into a torch Dataset."""

    def __init__(self, records: list[dict], processor):
        self.records   = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image  = Image.open(record["image"]).convert("RGB")
        label  = record["label"]   # "yes" or "no"

        # Build a chat-style conversation matching the model's expected format.
        # The model is instruction-tuned, so we frame training as completing an
        # assistant turn.
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label}],
            },
        ]

        # apply_chat_template renders the conversation into a token string
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

        encoding = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )

        # Squeeze batch dim added by processor
        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        pixel_values   = encoding["pixel_values"].squeeze(0)

        # Labels: copy of input_ids, but mask everything before the assistant
        # answer so we only compute loss on the answer token(s).
        labels = input_ids.clone()
        # Find where the assistant answer starts by locating the last newline
        # before the label token (rough heuristic — works for short answers)
        # Everything up to and including the last non-pad token before the
        # answer is masked with -100.
        label_token_ids = self.processor.tokenizer(label, add_special_tokens=False)["input_ids"]
        n_label_tokens  = len(label_token_ids)
        # Mask all but the last n_label_tokens (the answer) + EOS
        labels[: -n_label_tokens - 1] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "pixel_values":   pixel_values,
            "labels":         labels,
        }


def main():
    # ── Sanity checks ──────────────────────────────────────────────────────────
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"{TRAIN_FILE} not found — run prepare_dataset.py first.")
    if not VAL_FILE.exists():
        raise FileNotFoundError(f"{VAL_FILE} not found — run prepare_dataset.py first.")

    train_records = load_jsonl(TRAIN_FILE)
    val_records   = load_jsonl(VAL_FILE)
    print(f"Train samples: {len(train_records)}, Val samples: {len(val_records)}")

    # ── Quantization config (QLoRA = LoRA + 4-bit) ────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4 — best for LLM weights
        bnb_4bit_use_double_quant=True,      # quantize the quantization constants too
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # ── Load model + processor ────────────────────────────────────────────────
    print(f"Loading {MODEL_ID} in 4-bit...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model     = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    # ── Apply LoRA ─────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expected output: ~0.2–0.5% of parameters are trainable — that's normal for LoRA.

    # ── Datasets ───────────────────────────────────────────────────────────────
    train_dataset = FishDataset(train_records, processor)
    val_dataset   = FishDataset(val_records,   processor)

    # ── Training arguments ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",   # set to "wandb" if you want experiment tracking
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("\nStarting training...")
    trainer.train()

    # ── Save LoRA adapter only (small!) ───────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    print(f"\nLoRA adapter saved to {OUTPUT_DIR}/")
    print("Next step: python inference.py <path-to-image>")


if __name__ == "__main__":
    main()
