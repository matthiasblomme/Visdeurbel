"""
STEP 2: Fine-tune Granite Vision 3.2-2B with LoRA.

Reads dataset/train.jsonl and dataset/val.jsonl (created by prepare_dataset.py),
applies LoRA adapters to the language backbone, trains using TRL's SFTTrainer,
and saves the adapter to output/granite-fish-lora/.

Usage:
    python train.py

Requirements:
    - CUDA GPU with ~12-16 GB VRAM
    - pip install -r requirements.txt
"""

import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# ── Hyperparameters (adjust as needed) ─────────────────────────────────────────
MODEL_ID        = "ibm-granite/granite-vision-3.2-2b"
OUTPUT_DIR      = Path("output/granite-fish-lora")
TRAIN_FILE      = Path("dataset/train.jsonl")
VAL_FILE        = Path("dataset/val.jsonl")
NUM_EPOCHS      = 3
LEARNING_RATE   = 2e-4
BATCH_SIZE      = 1       # keep at 1 for most consumer GPUs
GRAD_ACCUM      = 4       # effective batch = BATCH_SIZE * GRAD_ACCUM

# ── LoRA config ─────────────────────────────────────────────────────────────────
# Applied only to the language backbone — vision encoder stays frozen.
# rank=16 is a good default: higher = more capacity but more VRAM/time.
LORA_RANK       = 16
LORA_ALPHA      = 32      # scaling factor; typically 2× rank
LORA_DROPOUT    = 0.05
LORA_TARGETS    = ["q_proj", "v_proj", "k_proj", "o_proj"]
# ───────────────────────────────────────────────────────────────────────────────

PROMPT = (
    "You are reviewing a frame from a low-visibility underwater monitoring camera to decide whether a fish is visible.\n\n"
    "Context:\n"
    "- The water is murky, noisy, low-contrast, and may contain haze, sediment, blur, shadows, and reflections.\n"
    "- Fish may appear from the side, head-on, tail-on, partially cropped, very blurry, or very close to the lens.\n"
    "- Ignore any green boxes, timestamps, labels, or overlays.\n\n"
    "Answer YES only if there is a plausible fish form present.\n"
    "A plausible fish form means one or more of these:\n"
    "- an elongated or tapered body\n"
    "- a coherent curved body mass\n"
    "- a head/body/tail relationship\n"
    "- a fin, tail, or fish-like silhouette\n"
    "- a partial but still believable fish-shaped body\n\n"
    "Answer NO if the image shows only:\n"
    "- uniform murk or haze\n"
    "- sediment clouds or floating particles\n"
    "- vague shadow patches\n"
    "- reflections or light artifacts\n"
    "- shapeless dark blobs without a coherent fish form\n\n"
    "Important rule:\n"
    "Do NOT require a perfect, sharp fish.\n"
    "Do NOT answer YES for a vague blob alone.\n"
    "Answer YES when there is a believable fish-like structure, even if partial or blurry.\n"
    "Answer NO when the shape is only ambiguous murk or debris.\n\n"
    "Reply with exactly one word: YES or NO"
)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(records: list[dict], processor):
    """Convert JSONL records into the chat format expected by SFTTrainer."""
    samples = []
    for r in records:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": r["image"]},
                    {"type": "text", "text": PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": r["label"]}],
            },
        ]
        samples.append({"messages": messages})
    return samples


def collate_fn(examples, processor):
    """Custom collator that processes images and text for the vision model."""
    texts = []
    images = []
    for ex in examples:
        msg = ex["messages"]
        # Extract image path from the user message
        image_path = msg[0]["content"][0]["image"]
        image = Image.open(image_path).convert("RGB")
        images.append(image)

        # Build the chat text (replace image entry with placeholder for template)
        chat_msg = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": msg[0]["content"][1]["text"]},
                ],
            },
            msg[1],  # assistant message
        ]
        text = processor.apply_chat_template(chat_msg, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    batch = processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding=True,
    )

    # Create labels: mask everything except the assistant answer tokens
    labels = batch["input_ids"].clone()
    # Mask padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # For each sample, mask everything before the assistant answer
    for i, ex in enumerate(examples):
        answer = ex["messages"][1]["content"][0]["text"]
        answer_ids = processor.tokenizer(answer, add_special_tokens=False)["input_ids"]
        n_answer = len(answer_ids)
        # Mask all but the last n_answer tokens + EOS
        labels[i, : -(n_answer + 1)] = -100

    batch["labels"] = labels
    return batch


def main():
    # ── Sanity checks ──────────────────────────────────────────────────────────
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"{TRAIN_FILE} not found — run prepare_dataset.py first.")
    if not VAL_FILE.exists():
        raise FileNotFoundError(f"{VAL_FILE} not found — run prepare_dataset.py first.")

    train_records = load_jsonl(TRAIN_FILE)
    val_records   = load_jsonl(VAL_FILE)
    print(f"Train samples: {len(train_records)}, Val samples: {len(val_records)}")

    # ── Load model + processor ────────────────────────────────────────────────
    print(f"Loading {MODEL_ID} in fp16...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False

    # ── LoRA config ───────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Build datasets ────────────────────────────────────────────────────────
    train_data = build_dataset(train_records, processor)
    val_data   = build_dataset(val_records,   processor)

    # ── Training config ───────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        data_collator=lambda examples: collate_fn(examples, processor),
    )
    trainer.model.print_trainable_parameters()

    print("\nStarting training...")
    trainer.train()

    # ── Save LoRA adapter only (small!) ───────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    print(f"\nLoRA adapter saved to {OUTPUT_DIR}/")
    print("Next step: python inference.py <path-to-image>")


if __name__ == "__main__":
    main()
