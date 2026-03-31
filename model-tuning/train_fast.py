"""
STEP 2 (fast variant): Fine-tune Granite Vision 3.2-2B with LoRA.

Same approach as train.py but optimized for speed:
- No mid-training evaluation (saves ~6+ hours of eval passes)
- No mid-training checkpoint saves (avoids memory spikes)
- Adapter is saved only at the end

Expected runtime: ~40 minutes on a 16 GB consumer GPU (vs ~11 hours with train.py).
Use inference.py --eval separately after training to measure accuracy.

Usage:
    python train_fast.py

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
OUTPUT_DIR      = Path("output/granite-fish-lora-fast")
TRAIN_FILE      = Path("dataset/train.jsonl")
VAL_FILE        = Path("dataset/val.jsonl")
NUM_EPOCHS      = 3
LEARNING_RATE   = 2e-4
BATCH_SIZE      = 1
GRAD_ACCUM      = 4       # effective batch = BATCH_SIZE * GRAD_ACCUM

# ── LoRA config ────────────────────────────────────────────────────────────────
LORA_RANK       = 16
LORA_ALPHA      = 32
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
    texts = []
    images = []
    for ex in examples:
        msg = ex["messages"]
        image_path = msg[0]["content"][0]["image"]
        image = Image.open(image_path).convert("RGB")
        images.append(image)

        chat_msg = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": msg[0]["content"][1]["text"]},
                ],
            },
            msg[1],
        ]
        text = processor.apply_chat_template(chat_msg, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    batch = processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding=True,
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    for i, ex in enumerate(examples):
        answer = ex["messages"][1]["content"][0]["text"]
        answer_ids = processor.tokenizer(answer, add_special_tokens=False)["input_ids"]
        n_answer = len(answer_ids)
        labels[i, : -(n_answer + 1)] = -100

    batch["labels"] = labels
    return batch


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"{TRAIN_FILE} not found -- run prepare_dataset.py first.")

    # ── Hard VRAM cap (prevents system RAM spill and PC lockup) ───────────────
    # Limits PyTorch to 75% of available VRAM. If training exceeds this it
    # throws an OOM error and exits cleanly instead of freezing your machine.
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        torch.cuda.set_per_process_memory_fraction(0.75, 0)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({total_gb:.1f} GB)")
        print(f"VRAM cap: 75% = {total_gb * 0.75:.1f} GB")

    train_records = load_jsonl(TRAIN_FILE)
    print(f"Train samples: {len(train_records)}")

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

    # ── Build dataset ─────────────────────────────────────────────────────────
    train_data = build_dataset(train_records, processor)

    # ── Training config ───────────────────────────────────────────────────────
    # Key difference from train.py: no eval, no checkpoint saves during training.
    # This avoids the multi-hour eval pauses that made train.py take 11+ hours.
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
        logging_steps=10,
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
        args=training_args,
        train_dataset=train_data,
        peft_config=lora_config,
        data_collator=lambda examples: collate_fn(examples, processor),
    )
    trainer.model.print_trainable_parameters()

    print("\nStarting training (no eval pauses, ~40 min)...")
    trainer.train()

    # ── Save LoRA adapter only at the end ────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    print(f"\nLoRA adapter saved to {OUTPUT_DIR}/")
    print("Next step: python inference.py --eval")


if __name__ == "__main__":
    main()
