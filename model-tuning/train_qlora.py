"""
STEP 2 (QLoRA variant): Fine-tune Granite Vision 3.2-2B with 4-bit quantization.

Same task as train.py but loads the language model in 4-bit (QLoRA), cutting VRAM
usage roughly in half and training time from hours to ~30-40 minutes.

The SigLIP vision encoder is dequantized back to fp16 after loading because its
MultiheadAttention pooling head is incompatible with bitsandbytes 4-bit tensors
(a known issue with transformers 5.x + bitsandbytes on LLaVA-style models).

Usage:
    python train_qlora.py

Requirements:
    - CUDA GPU with ~8-12 GB VRAM (4-bit quantization)
    - pip install -r requirements.txt
"""

import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn as nn
import bitsandbytes as bnb
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# ── Hyperparameters ────────────────────────────────────────────────────────────
MODEL_ID        = "ibm-granite/granite-vision-3.2-2b"
OUTPUT_DIR      = Path("output/granite-fish-qlora")
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


def dequantize_vision_tower(model):
    """Replace 4-bit quantized layers in vision tower with fp16 equivalents.

    The crash is specifically in SigLIP's MultiheadAttention pooling head:
    F.multi_head_attention_forward accesses out_proj.weight directly, which fails
    when bitsandbytes has replaced it with a 4-bit tensor. We dequantize ONLY the
    layers inside the problematic pooling head (nn.MultiheadAttention modules),
    not the entire vision tower — keeping memory overhead minimal (~2 MB vs ~800 MB).
    """
    vision_tower = model.model.vision_tower
    patched = 0

    for name, module in vision_tower.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            # Patch out_proj if it was quantized
            if isinstance(module.out_proj, bnb.nn.Linear4bit):
                old = module.out_proj
                fp16_weight = bnb.functional.dequantize_4bit(
                    old.weight.data, old.weight.quant_state
                ).to(torch.float16)
                new_proj = nn.Linear(
                    old.in_features, old.out_features,
                    bias=old.bias is not None,
                    device=fp16_weight.device, dtype=torch.float16,
                )
                new_proj.weight.data.copy_(fp16_weight)
                if old.bias is not None:
                    new_proj.bias.data.copy_(old.bias.data.to(torch.float16))
                new_proj.requires_grad_(False)
                module.out_proj = new_proj
                patched += 1

    # Freeze entire vision tower
    for p in vision_tower.parameters():
        p.requires_grad = False

    print(f"  Patched {patched} MultiheadAttention out_proj layers to fp16")


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
    if not VAL_FILE.exists():
        raise FileNotFoundError(f"{VAL_FILE} not found -- run prepare_dataset.py first.")

    train_records = load_jsonl(TRAIN_FILE)
    val_records   = load_jsonl(VAL_FILE)
    print(f"Train samples: {len(train_records)}, Val samples: {len(val_records)}")

    # ── 4-bit quantization config ──────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # ── Load model + processor ─────────────────────────────────────────────────
    print(f"Loading {MODEL_ID} in 4-bit...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )

    # ── Fix: dequantize vision tower (workaround for SigLIP + bnb issue) ──────
    print("Dequantizing vision tower...")
    dequantize_vision_tower(model)

    # ── Prepare for k-bit training ─────────────────────────────────────────────
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ── LoRA config ────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Build datasets ─────────────────────────────────────────────────────────
    train_data = build_dataset(train_records, processor)
    val_data   = build_dataset(val_records,   processor)

    # ── Training config ────────────────────────────────────────────────────────
    # eval_strategy="no" skips validation during training (saves hours).
    # Use inference_qlora.py --eval after training to measure accuracy.
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
        bf16=True,
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

    print("\nStarting QLoRA training...")
    trainer.train()

    # ── Save LoRA adapter ─────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    print(f"\nLoRA adapter saved to {OUTPUT_DIR}/")
    print("Next step: python inference_qlora.py --eval")


if __name__ == "__main__":
    main()
