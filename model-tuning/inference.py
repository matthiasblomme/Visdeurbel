"""
STEP 3: Run inference with the fine-tuned model.

Loads the base Granite Vision model + your LoRA adapter and classifies
one or more images as fish / no fish.

Usage:
    # Single image
    python inference.py path/to/image.jpg

    # Batch evaluation against val.jsonl (prints accuracy + full report)
    python inference.py --eval

    # Use base model only (no adapter) — useful for comparing before/after
    python inference.py --no-adapter path/to/image.jpg
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID    = "ibm-granite/granite-vision-3.2-2b"
ADAPTER_DIR = Path("output/granite-fish-lora")
VAL_FILE    = Path("dataset/val.jsonl")
MAX_NEW_TOKENS = 5   # "yes" or "no" is 1 token; a little headroom
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


def load_model(use_adapter: bool):
    processor = AutoProcessor.from_pretrained(
        str(ADAPTER_DIR) if use_adapter else MODEL_ID
    )
    base = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    if use_adapter:
        if not ADAPTER_DIR.exists():
            raise FileNotFoundError(
                f"Adapter not found at {ADAPTER_DIR}. Run train.py first."
            )
        model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        print(f"Loaded LoRA adapter from {ADAPTER_DIR}")
    else:
        model = base
        print("Using base model (no adapter)")

    model.eval()
    return model, processor


def predict(image_path: str, model, processor) -> str:
    """Returns 'yes', 'no', or the raw model output if unexpected."""
    image = Image.open(image_path).convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    # Decode only the newly generated tokens (skip the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    answer = processor.decode(new_tokens, skip_special_tokens=True).strip().lower()

    # Normalise to yes/no
    if answer.startswith("yes"):
        return "yes"
    elif answer.startswith("no"):
        return "no"
    return answer  # unexpected — return raw for debugging


def evaluate(model, processor):
    """Run batch evaluation against val.jsonl and print a report."""
    if not VAL_FILE.exists():
        print(f"{VAL_FILE} not found. Run prepare_dataset.py first.")
        return

    records = [json.loads(l) for l in VAL_FILE.read_text().splitlines() if l.strip()]
    tp = fp = tn = fn = 0

    for i, r in enumerate(records, 1):
        pred  = predict(r["image"], model, processor)
        truth = r["label"]
        match = pred == truth
        status = "OK" if match else "WRONG"

        if truth == "yes" and pred == "yes": tp += 1
        elif truth == "no"  and pred == "no":  tn += 1
        elif truth == "no"  and pred == "yes": fp += 1
        elif truth == "yes" and pred == "no":  fn += 1

        print(f"  [{i:3d}/{len(records)}] {status:5s}  truth={truth}  pred={pred}  {Path(r['image']).name}")

    total     = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"\n{'-'*50}")
    print(f"  Total   : {total}")
    print(f"  Accuracy : {accuracy:.1%}")
    print(f"  Precision: {precision:.1%}  (of predicted fish, how many were real fish)")
    print(f"  Recall   : {recall:.1%}  (of real fish, how many did we catch)")
    print(f"  F1       : {f1:.1%}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="Path to image file")
    parser.add_argument("--eval",       action="store_true", help="Evaluate on val.jsonl")
    parser.add_argument("--no-adapter", action="store_true", help="Skip loading LoRA adapter")
    args = parser.parse_args()

    if not args.image and not args.eval:
        parser.print_help()
        sys.exit(1)

    print("Loading model...")
    model, processor = load_model(use_adapter=not args.no_adapter)

    if args.eval:
        print(f"\nEvaluating on {VAL_FILE}...\n")
        evaluate(model, processor)
    else:
        result = predict(args.image, model, processor)
        print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
