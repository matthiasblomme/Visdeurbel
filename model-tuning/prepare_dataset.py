"""
STEP 1: Prepare the dataset from labeled images.

Scans images/fish/ and images/no_fish/, splits 80/20 into train/val,
and writes dataset/train.jsonl + dataset/val.jsonl.

Usage:
    python prepare_dataset.py
"""

import json
import os
import random
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
FISH_DIR    = Path("images/fish")
NO_FISH_DIR = Path("images/no_fish")
DATASET_DIR = Path("dataset")
TRAIN_FILE  = DATASET_DIR / "train.jsonl"
VAL_FILE    = DATASET_DIR / "val.jsonl"
VAL_SPLIT   = 0.2   # 20% of images go to validation
SEED        = 42
IMAGE_EXTS  = {".jpg", ".jpeg", ".png"}
# ───────────────────────────────────────────────────────────────────────────────


def load_images(folder: Path, label: str) -> list[dict]:
    """Return list of {image, label} dicts for all images in folder."""
    if not folder.exists():
        print(f"  [warn] folder not found: {folder}")
        return []
    records = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in IMAGE_EXTS:
            records.append({"image": str(f.resolve()), "label": label})
    return records


def split(records: list[dict], val_ratio: float, seed: int):
    """Shuffle and split into (train, val)."""
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * val_ratio))
    return shuffled[cut:], shuffled[:cut]


def write_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    print("Loading images...")
    fish    = load_images(FISH_DIR,    label="yes")
    no_fish = load_images(NO_FISH_DIR, label="no")

    print(f"  fish images   : {len(fish)}")
    print(f"  no_fish images: {len(no_fish)}")

    if not fish and not no_fish:
        print("\nNo images found. Add images to images/fish/ and images/no_fish/ first.")
        return

    fish_train, fish_val       = split(fish,    VAL_SPLIT, SEED)
    no_fish_train, no_fish_val = split(no_fish, VAL_SPLIT, SEED)

    train = fish_train + no_fish_train
    val   = fish_val   + no_fish_val

    # Shuffle combined splits so fish/no_fish aren't grouped
    rng = random.Random(SEED)
    rng.shuffle(train)
    rng.shuffle(val)

    write_jsonl(train, TRAIN_FILE)
    write_jsonl(val,   VAL_FILE)

    print(f"\nDataset written to {DATASET_DIR}/")
    print(f"  train: {len(train):4d} samples  ({len(fish_train)} fish, {len(no_fish_train)} no_fish)")
    print(f"  val  : {len(val):4d} samples  ({len(fish_val)} fish, {len(no_fish_val)} no_fish)")
    print("\nDone. Next step: python train.py")


if __name__ == "__main__":
    main()
