"""
STEP 1: Prepare the dataset from labeled images.

Scans FISH_DIR and NO_FISH_DIR, caps the majority class to avoid imbalance,
splits 80/20 into train/val, and writes dataset/train.jsonl + dataset/val.jsonl.

Usage:
    python prepare_dataset.py
"""

import json
import random
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
FISH_DIR    = Path("../snapshots/training-data/fish")
NO_FISH_DIR = Path("../snapshots/training-data/not-fish")
DATASET_DIR = Path("dataset")
TRAIN_FILE  = DATASET_DIR / "train.jsonl"
VAL_FILE    = DATASET_DIR / "val.jsonl"
VAL_SPLIT   = 0.2    # 20% of images go to validation
SEED        = 42
IMAGE_EXTS  = {".jpg", ".jpeg", ".png"}

# Cap the majority class at this multiple of the minority class.
# 1:3 ratio (fish : not-fish) keeps training balanced enough while
# still giving the model plenty of negative examples to learn from.
# Set to None to use all images (not recommended if heavily imbalanced).
MAX_MAJORITY_RATIO = 3
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
    print(f"  not-fish images: {len(no_fish)} (raw)")

    if not fish and not no_fish:
        print("\nNo images found. Check FISH_DIR and NO_FISH_DIR paths.")
        return

    # ── Balance: cap the majority class ───────────────────────────────────────
    if MAX_MAJORITY_RATIO is not None:
        minority = min(len(fish), len(no_fish))
        cap      = minority * MAX_MAJORITY_RATIO
        rng      = random.Random(SEED)

        if len(fish) > cap:
            fish = rng.sample(fish, cap)
            print(f"  fish capped to  : {len(fish)} (ratio {MAX_MAJORITY_RATIO}:1 vs not-fish)")
        elif len(no_fish) > cap:
            no_fish = rng.sample(no_fish, cap)
            print(f"  not-fish capped to: {len(no_fish)} (ratio {MAX_MAJORITY_RATIO}:1 vs fish)")

    print(f"\n  Using: {len(fish)} fish, {len(no_fish)} not-fish")
    ratio = len(no_fish) / len(fish) if fish else 0
    print(f"  Class ratio: 1 : {ratio:.1f}  (fish : not-fish)")

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
    print(f"  train: {len(train):4d} samples  ({len(fish_train)} fish, {len(no_fish_train)} not-fish)")
    print(f"  val  : {len(val):4d} samples  ({len(fish_val)} fish, {len(no_fish_val)} not-fish)")
    print("\nDone. Next step: python train.py")


if __name__ == "__main__":
    main()
