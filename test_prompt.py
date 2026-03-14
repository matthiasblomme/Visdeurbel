"""
Prompt testing script for fish detection.
Tests the vision LLM prompt against labeled images before baking into Docker.

Usage:
    python test_prompt.py

Reads from:
    snapshots/fish/   — images that SHOULD return "yes"
    snapshots/false/  — images that SHOULD return "no"

Edit PROMPTS below to try different phrasings.
"""

import base64
import os
import requests

# ---------------------------------------------------------------------------
# Configure the model endpoint  (override with --model <name>)
# ---------------------------------------------------------------------------
# Ollama (already running locally):
API_URL = "http://localhost:11434/v1/chat/completions"
MODEL = "granite3.2-vision:2b"

# llama-server (when Docker llm service is running):
# API_URL = "http://localhost:8080/v1/chat/completions"
# MODEL = "granite-vision"

FISH_DIR = "snapshots/fish"
FALSE_DIR = "snapshots/false"

# ---------------------------------------------------------------------------
# Prompts to evaluate — add/edit entries to compare
# ---------------------------------------------------------------------------
PROMPTS = {
    "v1_simple": (
        "This is a frame from an underwater riverbed camera. "
        "Is there a fish visible in this image? "
        "Answer only 'yes' or 'no'."
    ),
    "v2_strict": (
        "You are reviewing frames from an underwater camera aimed at a riverbed. "
        "The camera sometimes detects false positives such as floating particles, "
        "sediment clouds, water shimmer, or shadows. "
        "Is there clearly a fish (with a distinct body shape) visible in this image? "
        "Answer only 'yes' or 'no'."
    ),
    "v3_shape_hint": (
        "This is a frame from an underwater riverbed camera. "
        "A fish detection algorithm has flagged this frame. "
        "Do you see an elongated animal with a fish-like body in the image? "
        "Answer only 'yes' or 'no'."
    ),
    "v4_precision": (
        "This is a frame from an underwater riverbed camera. "
        "A motion detector flagged this frame, but many flags are false positives "
        "(water shimmer, sediment particles, floating debris). "
        "Only answer 'yes' if you can clearly identify a fish with a recognizable body, fins, or tail. "
        "If you are uncertain or the object could be debris, answer 'no'. "
        "Answer only 'yes' or 'no'."
    ),
    "v5_user": (
        "You are verifying possible fish sightings from a low-visibility underwater monitoring camera.\n\n"
        "Context:\n"
        "- The camera operates underwater in turbid water.\n"
        "- Frames are often foggy, low-contrast, noisy, and partially obscured.\n"
        "- Fish may appear close to the lens and only partly inside the frame.\n"
        "- However, many false positives are caused by blur, shadow, haze, sediment, uneven lighting, and indistinct underwater shapes.\n"
        "- Green boxes, timestamps, labels, and other overlays must be ignored.\n\n"
        "Your job:\n"
        "Answer whether a real fish, or an identifiable part of a real fish, is visible in the image.\n\n"
        "Count as YES only if there is a believable fish feature, such as:\n"
        "- fish body contour\n"
        "- curved body mass with coherent outline\n"
        "- head/body shape\n"
        "- tail or fin structure\n"
        "- partial silhouette that is still recognizably fish-like\n\n"
        "Count as NO if the image shows only:\n"
        "- blur\n"
        "- haze\n"
        "- murky gradients\n"
        "- amorphous dark patches\n"
        "- uncertain silhouettes\n"
        "- shadows or artifacts\n"
        "- any shape that is not clearly fish-like\n\n"
        "Important:\n"
        "- Do not infer or assume.\n"
        "- Do not reward weak resemblance.\n"
        "- If uncertain, answer NO.\n"
        "- Use a strict threshold for YES.\n\n"
        "Reply with exactly one word only:\n"
        "YES\n"
        "NO"
    ),
    "v6_balanced": (
        "You are verifying possible fish sightings from a low-visibility underwater monitoring camera.\n\n"
        "Context:\n"
        "- The camera operates underwater in turbid, murky water with low contrast and noise.\n"
        "- Fish may appear at any angle: from the side, head-on, tail-on, or partially outside the frame.\n"
        "- Fish close to the lens may fill much of the frame and look blurry or distorted.\n"
        "- False positives include: floating sediment, haze blobs, shadows, reflections, and amorphous shapes.\n"
        "- Ignore any green detection boxes, text overlays, or timestamps in the image.\n\n"
        "Answer YES if you see any of the following:\n"
        "- A fish body or part of one (even if blurry, partial, or at an unusual angle)\n"
        "- A recognizable fish silhouette — elongated shape, curved body, fin, tail, or head\n"
        "- An organic-looking creature shape that is consistent with a fish\n\n"
        "Answer NO if the image shows only:\n"
        "- Uniform murk, haze, or gradient without any distinct shape\n"
        "- Clearly non-biological objects: sediment cloud, shadow patch, light artifact\n"
        "- Shapes that are geometric, square, or obviously not animal\n\n"
        "Remember: the image quality is intentionally poor. A fish does not need to be perfectly visible to count.\n"
        "If there is a reasonable chance the shape is a fish, answer YES.\n\n"
        "Reply with exactly one word: YES or NO"
    ),
    "v7_green_box": (
        "You are verifying possible fish sightings from a low-visibility underwater monitoring camera.\n\n"
        "IMPORTANT: The image contains a green rectangle drawn by a motion detection algorithm. "
        "This green box marks the exact region where movement was detected. "
        "Focus your analysis on the object or shape inside or near this green rectangle.\n\n"
        "Context:\n"
        "- The camera operates underwater with low visibility, turbid water, and noise.\n"
        "- Fish may appear at any angle and may only be partially inside the frame.\n"
        "- Fish close to the lens can appear large, blurry, and fill most of the frame.\n"
        "- Common false positives: sediment clouds, shadow blobs, water shimmer.\n\n"
        "Look at the shape inside the green box and decide:\n"
        "- Does it have an organic, curved, or elongated body shape?\n"
        "- Could it be a fish, or part of one, even if blurry or at an odd angle?\n\n"
        "If YES: the shape in the green box looks like it could be a fish or part of a fish.\n"
        "If NO: the shape is clearly non-biological (haze, shadow, sediment, artifacts).\n\n"
        "Reply with exactly one word: YES or NO"
    ),
}


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def ask(prompt: str, b64: str) -> str:
    # Use Ollama native /api/chat when on localhost:11434 (more reliable for vision)
    # Fall back to OpenAI-compat format for llama-server on :8080
    if "11434" in API_URL:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a binary classifier. You ONLY respond with the single word YES or NO. No explanations, no punctuation, nothing else.",
                    },
                    {"role": "user", "content": prompt, "images": [b64]},
                ],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 3},
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip().lower()
    else:
        resp = requests.post(
            API_URL,
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": 10,
                "temperature": 0.0,
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip().lower()


def load_images(folder: str):
    folder = os.path.join(os.path.dirname(__file__), folder)
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def run_prompt(name: str, prompt: str, fish_images, false_images):
    print(f"\n{'='*60}")
    print(f"PROMPT: {name}")
    print(f"{'='*60}")

    tp = fp = tn = fn = 0  # true/false positives/negatives

    print(f"\n  [FISH folder - should all return 'yes']")
    for path in fish_images:
        b64 = encode_image(path)
        answer = ask(prompt, b64)
        correct = "yes" in answer
        mark = "OK" if correct else "!!"
        if correct:
            tp += 1
        else:
            fn += 1
        print(f"    {mark}  {os.path.basename(path):35s}  -> '{answer}'")

    print(f"\n  [FALSE folder - should all return 'no']")
    for path in false_images:
        b64 = encode_image(path)
        answer = ask(prompt, b64)
        correct = "no" in answer and "yes" not in answer
        mark = "OK" if correct else "!!"
        if correct:
            tn += 1
        else:
            fp += 1
        print(f"    {mark}  {os.path.basename(path):35s}  -> '{answer}'")

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total * 100 if total else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) else 0

    print(f"\n  Results: {tp+tn}/{total} correct  "
          f"accuracy={accuracy:.0f}%  "
          f"precision={precision:.0f}%  "
          f"recall={recall:.0f}%")
    print(f"  (TP={tp} FN={fn} TN={tn} FP={fp})")
    return accuracy


def main():
    import sys
    global MODEL

    single = "--single" in sys.argv  # quick smoke-test with 1 image each

    # --n <int>  limit to first N images per folder
    n_limit = None
    if "--n" in sys.argv:
        n_limit = int(sys.argv[sys.argv.index("--n") + 1])

    # --model <name>  override the model
    if "--model" in sys.argv:
        MODEL = sys.argv[sys.argv.index("--model") + 1]

    fish_images = load_images(FISH_DIR)
    false_images = load_images(FALSE_DIR)
    if single:
        fish_images = fish_images[:1]
        false_images = false_images[:1]
    elif n_limit:
        fish_images = fish_images[:n_limit]
        false_images = false_images[:n_limit]
    print(f"Loaded {len(fish_images)} fish images, {len(false_images)} false-positive images")
    print(f"Model: {MODEL}  @  {API_URL}")

    # Use --prompt <name> to test a single prompt; otherwise tests all
    prompt_filter = None
    if "--prompt" in sys.argv:
        idx = sys.argv.index("--prompt")
        prompt_filter = sys.argv[idx + 1]

    scores = {}
    for name, prompt in PROMPTS.items():
        if prompt_filter and name != prompt_filter:
            continue
        scores[name] = run_prompt(name, prompt, fish_images, false_images)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, acc in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {acc:5.1f}%  {name}")
    best = max(scores, key=scores.get)
    print(f"\nBest prompt: {best}")


if __name__ == "__main__":
    main()
