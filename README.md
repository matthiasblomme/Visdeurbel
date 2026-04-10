# Visdeurbel

A real-time fish detection system for an underwater riverbed camera, built in three phases.

The camera watches a stream 24/7. The goal is a Telegram notification whenever a fish swims past, without too many false alarms and without missing too many real ones.

---

## The Story

### Phase 1 — Motion Detection

The first question was: can we even tell when something moves?

OpenCV's background subtraction (`MOG2`) watches each frame and flags anything that deviates from the static riverbed background. When a blob shows up that's the right size and shape to be a fish, an annotated snapshot is saved. That's it — fast, lightweight, runs on any hardware, no AI involved.

The problem: the camera sits underwater in a murky river. Sediment clouds, light shimmer, plant movement and passing debris all trigger the detector. The raw hit rate was too noisy to send straight to Telegram.

### Phase 2 — LLM Verification

The second question was: can we filter out the false alarms?

Rather than tuning more OpenCV parameters, we added IBM Granite Vision 3.2-2B as a second opinion. Every motion hit goes to the vision model with a detailed prompt: "Is there actually a fish here, or is this sediment/shimmer/debris?" Only confirmed detections get a Telegram notification.

Prompt engineering got us to about 58% accuracy on the hard cases — the murky, ambiguous frames where a blob might or might not be a fish. Good enough to prove the concept, not good enough to run unattended.

### Phase 3 — Fine-Tuning

The third question was: can we make the model actually good at *this specific camera*?

We collected labeled frames directly from the Visdeurbel stream — fish and no-fish — and fine-tuned Granite Vision using LoRA on our own footage. The model learned what sediment clouds look like in this particular stream, what the shimmer patterns look like, what a real fish silhouette looks like at this depth and angle.

Result: 94.6% F1 on the validation set. The fine-tuned model is exported to GGUF and served via llama.cpp inside Docker, on the same machine the detector already runs on.

---

## Running It

Copy `.env.example` to `.env` and fill in your Telegram credentials, then:

```bash
docker compose build
docker compose up
```

The first build downloads the vision model (~2.5 GB) and bakes it into the image. After that, starts in seconds.

---

## Repository Structure

```
Visdeurbel/
│
├── main.py               — main loop: reads stream, runs detector, calls verifier, sends Telegram
├── detector.py           — OpenCV MOG2 motion detection and blob filtering
├── verifier.py           — sends snapshot to llama-server, returns fish/no-fish
├── notifier.py           — Telegram bot integration
├── config.py             — all tunable settings (thresholds, URLs, cooldown, etc.)
│
├── Dockerfile            — container for the detector (Python + OpenCV)
├── Dockerfile.llm        — container for the LLM (llama.cpp server-cuda + model weights)
├── docker-compose.yml    — wires the two containers together with GPU passthrough and health check
│
├── models/               — place a fine-tuned granite-vision.gguf here to use it in Docker
│                           (if absent, the build downloads the base model from HuggingFace)
│
├── prompt-tuning/        — prompt experiments from Phase 2
│   └── prompt-test-results.md
│
└── model-tuning/         — fine-tuning pipeline from Phase 3
    ├── prepare_dataset.py — scans labeled image folders, creates train/val JSONL split
    ├── train_fast.py      — recommended training script (~40 min, fp16 LoRA, no eval pauses)
    ├── train.py           — original training script (with per-epoch eval, slow)
    ├── inference.py       — test the fine-tuned adapter on single images or the full val set
    ├── export_gguf.py     — merges the LoRA adapter and prepares it for GGUF conversion
    ├── night_run.bat      — convenience script: train then evaluate, safe to run before sleep
    ├── encountered_issues.md — log of every issue hit during setup, with causes and fixes
    └── how-to-custom-train.md — explainer on LoRA, QLoRA, and the training setup
```

---

## If You're Trying to Reproduce This

The path was not smooth. Setting up LoRA fine-tuning of a vision-language model on a consumer Windows machine with a brand-new GPU architecture involved 13 documented issues — wrong PyTorch builds, quantization incompatibilities, 11-hour training runs caused by a single config flag, and a GPU that locked the system under sustained load.

All of it is logged in [model-tuning/encountered_issues.md](model-tuning/encountered_issues.md), with the exact error, the cause, and what fixed it. If you're attempting something similar, it's worth a read before you start.

---

## Blog Series

The full story with technical detail is documented in three posts:

- **Part 1** — Motion detection with OpenCV
- **Part 2** — Adding a vision LLM as a second-pass filter
- **Part 3** — Fine-tuning Granite Vision on our own footage
