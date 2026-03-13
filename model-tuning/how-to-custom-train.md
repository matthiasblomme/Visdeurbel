# Fine-tuning a Vision LLM on Your Own Data: A Practical Guide

*How we taught IBM Granite Vision to recognize fish in an underwater camera feed — without a data center.*

---

## The Problem with Prompt Engineering

When you first connect a vision model to a real-world detection task, prompt engineering feels like magic. You write "Is there a fish in this image? Answer yes or no." and it works — sometimes. But as your use case gets more specific, prompting hits a ceiling.

The model was trained on the whole internet. It has a general idea of what a fish looks like, but it has never seen the murky, low-contrast footage of your specific riverbed camera. It doesn't know that the sediment clouds and water shimmer are *not* fish. It doesn't know the particular shapes and colors of the species that swim through your stream.

Fine-tuning is how you close that gap.

---

## What Is Fine-Tuning?

A large language model (or vision-language model) is pre-trained on massive, general datasets. Fine-tuning takes that pre-trained model and continues training it on a small, curated dataset specific to your task.

Think of it like this: the base model is a university graduate with broad knowledge. Fine-tuning is on-the-job training. You don't retrain them from scratch — you just teach them the specifics of your particular job.

In our case: show the model hundreds of labeled frames from the Visdeurbel stream, each tagged as "fish" or "no fish," and it learns the visual patterns that distinguish them in *this specific context*.

---

## Why Not Just Fine-Tune the Whole Model?

Granite Vision 3.2-2B has about 2 billion parameters. Training all of them would require:

- A multi-GPU setup (think: 4× A100s, ~$10/hour in the cloud)
- Days of training time
- Saving a 4+ GB checkpoint

For most practical fine-tuning tasks, you don't need this. Enter **LoRA**.

---

## LoRA: Fine-Tuning Without Full Retraining

**LoRA** stands for *Low-Rank Adaptation*. The core idea is elegant:

Instead of updating all the weights in the model's attention layers, you inject small *adapter matrices* alongside them. These adapters are tiny — maybe 0.3% of the total parameters — but they're enough to shift the model's behavior significantly for your task.

Here's the math intuition: a weight matrix `W` (say, 4096×4096) has millions of parameters. LoRA approximates the *change* to that matrix as a product of two smaller matrices:

```
ΔW ≈ A × B
     (4096×16) × (16×4096)
```

Where 16 is the **rank** (`r`). You only train `A` and `B`. The original `W` stays frozen.

The savings are dramatic:
- Full fine-tune: ~400M parameters to train
- LoRA (rank=16): ~2M parameters to train

And crucially: the LoRA adapter is a separate file (~50–200 MB) that sits on top of the base model. You can share it, version it, and swap it in/out at runtime.

---

## QLoRA: Making It Fit on a Consumer GPU

LoRA reduces the *trainable* parameters, but you still need to load the full model into GPU memory to do the forward pass. For a 2B model in 16-bit precision, that's ~4 GB of VRAM — tight on most consumer cards.

**QLoRA** adds one more trick: quantize the base model weights to **4-bit** before loading. The model weights are compressed from 16 bits to 4 bits per parameter using a format called NormalFloat4 (NF4), which is optimized for the distribution of neural network weights.

The result: the 2B model fits in ~3–4 GB of VRAM. Combined with LoRA's small adapter, you can fine-tune Granite Vision on a single consumer GPU with 12 GB VRAM (e.g., RTX 3060/3080/4070).

The training computation still happens in bfloat16 precision — only the stored weights are quantized.

---

## Our Dataset

We collected labeled frames directly from the Visdeurbel stream over several weeks. Two categories:

- **`images/fish/`** — frames where a fish is clearly visible (true positives from our existing detection pipeline, manually verified)
- **`images/no_fish/`** — frames that triggered the motion detector but contained no fish (false positives: sediment clouds, light changes, plant movement)

Images are 640×360 JPEG, the same resolution the stream is processed at.

We used an 80/20 train/validation split (see `prepare_dataset.py`), with random shuffling seeded for reproducibility.

Even 100–200 labeled images per class is enough to meaningfully shift model behavior. More is better, but the returns diminish past a few hundred.

---

## The Training Setup

### Model: `ibm-granite/granite-vision-3.2-2b`

IBM's Granite Vision is a compact (2B parameter) vision-language model built for instruction following. It's based on the LLaVA-Next architecture: a visual encoder (processes the image into tokens) feeds into a language model backbone (generates text). We apply LoRA only to the language backbone — the vision encoder stays frozen.

### Task framing

We format each training sample as a chat conversation:

```
User: [image] This is a frame from an underwater riverbed camera.
      Is there a fish visible in this image? Answer only yes or no.
Assistant: yes
```

This mirrors exactly the prompt used in the production `verifier.py`, so the fine-tuned adapter integrates seamlessly without changing any production code.

### Key hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `lora_rank` | 16 | Balances capacity vs. efficiency; 8 for very small datasets, 32 for larger |
| `lora_alpha` | 32 | Scaling factor = 2× rank is a common starting point |
| `lora_dropout` | 0.05 | Light regularization to prevent overfitting on small datasets |
| `learning_rate` | 2e-4 | Standard for LoRA; higher than full fine-tune because adapters train from random init |
| `epochs` | 3 | Enough for a small dataset; watch val loss to catch overfitting |
| `batch_size` | 1 + grad_accum=4 | Effective batch of 4; keeps VRAM usage low |
| `max_seq_len` | 512 | More than enough for image tokens + short prompt + "yes"/"no" |

---

## Running the Training Pipeline

### 1. Install dependencies

```bash
cd model-tuning
pip install -r requirements.txt
```

### 2. Add your images

```
model-tuning/
└── images/
    ├── fish/      ← copy fish frames here
    └── no_fish/   ← copy false-positive frames here
```

You can copy directly from the main project's `snapshots/` folder:
```bash
cp ../snapshots/fish/*.jpg images/fish/
cp ../snapshots/false/*.jpg images/no_fish/
```

### 3. Prepare the dataset

```bash
python prepare_dataset.py
```

Output:
```
  fish images   : 120
  no_fish images: 85

Dataset written to dataset/
  train: 164 samples  (96 fish, 68 no_fish)
  val  :  41 samples  (24 fish, 17 no_fish)
```

### 4. Train

```bash
python train.py
```

You'll see output like:
```
Loading ibm-granite/granite-vision-3.2-2b in 4-bit...
trainable params: 2,097,152 || all params: 668,442,624 || trainable%: 0.3137
Starting training...
{'loss': 0.6821, 'epoch': 1.0}
{'eval_loss': 0.4103, 'epoch': 1.0}
{'loss': 0.3244, 'epoch': 2.0}
...
LoRA adapter saved to output/granite-fish-lora/
```

Training 164 samples for 3 epochs takes roughly **10–20 minutes** on a modern consumer GPU.

### 5. Evaluate

```bash
# Compare base model vs fine-tuned on your validation set
python inference.py --eval --no-adapter   # base model
python inference.py --eval                # fine-tuned
```

### 6. Test on a single image

```bash
python inference.py images/fish/some_frame.jpg
# Result: yes
```

---

## Interpreting the Results

When evaluating fish detection, **recall matters more than precision**. Missing a fish (false negative) means a missed notification. Flagging a non-fish frame (false positive) just means a slightly annoying extra Telegram message that gets filtered by the LLM verifier downstream.

| Metric | What it means for us |
|--------|---------------------|
| **Accuracy** | Overall correct rate — good headline number |
| **Precision** | Of frames we called "fish", how many really were? |
| **Recall** | Of all fish frames, how many did we catch? |
| **F1** | Harmonic mean of precision and recall — use this to compare models |

A well-tuned model on this task should reach F1 > 0.85 with 200+ labeled samples.

---

## Integrating the Adapter with the Production Pipeline

The fine-tuned LoRA adapter is a drop-in replacement for the base model in `verifier.py`. The only change needed is to load the adapter on top of the base model before serving:

```python
from peft import PeftModel

model = LlavaNextForConditionalGeneration.from_pretrained(BASE_MODEL_ID, ...)
model = PeftModel.from_pretrained(model, "model-tuning/output/granite-fish-lora")
```

Alternatively, serve it via Ollama by first merging the adapter weights back into the base model:

```python
merged = model.merge_and_unload()
merged.save_pretrained("output/granite-fish-merged")
```

Then convert the merged model to GGUF format for llama.cpp using the standard conversion tools.

---

## What's Next

- **More data** — the single biggest lever. Every 2× increase in labeled samples improves accuracy.
- **Data augmentation** — horizontal flips, brightness jitter, and slight crops on the training images to improve robustness.
- **Active learning** — use the current model's uncertain predictions (near 50/50 confidence) to guide what to label next.
- **Rank tuning** — if overfitting is a problem, reduce `lora_rank` to 8. If the model plateaus early, try rank 32.
