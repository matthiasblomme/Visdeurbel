# Issues Encountered During QLoRA Fine-Tuning Setup

A log of every issue we hit while setting up LoRA/QLoRA fine-tuning of IBM Granite Vision 3.2-2B on a consumer Windows machine. Useful as blog content for others attempting the same.

---

## 1. `ValueError: Your setup doesn't support bf16/gpu`

**When:** First run of `train.py` on an NVIDIA RTX 5070 Ti (Blackwell architecture).

**Error:**
```
ValueError: Your setup doesn't support bf16/gpu. You need to assign use_cpu if you want to train the model on CPU.
```

**Cause:** The 5070 Ti (Blackwell / SM_120) wasn't exposing `bfloat16` support through the installed PyTorch/CUDA combination. This can happen with very new GPU architectures where driver/toolkit support is still maturing.

**Fix:** Switched all three occurrences of `bfloat16` to `float16`:
- `BitsAndBytesConfig(bnb_4bit_compute_dtype=torch.float16)` (was `torch.bfloat16`)
- `model.from_pretrained(torch_dtype=torch.float16)` (was `torch.bfloat16`)
- `TrainingArguments(fp16=True)` (was `bf16=True`)

`float16` works fine for QLoRA training — no meaningful quality difference for a binary classification task.

---

## 2. Unauthenticated Hugging Face requests (rate-limited downloads)

**When:** Downloading the Granite Vision model weights from Hugging Face Hub.

**Warning:**
```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

**Fix:** Created a `.env` file (gitignored) with `HF_TOKEN=hf_...` and added `python-dotenv` to load it at the top of `train.py` and `inference.py`:
```python
from dotenv import load_dotenv
load_dotenv()
```

The `huggingface_hub` library automatically picks up the `HF_TOKEN` environment variable.

---

## 3. `ValueError: Mismatch in image token count` (truncation)

**When:** First training batch — the processor prepares the input but the model rejects it.

**Error:**
```
ValueError: Mismatch in `image` token count between text and `input_ids`.
Got ids=[470] and text=[2052]. Likely due to `truncation='max_length'`.
Please disable truncation or increase `max_length`.
```

**Cause:** `MAX_SEQ_LEN` was set to 512 tokens. The LLaVA-Next processor generates ~2048 image tokens for a single 640x360 image. With `truncation=True` and `max_length=512`, most image tokens were being chopped off, causing a mismatch between what the text expected and what `input_ids` contained.

**Fix:** Increased `MAX_SEQ_LEN` from 512 to 4096:
```python
MAX_SEQ_LEN = 4096  # must fit image tokens (~2048) + prompt + answer
```

**Lesson:** Vision-language models tokenize images into hundreds or thousands of tokens. Always check how many tokens your image produces before setting `max_length`. You can inspect this with:
```python
encoding = processor(images=image, text=text, return_tensors="pt")
print(encoding["input_ids"].shape)  # check token count
```

---

## 4. `TypeError: 'NoneType' object is not iterable` on `image_sizes`

**When:** Forward pass during first training step.

**Error:**
```
File ".../modeling_llava_next.py", line 381, in get_image_features
    for imsize in image_sizes
                  ^^^^^^^^^^^
TypeError: 'NoneType' object is not iterable
```

**Cause:** LLaVA-Next requires `image_sizes` (the original image dimensions) to properly handle multi-resolution image encoding. The processor outputs this field, but our `FishDataset.__getitem__` wasn't including it in the returned dictionary — so the model received `image_sizes=None`.

**Fix:** Added `image_sizes` to the dataset output:
```python
image_sizes = encoding["image_sizes"].squeeze(0)
# ...
return {
    "input_ids":      input_ids,
    "attention_mask": attention_mask,
    "pixel_values":   pixel_values,
    "image_sizes":    image_sizes,   # ← was missing
    "labels":         labels,
}
```

**Lesson:** When fine-tuning vision-language models, check what fields the processor outputs and make sure ALL of them are passed through to the model. Different architectures need different fields (`image_sizes` for LLaVA-Next, `pixel_attention_mask` for others, etc.).

---

## 5. PyTorch installed without CUDA (CPU-only)

**When:** Training ran on CPU instead of GPU — no GPU load visible.

**Diagnosis:**
```python
import torch
print(torch.__version__)        # 2.10.0+cpu  ← the "+cpu" suffix is the giveaway
print(torch.cuda.is_available()) # False
```

**Cause:** `pip install torch` defaults to the CPU-only build. For CUDA support, you need to install from PyTorch's CUDA-specific index. The RTX 5070 Ti (Blackwell) specifically requires CUDA 12.8+.

**Fix:**
```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

The `--force-reinstall` is important — without it, pip sees the version as already satisfied and skips the download.

**How to verify:**
```python
import torch
print(torch.__version__)          # should show 2.x.x+cu128
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 5070 Ti
```

**Lesson:** Always verify CUDA after installing PyTorch, especially on new hardware. The `+cpu` or `+cu1xx` suffix in `torch.__version__` tells you immediately which build you have.

---

## 6. `RuntimeError: self and mat2 must have the same dtype, but got Half and Byte`

**When:** Forward pass during training with 4-bit quantization (bitsandbytes).

**Error:**
```
RuntimeError: self and mat2 must have the same dtype, but got Half and Byte
```

**Cause:** When loading the model with `BitsAndBytesConfig(load_in_4bit=True)`, bitsandbytes quantizes *all* linear layers — including those in the SigLIP vision encoder. The SigLIP attention pooling head uses `F.multi_head_attention_forward`, which doesn't support mixed 4-bit/fp16 operations.

**What we tried (didn't work):**
- `llm_int8_skip_modules=["vision_tower"]` — despite the documentation, this didn't prevent quantization of the vision tower with the current bitsandbytes version.
- `model.model.vision_tower = model.model.vision_tower.to(torch.float16)` — doesn't undo bitsandbytes' `Linear4bit` module replacements; the `.to()` call is silently ignored on quantized layers.

**Fix:** Dropped 4-bit quantization entirely. The 2B model in fp16 is only ~4 GB, which fits comfortably in 16 GB VRAM. Used plain LoRA instead of QLoRA:
```python
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map={"": 0},     # force all on GPU
    torch_dtype=torch.float16,
)
```

**Lesson:** Vision-language models with separate vision encoders (LLaVA-Next, LLaVA, etc.) are tricky to quantize. The vision encoder often uses architecture patterns (e.g., `MultiheadAttention`) that don't support 4-bit weights. For small models (≤3B), fp16 without quantization is often the simplest path. For larger models, you may need to manually dequantize the vision tower or use frameworks like `unsloth` that handle this correctly.

---

## 7. Training stuck at 0% / extremely slow (~5 min/step)

**When:** First successful training launch with `device_map="auto"` in fp16.

**Symptom:** Training appeared to start (progress bar showed 0/228) but each step took 5+ minutes. Left overnight, the GPU was peaking for 12+ hours with minimal progress.

**Cause:** Two compounding issues:
1. `device_map="auto"` was silently offloading some model layers to CPU, causing constant GPU↔CPU data transfers.
2. `padding="max_length"` with `max_length=4096` was padding every sample to 4096 tokens, even though actual content was ~2500 tokens — wasting 40% of compute per sample.

**Fix:**
1. Force all layers on GPU: `device_map={"": 0}` instead of `"auto"`
2. Switched to TRL's `SFTTrainer` with `padding=True` (pads to longest in batch, not max_length)
3. Enabled gradient checkpointing: `gradient_checkpointing=True`

Result: ~10 seconds/step → ~38 minutes total for 3 epochs.

**Lesson:** `device_map="auto"` is designed for models that *don't fit* in GPU memory. For a 2B fp16 model on a 16 GB GPU, it's counterproductive — it adds overhead from the auto-sharding logic. Use `{"": 0}` to pin everything to GPU 0 when the model fits.
