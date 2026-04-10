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

---

## 8. `eval_strategy="epoch"` turned a 40-minute training run into 11 hours

**When:** First complete training run with `train.py` using `eval_strategy="epoch"`.

**Symptom:** Training completed (228 steps, ~10s/step = ~38 minutes of actual training), but the total wall-clock time was **11 hours 9 minutes**. The progress bar showed the training steps moving normally, but enormous pauses appeared between epochs.

**Cause:** `eval_strategy="epoch"` runs the full validation set after every epoch. With 74 validation samples each requiring a full forward pass on an fp16 model (no quantization, ~4 GB), each eval pass took **~3 hours**. Three epochs = three eval passes = ~9 hours of eval on top of ~38 minutes of training.

The slowness wasn't compute-bound — it was memory-bound. The full fp16 model leaves little VRAM headroom, and validation forward passes (no gradient checkpointing) caused repeated GPU↔RAM pressure.

**Fix:** Removed eval from the training loop entirely. Created `train_fast.py` with:
```python
eval_strategy="no",
save_strategy="no",
```
Run evaluation separately after training with `inference.py --eval`. Total training time: ~40 minutes.

**Lesson:** For consumer GPU fine-tuning of vision models, separate training from evaluation. In-loop eval has the same memory footprint as training but without gradient checkpointing savings, making it disproportionately slow. Evaluate once after training is done.

---

## 9. RTX 5070 Ti (Blackwell) locks up the system under training load

**When:** Every training attempt launched programmatically (as a background process) on a fresh Windows 11 install with an RTX 5070 Ti.

**Symptom:** System becomes completely unresponsive within minutes of training start. Hard reboot required. This happened consistently with fp16 training, QLoRA attempts, and Unsloth runs.

**Cause (likely):** The RTX 5070 Ti is Blackwell architecture (SM_120), released February 2025. PyTorch + bitsandbytes support for Blackwell is still immature on Windows as of early 2026. When background-launched training processes are killed, GPU memory is not reliably released — subsequent runs start with an already-pressured GPU state, causing memory spill into system RAM and OS-level lockups.

One confirmed run **did complete successfully** (the 11-hour `train.py` run): it was launched directly from a user terminal on a clean boot, with no prior killed processes. This confirms the hardware can handle it — the instability is in how VRAM state accumulates across killed/relaunched processes.

**Mitigations applied:**
- Added `torch.cuda.set_per_process_memory_fraction(0.75)` to cap VRAM at 12 GB, forcing a clean OOM crash instead of silent RAM spill
- Switched to `train_fast.py` to minimize total runtime (~40 min vs 11 hours)
- Recommended running only on a **fresh reboot** with all other GPU apps closed
- Recommended running from the user's own terminal (not programmatically), so Ctrl+C releases GPU memory cleanly

**Lesson:** New GPU architectures on Windows need time for the software ecosystem to catch up. If you're hitting mysterious lockups with an RTX 40/50-series card, check whether your PyTorch build explicitly targets your CUDA compute capability. For Blackwell, use `--index-url https://download.pytorch.org/whl/cu128`.

---

## 10. `ValueError: Can not map tensor 'image_newline'` during GGUF conversion

**When:** Running `convert_hf_to_gguf.py` against the merged model output from `export_gguf.py`.

**Error:**
```
ValueError: Can not map tensor 'image_newline'
```

**Cause:** `model.merge_and_unload()` on a `LlavaNextForConditionalGeneration` saves the entire multimodal model, including the SigLIP vision tower, the multimodal projector, and the `image_newline` embedding tensor. The GGUF converter only understands `GraniteForCausalLM` (the language backbone), not the LLaVA-Next scaffolding around it.

**Fix:** Save only the language backbone, not the full model. The backbone is nested at `model.model.language_model` (a `GraniteModel`) and the LM head is at `model.lm_head`. Assemble a clean `GraniteForCausalLM` from these two parts before saving:

```python
from transformers import GraniteForCausalLM

backbone = model.model.language_model
lm_head  = model.lm_head

causal_lm = GraniteForCausalLM(backbone.config)
causal_lm.model.load_state_dict(backbone.state_dict())
causal_lm.lm_head.load_state_dict(lm_head.state_dict())
causal_lm.save_pretrained("output/granite-fish-lora-merged")
```

The mmproj (vision encoder) does not need to be touched — LoRA only trained the language backbone.

---

## 11. `ERROR: Model GraniteModel is not supported` during GGUF conversion

**When:** After first fix attempt (saving `model.model.language_model` directly).

**Error:**
```
ERROR:hf-to-gguf:Model GraniteModel is not supported
```

**Cause:** `model.model.language_model` is a `GraniteModel` (the bare transformer stack, no LM head). The converter requires a `GraniteForCausalLM` — the full causal LM wrapper including the output projection head (`lm_head`). Saving `GraniteModel` directly produces a config with `architectures: ["GraniteModel"]` which the converter rejects.

**Fix:** Same as issue 10 above — assemble `GraniteForCausalLM` explicitly from the backbone + head before saving. The `export_gguf.py` script handles this correctly.

---

## 12. `q4_k_m` is not a valid `--outtype` for `convert_hf_to_gguf.py`

**When:** Running GGUF conversion with `--outtype q4_k_m`.

**Error:**
```
convert_hf_to_gguf.py: error: argument --outtype: invalid choice: 'q4_k_m'
(choose from f32, f16, bf16, q8_0, tq1_0, tq2_0, auto)
```

**Cause:** `convert_hf_to_gguf.py` only converts to a limited set of types. `q4_k_m` is a k-quant format that requires a separate `llama-quantize` post-processing step. The converter itself doesn't support it directly.

**Fix:** Use `q8_0` as the output type. It is supported directly by the converter, halves the file size versus fp16, and runs comfortably in 16 GB VRAM for a 2B model. If you need smaller, convert to `q8_0` first and then run `llama-quantize` to get `q4_k_m`.

---

## 13. `ghcr.io/ggerganov/llama.cpp:server-cuda` not found

**When:** `docker compose build` for the `llm` service.

**Error:**
```
ERROR: ghcr.io/ggerganov/llama.cpp:server-cuda: not found
failed to resolve source metadata for ghcr.io/ggerganov/llama.cpp:server-cuda
```

**Cause:** The llama.cpp project moved its GitHub organisation from `ggerganov` to `ggml-org`. All container images are now published under `ghcr.io/ggml-org/llama.cpp`. The old `ggerganov` registry returns `manifest unknown` for all tags.

**Fix:** Update `Dockerfile.llm`:
```dockerfile
FROM ghcr.io/ggml-org/llama.cpp:server-cuda
```

**Also:** The `requirements-convert_hf_to_gguf.txt` file pins `torch~=2.6.0`, which conflicts with a newer PyTorch install. Instead of installing from that file, install only the packages actually needed by the converter:
```bash
pip install gguf transformers sentencepiece protobuf
```
