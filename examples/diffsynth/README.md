# Dual-GPU FLUX.2 LoRA training in DiffSynth-Studio

Drop-in helper for [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)'s FLUX.2 LoRA training scripts (`examples/flux2/model_training/`). Splits `Flux2DiT` across two CUDA devices so the model fits on pairs of 24 GB consumer GPUs (2× RTX 3090, 2× RTX 4090, 2× RTX 5090).

## Why this port is small

- `Flux2DiT` (in `diffsynth/models/flux2_dit.py`) is structurally identical to the diffusers `Flux2Transformer2DModel`. Same field names, same forward shape.
- DiffSynth uses **PEFT** for LoRA injection (`from peft import LoraConfig, inject_adapter_in_model` in `diffsynth/diffusion/training_module.py`). PEFT places LoRA params on the base layer's device automatically, so per-layer routing is free.
- The forward isn't overridden — pre-hooks bridge devices at the split point and at `norm_out` (the boundary back to cuda:0).

Total surface: one ~130-line helper file. No patches to DiffSynth itself required for the helper-based path.

## Usage

### 1. Copy the helper

```bash
curl -O https://raw.githubusercontent.com/genno-whittlery/flux2-dual-gpu-lora/main/examples/diffsynth/flux2_dual_gpu_diffsynth.py
```

### 2. Add 2 lines to your training script

In `examples/flux2/model_training/train.py` (or your derivative), after the `Flux2ImageTrainingModule` is built and LoRA injection has happened (which is inside `__init__` via `switch_pipe_to_training_mode`):

```python
from flux2_dual_gpu_diffsynth import enable_flux2_dual_gpu
enable_flux2_dual_gpu(training_module.pipe.dit)
```

If you use `accelerator.prepare(training_module, ...)`, pass `device_placement=False` for the training module so Accelerate doesn't flatten the distributed transformer back onto a single device:

```python
training_module = accelerator.prepare(
    training_module,
    device_placement=False,  # already distributed by enable_flux2_dual_gpu
)
```

### 3. Launch

```bash
export FLUX2_DUAL_GPU=true
# Optional: override the split index (default is num_single_blocks // 2)
# export FLUX2_DUAL_GPU_SPLIT_AT=24

accelerate launch --num_processes=1 examples/flux2/model_training/train.py \
    --model_paths "..." \
    --trainable_models "dit" \
    --lora_base_model "dit" \
    --lora_rank 16 \
    --task "sft" \
    ...
```

**Critical:** `--num_processes=1`. This is *model* parallelism (both GPUs cooperate on a single training step), not data parallelism (which would spawn one process per GPU). With `--num_processes=2` Accelerate would try to give each process its own GPU and the model wouldn't be split.

### 4. Verify the split landed

Add a print right after `enable_flux2_dual_gpu`:

```python
import torch
print(f"dit split_at = {getattr(training_module.pipe.dit, '_flux2_dual_gpu_split_at', 'NOT SPLIT')}")
for gpu in range(torch.cuda.device_count()):
    used_gb = torch.cuda.memory_allocated(gpu) / 1024**3
    print(f"  GPU{gpu}: {used_gb:.2f} GB allocated")
```

Expected on 2× RTX 5090, FLUX.2-dev fp8, rank 16:

```
dit split_at = 24
  GPU0: ~21 GB allocated
  GPU1: ~12 GB allocated
```

If both GPUs aren't being used, either `FLUX2_DUAL_GPU` is unset, `device_placement=False` was missed, or `enable_flux2_dual_gpu` was called before LoRA injection (call it after).

## Validation status

- **The split shape is validated end-to-end on ai-toolkit** ([PR #829](https://github.com/ostris/ai-toolkit/pull/829)). The math (16 GB per side for the FLUX.2-dev transformer at fp8, ~18 MB activation per PCIe boundary crossing) applies identically here.
- **The diffsynth-specific port is not yet end-to-end tested** — DiffSynth has its own training loop and accelerate config that may have edge cases. Please open an issue if you hit one.

## DiffSynth-specific considerations

- **DiffSynth's training pipeline supports FSDP**. The dual-GPU split here is *not* FSDP — it's a static model-parallel split across exactly two GPUs. Disable FSDP in your accelerate config (`fsdp: false` or no FSDP block) when using this helper.
- **DiffSynth's `pipe.dit` is loaded via `Flux2ImagePipeline.from_pretrained`** with the `device` argument. Pass `device="cpu"` to that call if you can — the helper will redistribute it across cuda:0/cuda:1. If you pass `device="cuda"`, the helper still works (it overrides placement after the fact), but you'll spike memory on cuda:0 briefly during load before the split lands.
- **`offload_models` / `fp8_models` in the training config** are orthogonal to this helper. Use `fp8_models=["dit"]` if you want fp8 weights (recommended on 24 GB cards). `offload_models` is for moving inactive models (vae / text encoder) to CPU — also orthogonal, and works with the split.

## Upstream PR

Not yet submitted. The DiffSynth-Studio repo is modelscope/Alibaba — primary language is Mandarin in code comments / issue threads. A bilingual PR + a brief Chinese-language `README_CN.md` in the same directory would be the right shape for that community.
