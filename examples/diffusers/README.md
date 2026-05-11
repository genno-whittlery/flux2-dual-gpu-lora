# Dual-GPU FLUX.2 LoRA training in diffusers

Drop-in helper for HuggingFace [diffusers](https://github.com/huggingface/diffusers)'s FLUX.2 LoRA training scripts (`examples/dreambooth/train_dreambooth_lora_flux2*.py`). Splits the FLUX.2 transformer across two CUDA devices so the model fits on pairs of 24 GB consumer GPUs (2× RTX 3090, 2× RTX 4090, 2× RTX 5090).

## Why this is smaller than the ai-toolkit / musubi-tuner patches

- **LoRA routing is automatic.** diffusers uses PEFT; PEFT places LoRA params on the base layer's device by default. We don't have to patch the LoRA wrapper.
- **Text encoder is offloaded by the existing `--offload` flag.** No runtime patch needed.
- **The transformer forward isn't overridden.** We use PyTorch [`register_forward_pre_hook`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook) to bridge devices at the split point — cleaner than rewriting the (complex, KV-cache-branching) `Flux2Transformer2DModel.forward`.

Total surface: one ~130-line helper file. No upstream patches to diffusers itself.

## Usage

### 1. Copy the helper into your project

```bash
curl -O https://raw.githubusercontent.com/genno-whittlery/flux2-dual-gpu-lora/main/examples/diffusers/flux2_dual_gpu_diffusers.py
```

Or `pip install` from this repo if you prefer (it's a single file with no dependencies beyond torch).

### 2. Add 3 lines to `train_dreambooth_lora_flux2.py`

Find the section where the transformer is loaded and the PEFT LoRA adapter is added (search for `add_adapter` or `LoraConfig`). After both calls, add:

```python
from flux2_dual_gpu_diffusers import enable_flux2_dual_gpu
transformer = enable_flux2_dual_gpu(transformer)
```

Find the `accelerator.prepare(...)` call (likely near the optimizer setup). Change it to skip device placement for the transformer:

```python
transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    transformer, optimizer, train_dataloader, lr_scheduler,
    device_placement=[False, True, True, True],
)
```

The `device_placement=[False, ...]` is the key change — the transformer is already distributed across cuda:0 and cuda:1, and we don't want Accelerate to flatten it onto a single device.

### 3. Launch

```bash
export FLUX2_DUAL_GPU=true
# Optional: override the split index (default is num_single_blocks // 2)
# export FLUX2_DUAL_GPU_SPLIT_AT=24

accelerate launch --num_processes=1 train_dreambooth_lora_flux2.py \
    --pretrained_model_name_or_path="black-forest-labs/FLUX.2-dev" \
    --instance_data_dir=path/to/data \
    --output_dir=path/to/output \
    --rank=16 \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1500 \
    --mixed_precision="bf16" \
    --offload \
    ...
```

**Critical:** `--num_processes=1`. This is *model* parallelism (both GPUs cooperate on a single training step), not data parallelism (which would spawn one process per GPU). If you launch with `--num_processes=2`, Accelerate will try to give each process its own GPU and the model won't be split at all.

### 4. Verify the split landed

Add this near the start of your training loop:

```python
print(f"transformer split_at = {getattr(transformer, '_flux2_dual_gpu_split_at', 'NOT SPLIT')}")
for gpu in range(torch.cuda.device_count()):
    used_gb = torch.cuda.memory_allocated(gpu) / 1024**3
    print(f"  GPU{gpu}: {used_gb:.2f} GB allocated")
```

Expected on 2× RTX 5090, FLUX.2-dev fp8mixed, rank 16:

```
transformer split_at = 24
  GPU0: ~21 GB allocated
  GPU1: ~12 GB allocated
```

If both GPUs aren't being used, either `FLUX2_DUAL_GPU` is unset or `device_placement=[False, ...]` was missed.

## Limitations

- **Not yet end-to-end validated on a diffusers training run.** The same split shape has been validated on ai-toolkit ([PR #829](https://github.com/ostris/ai-toolkit/pull/829)) and the porting plan for musubi-tuner is on a Genno fork awaiting comfy availability.
- **PEFT LoRA layers travel with their base layer's device automatically**, but if you wrap the LoRA with a custom module that intercepts forward, you'll need to ensure your wrapper preserves the device. The default PEFT path handles this transparently.
- **Single-microbatch pipeline-parallel.** Each batch occupies both GPUs in serial; no gpipe / 1F1B microbatching yet.
- **No mixed-precision regression testing**. The helper assumes the transformer's parameters are already in the dtype you want; we use `.to(device)` without changing dtype. If you load in bf16 you stay in bf16.

## How it compares to FSDP2

diffusers' default multi-GPU path uses FSDP2 (`fsdp_transformer_layer_cls_to_wrap`) which **shards** parameters across processes — each GPU holds 1/N of every layer, and parameters are gathered just-in-time during forward. That's a different distribution shape:

| | FSDP2 (built-in) | This helper |
|---|---|---|
| What gets split | every layer, sharded | layers, by index |
| Per-step PCIe traffic | high (all-gather every layer) | one ~18 MB activation at the split point |
| Useful when | many GPUs, very large model | exactly 2 GPUs, model just over single-card capacity |
| Setup cost | `accelerate config` + FSDP yaml | one env var |

For training FLUX.2 LoRAs on two consumer GPUs, this helper is the right shape. For 4+ datacenter GPUs, FSDP2 is the right shape. They solve different problems.
