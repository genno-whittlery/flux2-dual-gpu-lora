# Dual-GPU FLUX.2 LoRA training in musubi-tuner

`flux2_dual_gpu.py` — drop into `<musubi-tuner>/src/musubi_tuner/flux_2/`. Distributes the FLUX.2 transformer across two CUDA devices with one PCIe boundary mid-`single_blocks`, and routes per-layer LoRA modules to the device of the layer they wrap. Default behaviour unchanged when `FLUX2_DUAL_GPU` is unset.

## Why this port is clean

- Text encoders (Mistral 3 / Qwen 3) are pre-cached to disk via `flux_2_cache_text_encoder_outputs.py`, so the inner training loop never touches them — **no TE-on-CPU runtime patch needed**.
- `Flux2.move_to_device_except_swap_blocks` is the single canonical transformer-wide device-placement hook.
- `LoRAModule.apply_to` is a 3-line `forward` monkeypatch; a one-line device snapshot + lazy forward shim covers per-LoRA routing — except `apply_to`'s bound-method indirection makes instance-level shimming unreliable, so the device pin runs inline at the top of `LoRAModule.forward`.

## Integration

Three integration edits in `hv_train_network.py` (DDP-bypass when `FLUX2_DUAL_GPU=true`, CPU loading device, `device_placement=[False]` in the `accelerator.prepare` branch) and one inlined device-pin in `networks/lora.py`. The exact edits live in the raw patcher: [`patches/musubi/patch_musubi_dual_gpu.py`](../../patches/musubi/patch_musubi_dual_gpu.py). Hook-point analysis, complexity comparison, validation plan: [`docs/porting-musubi-tuner.md`](../../docs/porting-musubi-tuner.md).

## Env vars

| Variable | Default | Effect |
|---|---|---|
| `FLUX2_DUAL_GPU` | `false` | Set `true` to enable the dual-GPU path |
| `FLUX2_DUAL_GPU_SPLIT_AT` | `n_single // 2` (= 24) | Override the `single_blocks` split index |

## Status — ✅ validated end-to-end (2× RTX 5090, 2026-05-11)

10/10 steps on the sumi v8 dataset (32 imgs @ 512²) with `--fp8_base --fp8_scaled --gradient_checkpointing`, loss 0.526 → 0.545 monotonic, **2.22 s/it sustained** — ~21% faster than ai-toolkit on the same hardware (different attention path + LoRA wrapper overhead), both checkpoints saved. Split shape matches the other ports exactly: 20.7 GB cuda:0 / 12.6 GB cuda:1. Consume via the fork branch [`genno-whittlery/musubi-tuner:dual-gpu-flux2`](https://github.com/genno-whittlery/musubi-tuner/tree/dual-gpu-flux2); upstream PR to kohya-ss/musubi-tuner pending.

## Launch

```bash
export FLUX2_DUAL_GPU=true
export CUDA_VISIBLE_DEVICES=0,1
# (text encoder outputs already cached via flux_2_cache_text_encoder_outputs.py)
accelerate launch --num_processes=1 src/musubi_tuner/flux_2_train_network.py \
    --fp8_base --fp8_scaled --gradient_checkpointing \
    --network_module networks.lora --network_dim 16 \
    ...
```

`--num_processes=1` — this is *model* parallelism (both GPUs cooperate on one training step), not data parallelism.
