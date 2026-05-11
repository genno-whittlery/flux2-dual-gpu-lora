"""Dual-GPU model-parallel for FLUX.2 in HuggingFace diffusers.

Drop-in helper for the official diffusers FLUX.2 training scripts
(``examples/dreambooth/train_dreambooth_lora_flux2*.py``) that splits
``Flux2Transformer2DModel`` across two CUDA devices at the
``single_transformer_blocks`` midpoint. Enables FLUX.2 LoRA training on
pairs of 24+ GB consumer GPUs (2× RTX 3090, 2× RTX 4090, 2× RTX 5090) —
on a single 24 GB card the FLUX.2 transformer can't fit alongside its
activations even with WDDM paging.

Companion to the validated ai-toolkit patch
(https://github.com/genno-whittlery/flux2-dual-gpu-lora) and the
musubi-tuner port. This helper is intentionally minimal because
diffusers + PEFT make per-layer LoRA routing automatic.

Usage in a training script::

    from flux2_dual_gpu_diffusers import enable_flux2_dual_gpu

    # ...load transformer normally...
    transformer = Flux2Transformer2DModel.from_pretrained(...)

    # ...PEFT LoRA adapter added normally...
    transformer.add_adapter(lora_config)

    # Activate the split (env-gated; no-op when FLUX2_DUAL_GPU is unset).
    transformer = enable_flux2_dual_gpu(transformer)

    # When calling accelerator.prepare, skip device placement for the
    # transformer — it's already distributed across cuda:0/cuda:1.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler,
        device_placement=[False, True, True, True],
    )

Then::

    export FLUX2_DUAL_GPU=true
    export FLUX2_DUAL_GPU_SPLIT_AT=24  # optional; default is num_single // 2

    accelerate launch --num_processes=1 train_dreambooth_lora_flux2.py ...

Note the ``--num_processes=1`` — this is *model* parallelism, not data
parallelism. Both GPUs work together on a single training step; we
don't want Accelerate to spawn a process per GPU.

How it works:

The diffusers ``Flux2Transformer2DModel.forward`` is long and has
multiple branches (KV-cache modes, gradient-checkpointing variants),
so this helper doesn't override the forward. Instead it:

1. Manually places the transformer's submodules:
   pre-blocks scaffolding + ``transformer_blocks`` (double) +
   ``single_transformer_blocks[:split_at]`` → cuda:0;
   ``single_transformer_blocks[split_at:]`` → cuda:1;
   output layers (norm_out + proj_out) → cuda:0.
2. Registers ``forward_pre_hook``\\s on the split-point block and on
   ``norm_out`` to bridge devices: tensors crossing into the cuda:1
   region get moved to cuda:1; the result coming back for the output
   layers gets moved to cuda:0. Pre-hooks recursively walk nested
   tuples/lists/dicts (RoPE embeddings + attention kwargs need that).
3. Leaves PEFT-injected LoRA layers in place — PEFT places LoRA params
   on the base layer's device automatically, so per-layer routing is
   free as long as the base layout is correct.

One PCIe boundary per forward (~18 MB activation at 512², ~1 ms on
PCIe Gen5 x16) — same shape as the validated ai-toolkit version.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn


# ─── Env-gated public surface ───────────────────────────────────────────────

def is_dual_gpu_enabled() -> bool:
    """True iff ``FLUX2_DUAL_GPU=true`` in the environment."""
    return os.getenv("FLUX2_DUAL_GPU", "false").lower() == "true"


def get_split_at(num_single_blocks: int) -> int:
    """Single-blocks split index. Override via ``FLUX2_DUAL_GPU_SPLIT_AT``."""
    override = os.getenv("FLUX2_DUAL_GPU_SPLIT_AT")
    if override is not None:
        return int(override)
    return num_single_blocks // 2


def enable_flux2_dual_gpu(transformer: nn.Module) -> nn.Module:
    """Distribute the FLUX.2 transformer across cuda:0 and cuda:1.

    When ``FLUX2_DUAL_GPU`` is unset this is a no-op pass-through.

    Call after the transformer (and any PEFT LoRA adapters) are loaded
    and BEFORE ``accelerator.prepare`` (or use ``device_placement=False``
    when preparing the transformer specifically).

    Returns the (in-place modified) transformer.
    """
    if not is_dual_gpu_enabled():
        return transformer

    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"FLUX2_DUAL_GPU=true requires ≥2 CUDA devices, found "
            f"{torch.cuda.device_count()}."
        )

    num_single = len(transformer.single_transformer_blocks)
    split_at = get_split_at(num_single)
    if not 0 < split_at < num_single:
        raise RuntimeError(
            f"FLUX2_DUAL_GPU_SPLIT_AT={split_at} out of range "
            f"(transformer has {num_single} single blocks)."
        )

    cuda0 = torch.device("cuda:0")
    cuda1 = torch.device("cuda:1")

    # 1. Place pre-blocks scaffolding + all double_blocks + first half
    # of single_blocks on cuda:0. Output layers stay on cuda:0 too —
    # the second pre-hook below brings hidden_states back from cuda:1.
    transformer.x_embedder.to(cuda0)
    transformer.context_embedder.to(cuda0)
    transformer.time_guidance_embed.to(cuda0)
    transformer.pos_embed.to(cuda0)
    transformer.double_stream_modulation_img.to(cuda0)
    transformer.double_stream_modulation_txt.to(cuda0)
    transformer.single_stream_modulation.to(cuda0)
    for block in transformer.transformer_blocks:
        block.to(cuda0)
    for block in transformer.single_transformer_blocks[:split_at]:
        block.to(cuda0)
    for block in transformer.single_transformer_blocks[split_at:]:
        block.to(cuda1)
    transformer.norm_out.to(cuda0)
    transformer.proj_out.to(cuda0)

    # 2. Register pre-hooks for the cross-device boundaries.
    transformer.single_transformer_blocks[split_at].register_forward_pre_hook(
        _make_device_bridge_hook(cuda1), with_kwargs=True
    )
    transformer.norm_out.register_forward_pre_hook(
        _make_device_bridge_hook(cuda0), with_kwargs=True
    )

    transformer._flux2_dual_gpu_split_at = split_at  # marker for diagnostics
    return transformer


# ─── Internals ──────────────────────────────────────────────────────────────

def _move_to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors in nested tuple/list/dict to ``device``.

    No-op for tensors already on ``device`` (PyTorch's ``Tensor.to`` is
    an identity operation in that case, no kernel launched).
    """
    if torch.is_tensor(obj):
        return obj.to(device) if obj.device != device else obj
    if isinstance(obj, tuple):
        return tuple(_move_to_device(x, device) for x in obj)
    if isinstance(obj, list):
        return [_move_to_device(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _make_device_bridge_hook(target_device: torch.device):
    """Forward pre-hook that moves all tensor inputs to ``target_device``.

    Pre-hooks fire before a module's forward runs; rewriting (args,
    kwargs) here changes what the module receives. Used at the
    cuda:0→cuda:1 split point and again at the cuda:1→cuda:0 return
    boundary for the output layers.
    """
    def hook(module, args, kwargs):
        return (
            _move_to_device(args, target_device),
            _move_to_device(kwargs, target_device),
        )
    return hook
