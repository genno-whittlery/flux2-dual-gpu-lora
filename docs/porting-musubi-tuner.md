# Porting the dual-GPU patch to musubi-tuner

**Target:** [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) — kohya's FLUX.2 LoRA trainer. Recommended in [`trainer-survey.md`](./trainer-survey.md) as the highest-leverage next port target.

**Estimated effort:** 200–300 LOC, smaller than ai-toolkit's 460 because musubi's architecture is structurally cleaner for our purposes.

This is a **plan**, not yet implemented. Validation will need the same 2× RTX 5090 rack used for ai-toolkit.

## Why musubi is cleaner to port

| Concern | musubi-tuner | ai-toolkit (already done) |
|---|---|---|
| **Text encoder during training** | Pre-cached to disk via `flux_2_cache_text_encoder_outputs.py`. Inner training loop loads `ctx_vec` from a `.npy` per sample, never touches Mistral. | TE on GPU at runtime; required runtime patch to pin to `te_device_torch` and override `text_encoder_to`. |
| **LoRA wrapper** | `LoRAModule.apply_to()` is 3 lines (save `org_forward`, replace `forward`, drop `org_module` ref). | `ToolkitNetworkMixin.force_to` + `LoRANetwork.apply_to` + lazy forward-pin shim + `broadcast_and_multiply` align. |
| **Transformer device placement** | One canonical hook: `Flux2.move_to_device_except_swap_blocks(device)`. | Multiple `.to()` call sites scattered across `load_model`, pipeline composition, sample-time hooks. |
| **Forward shape** | `Flux2.forward(x, x_ids, timesteps, ctx, ctx_ids, guidance)` — identical double_blocks → merge → single_blocks → final_layer shape. | Same. |
| **Block-swap interaction** | `enable_block_swap()` already exists for RAM↔GPU paging; mutually exclusive with our cuda:0↔cuda:1 split — need a clean error path. | n/a |

## The four hook points

### 1. Transformer block distribution

**File:** `src/musubi_tuner/flux_2/flux2_models.py`
**Hook:** `Flux2.move_to_device_except_swap_blocks(device)` (line 575).

When `FLUX2_DUAL_GPU=true`, bypass the single-device `self.to(device)` call and instead distribute:
- `img_in`, `txt_in`, `time_in`, `pe_embedder`, `*_modulation*` → cuda:0
- `double_blocks[*]` → cuda:0 (all double blocks)
- `single_blocks[0:SPLIT_AT]` → cuda:0
- `single_blocks[SPLIT_AT:]` → cuda:1
- `final_layer` → cuda:1

`SPLIT_AT` defaults to `num_single_blocks // 2`, overridable via `FLUX2_DUAL_GPU_SPLIT_AT`.

### 2. Forward split-aware override

**File:** `src/musubi_tuner/flux_2/flux2_models.py`
**Hook:** `Flux2.forward()` (line 595).

Insert one `.to(cuda:1)` boundary inside the single_blocks loop at `block_idx == SPLIT_AT`:

```python
for block_idx, block in enumerate(self.single_blocks):
    if block_idx == SPLIT_AT and DUAL_GPU:
        img = img.to("cuda:1")
        pe = pe.to("cuda:1")
        single_block_mod = single_block_mod.to("cuda:1")
    ...
    img = block(img, pe, single_block_mod, attn_params)
```

After the loop, ensure `img` returns to the device of `vec` (cuda:0) for `final_layer`. Wait — `final_layer` lives on cuda:1 in our split. Re-check: simplest is to keep `final_layer` on cuda:0 and move `img` back, mirroring ai-toolkit. Decide during implementation; both work.

Cross-PCIe payload: same ~18 MB activation per forward at 512² as ai-toolkit (architecture is identical).

### 3. LoRA per-layer device routing

**File:** `src/musubi_tuner/networks/lora.py`
**Hook:** `LoRAModule.apply_to()` (line 98).

The current `apply_to`:

```python
def apply_to(self):
    self.org_forward = self.org_module.forward
    self.org_module.forward = self.forward
    del self.org_module
```

After `apply_to`, the LoRAModule's params (`lora_down`, `lora_up`) need to live on the device of `org_module`. Easiest path: before `del self.org_module`, snapshot `self._wrapped_device = next(self.org_module.parameters()).device`. Then have `LoRANetwork` walk all `LoRAModule` children after `apply_to` and call `m.to(m._wrapped_device)`.

Alternative path (less invasive): name-based routing. The lora_name encodes the block index (e.g., `lora_unet_double_blocks_0_img_attn_qkv`); parse the index and route to the device that owns that block. This works without touching `LoRAModule.apply_to`.

Verify in implementation:
- The LoRAModule's `forward` calls `self.lora_down(x)` and `self.lora_up(lx)`. `x` arrives on the device of the wrapped layer (cuda:0 or cuda:1 depending on split), so `lora_down.weight` must be on the same device. Once placement is correct, no further routing logic is needed.

### 4. Block-swap mutex

**File:** `src/musubi_tuner/flux_2/flux2_models.py`
**Hook:** `Flux2.enable_block_swap()` (line 500).

If `FLUX2_DUAL_GPU=true` and `blocks_to_swap > 0`, raise a clear error: "dual-GPU split and block-swap are mutually exclusive; the dual-GPU path already moves blocks off-GPU by splitting across two cards. Disable `--blocks_to_swap` or unset `FLUX2_DUAL_GPU`."

## Things we DON'T need to port

- **Mistral-on-CPU runtime patch.** Musubi caches TE outputs to disk via a separate script; the training loop is TE-free. Just remind users in the README to run `flux_2_cache_text_encoder_outputs.py` before training, which they already do.
- **`broadcast_and_multiply` device alignment.** Musubi's LoRA forward (`org_forwarded + lx * self.multiplier * scale`) keeps tensors on `x`'s device throughout — no network-level multiplier tensor that needs to migrate.
- **Pipeline composition `.to()` override.** Musubi doesn't have a pipeline object that re-collects the model; the trainer owns the transformer directly.

## Implementation shape

A single new file: `src/musubi_tuner/flux_2/flux2_dual_gpu.py`, mirroring the ai-toolkit `Flux2DualGPUMixin` pattern.

Public surface:

```python
def is_dual_gpu_enabled() -> bool: ...
def get_split_at(num_single_blocks: int) -> int: ...
def distribute_transformer(model: Flux2, dtype: torch.dtype) -> None: ...
def make_split_forward(num_single_blocks: int, split_at: int): ...
def route_lora_to_wrapped_device(network: LoRANetwork) -> None: ...
```

Three modifications to existing files:

1. `flux_2/flux2_models.py` — `Flux2.move_to_device_except_swap_blocks` checks the env var; gates `enable_block_swap` mutex.
2. `flux_2_train_network.py` — `Flux2NetworkTrainer.load_transformer` calls `distribute_transformer` after `load_flow_model` when the env var is set.
3. `networks/lora.py` — optional one-liner in `LoRAModule.apply_to` to snapshot `_wrapped_device` (only if we go the device-snapshot route).

## Validation plan

Mirror the ai-toolkit Suzurin/Fude path:
1. Smoke test: load model with `FLUX2_DUAL_GPU=true`, confirm no OOM, single forward+backward step completes.
2. Convergence test: train a character LoRA (rank 16, 512², 100 steps) and compare loss curve to ai-toolkit-trained reference.
3. Output quality: render keepers with the trained LoRA, confirm visual fidelity vs. ai-toolkit-trained version.
4. Step-rate benchmark: target ≥2 s/it sustained on 2× RTX 5090 (ai-toolkit hit 2.85; musubi's overhead may differ).

## Upstream submission strategy

Same as ai-toolkit:
- Fork to `genno-whittlery/musubi-tuner`
- Single-commit branch `dual-gpu-flux2`
- Open PR against kohya-ss main with the same body shape (problem → fix → validation → limitations → companion repo link)
- Comment on relevant open issues: musubi-tuner #923, #938 (24 GB OOM threads).

Kohya merges aggressively for FLUX trainers; expect faster review cycle than ai-toolkit.
