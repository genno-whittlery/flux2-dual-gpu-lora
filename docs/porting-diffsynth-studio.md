# Porting the dual-GPU FLUX.2 LoRA patch to DiffSynth-Studio

[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (modelscope, 12.4k stars) ships a FLUX.2-dev LoRA training recipe under `examples/flux2/model_training/`. It's structurally close to HuggingFace diffusers: same Flux2DiT field layout, same PEFT-based LoRA injection, same `accelerate.Accelerator` wrapper.

**Status (2026-05-11):** validated end-to-end on 2× RTX 5090 — the full `sft:data_process` → `sft:train` pipeline from `examples/flux2/model_training/lora/FLUX.2-dev.sh` runs through, **2.69 s/it sustained** across 15 training steps on 3 cached sumi-v8 images, LoRA checkpoint (`epoch-0.safetensors`, 270 MB at rank 32) saved.

This doc covers the three integration points and the one *novel* lesson the DiffSynth port surfaced: **the LoRA-skip `filter_fn` for `quantize_`**, which avoids the PEFT 0.19.1 × torchao 0.17 incompatibility that blocked the diffusers reference script.

## Why this port is easy

`Flux2DiT` (`diffsynth/models/flux2_dit.py`) is field-identical to diffusers' `Flux2Transformer2DModel`:

- `x_embedder`, `context_embedder`
- `time_guidance_embed`, `pos_embed`
- `double_stream_modulation_img`, `double_stream_modulation_txt`, `single_stream_modulation`
- `transformer_blocks` (double-stream, 8 blocks)
- `single_transformer_blocks` (single-stream, 48 blocks)
- `norm_out`, `proj_out`

The `forward` loop has the same loop-level-constant pattern as diffusers and OneTrainer: `temb_mod_params=single_stream_mod` and `image_rotary_emb=concat_rotary_emb` are passed to every single block. That means the **per-block pre-hook** pattern from the OneTrainer port carries over — a single hook on `single_transformer_blocks[split_at]` only bridges the first block's inputs to `cuda:1`; subsequent blocks receive the loop-level originals from `cuda:0` and crash with a device-mismatch error.

## The three integration points

### 1. `examples/flux2/model_training/train.py`

After `Flux2ImageTrainingModule` constructs the pipeline and runs `switch_pipe_to_training_mode` (which calls `add_lora_to_model` → PEFT `inject_adapter_in_model`), distribute the DiT:

```python
from flux2_dual_gpu_diffsynth import enable_flux2_dual_gpu, is_dual_gpu_enabled

# ... after model = Flux2ImageTrainingModule(...) ...

if is_dual_gpu_enabled():
    import gc
    from torchao.quantization import quantize_, Float8WeightOnlyConfig

    def _quant_filter(module, name):
        # Skip LoRA's own Linear submodules - quantizing them strips
        # requires_grad from lora_A.weight / lora_B.weight, breaking backward.
        if not isinstance(module, torch.nn.Linear):
            return False
        return "lora_A" not in name and "lora_B" not in name

    quantize_(model.pipe.dit, Float8WeightOnlyConfig(), filter_fn=_quant_filter)
    gc.collect()
    enable_flux2_dual_gpu(model.pipe.dit)
    model.pipe.device = torch.device("cuda:0")
```

The `model.pipe.device` pin matters because `Flux2ImageTrainingModule.forward` calls `transfer_data_to_device(inputs, self.pipe.device, ...)` to move incoming batch tensors. We pin to `cuda:0` since `x_embedder` lives there; the first hook bridges from `cuda:0` to `cuda:1` at `single_transformer_blocks[split_at]`.

Also force CPU initial load when `FLUX2_DUAL_GPU=true` is set:

```python
device="cpu" if (args.initialize_model_on_cpu or is_dual_gpu_enabled()) else accelerator.device,
```

The bf16 transformer is ~60 GB. If loaded on `cuda:0` first, it OOMs before we get a chance to split it.

### 2. `diffsynth/diffusion/runner.py`

In `launch_training_task`, skip the model-wide device move (would undo the split) and tell `accelerate` not to touch the model's device:

```python
_flux2_dual_gpu = os.environ.get("FLUX2_DUAL_GPU", "false").lower() == "true"
if not _flux2_dual_gpu:
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
else:
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler,
        device_placement=[False, True, True, True],
    )
```

This is the same shape as the musubi-tuner / OneTrainer / diffusers patches — `accelerate.prepare`'s `device_placement` list takes one bool per arg; `False` for the model preserves our manual split.

### 3. `examples/flux2/model_training/flux2_dual_gpu_diffsynth.py`

The helper itself (130 LOC). Distributes:

- `cuda:0`: `x_embedder`, `context_embedder`, `time_guidance_embed`, `pos_embed`, all three modulation modules, all 8 `transformer_blocks` (double-stream), first 24 `single_transformer_blocks`, `norm_out`, `proj_out`
- `cuda:1`: last 24 `single_transformer_blocks`

Registers a `forward_pre_hook` with `with_kwargs=True` on:

- **Every** `single_transformer_block` in `[split_at:]` — bridges loop-level constants (`temb_mod_params`, `image_rotary_emb`, `joint_attention_kwargs`) to `cuda:1` on each iteration
- `norm_out` — bridges the activation back to `cuda:0` for the final output layers

After fp8 weight-only quant, the placement is:

- `cuda:0`: 20.7 GB (matches diffusers / OneTrainer / musubi)
- `cuda:1`: 12.6 GB

About 10 GB headroom on each card for activations + LoRA optimizer state. Forward + backward observed at 1.26 + 1.43 s for a synthetic 512² / 256-text-token / rank-32 LoRA step with `use_gradient_checkpointing=True`.

## The PEFT × torchao gotcha and how this port avoids it

The diffusers reference script ran into an upstream blocker where `quantize_(transformer, Float8WeightOnlyConfig())` after `transformer.add_adapter(...)` produced a forward path that couldn't bridge devices in `WeightOnlyFloat8Tensor`'s dispatcher. Two things were happening at once in that path:

1. `quantize_` with no `filter_fn` was quantizing **every** `nn.Linear` it found — including PEFT's `lora_A` / `lora_B` submodules. That replaces `lora_A.weight` (a leaf `nn.Parameter` with `requires_grad=True`) with a `WeightOnlyFloat8Tensor` that has no `requires_grad`. Backward then fails at the loss with `element 0 of tensors does not require grad and does not have a grad_fn` — observed reproducibly here before the filter was added.

2. PEFT 0.19.1 has separate downstream code (`TorchaoLoraLinear`) that *would* run if the LoRA layer wrapped a torchao-quantized base, and that code path has its own constructor incompatibility (`__init__()` missing the `get_apply_tensor_subclass` kwarg). The diffusers session may have triggered this via a different order; the symptom looked like a cross-device dispatcher mismatch.

**The DiffSynth port avoids both** by passing a one-line `filter_fn` to `quantize_`:

```python
def _quant_filter(module, name):
    if not isinstance(module, torch.nn.Linear):
        return False
    return "lora_A" not in name and "lora_B" not in name
```

With this filter:

- The base layer inside `PEFT's LoraLinear.base_layer` (which is `nn.Linear`) gets quantized to `Float8` — halves base memory.
- The LoRA Linear submodules (`lora_A.default`, `lora_B.default`) stay `bf16` — `requires_grad` is preserved.
- PEFT never instantiates `TorchaoLoraLinear` because the quant happened **after** LoRA injection on a `bf16` base; PEFT's normal `LoraLinear` is the one that's active, and it's unaware of the `.data`-level swap that happened later.

This is the **novel finding** of the DiffSynth port. The same fix likely unblocks the diffusers reference script — not retested.

## Side-quest fixes encountered during validation

Two upstream compatibility issues surfaced while wiring this through end-to-end. Neither is dual-GPU-related — both would also bite single-GPU users — but they're documented here so anyone reproducing the validation knows what to expect.

### Mistral3 forward signature drift (transformers 5.8)

`diffsynth/models/flux2_text_encoder.py` subclasses `Mistral3ForConditionalGeneration` and explicitly passes 15 positional args to `super().forward(...)`. transformers 5.8 trimmed `output_attentions`, `output_hidden_states`, `return_dict`, `cache_position` from the positional signature — they're now `TransformersKwargs`-only. The mismatched call dies with `TypeError: forward() takes from 1 to 11 positional arguments but 15 were given`.

Patched in-place to forward args by name and re-inject `output_hidden_states` / `output_attentions` through `**kwargs` (the embedding pipeline depends on `output.hidden_states` being populated):

```python
def forward(self, input_ids=None, pixel_values=None, attention_mask=None, ...,
            output_hidden_states=None, output_attentions=None, ..., **kwargs):
    if output_hidden_states is not None:
        kwargs.setdefault("output_hidden_states", output_hidden_states)
    if output_attentions is not None:
        kwargs.setdefault("output_attentions", output_attentions)
    return super().forward(
        input_ids=input_ids, pixel_values=pixel_values,
        attention_mask=attention_mask, ...,
        **kwargs,
    )
```

### data_process can't fit Mistral-24B on a single 32 GB card

`diffsynth/diffusion/runner.py::launch_data_process_task` unconditionally calls `model.to(accelerator.device)` before encoding. For FLUX.2, the loaded models are Mistral-3-Small-24B (~48 GB bf16) and the VAE — Mistral alone overflows 32 GB. Added an env-var-gated CPU-offload branch:

```python
_data_process_on_cpu = os.environ.get("DIFFSYNTH_DATA_PROCESS_ON_CPU", "false").lower() == "true"
if not _data_process_on_cpu:
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
else:
    model, dataloader = accelerator.prepare(
        model, dataloader, device_placement=[False, True],
    )
```

Combined with `--initialize_model_on_cpu`, this lets the TE+VAE caching step run on CPU on cards that can't fit Mistral. It's slow (~1 min per image for 256-token captions at 512²), but it's a one-time pre-process before `sft:train` — and orthogonal to whether the subsequent training step uses one GPU or two.

## Setup

Per-trainer venv (consistent with the other trainer ports — keeps torch/peft/accelerate pins independent):

```powershell
# Match Python 3.12 to the existing trainers' venvs
C:\musubi-tuner\.venv\Scripts\python.exe -m venv C:\DiffSynth-Studio\.venv

# Blackwell-compatible torch (sm_120)
C:\DiffSynth-Studio\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu130 torchvision --index-url https://download.pytorch.org/whl/cu130

# DiffSynth in editable mode + torchao for the fp8 quant path
cd C:\DiffSynth-Studio
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install torchao==0.17
```

Drop `flux2_dual_gpu_diffsynth.py` into `examples/flux2/model_training/`, apply the train.py + runner.py edits, set `FLUX2_DUAL_GPU=true`, and launch:

```powershell
$env:FLUX2_DUAL_GPU = "true"
$env:DIFFSYNTH_MODEL_BASE_PATH = "F:\models\diffusers"   # if loading locally
$env:DIFFSYNTH_SKIP_DOWNLOAD = "true"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# Standard sft:data_process + sft:train flow from
# examples/flux2/model_training/lora/FLUX.2-dev.sh, unchanged.
```

The `sft:data_process` step (TE + VAE feature caching) is **orthogonal to the patch** — it doesn't touch the DiT. Run it with `--initialize_model_on_cpu` so Mistral text-encoder doesn't OOM on a single 32 GB card. After data_process produces the cache, the `sft:train` step is the one that benefits from the dual-GPU split.

## What was validated

Two layers of evidence:

**Smoke test** — a 60-second standalone script that loads the real FLUX.2-dev transformer weights through `Flux2ImagePipeline.from_pretrained` (transformer-only, the same shape `sft:train` uses), injects PEFT LoRA at rank 32, applies the filtered fp8 quant, distributes via `enable_flux2_dual_gpu`, and runs a synthetic forward + backward. Assertions:

- Device placement: `single_transformer_blocks[23]` on `cuda:0`, `[24]` on `cuda:1`, `norm_out` on `cuda:0`
- VRAM after distribute: 20.7 GB cuda:0 / 12.6 GB cuda:1
- Forward succeeds (output shape `[1, 1024, 128]` back on `cuda:0`)
- Backward succeeds (`loss=1.0781`)
- LoRA gradient devices: `['cuda:0', 'cuda:1']` — proves cross-device autograd

**End-to-end run** — the full `sft:data_process` → `sft:train` pipeline straight from `examples/flux2/model_training/lora/FLUX.2-dev.sh`, on 3 sumi-v8 images at 512² resolution, rank-32 LoRA on 7 target-module families, 1 epoch × 5-repeat dataset = 15 steps:

- `sft:data_process` on CPU-offloaded Mistral-24B: 3/3 images cached in 43s (~15 s/it)
- `sft:train` distribute step: same 20.7 / 12.6 GB shape
- `sft:train` steady-state: **2.69 s/it across 15/15 steps** (40s wall-clock)
- LoRA checkpoint: `epoch-0.safetensors`, 270 MB

This is the same bar that musubi-tuner and OneTrainer cleared in the same week.

## Upstream plan

Filing a PR to modelscope/DiffSynth-Studio after the helper is dual-described in English + Mandarin (matching their main README's bilingual convention). The PR will:

1. Add `examples/flux2/model_training/flux2_dual_gpu_diffsynth.py` as a new file
2. Add `FLUX2_DUAL_GPU` env-var-gated branch to `train.py` and `runner.py` (off by default — no behavior change for single-GPU users)
3. Document the LoRA-skip `filter_fn` lesson — likely the most generally useful finding for downstream FLUX.2 LoRA work
