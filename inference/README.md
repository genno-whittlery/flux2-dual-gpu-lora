# inference/ — dual-GPU FLUX.2 + LoRA rendering

Render FLUX.2-dev + ai-toolkit LoRAs on the same dual-GPU + Mistral-on-CPU setup the training patch enables. Exists because:

- **ComfyUI has no dual-GPU FLUX.2 path** — there's no way to render with the exact split + quantization the training ran in.
- **ai-toolkit's `run.py` can't either** — the "generate" / sample-during-training paths deadlock at the embed-cache/unload step on a dual-GPU + TE-on-CPU config, and `set_device_state` OOMs trying to move Mistral onto cuda:0.

The model-load / qfloat8-quantize / dual-GPU-split / `generate_images` code paths *inside* ai-toolkit work fine — this drives just those, plus the LoRA apply dance, plus a few small fixups:

1. **`set_device_state`** — coerce the text-encoder device entry to `te_device_torch` (CPU) so it doesn't try to park Mistral on cuda:0 on top of the ~21 GB transformer half.
2. **`transformers.PreTrainedModel.to(cuda…)` → no-op** — Mistral stays in system RAM.
3. **`optimum.quanto.qbytes_mm`** — move activations onto the quantized weight's device before the matmul (tolerates the cross-GPU split).
4. **`Flux2Pipeline._execution_device` → `cuda:0`** (a pipeline tweak, not a monkeypatch) — else it resolves to CPU (Mistral's device), the latent + position-id tensors get created on CPU, and feed CPU tensors to the cuda transformer.

Not a ComfyUI replacement — no node graph, no UI. A focused inference path for the one case ComfyUI can't cover.

## Layout

| File | What |
|---|---|
| `engine.py` | `load_flux2(base, lora=LoraSpec(path, rank, alpha))` → `(model, pipeline, network)`; `render(model, pipeline, {name: prompt}, out=…, network=…, strengths=(…,))`. Import this from other scripts (eval harness, dataset generation, …). |
| `cli.py` | One-off renders. `python cli.py --lora … --prompts prompts.json --out renders/ --strengths 0,0.6,1.0` (`--prompts` = a JSON `{name: prompt}` object). |
| `recipe.py` | Declarative YAML jobs (base + LoRA + prompt set + sampler params + strength sweep). `python recipe.py job.yaml` |

## Requirements

- Run inside **ai-toolkit's venv** (`<ai-toolkit>/venv/Scripts/python.exe` on Windows). Set `AITK_DIR` if ai-toolkit isn't at `C:\ai-toolkit`.
- The dual-GPU training patch must be installed in ai-toolkit (`flux2_model.py` from this repo, or merged upstream) — `FLUX2_DUAL_GPU=true` triggers the same `Flux2DualGPUMixin` the trainer uses.
- Two CUDA devices (`CUDA_VISIBLE_DEVICES=0,1`).
- `F:\models\diffusion_models\FLUX.2-dev` = the **raw BFL single-file checkpoint** (`flux2-dev.safetensors` + `ae.safetensors`), not a diffusers folder. Override with `--base` / `FLUX2_BASE`.

## Env vars

Inherits the training patch's env contract:

| Variable | Default here | Effect |
|---|---|---|
| `FLUX2_DUAL_GPU` | `true` | Enable the cuda:0+cuda:1 transformer split. |
| `FLUX2_TE_DEVICE` | `cpu` | Where Mistral lives (system RAM has room for the ~48 GB bf16 encoder). |
| `FLUX2_DUAL_GPU_SPLIT_AT` | `n_single // 2` (= 24) | Single-stream-block index to cut the pipeline at. |
| `FLUX2_BASE` | `F:\models\diffusion_models\FLUX.2-dev` | Base checkpoint path (also `--base`). |
| `AITK_DIR` | `C:\ai-toolkit` | ai-toolkit checkout location. |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Fragmentation guard. |

## Throughput

On 2× RTX 5090: ~80–140 s/image at 1024²/20 steps/cfg 4. The bottleneck is the per-prompt Mistral-on-CPU text encode under pagefile pressure, not the diffusion. (Validated rendering `fude-flux2-v3` and `suzurin-flux2-v2` LoRAs, 2026-05-12.)

## Not yet

- **img2img / reference-latent conditioning** — txt2img only. Reference conditioning (IPAdapter-style training-set generation from a locked hero image) is the planned next capability.
- **Multi-LoRA stacking** — one LoRA at a time.
