# patches/ — raw byte-surgery patchers

The drop-in **helper modules** (`flux2_dual_gpu.py` / `Flux2DualGpu.py` / etc.) live in [`examples/`](../examples/) and at the repo root (`flux2_model.py` for ai-toolkit). This directory holds the **one-shot patcher scripts** that actually inject those helpers + the few integration edits into a fresh checkout of each trainer.

These are the redeploy / reference artifacts — they encode *exactly which file, which line, what edit*. They were authored against specific checkouts on the dev box (Windows paths hard-coded inside, mostly `C:\…`); treat them as a precise record of the integration points rather than portable tools. CRLF line endings are preserved on purpose so the resulting diffs stay noise-free against the (CRLF) upstream files.

`base_sd_train_process_memory_probe.patch` — the only true unified diff here: an optional ai-toolkit diagnostic that emits `torch.cuda.memory_summary()` at user-specified relative training steps via `FLUX2_MEMORY_PROBE_STEPS`. Documented in the top-level README.

## Status legend

- ✅ **validated end-to-end** on 2× RTX 5090 — LoRA trained, checkpoints saved, step rate measured
- 🚧 **WIP / blocked** — the dual-GPU split itself works (distribute lands at the expected shape; forward/backward complete) but full real training is blocked on an upstream bug
- 🪦 **superseded** — kept as a breadcrumb; a later file in the same dir replaced it

## ai-toolkit (FLUX.2) — ✅ validated · 2.85 s/it · [PR #829](https://github.com/ostris/ai-toolkit/pull/829)

The flagship. The patch isn't a script here — it's the full-file replacement `flux2_model.py` (repo root) plus the `Flux2DualGPUMixin` in `flux2_dual_gpu.py`. See the top-level README. `patches/base_sd_train_process_memory_probe.patch` is the companion diagnostic diff.

## diffsynth/ — DiffSynth-Studio FLUX.2 — ✅ validated · 2.69 s/it · [PR #1434](https://github.com/modelscope/DiffSynth-Studio/pull/1434)

| File | Edits |
|---|---|
| `patch_diffsynth_dualgpu.py` | Injects the `flux2_dual_gpu_diffsynth` import + `enable_flux2_dual_gpu(model.pipe.dit)` into `examples/flux2/model_training/train.py`; forces CPU model-load; in `diffsynth/diffusion/runner.py` skips `model.to(accelerator.device)` and passes `device_placement=[False, …]` to `accelerator.prepare`. |
| `patch_diffsynth_addfp8.py` | Adds the `quantize_(…, Float8WeightOnlyConfig())` call with the LoRA-skip `filter_fn` (excludes `lora_A`/`lora_B` submodules — this is the workaround for the PEFT×torchao incompatibility that blocks the diffusers ref script). |
| `patch_diffsynth_te.py` 🪦 | First TE-device fix attempt. Superseded by `_te_v2.py`. |
| `patch_diffsynth_te_v2.py` | transformers 5.8 compat for `diffsynth/models/flux2_text_encoder.py` — kwargs-only `super().forward`, re-injecting `output_hidden_states` via `TransformersKwargs`. |
| `patch_diffsynth_data_process.py` | Applies the same `device_placement` fix to the `sft:data_process` task path (not just `sft:train`). |

Helper: [`examples/diffsynth/`](../examples/diffsynth/) · walkthrough: [`docs/porting-diffsynth-studio.md`](../docs/porting-diffsynth-studio.md).

## musubi/ — musubi-tuner FLUX.2 — ✅ validated · 2.22 s/it · upstream PR pending

| File | Edits |
|---|---|
| `patch_musubi_dual_gpu.py` | In `src/musubi_tuner/hv_train_network.py`: skip auto-enabling DDP when `FLUX2_DUAL_GPU=true`. (The full port also needs CPU loading-device + `device_placement=[False]` in the prepare branch + an inlined device-pin at the top of `LoRAModule.forward` in `networks/lora.py` — those live on the fork branch alongside the new `flux_2/flux2_dual_gpu.py` module.) |

Helper: [`examples/musubi/`](../examples/musubi/) · walkthrough: [`docs/porting-musubi-tuner.md`](../docs/porting-musubi-tuner.md) · branch: [`genno-whittlery/musubi-tuner:dual-gpu-flux2`](https://github.com/genno-whittlery/musubi-tuner/tree/dual-gpu-flux2).

## onetrainer/ — OneTrainer FLUX.2 — ✅ validated · 0.95 / 1.75 s/it · [PR #1450 closed-on-technicality](https://github.com/Nerogar/OneTrainer/pull/1450)

| File | Edits |
|---|---|
| `patch_onetrainer_perblock_hook.py` | In `modules/util/Flux2DualGpu.py`: register the device-bridge pre-hook on **every** `single_transformer_block` in the cuda:1 range, not just the boundary block — because `temb` is computed once and passed to each block in a loop; only transforming `block[split_at]`'s args leaves the loop-level `temb` on cuda:0 for subsequent blocks. |
| `patch_onetrainer_te_cpu.py` | In `modules/model/Flux2Model.py`: force the Mistral text encoder onto CPU; route `transformer_to(device)` through the distribute call. |
| `patch_ot_fp8.ps1` | PowerShell helper to configure OneTrainer's fp8 / int8 W8A8 dit-quantization mode. |

Helper: [`examples/onetrainer/`](../examples/onetrainer/) (incl. the `mgds/EncodeMistralText.py` data-loader fix — branch [`genno-whittlery/mgds:te-device-fix`](https://github.com/genno-whittlery/mgds/tree/te-device-fix)) · branch: [`genno-whittlery/OneTrainer:dual-gpu-flux2`](https://github.com/genno-whittlery/OneTrainer/tree/dual-gpu-flux2).

## diffusers/ — HuggingFace `train_dreambooth_lora_flux2*.py` — 🚧 split works, training blocked on PEFT×torchao upstream bug

The split distribution lands end-to-end (20.7 GB cuda:0 / 12.6 GB cuda:1, forward succeeds), but the training step dies at the LoRA forward: PEFT 0.19's `TorchaoLoraLinear` dispatcher doesn't bridge CPU/CUDA when the base weight is a torchao 0.17 `WeightOnlyFloat8Tensor`. Not fixable from a single-script patch — needs an upstream PEFT (or torchao) fix. The DiffSynth port's one-line `filter_fn` workaround (skip `lora_A`/`lora_B` in the quant pass) **likely unblocks this too — not retested**. The other validated trainers (ai-toolkit, musubi, OneTrainer) use weight-only fp8 paths that don't go through PEFT's torchao dispatcher, so the bug doesn't surface there.

| File | Edits |
|---|---|
| `patch_diffusers_train_script.py` 🪦 / `patch_diffusers_train_script_v2.py` | Wire `enable_flux2_dual_gpu(transformer)` + `device_placement=False` into `examples/dreambooth/train_dreambooth_lora_flux2.py`. v2 is current. |
| `patch_diffusers_fp8_first.py` 🪦 / `patch_diffusers_fp8_weightonly.py` | Move fp8 quant before distribute (60 GB bf16 / 32 GB doesn't fit even split); swap `convert_to_float8_training` (fp8 *compute*, weights stay bf16 in memory) for `quantize_(…, Float8WeightOnlyConfig())` (real fp8 *storage*, halves memory). `_weightonly` is current. |
| `patch_diffusers_lora_first.py` | Order LoRA adapter add before the fp8 quant pass. |
| `patch_diffusers_cpu_safety.py` | Guard against the example script's stray `.to(accelerator.device)` calls that would collapse the split. |

Helper: [`examples/diffusers/`](../examples/diffusers/).

## common/ — cross-trainer building blocks

| File | What |
|---|---|
| `patch_encode_mistral_te_device.py` | The mgds `EncodeMistralText` fix as a patcher — move tokenizer outputs to `text_encoder.device` before the forward. (Consumed copy: `examples/onetrainer/EncodeMistralText.py`.) |
| `patch_lora_forward_pin.py` | Inline the per-LoRA device-pin at the top of a `LoRAModule.forward` (the reliable form when bound-method indirection in `apply_to` defeats instance-level shimming). |
| `patch_loading_device.py` | Force a trainer's model-construction loading device to CPU when the dual-GPU env var is set. |
| `patch_prepare_branch.py` | Insert `device_placement=[False, …]` into an `accelerator.prepare(...)` call so Accelerate doesn't flatten the distributed transformer back onto one device. |

## wan-smoke/ — Wan 2.x dual-GPU smoke harness — 🚧 synthetic smoke OK, real training gated on [DiffSynth #1063](https://github.com/modelscope/DiffSynth-Studio/issues/1063)

| File | What |
|---|---|
| `run_wan_dualgpu_smoke.py` / `run_wan_dualgpu_smoke.ps1` | Synthetic smoke: build `WanModel`, enable the dual-GPU split, run a forward + backward on dummy inputs, assert LoRA gradients land on both cuda:0 and cuda:1. |
| `wan_dualgpu_smoke_v3.txt` | The last smoke run's log — forward + backward across the split complete; the *real-training* invocation then fails on the upstream `WanModel.patchify` shape/arity bug (the dual-GPU code is not in that call path). |
| `patch_train_redirect.ps1` / `patch_train_t5vae_move.ps1` | Operational helpers for pointing DiffSynth's Wan `train.py` at local model paths and moving the T5 / VAE off the training GPUs. |
| `check_patchify.ps1` / `verify_patches.ps1` | Sanity checks — confirm the upstream patchify bug reproduces, and that the dual-GPU + fp8 edits are present in the target checkout. |

Helper: [`examples/diffsynth/wan_dual_gpu_diffsynth.py`](../examples/diffsynth/wan_dual_gpu_diffsynth.py) · PRs: [#1435](https://github.com/modelscope/DiffSynth-Studio/pull/1435) (patchify), [#1436](https://github.com/modelscope/DiffSynth-Studio/pull/1436) (helper).
