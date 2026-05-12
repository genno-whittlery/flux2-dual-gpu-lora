"""Patch DiffSynth-Studio for dual-GPU FLUX.2 LoRA training.

Two surgical edits:

1. ``examples/flux2/model_training/train.py``
   - inject ``flux2_dual_gpu_diffsynth`` import
   - force model load on CPU when FLUX2_DUAL_GPU=true (so the 60 GB bf16
     transformer doesn't pre-allocate on cuda:0 before split)
   - call ``enable_flux2_dual_gpu(model.pipe.dit)`` after construction
     and move dataloader-side scaffolding (vae cache is on-disk in
     sft:train; no TE/VAE in memory)

2. ``diffsynth/diffusion/runner.py``
   - skip ``model.to(accelerator.device)`` when dual-GPU (would undo
     the split)
   - pass ``device_placement=[False, True, True, True]`` to
     ``accelerator.prepare`` so the model isn't moved
"""
from pathlib import Path

# ─── train.py ────────────────────────────────────────────────────────────
train_path = Path(r"C:\DiffSynth-Studio\examples\flux2\model_training\train.py")
data = train_path.read_bytes()

old_imports = (
    b"import torch, os, argparse, accelerate\n"
    b"from diffsynth.core import UnifiedDataset\n"
    b"from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig\n"
    b"from diffsynth.diffusion import *\n"
)
new_imports = (
    b"import torch, os, argparse, accelerate\n"
    b"from diffsynth.core import UnifiedDataset\n"
    b"from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig\n"
    b"from diffsynth.diffusion import *\n"
    b"from flux2_dual_gpu_diffsynth import enable_flux2_dual_gpu, is_dual_gpu_enabled\n"
)

old_device = b"        device=\"cpu\" if args.initialize_model_on_cpu else accelerator.device,\n"
new_device = (
    b"        # Dual-GPU: force CPU load so the 60 GB bf16 transformer doesn't\n"
    b"        # pre-allocate on cuda:0 before we get a chance to split it.\n"
    b"        device=\"cpu\" if (args.initialize_model_on_cpu or is_dual_gpu_enabled()) else accelerator.device,\n"
)

old_model_logger = (
    b"    model_logger = ModelLogger(\n"
    b"        args.output_path,\n"
    b"        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,\n"
    b"    )\n"
)
new_model_logger = (
    b"    # Dual-GPU split: gated on FLUX2_DUAL_GPU env var. Distributes\n"
    b"    # pipe.dit across cuda:0 and cuda:1 at the single_transformer_blocks\n"
    b"    # midpoint. Called AFTER LoRA injection (which happened inside\n"
    b"    # Flux2ImageTrainingModule.__init__) so PEFT LoRA params follow\n"
    b"    # the base layer's device automatically when block.to() runs.\n"
    b"    if is_dual_gpu_enabled():\n"
    b"        for _i in range(torch.cuda.device_count()):\n"
    b"            _free, _total = torch.cuda.mem_get_info(_i)\n"
    b"            print(f\"[dual-gpu] pre-distribute cuda:{_i} free={_free/1024**3:.1f}G / {_total/1024**3:.1f}G\")\n"
    b"        enable_flux2_dual_gpu(model.pipe.dit)\n"
    b"        # Pipeline tracks a logical device for input transfer (forward()\n"
    b"        # calls transfer_data_to_device(inputs, pipe.device, ...)). Pin it\n"
    b"        # to cuda:0 since x_embedder lives there.\n"
    b"        model.pipe.device = torch.device(\"cuda:0\")\n"
    b"        for _i in range(torch.cuda.device_count()):\n"
    b"            _free, _total = torch.cuda.mem_get_info(_i)\n"
    b"            print(f\"[dual-gpu] post-distribute cuda:{_i} free={_free/1024**3:.1f}G / {_total/1024**3:.1f}G\")\n"
    b"\n"
    b"    model_logger = ModelLogger(\n"
    b"        args.output_path,\n"
    b"        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,\n"
    b"    )\n"
)

changed = False
for label, old, new in [
    ("train.py: imports", old_imports, new_imports),
    ("train.py: device init", old_device, new_device),
    ("train.py: enable_flux2_dual_gpu", old_model_logger, new_model_logger),
]:
    if new in data:
        print(f"[{label}] ALREADY_PATCHED")
    elif old in data:
        data = data.replace(old, new, 1)
        changed = True
        print(f"[{label}] PATCHED")
    else:
        print(f"[{label}] PATTERN_NOT_FOUND")
        raise SystemExit(1)

if changed:
    train_path.write_bytes(data)
    print(f"WROTE {train_path}")


# ─── runner.py ────────────────────────────────────────────────────────────
runner_path = Path(r"C:\DiffSynth-Studio\diffsynth\diffusion\runner.py")
data = runner_path.read_bytes()

old_runner = (
    b"    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)\n"
    b"    model.to(device=accelerator.device)\n"
    b"    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)\n"
)
new_runner = (
    b"    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)\n"
    b"    # Dual-GPU model-parallel: skip the device move that would undo our\n"
    b"    # manual split, and tell accelerate not to touch the model's device.\n"
    b"    _flux2_dual_gpu = os.environ.get(\"FLUX2_DUAL_GPU\", \"false\").lower() == \"true\"\n"
    b"    if not _flux2_dual_gpu:\n"
    b"        model.to(device=accelerator.device)\n"
    b"        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)\n"
    b"    else:\n"
    b"        model, optimizer, dataloader, scheduler = accelerator.prepare(\n"
    b"            model, optimizer, dataloader, scheduler,\n"
    b"            device_placement=[False, True, True, True],\n"
    b"        )\n"
)

if new_runner in data:
    print("[runner.py] ALREADY_PATCHED")
elif old_runner in data:
    runner_path.write_bytes(data.replace(old_runner, new_runner, 1))
    print(f"[runner.py] PATCHED — WROTE {runner_path}")
else:
    print("[runner.py] PATTERN_NOT_FOUND")
    raise SystemExit(1)
