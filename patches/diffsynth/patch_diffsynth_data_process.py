"""Patch launch_data_process_task to keep TE on CPU when initialize_model_on_cpu.

FLUX.2's Mistral-3-Small-24B text encoder is ~48 GB in bf16 and doesn't
fit on a single 32 GB consumer GPU. Without this patch, the data_process
step's ``model.to(accelerator.device)`` line OOMs at load.

The detection is via env var ``DIFFSYNTH_DATA_PROCESS_ON_CPU=true``,
which the user/launcher script sets when running data_process on a card
with insufficient VRAM. Honors the same convention as the dual-GPU
patch and keeps the diff small.
"""
from pathlib import Path

path = Path(r"C:\DiffSynth-Studio\diffsynth\diffusion\runner.py")
data = path.read_bytes()

old = (
    b"    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)\n"
    b"    model.to(device=accelerator.device)\n"
    b"    model, dataloader = accelerator.prepare(model, dataloader)\n"
)
new = (
    b"    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)\n"
    b"    # Keep TE/VAE on CPU when explicitly requested. FLUX.2's Mistral-24B TE\n"
    b"    # is ~48 GB bf16; data_process OOMs on cards <48 GB without CPU offload.\n"
    b"    _data_process_on_cpu = os.environ.get('DIFFSYNTH_DATA_PROCESS_ON_CPU', 'false').lower() == 'true'\n"
    b"    if not _data_process_on_cpu:\n"
    b"        model.to(device=accelerator.device)\n"
    b"        model, dataloader = accelerator.prepare(model, dataloader)\n"
    b"    else:\n"
    b"        model, dataloader = accelerator.prepare(\n"
    b"            model, dataloader, device_placement=[False, True],\n"
    b"        )\n"
)

if new in data:
    print("ALREADY_PATCHED")
elif old in data:
    path.write_bytes(data.replace(old, new, 1))
    print(f"PATCHED -- WROTE {path}")
else:
    print("PATTERN_NOT_FOUND")
    raise SystemExit(1)
