"""Inject the dual-GPU helper into diffusers' train_dreambooth_lora_flux2.py.

Three insertions:
  1. Import the helper at module top (after the existing diffusers imports).
  2. Call enable_flux2_dual_gpu(transformer) after transformer.add_adapter().
  3. Pass device_placement=[False, True, True, True] to accelerator.prepare
     when FLUX2_DUAL_GPU=true (so it doesn't move our distributed transformer
     back onto a single device).
"""
from pathlib import Path

path = Path(r"C:\diffusers-flux2\diffusers\examples\dreambooth\train_dreambooth_lora_flux2.py")
data = path.read_bytes()

# Insertion 1: import. Put it right after the diffusers.utils import line.
old1 = b"from diffusers.utils.torch_utils import is_compiled_module\n"
new1 = (
    b"from diffusers.utils.torch_utils import is_compiled_module\n"
    b"\n"
    b"# Dual-GPU model-parallel helper: see flux2_dual_gpu_diffusers.py.\n"
    b"# No-op unless FLUX2_DUAL_GPU=true is set in the environment.\n"
    b"import sys as _sys\n"
    b"if r\"C:\\diffusers-flux2\" not in _sys.path:\n"
    b"    _sys.path.insert(0, r\"C:\\diffusers-flux2\")\n"
    b"from flux2_dual_gpu_diffusers import enable_flux2_dual_gpu, is_dual_gpu_enabled\n"
)

# Insertion 2: distribute after add_adapter. The line we anchor on is at ~1287.
old2 = b"    transformer.add_adapter(transformer_lora_config)\n"
new2 = (
    b"    transformer.add_adapter(transformer_lora_config)\n"
    b"    # Dual-GPU split: gated on FLUX2_DUAL_GPU env var.\n"
    b"    transformer = enable_flux2_dual_gpu(transformer)\n"
)

# Insertion 3: pass device_placement to accelerator.prepare.
old3 = (
    b"    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(\n"
    b"        transformer, optimizer, train_dataloader, lr_scheduler\n"
    b"    )\n"
)
new3 = (
    b"    if is_dual_gpu_enabled():\n"
    b"        # transformer is already split across cuda:0/cuda:1; don't let\n"
    b"        # accelerator.prepare move it back onto a single device.\n"
    b"        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(\n"
    b"            transformer, optimizer, train_dataloader, lr_scheduler,\n"
    b"            device_placement=[False, True, True, True],\n"
    b"        )\n"
    b"    else:\n"
    b"        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(\n"
    b"            transformer, optimizer, train_dataloader, lr_scheduler\n"
    b"        )\n"
)

changed = False
for label, old, new in [("import", old1, new1), ("add_adapter", old2, new2), ("prepare", old3, new3)]:
    if new in data:
        print(f"[{label}] ALREADY_PATCHED")
    elif old in data:
        data = data.replace(old, new, 1)
        changed = True
        print(f"[{label}] PATCHED")
    else:
        print(f"[{label}] PATTERN_NOT_FOUND")

if changed:
    path.write_bytes(data)
    print("WROTE", path)
