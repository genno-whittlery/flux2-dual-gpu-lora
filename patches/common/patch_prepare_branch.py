"""Add an FLUX2_DUAL_GPU branch in the accelerator.prepare call.

Mirrors the existing blocks_to_swap branch: passes device_placement=[False]
so accelerate doesn't move the model to a single device, then calls
move_to_device_except_swap_blocks (which our patch routes to
distribute_flux2_transformer + install_split_forward when FLUX2_DUAL_GPU is on).
"""
from pathlib import Path

path = Path(r"C:\musubi-tuner\src\musubi_tuner\hv_train_network.py")
data = path.read_bytes()

old = (
    b"        if blocks_to_swap > 0:\n"
    b"            transformer = accelerator.prepare(transformer, device_placement=[not blocks_to_swap > 0])\n"
    b"            accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)  # reduce peak memory usage\n"
    b"            accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()\n"
    b"        else:\n"
    b"            transformer = accelerator.prepare(transformer)"
)
new = (
    b"        if blocks_to_swap > 0:\n"
    b"            transformer = accelerator.prepare(transformer, device_placement=[not blocks_to_swap > 0])\n"
    b"            accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)  # reduce peak memory usage\n"
    b"            accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()\n"
    b'        elif os.environ.get("FLUX2_DUAL_GPU", "false").lower() == "true":\n'
    b"            # Dual-GPU model-parallel: prepare without device move, then run our\n"
    b"            # split-aware placement via move_to_device_except_swap_blocks (which\n"
    b"            # routes to distribute_flux2_transformer + install_split_forward).\n"
    b"            transformer = accelerator.prepare(transformer, device_placement=[False])\n"
    b"            accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)\n"
    b"        else:\n"
    b"            transformer = accelerator.prepare(transformer)"
)
if new in data:
    print("ALREADY_PATCHED")
elif old in data:
    path.write_bytes(data.replace(old, new))
    print("PATCHED")
else:
    raise SystemExit("PATTERN_NOT_FOUND")
