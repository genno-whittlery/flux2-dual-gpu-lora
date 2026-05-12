"""Force loading_device=cpu when FLUX2_DUAL_GPU=true.

The fp8_scaled quantization step peaks well above 32 GB when loading the
60 GB FLUX.2 transformer onto a single 5090. Loading to CPU first (where
we have 128 GB) lets the quant happen freely, then distribute_flux2_transformer
moves the already-quantized fp8 tensors to cuda:0/cuda:1 cheaply.
"""
from pathlib import Path

path = Path(r"C:\musubi-tuner\src\musubi_tuner\hv_train_network.py")
data = path.read_bytes()

old = b'loading_device = "cpu" if blocks_to_swap > 0 else accelerator.device'
new = (
    b'loading_device = "cpu" if (blocks_to_swap > 0 or '
    b'os.environ.get("FLUX2_DUAL_GPU", "false").lower() == "true") '
    b'else accelerator.device'
)
if new in data:
    print("ALREADY_PATCHED")
elif old in data:
    path.write_bytes(data.replace(old, new))
    print("PATCHED")
else:
    raise SystemExit("PATTERN_NOT_FOUND")
