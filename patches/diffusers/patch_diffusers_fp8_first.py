"""Run torchao fp8 conversion BEFORE the dual-GPU distribute, not after.

The diffusers FLUX.2 transformer is 60 GB in bf16 — too large for any
2x32GB split (best case puts 38 GB on cuda:0). fp8 weight-only quant
halves it to ~30 GB, which fits.

Place the convert_to_float8_training call inside our FLUX2_DUAL_GPU
branch (model still on CPU), then skip the existing later call to avoid
double-converting.
"""
from pathlib import Path

path = Path(r"C:\diffusers-flux2\diffusers\examples\dreambooth\train_dreambooth_lora_flux2.py")
data = path.read_bytes()

# 1. Insert fp8 conversion in our dual-GPU branch, before enable.
old1 = (
    b"        transformer.to(\"cpu\")\n"
    b"        if not args.remote_text_encoder and text_encoder is not None:\n"
    b"            text_encoder.to(\"cpu\")\n"
    b"        vae.to(\"cpu\")\n"
    b"        _gc.collect()\n"
    b"        torch.cuda.empty_cache()\n"
)
new1 = (
    b"        transformer.to(\"cpu\")\n"
    b"        if not args.remote_text_encoder and text_encoder is not None:\n"
    b"            text_encoder.to(\"cpu\")\n"
    b"        vae.to(\"cpu\")\n"
    b"        _gc.collect()\n"
    b"        torch.cuda.empty_cache()\n"
    b"        # fp8 quantization MUST happen before distribute when dual-GPU is on:\n"
    b"        # bf16 60 GB / 32 GB GPU can't fit even split 50/50. fp8 halves it.\n"
    b"        if args.do_fp8_training:\n"
    b"            from torchao.float8 import Float8LinearConfig, convert_to_float8_training\n"
    b"            convert_to_float8_training(\n"
    b"                transformer, module_filter_fn=module_filter_fn,\n"
    b"                config=Float8LinearConfig(pad_inner_dim=True),\n"
    b"            )\n"
    b"            print(\"[dual-gpu] fp8 conversion done on CPU\")\n"
)

# 2. Skip the existing fp8 call when dual-GPU is on (already converted above).
old2 = (
    b"    if args.do_fp8_training:\n"
    b"        convert_to_float8_training(\n"
    b"            transformer, module_filter_fn=module_filter_fn, config=Float8LinearConfig(pad_inner_dim=True)\n"
    b"        )\n"
)
new2 = (
    b"    if args.do_fp8_training and not is_dual_gpu_enabled():\n"
    b"        # Dual-GPU branch already ran fp8 conversion BEFORE distribute.\n"
    b"        convert_to_float8_training(\n"
    b"            transformer, module_filter_fn=module_filter_fn, config=Float8LinearConfig(pad_inner_dim=True)\n"
    b"        )\n"
)

changed = False
for label, old, new in [
    ("fp8 inside dual-gpu branch", old1, new1),
    ("skip duplicate fp8 call", old2, new2),
]:
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
