"""Swap convert_to_float8_training (fp8 compute) for quantize_ with
Float8WeightOnlyConfig (real fp8 storage that halves memory).
"""
from pathlib import Path

path = Path(r"C:\diffusers-flux2\diffusers\examples\dreambooth\train_dreambooth_lora_flux2.py")
data = path.read_bytes()

old = (
    b"        # fp8 quantization MUST happen before distribute when dual-GPU is on:\n"
    b"        # bf16 60 GB / 32 GB GPU can't fit even split 50/50. fp8 halves it.\n"
    b"        if args.do_fp8_training:\n"
    b"            from torchao.float8 import Float8LinearConfig, convert_to_float8_training\n"
    b"            convert_to_float8_training(\n"
    b"                transformer, module_filter_fn=module_filter_fn,\n"
    b"                config=Float8LinearConfig(pad_inner_dim=True),\n"
    b"            )\n"
    b'            print("[dual-gpu] fp8 conversion done on CPU")\n'
)
new = (
    b"        # fp8 WEIGHT-ONLY quant MUST happen before distribute when dual-GPU is on:\n"
    b"        # bf16 60 GB / 32 GB GPU can't fit even split 50/50. Float8WeightOnlyConfig\n"
    b"        # halves storage to ~30 GB. This is DIFFERENT from --do_fp8_training\n"
    b"        # (fp8 compute, weights stay bf16 = no memory savings).\n"
    b"        if args.do_fp8_training:\n"
    b"            from torchao.quantization import quantize_, Float8WeightOnlyConfig\n"
    b"            quantize_(transformer, Float8WeightOnlyConfig())\n"
    b"            _gc.collect()\n"
    b"            torch.cuda.empty_cache()\n"
    b'            print("[dual-gpu] fp8 weight-only quantization done on CPU")\n'
)
if new in data:
    print("ALREADY_PATCHED")
elif old in data:
    path.write_bytes(data.replace(old, new))
    print("PATCHED")
else:
    raise SystemExit("PATTERN_NOT_FOUND")
