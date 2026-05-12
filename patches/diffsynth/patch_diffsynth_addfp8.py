"""Patch: insert fp8 weight-only quant BEFORE enable_flux2_dual_gpu in train.py.

Without this, the bf16 FLUX.2 transformer (~60 GB) cannot fit on two
32 GB cards even after a midpoint split -- 8 double_blocks plus the first
half of single_blocks alone already exceeds the cuda:0 budget. The fp8
filter excludes LoRA's lora_A/lora_B submodules so their requires_grad
survives quant (backward pass breaks otherwise).
"""
from pathlib import Path

path = Path(r"C:\DiffSynth-Studio\examples\flux2\model_training\train.py")
data = path.read_bytes()

old_block = (
    b"        for _i in range(torch.cuda.device_count()):\n"
    b"            _free, _total = torch.cuda.mem_get_info(_i)\n"
    b"            print(f\"[dual-gpu] pre-distribute cuda:{_i} free={_free/1024**3:.1f}G / {_total/1024**3:.1f}G\")\n"
    b"        enable_flux2_dual_gpu(model.pipe.dit)\n"
)
new_block = (
    b"        for _i in range(torch.cuda.device_count()):\n"
    b"            _free, _total = torch.cuda.mem_get_info(_i)\n"
    b"            print(f\"[dual-gpu] pre-distribute cuda:{_i} free={_free/1024**3:.1f}G / {_total/1024**3:.1f}G\")\n"
    b"        # fp8 weight-only quant on CPU BEFORE distribute. Without this the\n"
    b"        # bf16 transformer (~60 GB) overflows a 32 GB card on either side.\n"
    b"        # Filter excludes LoRA's lora_A/lora_B Linear submodules -- quantizing\n"
    b"        # them strips requires_grad and breaks backward.\n"
    b"        import gc as _gc\n"
    b"        from torchao.quantization import quantize_, Float8WeightOnlyConfig\n"
    b"        def _quant_filter(module, name):\n"
    b"            if not isinstance(module, torch.nn.Linear):\n"
    b"                return False\n"
    b"            return \"lora_A\" not in name and \"lora_B\" not in name\n"
    b"        quantize_(model.pipe.dit, Float8WeightOnlyConfig(), filter_fn=_quant_filter)\n"
    b"        _gc.collect()\n"
    b"        print(\"[dual-gpu] fp8 weight-only quant done (LoRA params unmodified)\")\n"
    b"        enable_flux2_dual_gpu(model.pipe.dit)\n"
)

if new_block in data:
    print("ALREADY_PATCHED")
elif old_block in data:
    path.write_bytes(data.replace(old_block, new_block, 1))
    print(f"PATCHED -- WROTE {path}")
else:
    print("PATTERN_NOT_FOUND")
    raise SystemExit(1)
