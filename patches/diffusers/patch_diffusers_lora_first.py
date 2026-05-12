"""Move the dual-GPU block (fp8 quant + distribute) AFTER add_adapter.

PEFT 0.19.1's torchao dispatcher (TorchaoLoraLinear) is broken with
torchao 0.17 — it calls __init__ without the required get_apply_tensor_subclass
kwarg. The workaround: add LoRA on normal bf16 weights FIRST (PEFT uses
normal LoraLinear, no torchao dispatch), then fp8-quant the base weights
(LoRA layers untouched, PEFT's LoraLinear still works because base.weight
is .data-replaced), THEN distribute.
"""
from pathlib import Path

path = Path(r"C:\diffusers-flux2\diffusers\examples\dreambooth\train_dreambooth_lora_flux2.py")
data = path.read_bytes()

# Block to remove (the early dual-GPU pull + quant + enable).
old_block = (
    b"    # Dual-GPU split: gated on FLUX2_DUAL_GPU env var. Must run BEFORE\n"
    b"    # the transformer.to() below - that .to() would OOM on a single 5090.\n"
    b"    if is_dual_gpu_enabled():\n"
    b"        # Defensive: from_pretrained with accelerate present can load\n"
    b"        # weights to cuda:0 via auto-dispatch. Pull EVERYTHING back to CPU\n"
    b"        # so enable_flux2_dual_gpu has a clean GPU to distribute into.\n"
    b"        import gc as _gc\n"
    b"        for _i in range(torch.cuda.device_count()):\n"
    b"            _free, _total = torch.cuda.mem_get_info(_i)\n"
    b"            print(f\"[dual-gpu] pre-distribute cuda:{_i} free={_free/1024**3:.1f}G / {_total/1024**3:.1f}G\")\n"
    b"        transformer.to(\"cpu\")\n"
    b"        if not args.remote_text_encoder and text_encoder is not None:\n"
    b"            text_encoder.to(\"cpu\")\n"
    b"        vae.to(\"cpu\")\n"
    b"        _gc.collect()\n"
    b"        torch.cuda.empty_cache()\n"
    b"        # fp8 WEIGHT-ONLY quant MUST happen before distribute when dual-GPU is on:\n"
    b"        # bf16 60 GB / 32 GB GPU can't fit even split 50/50. Float8WeightOnlyConfig\n"
    b"        # halves storage to ~30 GB. This is DIFFERENT from --do_fp8_training\n"
    b"        # (fp8 compute, weights stay bf16 = no memory savings).\n"
    b"        if args.do_fp8_training:\n"
    b"            from torchao.quantization import quantize_, Float8WeightOnlyConfig\n"
    b"            quantize_(transformer, Float8WeightOnlyConfig())\n"
    b"            _gc.collect()\n"
    b"            torch.cuda.empty_cache()\n"
    b"            print(\"[dual-gpu] fp8 weight-only quantization done on CPU\")\n"
    b"        for _i in range(torch.cuda.device_count()):\n"
    b"            _free, _total = torch.cuda.mem_get_info(_i)\n"
    b"            print(f\"[dual-gpu] post-cpu-pull cuda:{_i} free={_free/1024**3:.1f}G / {_total/1024**3:.1f}G\")\n"
    b"    transformer = enable_flux2_dual_gpu(transformer)\n"
)

# Replacement at original site: just a placeholder comment so line numbers don't shift wildly.
new_block_early = (
    b"    # Dual-GPU block moved to after add_adapter (PEFT's torchao dispatcher is\n"
    b"    # broken with torchao 0.17; need LoRA on bf16 weights, then quant base).\n"
)

# After add_adapter — insert the moved block.
old_after = b"    transformer.add_adapter(transformer_lora_config)\n"
new_after = (
    b"    transformer.add_adapter(transformer_lora_config)\n"
    b"\n"
    b"    # Dual-GPU: LoRA-first, then fp8-quant base, then distribute.\n"
    b"    if is_dual_gpu_enabled():\n"
    b"        import gc as _gc\n"
    b"        transformer.to(\"cpu\")\n"
    b"        if not args.remote_text_encoder and text_encoder is not None:\n"
    b"            text_encoder.to(\"cpu\")\n"
    b"        vae.to(\"cpu\")\n"
    b"        _gc.collect()\n"
    b"        torch.cuda.empty_cache()\n"
    b"        if args.do_fp8_training:\n"
    b"            from torchao.quantization import quantize_, Float8WeightOnlyConfig\n"
    b"            quantize_(transformer, Float8WeightOnlyConfig())\n"
    b"            _gc.collect()\n"
    b"            torch.cuda.empty_cache()\n"
    b'            print("[dual-gpu] fp8 weight-only quant done (post-add_adapter)")\n'
    b"        transformer = enable_flux2_dual_gpu(transformer)\n"
    b"        for _i in range(torch.cuda.device_count()):\n"
    b"            _free, _total = torch.cuda.mem_get_info(_i)\n"
    b'            print(f"[dual-gpu] post-distribute cuda:{_i} free={_free/1024**3:.1f}G / {_total/1024**3:.1f}G")\n'
)

changed = False
for label, old, new in [
    ("remove early dual-gpu block", old_block, new_block_early),
    ("insert dual-gpu after add_adapter", old_after, new_after),
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
