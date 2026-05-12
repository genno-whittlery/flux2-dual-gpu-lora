"""Force the transformer to CPU right before enable_flux2_dual_gpu.

Defensive: from_pretrained with accelerate installed can load weights
directly to cuda:0 (auto-dispatch). If that's already happened, the
distribute step inside enable_flux2_dual_gpu can OOM because the model
is sitting on cuda:0 AND we're trying to move blocks there one at a time.
Pull back to CPU first, then distribute from CPU.
"""
from pathlib import Path

path = Path(r"C:\diffusers-flux2\diffusers\examples\dreambooth\train_dreambooth_lora_flux2.py")
data = path.read_bytes()

old = (
    b"    # Dual-GPU split: gated on FLUX2_DUAL_GPU env var. Must run BEFORE\n"
    b"    # the transformer.to() below - that .to() would OOM on a single 5090.\n"
    b"    transformer = enable_flux2_dual_gpu(transformer)\n"
)
new = (
    b"    # Dual-GPU split: gated on FLUX2_DUAL_GPU env var. Must run BEFORE\n"
    b"    # the transformer.to() below - that .to() would OOM on a single 5090.\n"
    b"    if is_dual_gpu_enabled():\n"
    b"        # Defensive: from_pretrained with accelerate present can load\n"
    b"        # weights directly to cuda:0 (auto-dispatch). Pull back to CPU so\n"
    b"        # the per-block placement below has room to work.\n"
    b"        transformer.to(\"cpu\")\n"
    b"        torch.cuda.empty_cache()\n"
    b"    transformer = enable_flux2_dual_gpu(transformer)\n"
)
if new in data:
    print("ALREADY_PATCHED")
elif old in data:
    path.write_bytes(data.replace(old, new))
    print("PATCHED")
else:
    raise SystemExit("PATTERN_NOT_FOUND")
