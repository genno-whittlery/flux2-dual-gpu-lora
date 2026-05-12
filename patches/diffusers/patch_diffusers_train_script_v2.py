"""Second-round patch for diffusers' train_dreambooth_lora_flux2.py.

The first patch added a no-op enable call after add_adapter (line 1287),
but the script hits `transformer.to(accelerator.device)` at line 1252 which
OOMs before reaching the adapter step. This patch:

1. Removes the misplaced enable call after add_adapter (we put it earlier).
2. Adds the enable call right after `transformer.requires_grad_(False)` -
   distribute the transformer while it's still on CPU.
3. Wraps `transformer.to(**transformer_to_kwargs)` in a dual-GPU guard so
   it doesn't move the distributed transformer back onto a single GPU.
4. Wraps `text_encoder.to(**to_kwargs)` in a dual-GPU guard so Mistral
   stays on CPU (we don't have room on cuda:0 with the transformer split).
"""
from pathlib import Path

path = Path(r"C:\diffusers-flux2\diffusers\examples\dreambooth\train_dreambooth_lora_flux2.py")
data = path.read_bytes()

# 1. Remove the no-op enable that landed after add_adapter (idempotent - we'll
#    add it in the right place in step 2). The lines added by the previous
#    patch are exact, so we delete them.
old1 = (
    b"    transformer.add_adapter(transformer_lora_config)\n"
    b"    # Dual-GPU split: gated on FLUX2_DUAL_GPU env var.\n"
    b"    transformer = enable_flux2_dual_gpu(transformer)\n"
)
new1 = b"    transformer.add_adapter(transformer_lora_config)\n"

# 2. Insert the enable call right after `transformer.requires_grad_(False)`.
#    This puts it BEFORE the OOM-causing transformer.to() at line 1252.
old2 = (
    b"    # We only train the additional adapter LoRA layers\n"
    b"    transformer.requires_grad_(False)\n"
    b"    vae.requires_grad_(False)\n"
)
new2 = (
    b"    # We only train the additional adapter LoRA layers\n"
    b"    transformer.requires_grad_(False)\n"
    b"    vae.requires_grad_(False)\n"
    b"\n"
    b"    # Dual-GPU split: gated on FLUX2_DUAL_GPU env var. Must run BEFORE\n"
    b"    # the transformer.to() below - that .to() would OOM on a single 5090.\n"
    b"    transformer = enable_flux2_dual_gpu(transformer)\n"
)

# 3. Guard the transformer.to call so it doesn't run under dual-GPU.
old3 = (
    b"    is_fsdp = getattr(accelerator.state, \"fsdp_plugin\", None) is not None\n"
    b"    if not is_fsdp:\n"
    b"        transformer.to(**transformer_to_kwargs)\n"
)
new3 = (
    b"    is_fsdp = getattr(accelerator.state, \"fsdp_plugin\", None) is not None\n"
    b"    if not is_fsdp and not is_dual_gpu_enabled():\n"
    b"        # Dual-GPU skips this - enable_flux2_dual_gpu has already placed\n"
    b"        # the transformer across cuda:0/cuda:1 with the correct dtype.\n"
    b"        transformer.to(**transformer_to_kwargs)\n"
)

# 4. Guard text_encoder.to to keep Mistral on CPU under dual-GPU.
old4 = (
    b"    if not args.remote_text_encoder:\n"
    b"        text_encoder.to(**to_kwargs)\n"
)
new4 = (
    b"    if not args.remote_text_encoder:\n"
    b"        if is_dual_gpu_enabled():\n"
    b"            # Mistral on CPU: transformer split already occupies cuda:0.\n"
    b"            text_encoder.to(dtype=weight_dtype)\n"
    b"        else:\n"
    b"            text_encoder.to(**to_kwargs)\n"
)

changed = False
for label, old, new in [
    ("remove dup enable", old1, new1),
    ("enable after requires_grad", old2, new2),
    ("guard transformer.to", old3, new3),
    ("guard text_encoder.to", old4, new4),
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
