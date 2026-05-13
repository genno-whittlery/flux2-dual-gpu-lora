"""Idempotent patch: wire Wan22DualGPUMixin into ai-toolkit's Wan trainer.

Companion to ``wan22_dual_gpu.py`` at the repo root, which is the drop-in
mixin module — copy that file to
``<ai-toolkit>/extensions_built_in/diffusion_models/wan22/wan22_dual_gpu.py``
first. This script then makes two surgical edits so the mixin actually runs:

  1. ``wan22_5b_model.py``  — import the mixin and prepend it to the
     ``Wan225bModel`` MRO so ``setup_dual_gpu_distribution`` resolves on
     the class.
  2. ``wan21.py``  — soften the existing
     ``raise ValueError("Splitting model over gpus is not supported for
     Wan2.1 models")``; when ``WAN_DUAL_GPU=true`` *and* the model class
     provides ``setup_dual_gpu_distribution`` (i.e. it inherits the
     mixin), keep the transformer on CPU through quantize, then call the
     mixin's distribution method.

Env gate stays at ``WAN_DUAL_GPU=true`` (mirrors ``FLUX2_DUAL_GPU``).
Idempotent: running twice is a no-op (the script checks for the
post-patch markers before editing).

Adjust ``AI_TOOLKIT_ROOT`` below if your ai-toolkit checkout lives
somewhere other than ``C:\\ai-toolkit``.
"""
from pathlib import Path
import sys

AI_TOOLKIT_ROOT = Path(r"C:\ai-toolkit")


# ── Edit 1: wan22_5b_model.py — add mixin to MRO ──────────────────────────────

p_5b = AI_TOOLKIT_ROOT / "extensions_built_in" / "diffusion_models" / "wan22" / "wan22_5b_model.py"
src_5b = p_5b.read_text()

import_marker = "from .wan22_dual_gpu import Wan22DualGPUMixin"
class_after = "class Wan225bModel(Wan22DualGPUMixin, Wan21):"
class_before = "class Wan225bModel(Wan21):"

if import_marker in src_5b and class_after in src_5b:
    print(f"wan22_5b_model.py: ALREADY_PATCHED")
else:
    if class_before not in src_5b:
        sys.exit(f"PATTERN_NOT_FOUND in {p_5b} (expected '{class_before}')")
    # Inject the import after the last existing relative import we know about.
    import_anchor = (
        "from toolkit.models.wan21.wan_utils import add_first_frame_conditioning_v22"
    )
    if import_anchor not in src_5b:
        sys.exit(f"PATTERN_NOT_FOUND in {p_5b} (expected anchor '{import_anchor}')")
    src_5b = src_5b.replace(
        import_anchor,
        import_anchor + "\n" + import_marker,
    )
    src_5b = src_5b.replace(class_before, class_after)
    p_5b.write_text(src_5b)
    print(f"wan22_5b_model.py: PATCHED")


# ── Edit 2: wan21.py — soften raise + call setup_dual_gpu_distribution ────────

p_wan21 = AI_TOOLKIT_ROOT / "toolkit" / "models" / "wan21" / "wan21.py"
src_wan21 = p_wan21.read_text()

marker_use_dual_gpu = "use_dual_gpu = (\n            hasattr(self, 'setup_dual_gpu_distribution')"
marker_call = "if use_dual_gpu:\n            self.setup_dual_gpu_distribution(transformer, dtype)"

if marker_use_dual_gpu in src_wan21 and marker_call in src_wan21:
    print(f"wan21.py: ALREADY_PATCHED")
else:
    # Edit A: replace the raise with the conditional dispatch block.
    old_a = (
        "        if self.model_config.split_model_over_gpus:\n"
        "            raise ValueError(\n"
        "                \"Splitting model over gpus is not supported for Wan2.1 models\")\n"
        "\n"
        "        if self.model_config.low_vram:\n"
        "            # quantize on the device\n"
        "            transformer.to('cpu', dtype=dtype)\n"
        "            flush()\n"
        "        else:\n"
        "            transformer.to(self.device_torch, dtype=dtype)\n"
        "            flush()\n"
    )
    new_a = (
        "        # Dual-GPU model-parallel path is supplied by a subclass mixin\n"
        "        # (Wan22DualGPUMixin on Wan225bModel). Activates when WAN_DUAL_GPU=true\n"
        "        # in the environment; keeps the transformer on CPU through quantize so\n"
        "        # the mixin can distribute modules across cuda:0/cuda:1 afterward.\n"
        "        use_dual_gpu = (\n"
        "            hasattr(self, 'setup_dual_gpu_distribution')\n"
        "            and os.getenv(\"WAN_DUAL_GPU\", \"false\").lower() == \"true\"\n"
        "        )\n"
        "        if self.model_config.split_model_over_gpus and not use_dual_gpu:\n"
        "            raise ValueError(\n"
        "                \"Splitting model over gpus is not supported for Wan2.1 models\")\n"
        "\n"
        "        if self.model_config.low_vram or use_dual_gpu:\n"
        "            # quantize on CPU; for dual-GPU, distribution happens post-quant\n"
        "            transformer.to('cpu', dtype=dtype)\n"
        "            flush()\n"
        "        else:\n"
        "            transformer.to(self.device_torch, dtype=dtype)\n"
        "            flush()\n"
    )
    if old_a not in src_wan21:
        sys.exit(f"PATTERN_NOT_FOUND in {p_wan21} (Edit A — raise+to block)")
    src_wan21 = src_wan21.replace(old_a, new_a)

    # Edit B: call setup_dual_gpu_distribution after quantize_model.
    old_b = (
        "        if self.model_config.quantize:\n"
        "            self.print_and_status_update(\"Quantizing Transformer\")\n"
        "            quantize_model(self, transformer)\n"
        "            flush()\n"
        "        \n"
        "        if self.model_config.layer_offloading and self.model_config.layer_offloading_transformer_percent > 0:"
    )
    new_b = (
        "        if self.model_config.quantize:\n"
        "            self.print_and_status_update(\"Quantizing Transformer\")\n"
        "            quantize_model(self, transformer)\n"
        "            flush()\n"
        "\n"
        "        if use_dual_gpu:\n"
        "            self.setup_dual_gpu_distribution(transformer, dtype)\n"
        "            flush()\n"
        "\n"
        "        if self.model_config.layer_offloading and self.model_config.layer_offloading_transformer_percent > 0:"
    )
    if old_b not in src_wan21:
        sys.exit(f"PATTERN_NOT_FOUND in {p_wan21} (Edit B — quantize+offload)")
    src_wan21 = src_wan21.replace(old_b, new_b)

    p_wan21.write_text(src_wan21)
    print(f"wan21.py: PATCHED")
