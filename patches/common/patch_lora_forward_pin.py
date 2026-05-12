"""Inject device-pin at the top of LoRAModule.forward.

Replacing lora.forward via instance assignment doesn't work because
LoRAModule.apply_to does `self.org_module.forward = self.forward`, which
captures the original bound method by reference. The wrapped layer always
calls the original LoRAModule.forward — not whatever we monkey-patch onto
the instance after.

The reliable fix is to modify LoRAModule.forward directly so the device
pin runs at the very top of every call. The pin is a no-op outside dual-GPU
mode (env var unset → early return).
"""
from pathlib import Path

path = Path(r"C:\musubi-tuner\src\musubi_tuner\networks\lora.py")
data = path.read_bytes()

# Insert right after the def line. The patched form is opt-in via FLUX2_DUAL_GPU
# so it doesn't perturb non-dual-GPU paths at all.
marker = b"    def forward(self, x):\n        org_forwarded = self.org_forward(x)\n"
inject = (
    b"    def forward(self, x):\n"
    b"        # Dual-GPU model-parallel pin: when FLUX2_DUAL_GPU=true, the\n"
    b"        # wrapped layer may live on a different device than this LoRA\n"
    b"        # (e.g., layer on cuda:1, LoRA was prepared on cuda:0 by\n"
    b"        # accelerator.prepare(network)). The bound-method indirection\n"
    b"        # in apply_to makes instance-level forward-shimming unreliable,\n"
    b"        # so we do the pin inline here.\n"
    b'        import os as _os\n'
    b'        if _os.environ.get("FLUX2_DUAL_GPU", "false").lower() == "true":\n'
    b"            try:\n"
    b"                _wrapped_module = self.org_forward.__self__\n"
    b"                _target_device = next(_wrapped_module.parameters()).device\n"
    b"                _current_device = next(self.parameters()).device\n"
    b"                if _current_device != _target_device:\n"
    b"                    self.to(_target_device)\n"
    b"            except (AttributeError, StopIteration):\n"
    b"                pass\n"
    b"        org_forwarded = self.org_forward(x)\n"
)
if inject in data:
    print("ALREADY_PATCHED")
elif marker in data:
    path.write_bytes(data.replace(marker, inject))
    print("PATCHED")
else:
    raise SystemExit("PATTERN_NOT_FOUND")
