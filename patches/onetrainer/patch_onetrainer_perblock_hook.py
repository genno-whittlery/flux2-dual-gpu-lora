"""Register the device-bridge pre-hook on EVERY single_transformer_block
in [split_at:], not just the boundary block.

Modulation (temb) is computed once and passed to each single_transformer_block
in a loop. The boundary pre-hook moves it to cuda:1 on entry to block[split_at],
but when the loop then calls block[split_at+1], the caller-side `temb` is still
the original cuda:0 tensor — only block[split_at]'s args were transformed, not
the loop-level temb. Registering the same hook on every block in the cuda:1
range ensures all subsequent calls also see their args on cuda:1.
"""
from pathlib import Path

path = Path(r"C:\OneTrainer\modules\util\Flux2DualGpu.py")
data = path.read_bytes()

old = (
    b"    transformer.single_transformer_blocks[split_at].register_forward_pre_hook(\n"
    b"        _make_device_bridge_hook(cuda1), with_kwargs=True\n"
    b"    )\n"
)
new = (
    b"    # Per-block pre-hook: every single_transformer_block on cuda:1\n"
    b"    # needs its args moved (temb is shared across the loop, so only\n"
    b"    # hooking the boundary block leaves temb on cuda:0 for subsequent\n"
    b"    # blocks).\n"
    b"    for block in transformer.single_transformer_blocks[split_at:]:\n"
    b"        block.register_forward_pre_hook(\n"
    b"            _make_device_bridge_hook(cuda1), with_kwargs=True\n"
    b"        )\n"
)
if new in data:
    print("ALREADY_PATCHED")
elif old in data:
    path.write_bytes(data.replace(old, new))
    print("PATCHED")
else:
    raise SystemExit("PATTERN_NOT_FOUND")
