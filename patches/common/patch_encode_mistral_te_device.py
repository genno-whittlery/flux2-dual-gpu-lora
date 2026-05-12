"""Move tokens to the text_encoder's device before the forward call.

mgds.EncodeMistralText assumes tokens and text_encoder are on the same device.
With FLUX2_DUAL_GPU=true our OneTrainer patch puts Mistral on CPU but tokens
are still produced on cuda:0 by the upstream Tokenize module. F.embedding
then fails with "Expected all tensors to be on the same device".

This patch makes the encoder a device-agnostic step by always moving inputs
to the text_encoder's first-parameter device. Harmless on single-device
setups (`.to(same_device)` is a no-op).
"""
from pathlib import Path

path = Path(r"C:\OneTrainer\.venv\src\mgds\src\mgds\pipelineModules\EncodeMistralText.py")
data = path.read_bytes()

old = (
    b"        with self._all_contexts(self.autocast_contexts):\n"
    b"            text_encoder_output = self.text_encoder(\n"
    b"                tokens,\n"
    b"                attention_mask=tokens_attention_mask.float(),\n"
)
new = (
    b"        # Move inputs to the text_encoder's device (CPU under dual-GPU)\n"
    b"        _te_device = next(self.text_encoder.parameters()).device\n"
    b"        tokens = tokens.to(_te_device)\n"
    b"        if tokens_attention_mask is not None:\n"
    b"            tokens_attention_mask = tokens_attention_mask.to(_te_device)\n"
    b"        with self._all_contexts(self.autocast_contexts):\n"
    b"            text_encoder_output = self.text_encoder(\n"
    b"                tokens,\n"
    b"                attention_mask=tokens_attention_mask.float(),\n"
)
if new in data:
    print("ALREADY_PATCHED")
elif old in data:
    path.write_bytes(data.replace(old, new))
    print("PATCHED")
else:
    raise SystemExit("PATTERN_NOT_FOUND")
