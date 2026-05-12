"""Force text_encoder to CPU when FLUX2_DUAL_GPU=true.

OneTrainer calls text_encoder_to(self.train_device) during prepare_text_caching,
which dumps the 44 GB Mistral on cuda:0 — but cuda:0 already holds half the
transformer from our dual-GPU split. Force CPU instead; Mistral encodes 32
captions in ~7 minutes on CPU (validated with musubi).
"""
from pathlib import Path

path = Path(r"C:\OneTrainer\modules\model\Flux2Model.py")
data = path.read_bytes()

old = (
    b"    def text_encoder_to(self, device: torch.device):\n"
    b"        if self.text_encoder is not None:\n"
)
new = (
    b"    def text_encoder_to(self, device: torch.device):\n"
    b"        # FLUX2_DUAL_GPU=true: text encoder runs on CPU to avoid\n"
    b"        # contending with the transformer split that's already\n"
    b"        # occupying cuda:0. Mistral is frozen during LoRA training\n"
    b"        # and only encodes captions once before caching, so CPU is\n"
    b"        # fine performance-wise.\n"
    b'        import os as _os\n'
    b'        if _os.environ.get("FLUX2_DUAL_GPU", "false").lower() == "true":\n'
    b'            device = torch.device("cpu")\n'
    b"        if self.text_encoder is not None:\n"
)
if new in data:
    print("ALREADY_PATCHED")
elif old in data:
    path.write_bytes(data.replace(old, new))
    print("PATCHED")
else:
    raise SystemExit("PATTERN_NOT_FOUND")
