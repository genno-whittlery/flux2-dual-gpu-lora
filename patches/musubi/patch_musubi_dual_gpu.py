"""One-shot patch: musubi-tuner hv_train_network.py.

Skip auto-enabling DDP when FLUX2_DUAL_GPU=true. Preserves the original
file's line endings (Windows CRLF) so diff-noise stays away.
"""
from pathlib import Path

path = Path(r"C:\musubi-tuner\src\musubi_tuner\hv_train_network.py")
data = path.read_bytes()

old = b"            if torch.cuda.device_count() > 1\r\n            else None"
new = (
    b'            if torch.cuda.device_count() > 1 and '
    b'os.environ.get("FLUX2_DUAL_GPU", "false").lower() != "true"\r\n'
    b"            else None"
)
if old not in data:
    # already patched or file uses LF
    if new in data:
        print("ALREADY_PATCHED")
        raise SystemExit(0)
    old_lf = old.replace(b"\r\n", b"\n")
    new_lf = new.replace(b"\r\n", b"\n")
    if old_lf in data:
        data = data.replace(old_lf, new_lf)
        path.write_bytes(data)
        print("PATCHED_LF")
        raise SystemExit(0)
    raise SystemExit("PATTERN_NOT_FOUND")
data = data.replace(old, new)
path.write_bytes(data)
print("PATCHED_CRLF")
