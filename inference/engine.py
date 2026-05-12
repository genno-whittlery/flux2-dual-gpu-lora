"""Dual-GPU FLUX.2 (+ ai-toolkit LoRA) inference engine.

Drives ai-toolkit's ``Flux2Model`` / ``LoRASpecialNetwork`` / ``generate_images``
directly, so it runs on the same dual-GPU + Mistral-on-CPU setup the training patch
enables — which ai-toolkit's own ``run.py`` "generate" / "sample-during-training"
paths cannot (they deadlock at the embed-cache/unload step and OOM trying to put
Mistral on cuda:0). ComfyUI has no dual-GPU FLUX.2 path at all, so this is the only
way to render a LoRA trained with this patch under the exact same split + quantization
the training ran in (fp8 weight-only transformer across cuda:0+cuda:1, Mistral bf16 in
system RAM).

Run inside ai-toolkit's venv. Set ``AITK_DIR`` if ai-toolkit isn't at ``C:\\ai-toolkit``.

Public API
----------
    load_flux2(base=..., lora=LoraSpec(path, rank, alpha) | None) -> (model, pipeline, network|None)
    render(model, pipeline, prompts={name: text}, out=..., network=..., strengths=(...,), ...) -> int
"""
import os
import sys
from dataclasses import dataclass

# --- env must be set BEFORE importing ai-toolkit ---
os.environ.setdefault("FLUX2_DUAL_GPU", "true")
os.environ.setdefault("FLUX2_TE_DEVICE", "cpu")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_AITK = os.environ.get("AITK_DIR", r"C:\ai-toolkit")
if _AITK not in sys.path:
    sys.path.insert(0, _AITK)

import torch  # noqa: E402
import toolkit.models.base_model as _bm  # noqa: E402

DEFAULT_BASE = os.environ.get("FLUX2_BASE", r"F:\models\diffusion_models\FLUX.2-dev")

# ------------------------------------------------------------------ monkeypatches
# 1. keep the (CPU-resident) text encoder off cuda inside set_device_state — the
#    train-device preset hardcodes Mistral onto device_torch (cuda:0), which OOMs
#    on top of the ~21 GB transformer half.
_BM = getattr(_bm, "BaseModel", None) or getattr(_bm, "StableDiffusion", None)
if _BM is not None and hasattr(_BM, "set_device_state"):
    _orig_sds = _BM.set_device_state

    def _patched_sds(self, state):
        te = getattr(self, "te_device_torch", None)
        if te is not None and isinstance(state, dict) and isinstance(state.get("text_encoder"), (dict, list)):
            ts = state["text_encoder"]
            for s in (ts if isinstance(ts, list) else [ts]):
                if isinstance(s, dict):
                    s["device"] = str(te)
        return _orig_sds(self, state)

    _BM.set_device_state = _patched_sds

# 2. block transformers.PreTrainedModel.to(cuda...) so Mistral stays in system RAM
import transformers.modeling_utils as _tmu  # noqa: E402

_orig_to = _tmu.PreTrainedModel.to


def _to_no_cuda(self, *a, **k):
    d = k.get("device")
    if d is None and a:
        a0 = a[0]
        if isinstance(a0, str) or hasattr(a0, "type"):
            d = a0
    if d is not None and "cuda" in str(d):
        return self
    return _orig_to(self, *a, **k)


_tmu.PreTrainedModel.to = _to_no_cuda

# 3. tolerate cross-device quanto: move activations onto the quantized weight's device
try:
    import optimum.quanto.library.qbytes_mm as _qmm

    _orig_qbm = _qmm.qbytes_mm

    def _coerce_qbm(activations, weights, output_scales):
        try:
            wd = weights.device
            if activations.device != wd:
                activations = activations.to(wd)
            if hasattr(output_scales, "device") and output_scales.device != wd:
                output_scales = output_scales.to(wd)
        except Exception:
            pass
        return _orig_qbm(activations, weights, output_scales)

    _qmm.qbytes_mm = _coerce_qbm
    QUANTO_PATCH = "patched"
except Exception as e:  # quanto layout varies by version; non-fatal
    QUANTO_PATCH = f"skip ({e!r})"


@dataclass
class LoraSpec:
    path: str
    rank: int = 16
    alpha: int = 16


def _log(*a):
    print("[flux2-infer]", *a, flush=True)


def load_flux2(base=DEFAULT_BASE, lora=None, *, primary_device="cuda:0", dtype="bf16", log=_log):
    """Load FLUX.2 (qfloat8 transformer, dual-GPU split, Mistral on CPU); optionally apply a LoRA.

    Returns ``(model, pipeline, network)`` where ``network`` is the ``LoRASpecialNetwork`` or ``None``.
    """
    from toolkit.config_modules import ModelConfig, NetworkConfig
    from toolkit.util.get_model import get_model_class
    from toolkit.lora_special import LoRASpecialNetwork

    log(f"dual_gpu={os.environ.get('FLUX2_DUAL_GPU')} te_device={os.environ.get('FLUX2_TE_DEVICE')} "
        f"quanto={QUANTO_PATCH} cuda_devices={torch.cuda.device_count()}")
    log(f"base={base}" + (f"  lora={lora.path} (rank {lora.rank}/alpha {lora.alpha})" if lora else "  (no lora)"))

    mc = ModelConfig(name_or_path=base, arch="flux2", dtype=dtype, quantize=True, quantize_te=False)
    ModelClass = get_model_class(mc)
    log(f"model class = {ModelClass.__name__}")
    sd = ModelClass(primary_device, mc, dtype)
    sd.load_model()
    log("model loaded")

    net = None
    if lora is not None:
        nc = NetworkConfig(type="lora", linear=lora.rank, linear_alpha=lora.alpha, transformer_only=True)
        nk = dict(nc.network_kwargs or {})
        if hasattr(sd, "target_lora_modules"):
            nk["target_lin_modules"] = sd.target_lora_modules
        unet = getattr(sd, "unet", None) or getattr(sd, "transformer", None) or sd.get_model_to_train()
        te = sd.text_encoder
        net = LoRASpecialNetwork(
            text_encoder=te,
            unet=sd.get_model_to_train(),
            lora_dim=nc.linear, multiplier=1.0, alpha=nc.linear_alpha,
            train_unet=True, train_text_encoder=False,
            conv_lora_dim=nc.conv, conv_alpha=nc.conv_alpha,
            is_sdxl=False, is_v2=False, is_v3=False, is_pixart=False, is_auraflow=False,
            is_flux=False, is_lumina2=False, is_ssd=False, is_vega=False,
            dropout=nc.dropout, use_text_encoder_1=True, use_text_encoder_2=True,
            use_bias=False, is_lorm=False,
            network_config=nc, network_type=nc.type, transformer_only=nc.transformer_only,
            is_transformer=getattr(sd, "is_transformer", True), base_model=sd, **nk,
        )
        net.force_to(sd.device_torch, dtype=torch.float32)
        sd.network = net
        if hasattr(net, "_update_torch_multiplier"):
            net._update_torch_multiplier()
        net.apply_to(te, unet, False, True)
        net.can_merge_in = False
        extra = net.load_weights(lora.path)
        log(f"LoRA applied + weights loaded (load_weights -> {extra!r})")

    pipeline = sd.get_generation_pipeline()
    # The diffusers Flux2Pipeline's _execution_device resolves to CPU (Mistral lives there),
    # which would create the latent + position ids on CPU and feed CPU tensors to the cuda
    # transformer. Pin it to the transformer's primary half. Mistral encoding is forced onto
    # te_device_torch (CPU) inside get_prompt_embeds regardless, so this is safe.
    _dev = torch.device(primary_device)
    try:
        type(pipeline)._execution_device = property(lambda self: _dev)
    except Exception as e:
        log(f"warning: could not pin _execution_device: {e!r}")
    log(f"generation pipeline ready (_execution_device -> {primary_device})")
    return sd, pipeline, net


def render(sd, pipeline, prompts, *, out, network=None, strengths=(1.0,),
           width=1024, height=1024, steps=20, cfg=4.0, seed=42, ext="png", log=_log):
    """Render ``{name: prompt}`` at each LoRA strength into ``out`` as ``{name}__s{NNN}.{ext}``.

    Returns the total number of images written.
    """
    from toolkit.config_modules import GenerateImageConfig

    os.makedirs(out, exist_ok=True)
    total = 0
    for strength in strengths:
        if network is not None:
            network.multiplier = strength
        configs = []
        for name, prompt in prompts.items():
            tag = f"{name}__s{int(round(strength * 100)):03d}"
            configs.append(GenerateImageConfig(
                prompt=prompt, width=width, height=height,
                num_inference_steps=steps, guidance_scale=cfg,
                negative_prompt="", seed=seed, network_multiplier=strength,
                output_path=os.path.join(out, f"{tag}.{ext}"), output_ext=f".{ext}",
            ))
        log(f"strength {strength}: generating {len(configs)} images -> {out}")
        sd.generate_images(configs, pipeline=pipeline)
        total += len(configs)
    log(f"done: {total} images -> {out}")
    return total
