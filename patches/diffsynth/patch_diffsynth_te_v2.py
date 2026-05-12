"""Patch Flux2TextEncoder.forward (v2): forward output_hidden_states/output_attentions via kwargs.

v1 stripped output_hidden_states from the super().forward call entirely
because transformers 5.8 dropped it from positional args. But it's still
accepted via **kwargs (it's annotated on TransformersKwargs). Without
passing it through, output.hidden_states is None and DiffSynth crashes
in get_mistral_3_small_prompt_embeds at:
    out = torch.stack([output.hidden_states[k] for k in layers], dim=1)
"""
from pathlib import Path

path = Path(r"C:\DiffSynth-Studio\diffsynth\models\flux2_text_encoder.py")
data = path.read_bytes()

old = (
    b"        # transformers 5.8 trimmed Mistral3's positional args; pass everything\n"
    b"        # as kwargs so the call survives across transformers versions.\n"
    b"        return super().forward(\n"
    b"            input_ids=input_ids,\n"
    b"            pixel_values=pixel_values,\n"
    b"            attention_mask=attention_mask,\n"
    b"            position_ids=position_ids,\n"
    b"            past_key_values=past_key_values,\n"
    b"            inputs_embeds=inputs_embeds,\n"
    b"            labels=labels,\n"
    b"            use_cache=use_cache,\n"
    b"            logits_to_keep=logits_to_keep,\n"
    b"            image_sizes=image_sizes,\n"
    b"            **kwargs,\n"
    b"        )"
)
new = (
    b"        # transformers 5.8 trimmed Mistral3's positional args; pass everything\n"
    b"        # as kwargs so the call survives across transformers versions. The dropped\n"
    b"        # output_hidden_states / output_attentions are still honored via\n"
    b"        # TransformersKwargs -> forward them through **kwargs.\n"
    b"        if output_hidden_states is not None:\n"
    b"            kwargs.setdefault(\"output_hidden_states\", output_hidden_states)\n"
    b"        if output_attentions is not None:\n"
    b"            kwargs.setdefault(\"output_attentions\", output_attentions)\n"
    b"        return super().forward(\n"
    b"            input_ids=input_ids,\n"
    b"            pixel_values=pixel_values,\n"
    b"            attention_mask=attention_mask,\n"
    b"            position_ids=position_ids,\n"
    b"            past_key_values=past_key_values,\n"
    b"            inputs_embeds=inputs_embeds,\n"
    b"            labels=labels,\n"
    b"            use_cache=use_cache,\n"
    b"            logits_to_keep=logits_to_keep,\n"
    b"            image_sizes=image_sizes,\n"
    b"            **kwargs,\n"
    b"        )"
)

if new in data:
    print("ALREADY_PATCHED")
elif old in data:
    path.write_bytes(data.replace(old, new, 1))
    print(f"PATCHED -- WROTE {path}")
else:
    print("PATTERN_NOT_FOUND")
    raise SystemExit(1)
