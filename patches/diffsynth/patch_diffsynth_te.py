"""Patch Flux2TextEncoder.forward to use kwargs-only super().forward call.

transformers 5.8 dropped output_attentions / output_hidden_states /
return_dict / cache_position from Mistral3ForConditionalGeneration's
positional args; DiffSynth's wrapper passes them positionally and
crashes with "takes 1 to 11 positional arguments but 15 were given".
"""
from pathlib import Path

path = Path(r"C:\DiffSynth-Studio\diffsynth\models\flux2_text_encoder.py")
data = path.read_bytes()

old = (
    b"        return super().forward(input_ids, pixel_values, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, cache_position, logits_to_keep, image_sizes, **kwargs)"
)
new = (
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

if new in data:
    print("ALREADY_PATCHED")
elif old in data:
    path.write_bytes(data.replace(old, new, 1))
    print(f"PATCHED -- WROTE {path}")
else:
    print("PATTERN_NOT_FOUND")
    raise SystemExit(1)
