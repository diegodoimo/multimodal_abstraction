"""Attention-knockout and layer-skipping patches for InternVL2 (interventions and --no-image compositionality).

``patch`` replaces InternLM2Model.forward and InternVLChatModel.generate: generate records the image
span of each prompt, and in the layers of ``layers_to_skip`` the forward masks the attention of the
text that follows the image to the image span. Requires eager attention
(4D additive mask). ``ablation_mode="layer"`` replaces the layers with identity modules instead. The
image-removed benchmark baselines use patch_internvl_eval.py.
"""

from functools import partialmethod
from typing import List, Optional, Tuple, Union

import torch
from transformers import GenerationConfig
from transformers.modeling_outputs import BaseModelOutputWithPast

from utils.models.layer_skipping import apply_layer_skip


def ilm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    """InternLM2Model.forward (eager attention) with the ablation mask in ``layers_to_skip``.

    ``layers_to_skip`` and ``ablation_mode`` are bound by ``patch`` through partialmethod.
    """
    layers_to_skip = kwargs.pop("layers_to_skip", [])
    ablation_mode = kwargs.pop("ablation_mode", None)

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.tok_embeddings(input_ids)

    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
    causal_mask = self._prepare_decoder_attention_mask(
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )
    if causal_mask is None:
        raise RuntimeError("The InternVL2 ablation needs a 4D additive mask (eager attention).")

    # Ablation mask: the causal mask with the cells (queries after the image, keys in the image span
    # [start, end]) set to the dtype minimum. image_limits holds per sample (start, end) = positions of
    # <img> and </img>, set by ivl_generate (or by the caller of a plain forward).
    image_limits = getattr(self, "image_limits", None)
    image_mode = ablation_mode in ("image", "cross_modal")
    image_mask = None
    if layers_to_skip and image_mode and image_limits is not None:
        image_mask = causal_mask.clone()
        masked_value = (
            torch.finfo(image_mask.dtype).min
            if torch.is_floating_point(image_mask)
            else False
        )
        q_len = inputs_embeds.shape[1]
        for b, limit in enumerate(image_limits):
            if limit is None:
                continue
            start, end = limit
            if q_len > 1:
                # Prefill: block post-image-text queries from the image span.
                image_mask[b, :, end + 1 :, start : end + 1] = masked_value
            else:
                # Decode: single query sits after the image block.
                image_mask[b, :, :, start : end + 1] = masked_value

    hidden_states = inputs_embeds

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    is_prefill = inputs_embeds.shape[1] > 1
    stored_embeds = None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        layer_mask = causal_mask
        # "image": after the last skipped layer, write the image-span hidden states stored before
        # the block back (prefill only).
        does_restore = ablation_mode == "image"
        restore_source_present = image_mode and image_limits is not None
        if idx in layers_to_skip:
            if image_mask is not None:
                layer_mask = image_mask
            if (
                idx == layers_to_skip[0]
                and does_restore
                and is_prefill
                and restore_source_present
            ):
                stored_embeds = hidden_states.clone()
        elif (
            len(layers_to_skip) > 0
            and idx == layers_to_skip[-1] + 1
            and does_restore
            and is_prefill
            and stored_embeds is not None
            and restore_source_present
        ):
            for b, limit in enumerate(image_limits):
                if limit is not None:
                    start, end = limit
                    hidden_states[b, start : end + 1, :] = stored_embeds[b, start : end + 1, :]

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=layer_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


@torch.no_grad()
def ivl_generate(
    self,
    pixel_values: Optional[torch.FloatTensor] = None,
    input_ids: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    visual_features: Optional[torch.FloatTensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    output_hidden_states: Optional[bool] = None,
    **generate_kwargs,
) -> torch.LongTensor:
    """InternVLChatModel.generate that also stores the image spans read by ilm_forward."""
    assert self.img_context_token_id is not None
    # clear the positions of the previous call
    self.language_model.model.image_limits = None

    if pixel_values is not None:
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds = self.extract_feature(pixel_values)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if (input_ids == self.img_context_token_id).any():
            image_limits = []
            seq_len = input_ids.shape[1]
            for i in range(input_ids.shape[0]):
                indices = torch.where(input_ids[i] == self.img_context_token_id)[0]
                if indices.numel() > 0:
                    # +-1: the <img> / </img> delimiters around the IMG_CONTEXT tokens
                    start = max(0, indices[0].item() - 1)
                    end = min(seq_len - 1, indices[-1].item() + 1)
                    image_limits.append((start, end))
                else:
                    image_limits.append(None)

            self.language_model.model.image_limits = image_limits

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == self.img_context_token_id

        assert selected.sum() != 0
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)
    else:
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        use_cache=True,
        **generate_kwargs,
    )

    return outputs


_ABLATION_MODES = {"image", "cross_modal", "layer"}


def patch(model, layers_to_skip, ablation_mode="image"):
    """Apply an attention-knockout or layer-skipping ablation to the layers ``layers_to_skip``.

    Modes (``start``/``end`` = positions of <img> and </img>):
      * ``"image"``: text queries after the image cannot read the image span [start, end]; the image
        hidden states are restored after the block (interventions, --no-image compositionality).
      * ``"cross_modal"``: the same mask without the restore (interventions).
      * ``"layer"``: the layers are replaced with identity modules (text interventions).
    """
    if ablation_mode not in _ABLATION_MODES:
        raise ValueError(f"ablation_mode must be one of {sorted(_ABLATION_MODES)}, got {ablation_mode!r}")

    if ablation_mode == "layer":
        apply_layer_skip(model, layers_to_skip, model_type="internvl")
        return

    layers_to_skip = list(layers_to_skip or [])
    # InternLM2 uses `config.attn_implementation` (no leading underscore).
    inner_cfg = model.language_model.model.config
    attn_impl = getattr(
        inner_cfg,
        "attn_implementation",
        getattr(inner_cfg, "_attn_implementation", None),
    )
    if attn_impl != "eager":
        raise ValueError(
            f"ablation_mode={ablation_mode!r} needs eager attention in the InternLM2 "
            f"language model, got {attn_impl!r}."
        )

    # Patch the inner InternLM2Model (InternLM2ForCausalLM keeps its output head) and the
    # top-level generate, which sets the positions before each call.
    model.language_model.model.__class__.forward = partialmethod(
        ilm_forward,
        layers_to_skip=layers_to_skip,
        ablation_mode=ablation_mode,
    )
    model.__class__.generate = ivl_generate
