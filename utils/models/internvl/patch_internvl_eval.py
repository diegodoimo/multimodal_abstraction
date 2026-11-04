"""Image-removed InternVL2 forward for the benchmark baselines (scripts/finetuning/benchmarks.py --no-image).

``patch`` replaces InternLM2Model.forward and InternVLChatModel.generate: generate records the image
span (<img> ... </img>) of each prompt, and in the layers of ``layers_to_skip`` the forward masks
the image keys in the attention mask (the benchmarks pass all layers). Requires eager attention
(4D additive mask). The interventions use patch_internvl.py.
"""

from functools import partialmethod
from typing import List, Optional, Tuple, Union

import torch
from transformers import GenerationConfig
from transformers.modeling_outputs import BaseModelOutputWithPast


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
    """InternLM2Model.forward (eager attention) with the image keys masked in ``layers_to_skip``."""
    layers_to_skip = kwargs.get("layers_to_skip", [])
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

    if self.config.attn_implementation == "flash_attention_2":
        raise RuntimeError(
            "The image-removed InternVL2 forward needs eager attention (4D additive mask)."
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

    # During decoding the attention mask grows by one position per step; the new positions are text.
    if self.image_attention_mask.shape[-1] < attention_mask.shape[-1]:
        self.image_attention_mask = torch.cat(
            (
                self.image_attention_mask,
                torch.ones(self.image_attention_mask.shape[:-1] + (1,)).to(
                    self.image_attention_mask.device
                ),
            ),
            dim=-1,
        )
    image_mask = torch.logical_and(self.image_attention_mask, attention_mask)
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )
    image_mask = self._prepare_decoder_attention_mask(
        image_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds

    min_dtype = (
        torch.finfo(attention_mask.dtype).min
        if torch.is_floating_point(attention_mask)
        else False
    )
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        layer_mask = attention_mask
        past_key_value = past_key_values[idx] if past_key_values is not None else None
        if idx in layers_to_skip and self.image_limits is not None:
            layer_mask = image_mask.clone()
            for b, limit in enumerate(self.image_limits):
                if limit is not None:
                    start, end = limit
                    if layer_mask.shape[2] > 1:
                        layer_mask[b, :, end + 1 :, start : end + 1] = min_dtype
                    else:
                        layer_mask[b, :, :, start : end + 1] = min_dtype

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
    """InternVLChatModel.generate that also stores the image span read by ilm_forward."""
    assert self.img_context_token_id is not None
    if pixel_values is not None:
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds = self.extract_feature(pixel_values)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        if (input_ids == self.img_context_token_id).any():
            # (start, end) = positions of <img> and </img> around the IMG_CONTEXT tokens
            image_limits = []
            image_attention_mask = torch.ones_like(input_ids, dtype=torch.int)
            for i in range(input_ids.shape[0]):
                indices = torch.where(input_ids[i] == self.img_context_token_id)[0]
                if indices.numel() > 0:
                    start = indices[0].item() - 1
                    end = indices[-1].item() + 1
                    image_limits.append((start, end))
                    image_attention_mask[i, start : end + 1] = 0
                else:
                    image_limits.append(None)

            self.language_model.model.image_limits = image_limits
            self.language_model.model.image_attention_mask = image_attention_mask
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


def patch(model, layers_to_skip):
    """Install the image-removed forward and generate on the classes of an InternVLChatModel."""
    model.language_model.model.__class__.forward = partialmethod(
        ilm_forward, layers_to_skip=layers_to_skip
    )
    model.__class__.generate = ivl_generate
