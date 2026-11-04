"""Image-removed evaluation patch for LLaVA-1.5 (Llama language model) and pixtral (Mistral).

Used by scripts/finetuning/benchmarks.py for the image-blind baselines (``--no-image``): in the
patched layers, text queries after the image cannot attend to the image tokens. The interventions and
the image-blind compositionality runs use patch_llava.py.
"""

import torch
from typing import Unpack, Optional
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import TransformersKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llava.modeling_llava import LlavaModelOutputWithPast
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from functools import partialmethod


def llava_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    vision_feature_layer: int | list[int] | None = None,
    vision_feature_select_strategy: str | None = None,
    cache_position: torch.LongTensor | None = None,
    image_sizes: torch.Tensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | LlavaModelOutputWithPast:
    vision_feature_layer = (
        vision_feature_layer
        if vision_feature_layer is not None
        else self.config.vision_feature_layer
    )
    vision_feature_select_strategy = (
        vision_feature_select_strategy
        if vision_feature_select_strategy is not None
        else self.config.vision_feature_select_strategy
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        image_features = self.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            image_sizes=image_sizes,
        )
        image_features = torch.cat(image_features, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        special_image_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_features
        )
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    # image-token span of each sample at prefill; at decoding steps the image mask grows by one column
    if (input_ids == self.config.image_token_id).any():
        image_limits = []
        for i in range(input_ids.shape[0]):
            indices = torch.where(input_ids[i] == self.config.image_token_id)[0]
            if indices.numel() > 0:
                image_limits.append((indices[0].item(), indices[-1].item()))
            else:
                image_limits.append(None)
            self.image_attention_mask = (input_ids != self.config.image_token_id).to(
                torch.int
            )
    else:
        image_limits = None
        if self.image_attention_mask is not None:
            self.image_attention_mask = torch.cat(
                (
                    self.image_attention_mask,
                    torch.ones(self.image_attention_mask.shape[:-1] + (1,)).to(
                        self.image_attention_mask.device
                    ),
                ),
                dim=-1,
            )

    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        image_limits=image_limits,
        image_attention_mask=self.image_attention_mask,
        **kwargs,
    )

    return LlavaModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
    )


def llama_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    cache_position: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
    image_limits = kwargs.get("image_limits", None)
    image_attention_mask = kwargs.get("image_attention_mask", None)
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position: torch.Tensor = (
            torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            + past_seen_tokens
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    causal_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    image_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=image_attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    if causal_mask is not None and image_mask is not None:
        image_mask = torch.logical_and(image_mask, causal_mask)
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
    layers_to_skip = kwargs.get("layers_to_skip", [])
    if image_limits is not None:
        self.image_limits = image_limits
    for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        layer_mask = causal_mask
        if i in layers_to_skip and self.image_limits is not None:
            if i == layers_to_skip[0]:
                stored_embeds = hidden_states
            layer_mask = image_mask.clone()
            # False masks a boolean mask; an additive float mask (eager attention) needs -inf
            min_dtype = False if layer_mask.dtype == torch.bool else torch.finfo(layer_mask.dtype).min
            for b, limit in enumerate(self.image_limits):
                if limit is not None:
                    start, end = limit
                    if layer_mask.shape[2] > 1:
                        layer_mask[b, :, end + 1 :, start : end + 1] = min_dtype
                    else:
                        layer_mask[b, :, :, start : end + 1] = min_dtype

        elif (
            len(layers_to_skip) > 0
            and i == (layers_to_skip[-1] + 1)
            and self.image_limits is not None
        ):
            for b, limit in enumerate(self.image_limits):
                if limit is not None:
                    start, end = limit
                    hidden_states[b, start : end + 1, :] = stored_embeds[
                        b, start : end + 1, :
                    ]
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=layer_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )

def mistral_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        image_limits = kwargs.get("image_limits", None)
        image_attention_mask = kwargs.get("image_attention_mask", None)
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        image_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=image_attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        if causal_mask is not None and image_mask is not None:
            image_mask = torch.logical_and(image_mask, causal_mask)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        layers_to_skip = kwargs.get("layers_to_skip", [])
        if image_limits is not None:
            self.image_limits = image_limits
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_mask = causal_mask
            if i in layers_to_skip and self.image_limits is not None:
                if i == layers_to_skip[0]:
                    stored_embeds = hidden_states
                layer_mask = image_mask.clone()
                # False masks a boolean mask; an additive float mask (eager attention) needs -inf
                min_dtype = False if layer_mask.dtype == torch.bool else torch.finfo(layer_mask.dtype).min
                for b, limit in enumerate(self.image_limits):
                    if limit is not None:
                        start, end = limit
                        if layer_mask.shape[2] > 1:
                            layer_mask[b, :, end + 1 :, start : end + 1] = min_dtype
                        else:
                            layer_mask[b, :, :, start : end + 1] = min_dtype

            elif (
                len(layers_to_skip) > 0
                and i == (layers_to_skip[-1] + 1)
                and self.image_limits is not None
            ):
                for b, limit in enumerate(self.image_limits):
                    if limit is not None:
                        start, end = limit
                        hidden_states[b, start : end + 1, :] = stored_embeds[
                            b, start : end + 1, :
                        ]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


def patch(model, layers_to_skip):
    """Mask the image tokens for the text in ``layers_to_skip`` (all layers for the image-blind baselines)."""
    if 'Mistral' in model.language_model.__class__.__name__:
        model.language_model.__class__.forward = partialmethod(mistral_forward, layers_to_skip=layers_to_skip)
    else:
        model.language_model.__class__.forward = partialmethod(llama_forward, layers_to_skip=layers_to_skip)
    model.model.__class__.forward = llava_forward
