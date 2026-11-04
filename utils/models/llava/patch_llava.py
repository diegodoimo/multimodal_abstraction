"""Intervention patch for LLaVA-1.5 (and pixtral): image-token ablations in a set of decoder layers.

Used by scripts/interventions/ (Fig. 2, text-only appendix) and by the image-blind runs of
scripts/finetuning/compositionality.py. Modes (``patch``):

* ``"image"``: text queries after the image cannot attend to the image tokens in the skipped layers,
  and the image hidden states are reset to their value at the first skipped layer when the block ends
  (layer skip of the image tokens).
* ``"cross_modal"``: the same attention mask without the reset (attention knockout).
* ``"layer"``: the skipped decoder layers are replaced by identity modules (utils.models.layer_skipping).
"""

from functools import partialmethod
from typing import Unpack

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llava.modeling_llava import (
    LlavaModel,
    LlavaModelOutputWithPast,
)
from transformers.utils import TransformersKwargs

from utils.models.layer_skipping import apply_layer_skip


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

    # (first, last) position of the image tokens of each sample, computed at prefill and cached on the
    # language model for the decoding steps (whose input_ids no longer contain the image tokens)
    image_token_id = self.config.image_token_id
    past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
    if past_len == 0:
        self.language_model._cached_image_limits = None

    if input_ids is not None and (input_ids == image_token_id).any():
        image_limits = []
        for i in range(input_ids.shape[0]):
            indices = torch.where(input_ids[i] == image_token_id)[0]
            if indices.numel() > 0:
                image_limits.append((indices[0].item(), indices[-1].item()))
            else:
                image_limits.append(None)
        self.language_model._cached_image_limits = image_limits
    else:
        image_limits = getattr(self.language_model, "_cached_image_limits", None)

    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        image_limits=image_limits,
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
    # pop the patch's own kwargs so that they do not reach decoder_layer(**kwargs)
    image_limits = kwargs.pop("image_limits", None)
    layers_to_skip = kwargs.pop("layers_to_skip", [])
    ablation_mode = kwargs.pop("ablation_mode", None)
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

    # eager attention (enforced by patch()) gives a 4D additive mask that can be edited
    if causal_mask is None:
        raise RuntimeError("The LLaVA ablation patch needs a 4D attention mask: load the model with eager attention.")

    # mask of the skipped layers: queries after the image cannot attend to the image keys
    # (prefill); at decoding steps the single query cannot attend to them either
    image_mask = None
    if layers_to_skip and image_limits is not None:
        image_mask = causal_mask.clone()
        masked_value = torch.finfo(image_mask.dtype).min
        q_len = inputs_embeds.shape[1]
        for b, limit in enumerate(image_limits):
            if limit is None:
                continue
            start, end = limit
            if q_len > 1:
                image_mask[b, :, end + 1 :, start : end + 1] = masked_value
            else:
                image_mask[b, :, :, start : end + 1] = masked_value

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

    # the image hidden states exist only at prefill (afterwards they live in the KV cache)
    is_prefill = inputs_embeds.shape[1] > 1
    stored_embeds = None

    for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):

        layer_mask = causal_mask

        if i in layers_to_skip:
            if image_mask is not None:
                layer_mask = image_mask
            if (
                i == layers_to_skip[0]
                and ablation_mode == "image"
                and is_prefill
            ):
                stored_embeds = hidden_states.clone()

        elif (
            len(layers_to_skip) > 0
            and i == (layers_to_skip[-1] + 1)
            and ablation_mode == "image"
            and is_prefill
            and stored_embeds is not None
            and image_limits is not None
        ):
            # "image": reset the image hidden states to their value at the start of the block
            for b, limit in enumerate(image_limits):
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


_ABLATION_MODES = {"image", "cross_modal", "layer"}


def patch(model, layers_to_skip, ablation_mode="image"):
    """Apply ``ablation_mode`` (module docstring) to the decoder layers ``layers_to_skip`` of ``model``.

    "image" and "cross_modal" need eager attention; "layer" works with any attention implementation.
    """
    if ablation_mode not in _ABLATION_MODES:
        raise ValueError(
            f"ablation_mode must be one of {sorted(_ABLATION_MODES)}, "
            f"got {ablation_mode!r}."
        )

    if ablation_mode == "layer":
        apply_layer_skip(model, layers_to_skip, model_type="llava")
        return

    layers_to_skip = list(layers_to_skip or [])

    # only the 4D additive mask of eager attention can be edited cell by cell
    attn_impl = getattr(model.language_model.config, "_attn_implementation", None)
    if attn_impl != "eager":
        raise ValueError(
            f"ablation_mode={ablation_mode!r} needs eager attention, got {attn_impl!r}."
        )

    # patch the inner LlavaModel (the outer class keeps lm_head and generate()) and its language model
    model.language_model.__class__.forward = partialmethod(
        llama_forward,
        layers_to_skip=layers_to_skip,
        ablation_mode=ablation_mode,
    )
    LlavaModel.forward = llava_forward
