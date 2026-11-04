"""Image-removed Cambrian forward for the benchmark baselines (scripts/finetuning/benchmarks.py --no-image).

:func:`patch` replaces ``generate``, ``forward`` and ``prepare_inputs_labels_for_multimodal`` of
``CambrianLlamaForCausalLM`` and the inner ``forward``: in the layers of ``layers_to_skip`` (all layers
for the baselines) the positions after the image do not attend to it. The image-token spans are read
from the token ids before the multimodal preparation; the masks are built in sdpa's 4-D format, the
attention of the benchmark wrapper (cambrian.py). The interventions and the image-blind
compositionality use patch_cambrian.py.
"""

from functools import partialmethod
from typing import List, Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers import LlamaForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import GenerateOutput
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from utils.models.cambrian.cambrian_bridge import _ensure_cambrian_importable
from utils.models.cambrian.patch_cambrian import (
    aggregate_vision_features,
    cmm_prepare_inputs_labels_for_multimodal,
)

_ensure_cambrian_importable()

from cambrian.constants import IMAGE_TOKEN_INDEX  # noqa: E402


def _set_image_span(model, inputs: torch.Tensor) -> None:
    """Store on ``model`` the image span of each sample and a 2-D mask that is 0 on it."""
    image_limits = []
    image_attention_mask = torch.ones_like(inputs, dtype=torch.int)
    for i in range(inputs.shape[0]):
        indices = torch.where(inputs[i] == IMAGE_TOKEN_INDEX)[0]
        if indices.numel() > 0:
            start = indices[0].item()
            end = indices[-1].item()
            image_limits.append((start, end))
            image_attention_mask[i, start : end + 1] = 0
        else:
            image_limits.append(None)
    model.image_limits = image_limits
    model.image_attention_mask = image_attention_mask


def forward_cambrian(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    images: Optional[torch.FloatTensor] = None,
    image_aux_attention_masks_list: Optional[List[torch.Tensor]] = None,
    image_sizes: Optional[List[List[int]]] = None,
    return_dict: Optional[bool] = None,
    cache_position=None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    """``CambrianLlamaForCausalLM.forward`` for the image-removed baselines.

    The language model gets the vision features stored by the last :func:`generate_cambrian` with
    an image (none before the first one).
    """
    # The image span is read from the token ids before the multimodal preparation replaces them by
    # embeddings; input_ids is None at the first generation step and has no image token on the
    # later ones, where the span of the prefill is kept.
    if input_ids is not None:
        inputs = torch.stack(input_ids) if isinstance(input_ids, (list, tuple)) else input_ids
        if (inputs == IMAGE_TOKEN_INDEX).any():
            _set_image_span(self.model, inputs)

    if inputs_embeds is None:
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            vision_tower_aux_feature_list,
            vision_tower_aux_attention_masks_list,
            final_vision_feature_size,
            global_context_feature,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            image_aux_attention_masks_list,
            image_sizes,
        )
    if inputs_embeds is None and input_ids is not None:
        # text-only prompts and the later decoding steps: the preparation returns no embeddings
        inputs_embeds = self.get_model().embed_tokens(input_ids)

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    vision_kwargs = {}
    if hasattr(self, "vision_tower_aux_feature_list"):
        vision_kwargs = dict(
            vision_tower_aux_feature_list=self.vision_tower_aux_feature_list,
            vision_tower_aux_attention_masks_list=self.vision_tower_aux_attention_masks_list,
            final_vision_feature_size=self.final_vision_feature_size,
            global_context_feature=self.global_context_feature,
        )
    outputs = self.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        **vision_kwargs,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


@torch.no_grad()
def generate_cambrian(
    self,
    inputs: Optional[torch.Tensor] = None,
    images: Optional[torch.Tensor] = None,
    image_sizes: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    """``generate`` on the multimodal embeddings; stores the vision features and the image span."""
    position_ids = kwargs.pop("position_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    if "inputs_embeds" in kwargs:
        raise NotImplementedError("`inputs_embeds` is not supported")

    if images is not None:
        (
            inputs,
            position_ids,
            attention_mask,
            _,
            inputs_embeds,
            _,
            vision_tower_aux_feature_list,
            vision_tower_aux_attention_masks_list,
            final_vision_feature_size,
            global_context_feature,
        ) = self.prepare_inputs_labels_for_multimodal(
            inputs,
            position_ids,
            attention_mask,
            None,
            None,
            images,
            image_sizes=image_sizes,
        )
        self.vision_tower_aux_feature_list = vision_tower_aux_feature_list
        self.vision_tower_aux_attention_masks_list = (
            vision_tower_aux_attention_masks_list
        )
        self.final_vision_feature_size = final_vision_feature_size
        self.global_context_feature = global_context_feature
    else:
        inputs_embeds = self.get_model().embed_tokens(inputs)
    inputs = torch.stack(inputs)
    if (inputs == IMAGE_TOKEN_INDEX).any():
        _set_image_span(self.model, inputs)
    return LlamaForCausalLM.generate(
        self,
        position_ids=position_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )


def llama_forward(
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
    vision_tower_aux_feature_list: Optional[List[torch.FloatTensor]] = None,
    vision_tower_aux_attention_masks_list: Optional[List[torch.Tensor]] = None,
    final_vision_feature_size: Optional[List[tuple]] = None,
    global_context_feature: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    """``CambrianLlamaModel.forward`` with the image span masked in ``layers_to_skip`` (sdpa masks).

    The hidden states of the span are restored after the window; the SVA does not run in skipped layers.
    """
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

    past_key_values_length = 0
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

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
        inputs_embeds = self.embed_tokens(input_ids)

    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length + past_key_values_length),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
    if getattr(self, "image_attention_mask", None) is None:  # text-only prompt: nothing to mask
        self.image_attention_mask = torch.ones_like(attention_mask, dtype=torch.int)
    if self.image_attention_mask.shape[-1] < attention_mask.shape[-1]:
        # decoding step: the new token is not an image token
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
    attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )
    image_mask = _prepare_4d_causal_attention_mask_for_sdpa(
        image_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    cache_position = position_ids[0] if position_ids is not None else None
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None
    min_dtype = (
        torch.finfo(image_mask.dtype).min
        if torch.is_floating_point(image_mask)
        else False
    )
    for i, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        layer_mask = attention_mask
        if i in layers_to_skip and self.image_limits is not None:
            if i == layers_to_skip[0]:
                stored_embeds = hidden_states
            # queries after the image span do not see it (prefill), the decoding query never does
            layer_mask = image_mask.clone()
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

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=layer_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        if isinstance(layer_outputs, tuple):
            hidden_states = layer_outputs[0]
        else:
            hidden_states = layer_outputs

        if (
            not self.config.connector_only
            and vision_tower_aux_feature_list is not None
            and i not in layers_to_skip
        ):
            aggregate_vision_features(
                self,
                i,
                hidden_states,
                vision_tower_aux_feature_list,
                vision_tower_aux_attention_masks_list,
                final_vision_feature_size,
                global_context_feature,
            )

        if use_cache and isinstance(layer_outputs, tuple):
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions and isinstance(layer_outputs, tuple):
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache()
            if use_legacy_cache
            else next_decoder_cache
        )
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


def patch(model, layers_to_skip):
    """Install the image-removed forward on ``model``, masking the image in ``layers_to_skip``."""
    model.__class__.prepare_inputs_labels_for_multimodal = (
        cmm_prepare_inputs_labels_for_multimodal
    )
    model.model.__class__.forward = partialmethod(
        llama_forward, layers_to_skip=layers_to_skip
    )
    model.__class__.generate = generate_cambrian
    model.__class__.forward = forward_cambrian
