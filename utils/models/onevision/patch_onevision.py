"""Intervention patch for LLaVA-OneVision-1.5 (remote code): image-token ablations in a set of decoder layers.

Used by scripts/interventions/ (Fig. 2, text-only appendix) and by the image-blind runs of
scripts/finetuning/compositionality.py. The modes are those of utils.models.llava.patch_llava:

* ``"image"``: text queries after the image cannot attend to the image span in the skipped layers, and
  the image hidden states are reset to their value at the first skipped layer when the block ends.
* ``"cross_modal"``: the same attention mask without the reset (attention knockout).
* ``"layer"``: the skipped decoder layers are replaced by identity modules (utils.models.layer_skipping).

The image span includes the ``<|vision_start|>`` and ``<|vision_end|>`` tokens around the image tokens.
"""

from dataclasses import dataclass
from functools import partialmethod
from typing import List, Optional, Tuple, Union

import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.utils import is_torchdynamo_compiling

from utils.models.layer_skipping import apply_layer_skip


@dataclass
class LLaVAOneVision1_5_ModelOutputWithPast(ModelOutput):
    """
    Base class for Llava outputs, with hidden states and attentions.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


def onevision_forward(
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
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, LLaVAOneVision1_5_ModelOutputWithPast]:
    r"""
    pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
        The tensors corresponding to the input videos. Pixel values can be obtained using
        [`AutoImageProcessor`]. See [`Qwen2VLImageProcessor.__call__`] for details. [`Qwen2VLProcessor`] uses
        [`Qwen2VLImageProcessor`] for processing videos.
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """
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
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)
    # (start, end) of the image span of each sample, <|vision_start|> and <|vision_end|> included;
    # computed at prefill and cached on the language model for the decoding steps
    past_len = (
        past_key_values.get_seq_length() if past_key_values is not None else 0
    )
    if past_len == 0:
        self.language_model._cached_image_limits = None

    if input_ids is not None and (input_ids == self.config.image_token_id).any():
        image_limits = []
        seq_len = input_ids.shape[1]
        for i in range(input_ids.shape[0]):
            indices = torch.where(input_ids[i] == self.config.image_token_id)[0]
            if indices.numel() > 0:
                start = max(0, indices[0].item() - 1)
                end = min(seq_len - 1, indices[-1].item() + 1)
                image_limits.append((start, end))
            else:
                image_limits.append(None)
        self.language_model._cached_image_limits = image_limits
    else:
        image_limits = getattr(self.language_model, "_cached_image_limits", None)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        image_limits=image_limits,
    )

    output = LLaVAOneVision1_5_ModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    return output if return_dict else output.to_tuple()


def onevision_llm_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    image_limits = kwargs.pop("image_limits", None)
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

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(
            3, inputs_embeds.shape[0], -1
        )

    causal_mask = self._update_causal_mask(
        attention_mask,
        inputs_embeds,
        cache_position,
        past_key_values,
        output_attentions,
    )
    # eager attention (enforced by patch()) gives a 4D additive mask that can be edited
    if causal_mask is None:
        raise RuntimeError(
            "The OneVision ablation patch needs a 4D attention mask: load the model with eager attention."
        )

    # mask of the skipped layers: queries after the image span cannot attend to it (prefill);
    # at decoding steps the single query cannot attend to it either
    image_mask = None
    if layers_to_skip and image_limits is not None:
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
                image_mask[b, :, end + 1 :, start : end + 1] = masked_value
            else:
                image_mask[b, :, :, start : end + 1] = masked_value

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    # the image hidden states exist only at prefill (afterwards they live in the KV cache)
    is_prefill = inputs_embeds.shape[1] > 1
    stored_embeds = None

    for i, decoder_layer in enumerate(self.layers):
        layer_mask = causal_mask
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if i in layers_to_skip:
            if image_mask is not None:
                layer_mask = image_mask
            if (
                i == layers_to_skip[0]
                and ablation_mode == "image"
                and is_prefill
                and image_limits is not None
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

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=layer_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
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
        apply_layer_skip(model, layers_to_skip, model_type="onevision")
        return

    layers_to_skip = list(layers_to_skip or [])

    # only the 4D additive mask of eager attention can be edited cell by cell
    attn_impl = getattr(model.language_model.config, "_attn_implementation", None)
    if attn_impl != "eager":
        raise ValueError(
            f"ablation_mode={ablation_mode!r} needs eager attention, got {attn_impl!r}."
        )

    # the remote LLaVAOneVision1_5_Attention never sets self.num_heads, which its eager forward reads
    num_heads = model.language_model.config.num_attention_heads
    for layer in model.language_model.layers:
        if not hasattr(layer.self_attn, "num_heads"):
            layer.self_attn.num_heads = num_heads

    # patch the inner multimodal model and its text model; the outer class keeps lm_head and generate()
    model.language_model.__class__.forward = partialmethod(
        onevision_llm_forward,
        layers_to_skip=layers_to_skip,
        ablation_mode=ablation_mode,
    )
    model.model.__class__.forward = onevision_forward
