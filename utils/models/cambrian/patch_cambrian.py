"""Compatibility and layer-skipping patches for the vendored Cambrian model (third_party/cambrian).

* :func:`install_compat` replaces ``generate``, ``forward`` and ``prepare_inputs_labels_for_multimodal``
  of ``CambrianLlamaForCausalLM`` and the ``forward`` of the inner ``CambrianLlamaModel``, so that the
  vendored code runs on transformers 4.57 and the Spatial Vision Aggregator (SVA) receives the vision
  features on every call. ``cambrian_bridge.load_cambrian_pretrained`` installs it at load time.
* :func:`configure_layer_skip` sets the intervention on a window of decoder layers (``"image"``,
  ``"cross_modal"``, ``"layer"``), :func:`patch` does both (image-blind compositionality).

The replacements follow the GPU inference path of the vendored ``CambrianLlamaModel.forward`` and
``CambrianMetaForCausalLM.prepare_inputs_labels_for_multimodal`` for Cambrian-8B (SVA connector).
"""

from functools import partialmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import GenerateOutput
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from utils.models.cambrian.cambrian_bridge import _ensure_cambrian_importable
from utils.models.layer_skipping import apply_layer_skip

_ensure_cambrian_importable()

from cambrian.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX  # noqa: E402
from cambrian.model.cambrian_arch import unpad_image  # noqa: E402


@torch.no_grad()
def generate_cambrian(
    self,
    inputs: Optional[torch.Tensor] = None,
    images: Optional[torch.Tensor] = None,
    image_sizes: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    """``generate`` on the multimodal embeddings.

    Stores the vision features for the SVA layers (cleared for text-only prompts) and, on the inner
    model, the image span ``image_limits`` used by the ablations of :func:`llama_forward`.
    """
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
        # text-only prompt: drop the vision features of an earlier image prompt
        self.vision_tower_aux_feature_list = None
        self.vision_tower_aux_attention_masks_list = None
        self.final_vision_feature_size = None
        self.global_context_feature = None
    self.model.image_limits = None

    if images is not None:
        # the expanded ids carry IMAGE_TOKEN_INDEX over the whole image block (patches and the
        # image_newline of every row); the span is widened by one position on each side
        inputs = torch.stack(inputs)
        if (inputs == IMAGE_TOKEN_INDEX).any():
            image_limits = []
            seq_len = inputs.shape[1]
            for i in range(inputs.shape[0]):
                indices = torch.where(inputs[i] == IMAGE_TOKEN_INDEX)[0]
                if indices.numel() > 0:
                    start = max(0, indices[0].item() - 1)
                    end = min(seq_len - 1, indices[-1].item() + 1)
                    image_limits.append((start, end))
                else:
                    image_limits.append(None)
            self.model.image_limits = image_limits
    return LlamaForCausalLM.generate(
        self,
        position_ids=position_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )


def aggregate_vision_features(
    self,
    layer_idx: int,
    hidden_states: torch.Tensor,
    vision_tower_aux_feature_list: List[torch.Tensor],
    vision_tower_aux_attention_masks_list: List[torch.Tensor],
    final_vision_feature_size: List[tuple],
    global_context_feature: torch.Tensor,
) -> None:
    """SVA step of the vendored ``CambrianLlamaModel.forward`` after decoder layer ``layer_idx``.

    At the aggregator layers, the hidden states of the image block (without the newline column)
    query the vision features and are overwritten in place; no-op at the other layers.
    """
    cross_layers_start_idx = self.config.start_of_vision_sampler_layers
    cross_index_step = self.config.stride_of_vision_sampler_layers
    cross_layers_start_idx_list = [
        cross_layers_start_idx + cross_index * cross_index_step
        for cross_index in range(len(self.vision_sampler_layers))
    ]
    if layer_idx not in cross_layers_start_idx_list:
        return

    latent_query_start_idx = self.config.image_position
    bs = len(final_vision_feature_size)
    latent_query_num_list = []
    newline_embd_list = []
    latent_query_list = []
    for batch_i in range(bs):
        cur_h, cur_w = final_vision_feature_size[batch_i]

        cur_latent_query_num = cur_h * cur_w
        cur_latent_query_newline_num = cur_h * (cur_w + 1)
        cur_latent_query_with_newline = hidden_states[
            batch_i : batch_i + 1,
            latent_query_start_idx : latent_query_start_idx
            + cur_latent_query_newline_num,
            :,
        ].clone()

        cur_latent_query_with_newline = cur_latent_query_with_newline.view(
            1, cur_h, cur_w + 1, -1
        )
        cur_latent_query = cur_latent_query_with_newline[:, :, :-1, :]
        cur_newline_embd = cur_latent_query_with_newline[:, :, -1:, :]

        latent_query_num_list.append(cur_latent_query_num)
        latent_query_list.append(
            cur_latent_query.contiguous().view(cur_latent_query_num, 1, -1)
        )
        newline_embd_list.append(cur_newline_embd)

    latent_query = torch.cat(latent_query_list, 0)
    latent_query = self.vision_sampler_layers[
        (layer_idx - cross_layers_start_idx) // cross_index_step
    ](
        latent_query,
        global_context_feature,
        *vision_tower_aux_feature_list,
        *vision_tower_aux_attention_masks_list,
    )

    latent_query = torch.split(latent_query, latent_query_num_list, 0)
    for batch_i in range(bs):
        cur_h, cur_w = final_vision_feature_size[batch_i]
        cur_latent_query = latent_query[batch_i]
        cur_newline_embd = newline_embd_list[batch_i]
        cur_latent_query_newline_num = cur_h * (cur_w + 1)
        cur_latent_query = cur_latent_query.view(1, cur_h, cur_w, -1)
        cur_latent_query_with_newline = torch.cat(
            [cur_latent_query, cur_newline_embd], 2
        ).flatten(1, 2)
        hidden_states[
            batch_i : batch_i + 1,
            latent_query_start_idx : latent_query_start_idx
            + cur_latent_query_newline_num,
        ] = cur_latent_query_with_newline[:, :, :]


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
    """``CambrianLlamaModel.forward`` for transformers 4.57, with the ablations of the interventions.

    ``layers_to_skip`` and ``ablation_mode`` come from :func:`configure_layer_skip` (``partialmethod``).
    In the skipped layers, ``"image"`` and ``"cross_modal"`` block the attention of the positions after
    the image span to the span (4-D additive mask, eager attention); ``"image"`` also restores the
    hidden states of the span after the window. The SVA does not run in skipped layers.
    """
    # popped so that they do not reach the decoder layers
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

    # causal_lm_forward passes both: the multimodal preparation keeps the expanded input_ids,
    # and inputs_embeds already holds the image and text embeddings.
    if input_ids is not None and inputs_embeds is not None:
        input_ids = None
    if inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
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

    # The ablation edits a 4-D additive mask, which only eager attention uses as given.
    attn_impl = getattr(self.config, "_attn_implementation", None)
    if layers_to_skip and attn_impl != "eager":
        raise RuntimeError(
            f"Cambrian ablation requires attn_implementation='eager', got {attn_impl!r}."
        )

    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length + past_key_values_length),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
    causal_mask = _prepare_4d_causal_attention_mask(
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )
    if causal_mask is None:
        raise RuntimeError("_prepare_4d_causal_attention_mask returned None; a 4-D mask is required.")

    # Mask of the skipped layers: queries after the image span do not see the span. image_limits
    # is set on this (inner) model by generate_cambrian.
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
                # prefill: block post-image queries from the image key columns
                image_mask[b, :, end + 1 :, start : end + 1] = masked_value
            else:
                # decoding: the single query comes after the image block
                image_mask[b, :, :, start : end + 1] = masked_value

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    cache_position = position_ids[0] if position_ids is not None else None
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    # "image": hidden states of the image span saved before the window, written back after it
    restore = ablation_mode == "image" and image_limits is not None and inputs_embeds.shape[1] > 1
    stored_embeds = None

    for i, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        layer_mask = causal_mask
        if i in layers_to_skip:
            if image_mask is not None:
                layer_mask = image_mask
            if restore and i == layers_to_skip[0]:
                stored_embeds = hidden_states.clone()
        elif (
            restore
            and len(layers_to_skip) > 0
            and i == (layers_to_skip[-1] + 1)
            and stored_embeds is not None
        ):
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


def cmm_prepare_inputs_labels_for_multimodal(
    self,
    input_ids,
    position_ids,
    attention_mask,
    past_key_values,
    labels,
    images,
    image_aux_attention_masks_list=None,
    image_sizes=None,
):
    """Upstream ``prepare_inputs_labels_for_multimodal`` (GPU inference, SVA connector), except that it
    returns the expanded ``input_ids`` (IMAGE_TOKEN_INDEX over the image block) instead of None, and
    supports one image per sample and batch size 1.
    """
    vision_tower_aux_list = self.get_model().get_vision_tower_aux_list()
    if vision_tower_aux_list is None or images is None or input_ids.shape[1] == 1:
        return (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            None,
            labels,
            None,
            None,
            None,
            None,
        )

    image_aux_list = images

    bs = image_aux_list[0].shape[0]
    dtype = image_aux_list[0].dtype

    image_token_len = self.get_model().config.image_token_len
    query_num_list = self.get_model().config.query_num_list

    final_height = final_width = int(image_token_len**0.5)

    final_image_features_list = []

    image_aux_features_list = self.encode_images(image_aux_list)

    # project the features of each vision tower
    vision_tower_aux_feature_list = []
    for aux_i in range(len(vision_tower_aux_list)):
        image_aux_features = image_aux_features_list[aux_i]

        image_aux_features = getattr(
            self.get_model(), "mm_projector_aux_{}".format(aux_i)
        )(image_aux_features).to(dtype)
        if aux_i == 0:
            global_context_feature = image_aux_features.mean(1).view(bs, 1, 1, -1)

        vision_tower_aux_feature_list.append(image_aux_features)

    # vision sampling for each query group
    for query_group_i, query_num in enumerate(query_num_list):
        query_features_i = (
            self.get_model()
            .vision_query[query_group_i, :]
            .view(1, 1, 1, -1)
            .expand(bs, query_num, -1, -1)
        )
        global_context_feature_i = global_context_feature.expand(
            -1, query_num, 1, -1
        ).flatten(0, 1)
        query_side_len = int(query_num**0.5)
        (
            vision_tower_aux_feature_list_i,
            vision_tower_aux_attention_masks_list_i,
        ) = self.rearrange_vision_tower_features_inference(
            vision_tower_aux_feature_list, query_side_len, image_sizes
        )

        query_features_i = getattr(
            self.get_model(), "vision_sampler_{}".format(query_group_i)
        )(
            query_features_i.flatten(0, 1),
            global_context_feature_i,
            *vision_tower_aux_feature_list_i,
            *vision_tower_aux_attention_masks_list_i,
        )
        query_features_i = query_features_i.view(bs, query_num, -1)
        # interpolate to the final target size
        if query_side_len != final_height:
            query_features_i = (
                query_features_i.permute(0, 2, 1)
                .contiguous()
                .view(bs, -1, query_side_len, query_side_len)
            )
            query_features_i = F.interpolate(
                query_features_i.float(),
                size=(final_height, final_width),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=query_features_i.dtype)
            query_features_i = (
                query_features_i.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            )
        final_image_features_list.append(query_features_i)

    image_features = torch.cat(final_image_features_list, -1)
    image_features = self.get_model().mm_projector(image_features).to(dtype)

    # unpad to the image aspect ratio and append the image_newline embedding to every row
    image_features = image_features.view(bs, final_height, final_width, -1)
    image_features_unpadded = []
    final_size = []
    (
        vision_tower_aux_feature_list_final,
        vision_tower_aux_attention_masks_list_final,
    ) = self.rearrange_vision_tower_features_inference(
        vision_tower_aux_feature_list, final_height, image_sizes, unpad=True
    )
    global_context_feature_final = []
    for batch_i in range(bs):
        cur_image_feature = image_features[batch_i]
        image_size = image_sizes[batch_i]

        cur_image_feature = unpad_image(cur_image_feature.unsqueeze(0), image_size)

        cur_h, cur_w = cur_image_feature.shape[1:3]
        final_size.append((cur_h, cur_w))
        cur_image_feature = cur_image_feature.view(1, cur_h, cur_w, -1)
        cur_image_feature = torch.cat(
            (
                cur_image_feature,
                self.model.image_newline.view(1, 1, 1, -1)
                .expand(1, cur_h, 1, -1)
                .to(cur_image_feature.device),
            ),
            dim=2,
        )
        cur_image_feature = cur_image_feature.flatten(1, 2)
        image_features_unpadded.append(cur_image_feature.squeeze(0))

        cur_global_context_feature = global_context_feature[batch_i].expand(
            cur_h * cur_w, 1, -1
        )
        global_context_feature_final.append(cur_global_context_feature)
    global_context_feature_final = torch.cat(global_context_feature_final, 0)

    image_features = image_features_unpadded

    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(
            0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
        )
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask
    input_ids = [
        cur_input_ids[cur_attention_mask]
        for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
    ]
    labels = [
        cur_labels[cur_attention_mask]
        for cur_labels, cur_attention_mask in zip(labels, attention_mask)
    ]
    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat(
                [cur_input_embeds_1, cur_image_features[0:0]], dim=0
            )
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = (
            [-1]
            + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            + [cur_input_ids.shape[0]]
        )
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(
                cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]]
            )
            cur_labels_noim.append(
                cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
            )
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []
        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(
                    torch.full(
                        (cur_image_features.shape[0],),
                        IGNORE_INDEX,
                        device=cur_labels.device,
                        dtype=cur_labels.dtype,
                    )
                )
        assert num_images == 1, "Only one image per sample is supported."
        # expanded ids: IMAGE_TOKEN_INDEX over the image block
        input_ids[batch_idx] = torch.cat(
            [
                input_ids[batch_idx][: len(cur_new_input_embeds[0])],
                torch.tensor(
                    [IMAGE_TOKEN_INDEX] * len(cur_new_input_embeds[1]),
                    device=input_ids[batch_idx].device,
                ),
                input_ids[batch_idx][-len(cur_new_input_embeds[-1]) :],
            ]
        )
        cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)

    # Truncate sequences to max length as image embeddings can make the sequence longer
    tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
    if tokenizer_model_max_length is not None:
        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        input_ids = [x[:tokenizer_model_max_length] for x in input_ids]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    # Combine them (right padding, as in the Cambrian-8B config)
    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    assert batch_size == 1, "Batch size > 1 is not supported for Cambrian."

    new_input_embeds_padded = []
    new_labels_padded = torch.full(
        (batch_size, max_len),
        IGNORE_INDEX,
        dtype=new_labels[0].dtype,
        device=new_labels[0].device,
    )
    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    position_ids = torch.zeros(
        (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
    )

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        new_input_embeds_padded.append(
            torch.cat(
                (
                    cur_new_embed,
                    torch.zeros(
                        (max_len - cur_len, cur_new_embed.shape[1]),
                        dtype=cur_new_embed.dtype,
                        device=cur_new_embed.device,
                    ),
                ),
                dim=0,
            )
        )
        if cur_len > 0:
            new_labels_padded[i, :cur_len] = cur_new_labels
            attention_mask[i, :cur_len] = True
            position_ids[i, :cur_len] = torch.arange(
                0, cur_len, dtype=position_ids.dtype, device=position_ids.device
            )

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None

    return (
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        new_input_embeds,
        new_labels,
        vision_tower_aux_feature_list_final,
        vision_tower_aux_attention_masks_list_final,
        final_size,
        global_context_feature_final,
    )


_ABLATION_MODES = {"image", "cross_modal", "layer"}


def causal_lm_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
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
    **kwargs,
):
    """Replacement for the vendored ``CambrianLlamaForCausalLM.forward``.

    The vendored forward tests ``inputs_embeds is None`` after the multimodal preparation has set it,
    so a plain forward call with images never hands the fresh vision features to the SVA layers
    (they run without them, or with those of an earlier ``generate``). Every representation
    extraction calls the model this way. This version passes the features whenever it computed them,
    keeps them for the decoding steps of ``generate`` (which arrive with ``inputs_embeds``), and
    clears them on text-only calls.
    """
    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    vision_kwargs = {}
    if inputs_embeds is None:
        if images is not None:
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
            self.vision_tower_aux_feature_list = vision_tower_aux_feature_list
            self.vision_tower_aux_attention_masks_list = vision_tower_aux_attention_masks_list
            self.final_vision_feature_size = final_vision_feature_size
            self.global_context_feature = global_context_feature
        else:
            # text-only call: no vision features, none left over from an earlier image
            self.vision_tower_aux_feature_list = None
            self.vision_tower_aux_attention_masks_list = None
            self.final_vision_feature_size = None
            self.global_context_feature = None
    if getattr(self, "vision_tower_aux_feature_list", None) is not None:
        vision_kwargs = dict(
            vision_tower_aux_feature_list=self.vision_tower_aux_feature_list,
            vision_tower_aux_attention_masks_list=self.vision_tower_aux_attention_masks_list,
            final_vision_feature_size=self.final_vision_feature_size,
            global_context_feature=self.global_context_feature,
        )

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        **vision_kwargs,
        **kwargs,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states).float()

    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
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


def install_compat(model):
    """Install the class-level replacements that make the vendored Cambrian code run on transformers 4.57.

    Needed on every code path: the vendored ``CambrianLlamaModel.forward`` calls the decoder layers
    without ``position_embeddings`` and indexes their output as a tuple. Installs ``generate``,
    ``forward`` and ``prepare_inputs_labels_for_multimodal`` on ``CambrianLlamaForCausalLM`` and
    :func:`llama_forward` (no ablation) on the inner model. Idempotent.
    """
    model.__class__.prepare_inputs_labels_for_multimodal = (
        cmm_prepare_inputs_labels_for_multimodal
    )
    model.model.__class__.forward = llama_forward
    model.__class__.forward = causal_lm_forward
    model.__class__.generate = generate_cambrian


def configure_layer_skip(model, layers_to_skip, ablation_mode="image"):
    """Set the intervention on ``layers_to_skip``; requires :func:`install_compat` first.

    * ``"image"``: in the window, the positions after the image span do not attend to it, and the
      hidden states of the span are restored after the window (layer skip of the image tokens).
    * ``"cross_modal"``: the same attention knockout, without the restore.
    * ``"layer"``: the layers of the window are replaced by identity modules (text-only sweeps).

    ``"image"`` and ``"cross_modal"`` need eager attention.
    """
    if ablation_mode not in _ABLATION_MODES:
        raise ValueError(
            f"ablation_mode must be one of {sorted(_ABLATION_MODES)}, got {ablation_mode!r}."
        )

    layers_to_skip = list(layers_to_skip or [])

    if ablation_mode == "layer":
        apply_layer_skip(model, layers_to_skip, model_type="cambrian")
        # identity layers do the skip; the forward must not also mask these indices
        forward_layers_to_skip: list[int] = []
    else:
        forward_layers_to_skip = layers_to_skip

    if forward_layers_to_skip:
        attn_impl = getattr(model.model.config, "_attn_implementation", None)
        if attn_impl != "eager":
            raise ValueError(
                f"ablation_mode={ablation_mode!r} requires attn_implementation='eager', got {attn_impl!r}."
            )

    # re-install the inner forward with the ablation arguments bound
    model.model.__class__.forward = partialmethod(
        llama_forward,
        layers_to_skip=forward_layers_to_skip,
        ablation_mode=ablation_mode,
    )


def patch(model, layers_to_skip, ablation_mode="image"):
    """:func:`install_compat` followed by :func:`configure_layer_skip`."""
    install_compat(model)
    configure_layer_skip(model, layers_to_skip, ablation_mode=ablation_mode)
