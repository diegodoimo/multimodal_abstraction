"""Cambrian-8B image-only dataset and pooling support for the COCO extraction.

Cambrian tokenizes <image> as one IMAGE_TOKEN_INDEX (-200) placeholder, which the model expands
inside prepare_inputs_labels_for_multimodal into the image features plus one learned image_newline
embedding per row, so the hidden states are longer than input_ids. wrap_cambrian_for_pooling records
the expanded attention mask and a synthetic input_ids in the expanded space, with
IMAGE_TOKEN_SENTINEL on the image positions and IMAGE_NEWLINE_SENTINEL on the newline positions;
make_cambrian_pool_info_provider hands them to the extraction engine. Cambrian has no end-of-image
token: the text after the image starts after the last image position.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_TOKEN_SENTINEL = -200  # Cambrian's IMAGE_TOKEN_INDEX
# Kept out of the image mean (_imgavg).
IMAGE_NEWLINE_SENTINEL = -201


def _build_prompt(conv_mode: str, model_config) -> str:
    """Image-only user turn followed by the assistant header."""
    from cambrian.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN
    from cambrian.conversation import conv_templates

    if getattr(model_config, "mm_use_im_start_end", False):
        qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}"
    else:
        qs = DEFAULT_IMAGE_TOKEN
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def _tokenize_prompt(prompt: str, tokenizer, conv_mode: str) -> torch.Tensor:
    from cambrian.constants import IMAGE_TOKEN_INDEX
    from cambrian.mm_utils import tokenizer_image_token, tokenizer_image_token_llama3

    tokenize_fn = tokenizer_image_token_llama3 if conv_mode in {"llama_3", "llama_v3"} else tokenizer_image_token
    return tokenize_fn(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors="pt")


def _process_image_cpu(image: Image.Image, image_processor) -> List[torch.Tensor]:
    """cambrian.mm_utils.process_images for one RGB image, without moving it to the GPU (DataLoader
    workers): one (C, H_i, W_i) tensor per vision tower, padded to a square."""
    from cambrian.mm_utils import expand2square

    out: List[torch.Tensor] = []
    for proc in image_processor:
        cur = image
        if hasattr(proc, "image_mean"):
            target = proc.crop_size["height"]
            cur = expand2square(cur, tuple(int(x * 255) for x in proc.image_mean)).resize((target, target))
        out.append(proc.preprocess(cur, return_tensors="pt")["pixel_values"][0])
    return out


class CambrianImageDataset(Dataset):
    """Image-only prompts for Cambrian: input_ids with the -200 placeholder and fp16 per-tower images."""

    def __init__(self, records: List[Dict], tokenizer, image_processor, model_config, conv_mode: str):
        self.records = records
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_mode = conv_mode
        self.prompt = _build_prompt(conv_mode, model_config)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image = Image.open(self.records[idx]["image_path"]).convert("RGB")
        input_ids = _tokenize_prompt(self.prompt, self.tokenizer, self.conv_mode)
        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze(0)
        return {
            "text_prompts": self.prompt,
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            # fp16, the dtype of the loaded model
            "images_aux_list": [t.to(torch.float16) for t in _process_image_cpu(image, self.image_processor)],
            "image_size": image.size,  # (W, H)
        }


def cambrian_collator(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch of one sample (the expanded length and the image span differ between samples)."""
    if len(items) != 1:
        raise ValueError("CambrianImageDataset requires batch_size=1.")
    item = items[0]
    return {
        "input_ids": item["input_ids"].unsqueeze(0),
        "attention_mask": item["attention_mask"].unsqueeze(0),
        "images": [aux.unsqueeze(0) for aux in item["images_aux_list"]],
        "image_sizes": [item["image_size"]],
    }


def wrap_cambrian_for_pooling(model) -> None:
    """Wrap model.prepare_inputs_labels_for_multimodal so that every forward leaves
    model._cambrian_pool_aux = {"attention_mask_expanded": (B, T_exp) mask, "synthetic_input_ids":
    (B, T_exp) ids}, both on CPU. The synthetic ids hold IMAGE_TOKEN_SENTINEL on the image span,
    IMAGE_NEWLINE_SENTINEL on the per-row newline positions and the original ids elsewhere.
    Call after utils.models.cambrian.patch_cambrian.install_compat, which replaces the method.
    """
    original = model.prepare_inputs_labels_for_multimodal

    def patched(input_ids, position_ids, attention_mask, *args, **kwargs):
        # Placeholder position of each sample, before the expansion.
        orig_input_ids = None
        image_positions: List[Optional[int]] = []
        if input_ids is not None:
            orig_input_ids = input_ids.detach().cpu()
            for b in range(input_ids.shape[0]):
                pos = (input_ids[b] == IMAGE_TOKEN_SENTINEL).nonzero(as_tuple=True)[0]
                image_positions.append(pos[0].item() if len(pos) > 0 else None)

        result = original(input_ids, position_ids, attention_mask, *args, **kwargs)

        expanded_mask = result[2]
        inputs_embeds = result[4]
        final_size = result[8] if len(result) > 8 else None  # (rows, cols) of the image features per sample
        if (
            expanded_mask is not None
            and inputs_embeds is not None
            and image_positions
            and orig_input_ids is not None
        ):
            B, T_exp = inputs_embeds.shape[:2]
            # The image span length depends on the aspect ratio, so it is recovered from the
            # expansion; this is unambiguous only for one sample per batch.
            assert B == 1, "wrap_cambrian_for_pooling requires batch_size=1"
            actual_image_len = T_exp - (orig_input_ids.shape[1] - 1)
            synthetic = torch.zeros((B, T_exp), dtype=torch.long)
            for b, p in enumerate(image_positions):
                orig = orig_input_ids[b]
                T_orig = orig.shape[0]
                if p is None:  # no image: the sequence is not expanded
                    copy_len = min(T_orig, T_exp)
                    synthetic[b, :copy_len] = orig[:copy_len]
                    continue
                synthetic[b, :p] = orig[:p]
                end_image = min(p + actual_image_len, T_exp)
                synthetic[b, p:end_image] = IMAGE_TOKEN_SENTINEL
                # The features are flattened from (rows, cols + 1), the newline being the last column.
                if final_size is not None and b < len(final_size):
                    cur_h, cur_w = final_size[b]
                    if cur_h * (cur_w + 1) != actual_image_len:
                        raise RuntimeError(
                            f"final_size {(cur_h, cur_w)} implies an image span of {cur_h * (cur_w + 1)}, "
                            f"the expansion gives {actual_image_len}"
                        )
                    for r in range(cur_h):
                        nl_pos = p + r * (cur_w + 1) + cur_w
                        if nl_pos < end_image:
                            synthetic[b, nl_pos] = IMAGE_NEWLINE_SENTINEL
                # Text after the image: original ids after the placeholder.
                tail_dst_start = p + actual_image_len
                tail_src_start = p + 1
                tail_len = min(T_orig - tail_src_start, T_exp - tail_dst_start)
                if tail_len > 0:
                    synthetic[b, tail_dst_start:tail_dst_start + tail_len] = (
                        orig[tail_src_start:tail_src_start + tail_len]
                    )
            model._cambrian_pool_aux = {
                "attention_mask_expanded": expanded_mask.detach().cpu(),
                "synthetic_input_ids": synthetic,
            }
        return result

    model.prepare_inputs_labels_for_multimodal = patched


def make_cambrian_pool_info_provider(model):
    """pool_info_provider for the extraction engine: the mask and ids recorded by wrap_cambrian_for_pooling."""

    def provider(batch):
        aux = getattr(model, "_cambrian_pool_aux", None)
        if aux is None:
            raise RuntimeError("model._cambrian_pool_aux missing; call wrap_cambrian_for_pooling(model) first.")
        return {
            "mask": aux["attention_mask_expanded"] != 0,
            "input_ids": aux["synthetic_input_ids"],
        }

    return provider
