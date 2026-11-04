"""Cambrian helpers: model detection, conversation mode, device, and tokenized / image-preprocessed batch inputs."""

from __future__ import annotations

from typing import Any, Optional

import torch
from PIL import Image

from transformers import BatchEncoding, PreTrainedTokenizer

from utils.models.cambrian.cambrian_bridge import _ensure_cambrian_importable, load_cambrian_pretrained


def is_cambrian_model(model: Any) -> bool:
    """True if the model class is a Cambrian model."""
    return model is not None and "cambrian" in model.__class__.__name__.lower()


def resolve_cambrian_conv_mode(
    model: Any,
    model_name_or_path: Optional[str] = None,
    explicit_conv_mode: Optional[str] = None,
) -> str:
    """Conversation template of the vendored Cambrian code: ``llama_3`` for Cambrian-8B checkpoints."""
    if explicit_conv_mode:
        return explicit_conv_mode

    if model_name_or_path:
        model_name = model_name_or_path.lower()
    else:
        model_name = str(
            getattr(getattr(model, "config", None), "_name_or_path", "")
        ).lower()
    if "cambrian-8b" in model_name:
        return "llama_3"

    raise ValueError(
        "Cannot infer the Cambrian conversation mode for "
        f"'{model_name or model.__class__.__name__}'; pass it explicitly."
    )


def load_cambrian(model_path: str):
    """(model, tokenizer, image_processor, conv_mode) with eager attention (COCO retrieval, linguistic probing)."""
    model, tokenizer = load_cambrian_pretrained(model_path, attn_implementation="eager")
    image_processor = getattr(tokenizer, "image_processor", None)
    if image_processor is None:
        raise RuntimeError("Cambrian tokenizer has no image_processor after load_cambrian_pretrained.")
    conv_mode = resolve_cambrian_conv_mode(model=model, model_name_or_path=model_path)
    return model, tokenizer, image_processor, conv_mode


def get_cambrian_device(model: Any) -> torch.device:
    """First non-CPU device of the device map, else ``model.device``, else the device of the parameters."""
    device_map = getattr(model, "hf_device_map", {})
    for dev in device_map.values():
        if str(dev) not in {"cpu", "disk"}:
            return torch.device(dev if isinstance(dev, (int, str)) else dev)

    device = getattr(model, "device", None)
    if device:
        return torch.device(device)

    return next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")


def apply_chat_template(
    model: Any,
    text: str | list[str],
    conv_mode: str,
    has_image: bool = False,
    guide_text: str = "",
) -> list[str]:
    """Format questions with the Cambrian conversation template ``conv_mode``.

    The image placeholder goes before the question.
    """
    _ensure_cambrian_importable()
    from cambrian.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
    )
    from cambrian.conversation import conv_templates

    questions = [text] if isinstance(text, str) else text
    use_im_start_end = bool(getattr(model.config, "mm_use_im_start_end", False))

    prompts: list[str] = []
    for question in questions:
        # No newline before <|eot_id|>, as in the paper's Cambrian similarity data; the representation
        # prompts of the other models end with one (see utils/data/prompts.py).
        q = f"{question}\n{guide_text}\n" if guide_text else question
        q = q.strip()
        if has_image:
            if use_im_start_end:
                image_prefix = (
                    f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}"
                )
            else:
                image_prefix = DEFAULT_IMAGE_TOKEN
            q = f"{image_prefix}\n{q}"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], q)
        conv.append_message(conv.roles[1], None)
        prompts.append(conv.get_prompt())

    return prompts


def tokenize_cambrian_prompts(
    prompts: list[str],
    tokenizer: PreTrainedTokenizer,
    conv_mode: str,
    device: torch.device,
) -> BatchEncoding:
    """Tokenize prompts with the image placeholder as IMAGE_TOKEN_INDEX, left-padded."""
    _ensure_cambrian_importable()
    from cambrian.constants import IMAGE_TOKEN_INDEX
    from cambrian.mm_utils import tokenizer_image_token, tokenizer_image_token_llama3

    tokenize_fn = (
        tokenizer_image_token_llama3
        if conv_mode in {"llama_3", "llama_v3"}
        else tokenizer_image_token
    )
    ids_per_sample = [
        tokenize_fn(
            prompt,
            tokenizer,
            image_token_index=IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        for prompt in prompts
    ]

    if tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    elif tokenizer.eos_token_id is not None:
        pad_token_id = tokenizer.eos_token_id
    elif tokenizer.bos_token_id is not None:
        pad_token_id = tokenizer.bos_token_id
    else:
        pad_token_id = 0

    max_len = max(sample_ids.shape[0] for sample_ids in ids_per_sample)
    batch_size = len(ids_per_sample)

    input_ids = torch.full(
        (batch_size, max_len),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for idx, sample_ids in enumerate(ids_per_sample):
        seq_len = sample_ids.shape[0]
        input_ids[idx, -seq_len:] = sample_ids.to(device)
        attention_mask[idx, -seq_len:] = 1

    return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})


def preprocess_cambrian_images(
    images: list[Image.Image],
    image_processor: Any,
    model: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    """Preprocess images for every vision encoder (one tensor per encoder)."""
    _ensure_cambrian_importable()
    from cambrian.mm_utils import process_images

    # expand2square pads with the RGB image mean: convert L / P / RGBA images first
    images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]

    image_tensors = process_images(images, image_processor, model.config)
    if isinstance(image_tensors, list):
        return [tensor.to(device=device, dtype=dtype) for tensor in image_tensors]
    return image_tensors.to(device=device, dtype=dtype)


def prepare_cambrian_batch_inputs(
    model: Any,
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    images: Optional[list[Image.Image]] = None,
    guide_text: str = "",
    conv_mode: Optional[str] = None,
) -> dict[str, Any]:
    """Prompts, tokenized inputs and fp16 image tensors for a Cambrian batch.

    Returns a dict with keys ``tokenized``, ``prompts``, ``image_tensors``, ``image_sizes``,
    ``conv_mode``.
    """
    device = get_cambrian_device(model)
    has_image = images is not None
    selected_mode = resolve_cambrian_conv_mode(model=model, explicit_conv_mode=conv_mode)
    prompts = apply_chat_template(
        model=model,
        text=questions,
        conv_mode=selected_mode,
        has_image=has_image,
        guide_text=guide_text,
    )
    tokenized = tokenize_cambrian_prompts(
        prompts=prompts,
        tokenizer=tokenizer,
        conv_mode=selected_mode,
        device=device,
    )

    result: dict[str, Any] = {
        "tokenized": tokenized,
        "prompts": prompts,
        "image_tensors": None,
        "image_sizes": None,
        "conv_mode": selected_mode,
    }

    if not has_image:
        return result

    if any(image is None for image in images):
        raise ValueError("Cambrian multimodal batch cannot contain None images.")

    result["image_sizes"] = [image.size for image in images]
    result["image_tensors"] = preprocess_cambrian_images(
        images=images,
        image_processor=tokenizer.image_processor,
        model=model,
        device=device,
        dtype=torch.float16,
    )
    return result
