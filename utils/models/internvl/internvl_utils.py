"""Utilities for preparing InternVL2 image inputs and chat prompts and for generating responses."""

import copy
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import PreTrainedModel

from utils.data.prompts import format_user_turn_text

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# InternVL2 special tokens
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def build_transform(input_size: int) -> T.Compose:
    """RGB, bicubic resize to input_size x input_size, ImageNet normalisation."""
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    """Tile grid (columns, rows) closest to the image aspect ratio."""
    best_diff, best_ratio = float("inf"), (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_aspect)
        if diff < best_diff or (
            diff == best_diff and area > 0.5 * image_size**2 * ratio[0] * ratio[1]
        ):
            best_diff, best_ratio = diff, ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> List[Image.Image]:
    """Cut the image into the tile grid closest to its aspect ratio (InternVL's dynamic tiling),
    plus a thumbnail of the whole image when there is more than one tile and use_thumbnail is set."""
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j == n
        },
        key=lambda x: x[0] * x[1],
    )
    best_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_w, orig_h, image_size
    )
    target_w, target_h = image_size * best_ratio[0], image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]

    resized_img = image.resize((target_w, target_h))
    tiles = []
    for i in range(blocks):
        box = (
            (i % best_ratio[0]) * image_size,
            (i // best_ratio[0]) * image_size,
            ((i % best_ratio[0]) + 1) * image_size,
            ((i // best_ratio[0]) + 1) * image_size,
        )
        tiles.append(resized_img.crop(box))

    if use_thumbnail and len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def get_pixel_values(
    images: Union[Image.Image, List[Image.Image]],
    image_size: int = 448,
    max_tiles: int = 12,
) -> Tuple[torch.Tensor, List[int]]:
    """Tiles of all images (up to ``max_tiles`` each, plus a thumbnail) as one bf16 tensor, and the
    number of tiles of each image."""
    if isinstance(images, Image.Image):
        images = [images]

    all_tiles = []
    num_patches_list = []

    for image in images:
        tiles = dynamic_preprocess(
            image, image_size=image_size, max_num=max_tiles, use_thumbnail=True
        )
        num_patches_list.append(len(tiles))
        all_tiles.extend(tiles)

    transform = build_transform(image_size)
    pixel_values = torch.stack([transform(t) for t in all_tiles])
    return pixel_values.to(torch.bfloat16), num_patches_list


def apply_chat_template(
    model: PreTrainedModel,
    text: Union[str, List[str]],
    num_patches_list: Optional[List[int]] = None,
    guide_text: str = "",
) -> List[str]:
    """Prompts in InternVL's chat template (``model.conv_template`` with ``model.system_message``).

    Args:
        model: InternVL model.
        text: one string or a list of strings (one prompt each).
        num_patches_list: tiles per image; ``<image>`` (prepended when missing) becomes ``<img>``,
            ``model.num_image_token`` ``<IMG_CONTEXT>`` per tile and ``</img>``. None: text only.
        guide_text: added to the user turn by utils.data.prompts.format_user_turn_text, which
            also ends the turn with a newline.
    """
    if isinstance(text, str):
        text = [text]

    model.eval()
    queries = []

    for idx, txt in enumerate(text):
        txt = format_user_turn_text(txt, guide_text)

        has_image = num_patches_list is not None and idx < len(num_patches_list)
        if has_image and "<image>" not in txt:
            txt = "<image>\n" + txt

        template = copy.deepcopy(model.conv_template)
        template.system_message = model.system_message
        template.messages = []
        template.append_message(template.roles[0], txt)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if has_image:
            num_patches = num_patches_list[idx]
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)

        queries.append(query)

    return queries


def get_combined_embeddings(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    img_context_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Input embeddings with the IMG_CONTEXT positions replaced by the vision features (as in
    InternVLChatModel.forward); ``img_context_token_id`` defaults to ``model.img_context_token_id``."""
    model.eval()

    if img_context_token_id is None:
        if (
            hasattr(model, "img_context_token_id")
            and model.img_context_token_id is not None
        ):
            img_context_token_id = model.img_context_token_id
        else:
            raise ValueError(
                "img_context_token_id must be provided explicitly or available in model.img_context_token_id"
            )

    device = model.device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        vit_embeds = model.extract_feature(pixel_values.to(device))

    input_embeds = model.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape

    input_embeds_flat = input_embeds.reshape(B * N, C)
    input_ids_flat = input_ids.reshape(B * N)

    selected = input_ids_flat == img_context_token_id
    try:
        input_embeds_flat[selected] = input_embeds_flat[
            selected
        ] * 0.0 + vit_embeds.reshape(-1, C).to(input_embeds_flat.device)
    except Exception as e:
        vit_embeds = vit_embeds.reshape(-1, C).to(input_embeds_flat.device)
        print(
            f"warning: {e}, input_embeds[selected].shape={input_embeds_flat[selected].shape}, vit_embeds.shape={vit_embeds.shape}"
        )
        n_token = min(selected.sum(), vit_embeds.size(0))
        input_embeds_flat[selected][:n_token] = (
            input_embeds_flat[selected][:n_token] * 0.0 + vit_embeds[:n_token]
        )

    return input_embeds_flat.reshape(B, N, C)
