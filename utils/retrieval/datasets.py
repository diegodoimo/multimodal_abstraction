"""LLaVA detail_23k records, the per-family image-only and text-only datasets, and the collator of the COCO extraction."""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from PIL import Image

from utils.models.internvl.internvl_conversations import get_conv_template
from utils.models.internvl.internvl_utils import get_pixel_values as internvl_get_pixel_values


def load_and_validate_detail23k(json_path: str, image_dir: str, validate_images: bool = True) -> List[Dict]:
    """One record {image_path, description} per detail_23k entry; description is the assistant's
    detailed caption. With validate_images, raise if any COCO image is missing from image_dir."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def image_path(entry):
        return os.path.join(image_dir, f"COCO_train2014_{entry['image']}")

    if validate_images:
        missing = [image_path(entry) for entry in data if not os.path.isfile(image_path(entry))]
        if missing:
            sample = "\n  ".join(missing[:10])
            tail = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
            raise FileNotFoundError(f"{len(missing)} image(s) missing from {image_dir}:\n  {sample}{tail}")

    records = []
    for entry in data:
        gpt_turn = next(c for c in entry["conversations"] if c["from"] == "gpt")
        records.append({"image_path": image_path(entry), "description": gpt_turn["value"].strip()})
    return records


class DetailImageDataset(torch.utils.data.Dataset):
    """Image-only prompts for models with a Hugging Face processor (LLaVA-1.5, OneVision).

    Images are processed in the DataLoader workers; every tensor the processor returns is kept.
    """

    def __init__(self, records: List[Dict], processor):
        self.records = records
        self.processor = processor
        # The prompt does not depend on the image (the processor expands the image placeholder).
        self.prompt = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}]}], add_generation_prompt=True, tokenize=False
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        image = Image.open(self.records[idx]["image_path"]).convert("RGB")
        encoded = self.processor(text=self.prompt, images=image, return_tensors="pt")
        # input_ids / attention_mask lose their batch dimension (the collator re-pads them);
        # the image tensors are concatenated along dim 0 by the collator.
        item = {"text_prompts": self.prompt}
        for key, value in encoded.items():
            if not isinstance(value, torch.Tensor):
                continue
            item[key] = value[0] if key in ("input_ids", "attention_mask") else value
        return item


class InternVLImageDataset(torch.utils.data.Dataset):
    """Image-only prompts for InternVL2, built as in its model.chat.

    The internlm2-chat template renders the prompt, and the <image> placeholder is replaced with
    <img><IMG_CONTEXT> x (num_image_token * n_tiles)</img>, where n_tiles (at most 12) is the
    number of 448 px tiles of the image. image_flags (one per tile) is required by the forward.
    """

    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    PLACEHOLDER = "<image>"
    MAX_NUM_TILES = 12  # the InternVL2 default
    INPUT_SIZE = 448

    def __init__(self, records: List[Dict], tokenizer, num_image_token: int):
        self.records = records
        self.tokenizer = tokenizer
        self.num_image_token = num_image_token
        template = get_conv_template("internlm2-chat")
        template.append_message(template.roles[0], self.PLACEHOLDER)
        template.append_message(template.roles[1], None)
        self.prompt = template.get_prompt()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        image = Image.open(self.records[idx]["image_path"]).convert("RGB")
        pixel_values, num_patches_list = internvl_get_pixel_values(
            image, image_size=self.INPUT_SIZE, max_tiles=self.MAX_NUM_TILES
        )
        pixel_values = pixel_values.to(torch.bfloat16)  # dtype of the vision encoder weights
        n_tiles = num_patches_list[0]
        image_tokens = (
            self.IMG_START_TOKEN
            + self.IMG_CONTEXT_TOKEN * self.num_image_token * n_tiles
            + self.IMG_END_TOKEN
        )
        query = self.prompt.replace(self.PLACEHOLDER, image_tokens, 1)
        enc = self.tokenizer(query, return_tensors="pt")
        return {
            "text_prompts": self.prompt,
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "pixel_values": pixel_values,  # (n_tiles, 3, 448, 448)
            "image_flags": torch.ones((n_tiles, 1), dtype=torch.long),
        }


class TextOnlyDataset(torch.utils.data.Dataset):
    """The detailed caption as the user turn, rendered with the model's chat template (with the
    generation prompt), no image.
    family: "llava" or "onevision" (processor is a Hugging Face processor), "internvl" or
    "cambrian" (processor is the tokenizer; Cambrian also needs cambrian_conv_mode).
    """

    def __init__(self, records: List[Dict], processor, family: str, cambrian_conv_mode: str = None):
        self.records = records
        self.processor = processor
        self.family = family
        self.cambrian_conv_mode = cambrian_conv_mode
        self.tokenizer = processor.tokenizer if family in ("llava", "onevision") else processor
        if family == "cambrian":
            assert cambrian_conv_mode is not None, "cambrian_conv_mode is required for Cambrian"

    def _build_prompt(self, description: str) -> str:
        if self.family in ("llava", "onevision"):
            conv = [{"role": "user", "content": [{"type": "text", "text": description}]}]
            return self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        if self.family == "internvl":
            tmpl = get_conv_template("internlm2-chat")
            tmpl.append_message(tmpl.roles[0], description)
            tmpl.append_message(tmpl.roles[1], None)
            return tmpl.get_prompt()
        if self.family == "cambrian":
            from cambrian.conversation import conv_templates

            conv = conv_templates[self.cambrian_conv_mode].copy()
            conv.append_message(conv.roles[0], description)
            conv.append_message(conv.roles[1], None)
            return conv.get_prompt()
        raise ValueError(f"Unknown text-only family: {self.family}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        prompt = self._build_prompt(self.records[idx]["description"])
        enc = self.tokenizer(prompt, return_tensors="pt")
        return {"text_prompts": prompt, "input_ids": enc.input_ids[0], "attention_mask": enc.attention_mask[0]}


@dataclass
class DataCollatorForVisionLM:
    """Left-pads input_ids / attention_mask and concatenates every other tensor field along dim 0
    (pixel_values, image_grid_thw, image_flags)."""

    pad_token_id: int

    def __call__(self, instances: Sequence[Dict]) -> Dict:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(inst["input_ids"]) for inst in instances],
            batch_first=True, padding_value=self.pad_token_id, padding_side="left",
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(inst["attention_mask"]) for inst in instances],
            batch_first=True, padding_value=0, padding_side="left",
        )
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        for key, value in instances[0].items():
            if key in ("input_ids", "attention_mask") or not isinstance(value, torch.Tensor):
                continue
            batch[key] = torch.cat([torch.as_tensor(inst[key]) for inst in instances], dim=0)
        return batch
