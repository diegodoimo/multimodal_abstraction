"""Collate functions of the label-overlap experiment (scripts/probing/label_overlap.py).

One collate function per model family. Each takes a batch of one ``ClassificationDataset`` sample,
applies the model's chat template with the guide text, appends the formatted prompt to ``prompt_log``
when one is given, and returns the batch format of utils/representations/residual_stream.py followed by the labels.
"""
import hashlib

import numpy as np

from utils.models.cambrian.cambrian_utils import prepare_cambrian_batch_inputs
from utils.data.dataset import _build_llava_prompts
from utils.models.internvl.internvl_utils import apply_chat_template as internvl_apply_chat_template
from utils.models.internvl.internvl_utils import get_pixel_values


def _is_internvl(model_name: str) -> bool:
    return "intern" in model_name.lower()


def _is_cambrian(model_name: str) -> bool:
    return "cambrian" in model_name.lower()


def _build_sample_signature(sample_ids: list[str], labels: np.ndarray) -> str:
    """SHA-256 of the (sample id, label) pairs, used to validate the representation cache."""
    if len(sample_ids) != len(labels):
        raise ValueError(f"Sample IDs and labels length mismatch: {len(sample_ids)} vs {len(labels)}")
    joined = "\n".join(f"{sample_id}\t{label}" for sample_id, label in zip(sample_ids, labels))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _llava_collate_fn(
    batch: list[dict],
    processor,
    guide_text: str,
    prompt_log: list[str] | None = None,
) -> tuple:
    """LLaVA-1.5 and OneVision: (inputs, labels)."""
    texts = _build_llava_prompts(
        questions=[item["question"] for item in batch],
        images=[item["image"] for item in batch],
        processor=processor,
        guide_text=guide_text,
    )
    if prompt_log is not None:
        prompt_log.append(texts[0])
    return (
        processor(return_tensors="pt", text=texts, images=[item["image"] for item in batch]),
        [item["label"] for item in batch],
    )


def _internvl_collate_fn(
    batch: list[dict],
    model,
    tokenizer,
    guide_text: str,
    prompt_log: list[str] | None = None,
) -> tuple:
    """InternVL2: (tokenized, pixel_values, labels)."""
    item = batch[0]
    pixel_values, num_patches_list = get_pixel_values([item["image"]])
    query = internvl_apply_chat_template(
        model=model, text=[item["question"]], num_patches_list=num_patches_list, guide_text=guide_text
    )[0]
    if prompt_log is not None:
        prompt_log.append(query)
    tokenized = tokenizer(query, return_tensors="pt", padding=False)
    return tokenized, pixel_values, [item["label"]]


def _cambrian_collate_fn(
    batch: list[dict],
    model,
    tokenizer,
    guide_text: str,
    prompt_log: list[str] | None = None,
) -> tuple:
    """Cambrian: (tokenized, image_tensors, image_sizes, prompts, labels)."""
    item = batch[0]
    prepared = prepare_cambrian_batch_inputs(
        model=model,
        tokenizer=tokenizer,
        questions=[item["question"]],
        images=[item["image"]],
        guide_text=guide_text,
    )
    if prompt_log is not None:
        prompt_log.append(prepared["prompts"][0])
    return (
        prepared["tokenized"],
        prepared["image_tensors"],
        prepared["image_sizes"],
        prepared["prompts"],
        [item["label"]],
    )
