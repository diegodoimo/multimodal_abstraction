"""Image settings for lmms-eval's own InternVL2 and OneVision-1.5 wrappers, used by the multimodal
interventions (scripts/interventions/multimodal.py) so that they match the other experiments.

* InternVL2: up to 12 tiles per image, the InternVL2 default (lmms-eval's wrapper uses 6).
* OneVision-1.5: the checkpoint processor's pixel budget (lmms-eval's wrapper forces
  200,704 to 1,605,632 pixels).
"""
from __future__ import annotations

import functools
import json

from huggingface_hub import hf_hub_download

INTERNVL_MAX_TILES = 12


def use_internvl_tiles(max_num: int = INTERNVL_MAX_TILES) -> None:
    """Make lmms-eval's own InternVL2 wrapper cut images into up to ``max_num`` tiles (InternVL2 default: 12)."""
    import lmms_eval.models.simple.internvl2 as upstream

    if getattr(upstream.load_image, "_max_num", None) != max_num:
        original = getattr(upstream.load_image, "__wrapped__", upstream.load_image)
        patched = functools.partial(original, max_num=max_num)
        patched.__wrapped__ = original
        patched._max_num = max_num
        upstream.load_image = patched


def onevision_pixel_budget(model_path: str) -> dict:
    """min_pixels / max_pixels of the checkpoint's own image processor."""
    with open(hf_hub_download(model_path, "preprocessor_config.json")) as handle:
        config = json.load(handle)
    return {"min_pixels": config["min_pixels"], "max_pixels": config["max_pixels"]}
