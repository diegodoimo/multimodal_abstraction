"""Loading of Cambrian checkpoints through the vendored third_party/cambrian package."""

from __future__ import annotations

import importlib.util
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import torch


class CambrianBridgeError(RuntimeError):
    """Raised when the Cambrian package is unavailable or a checkpoint fails to load."""


def _suppress_logging() -> int:
    """Silence logging and warnings during loading; return the previous logging level."""
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    try:
        from transformers.utils import logging as tf_logging

        tf_logging.set_verbosity_error()
        tf_logging.disable_progress_bar()
    except Exception:
        pass
    return previous


def _ensure_cambrian_importable() -> None:
    """Make ``import cambrian`` resolve, adding third_party/cambrian to sys.path if needed."""
    if importlib.util.find_spec("cambrian") is not None:
        return

    root = Path(__file__).resolve().parents[3] / "third_party" / "cambrian"
    if not (root / "cambrian").exists():
        raise CambrianBridgeError(f"Vendored Cambrian package not found at {root}.")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if importlib.util.find_spec("cambrian") is None:
        raise CambrianBridgeError(f"Cambrian package not importable from {root}.")


def _patch_transformers_cache_compat() -> None:
    """Restore ``DynamicCache.get_usable_length`` (removed from recent transformers), used by the patched forwards."""
    try:
        from transformers.cache_utils import DynamicCache

        if not hasattr(DynamicCache, "get_usable_length"):
            DynamicCache.get_usable_length = (  # type: ignore[attr-defined]
                lambda self, *_args, **_kwargs: self.get_seq_length(0)
            )
    except ImportError:
        return


def load_cambrian_pretrained(
    model_name_or_path: str,
    cache_dir: str | None = None,
    device_map: str = "auto",
    attn_implementation: str = "sdpa",  # flash-attention fails on padded multimodal batches (cu_seqlens_q)
) -> Any:
    """Load a Cambrian checkpoint (fp16) with all its vision encoders and install the compatibility patch.

    Returns ``(model, tokenizer)``; the tokenizer carries the image processor as ``tokenizer.image_processor``
    and pads on the left.
    """
    _patch_transformers_cache_compat()

    previous_log_level = _suppress_logging()
    try:
        _ensure_cambrian_importable()
        from cambrian.model.builder import load_pretrained_model

        model_name = model_name_or_path.rstrip("/").split("/")[-1]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_flash_attn = attn_implementation == "flash_attention_2" and device == "cuda"

        try:
            load_kwargs: dict[str, Any] = {}
            if not use_flash_attn and attn_implementation:
                # the vendored loader only knows use_flash_attn; forward eager / sdpa explicitly
                load_kwargs["attn_implementation"] = attn_implementation

            tokenizer, model, image_processor, _ = load_pretrained_model(
                model_path=model_name_or_path,
                model_base=None,
                model_name=model_name,
                load_8bit=False,
                load_4bit=False,
                device_map=device_map,
                device=device,
                use_flash_attn=use_flash_attn,
                cache_dir=cache_dir,
                **load_kwargs,
            )
            # The vendored CambrianLlamaModel.forward does not run on transformers 4.57;
            # the (idempotent) patch is needed on every code path.
            from utils.models.cambrian.patch_cambrian import install_compat

            install_compat(model)
        except Exception as exc:
            raise CambrianBridgeError(
                f"Failed to load Cambrian model '{model_name_or_path}': {exc}"
            ) from exc
    finally:
        logging.disable(previous_log_level)

    if image_processor is not None and not hasattr(tokenizer, "image_processor"):
        tokenizer.image_processor = image_processor

    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    return model.eval(), tokenizer
