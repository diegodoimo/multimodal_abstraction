"""Model loading for the representation, probing and compositionality experiments.

``load_hf_model_and_processor_or_tokenizer`` loads, in eval mode with left padding: LLaVA-1.5 and pixtral
(``LlavaForConditionalGeneration``), LLaVA-OneVision-1.5 (remote code), InternVL2 (remote code; the
pre-trained weights are a subfolder of OpenGVLab/InternVL2-Pretrain-Models) and Cambrian (vendored code,
utils.models.cambrian). ``resolve_attn_implementation_for_model`` picks the attention implementation.
"""

import json
import os
import re
from typing import Optional, Tuple, Type, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    CLIPProcessor,
    LlavaForConditionalGeneration,
    LlavaNextProcessor,
    LlavaProcessor,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from utils.models.cambrian.cambrian_bridge import load_cambrian_pretrained
from utils.models.onevision.llavaonevision_compat import patch_flash_attn_varlen_import

# The OneVision-1.5 remote code imports flash_attn_varlen_func from transformers, which no longer exports
# it; applied on import so that every script importing utils can load OneVision.
patch_flash_attn_varlen_import()

# Type alias used in annotations (utils/data/dataset.py)
ProcessorType = Union[
    LlavaProcessor,
    LlavaNextProcessor,
    CLIPProcessor,
    AutoTokenizer,
    AutoProcessor,
]


def is_mllama_model_name_or_path(model_name_or_path: str) -> bool:
    """True for Llama-3.2-Vision (Mllama) checkpoints, by name or by a local config.json."""
    model_name_lower = model_name_or_path.lower()
    if "mllama" in model_name_lower:
        return True
    if "llama" in model_name_lower and "vision" in model_name_lower:
        return True

    config_path = os.path.join(os.path.expanduser(model_name_or_path), "config.json")
    if not os.path.isfile(config_path):
        return False

    try:
        with open(config_path) as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    architectures = config.get("architectures", [])
    if isinstance(architectures, str):
        architectures = [architectures]
    architecture_names = " ".join(str(name).lower() for name in architectures)
    return (
        str(config.get("model_type", "")).lower() == "mllama"
        or "mllama" in architecture_names
    )


def resolve_attn_implementation_for_model(
    model_name_or_path: str,
    requested: str,
) -> str:
    """Attention implementation for a model: flash-attention for InternVL2 only, sdpa for the others.

    InternVL2's code has no sdpa; with flash-attention requested it uses flash-attn when installed and
    eager otherwise. Every other model gets sdpa when flash-attention or "auto" is requested: LLaVA and
    OneVision run as fast with sdpa, and flash-attention fails for Cambrian (its vendored forward on
    padded multimodal batches) and pixtral (transformers 4.57 vision tower). An explicit "eager" or
    "sdpa" request is kept (the interventions need eager).
    """
    if "internvl" in model_name_or_path.lower():
        return "flash_attention_2" if requested == "auto" else requested
    if requested in ("auto", "flash_attention_2"):
        return "sdpa"

    if is_mllama_model_name_or_path(model_name_or_path) and requested != "sdpa":
        raise ValueError(
            "Mllama / Llama Vision models support only "
            f"`sdpa` attention, got `{requested}`."
        )
    return requested


def _get_vl_model_classes(model_name_or_path: str) -> Tuple[Type, Type]:
    """Processor and model classes of a LLaVA-1.5, pixtral or OneVision-1.5 checkpoint."""
    model_name_lower = model_name_or_path.lower()
    if "onevision" in model_name_lower:
        return AutoProcessor, AutoModelForCausalLM
    if "llava" in model_name_lower or "pixtral" in model_name_lower:
        return AutoProcessor, LlavaForConditionalGeneration
    raise ValueError(f"Unsupported model: {model_name_or_path}")


def _load_internvl2_model_and_tokenizer(
    model_name_or_path: str,
    cache_dir: Optional[str],
    device_map: str,
    dtype: torch.dtype,
    use_flash_attn: bool,
    skip_processor: bool,
    subfolder: Optional[str] = None,
    code_repo: Optional[str] = None,
):
    """InternVL2 model (``InternVLChatModel``) and tokenizer (``InternLM2Tokenizer``) from the remote code.

    ``code_repo`` is the repository of the Python code when it differs from the one of the weights
    (the pre-trained weights in a subfolder ship no code).
    """
    code_source = code_repo if code_repo else model_name_or_path
    weights = model_name_or_path + (f"/{subfolder}" if subfolder else "")

    tokenizer = None
    if not skip_processor:
        print(f"Loading tokenizer: weights {weights}, code {code_source}")
        InternLM2Tokenizer = get_class_from_dynamic_module(
            "tokenization_internlm2.InternLM2Tokenizer",
            code_source,
            cache_dir=cache_dir,
        )
        tokenizer = InternLM2Tokenizer.from_pretrained(
            model_name_or_path,
            subfolder=subfolder,
            cache_dir=cache_dir,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: weights {weights}, code {code_source}")
    InternVLChatModel = get_class_from_dynamic_module(
        "modeling_internvl_chat.InternVLChatModel",
        code_source,
        cache_dir=cache_dir,
    )
    # GenerationMixin for InternLM2ForCausalLM on transformers >= 4.50
    from utils.models.internvl.internvl2_compat import patch_internvl_classes_pre_instantiation

    patch_internvl_classes_pre_instantiation(InternVLChatModel)

    model = InternVLChatModel.from_pretrained(
        model_name_or_path,
        subfolder=subfolder if subfolder is not None else "",
        cache_dir=cache_dir,
        device_map=device_map,
        dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=use_flash_attn,
    )

    if skip_processor:
        return model
    return model, tokenizer


def _configure_padding(processor, model) -> None:
    """Pad on the left; add a pad token (and resize the embeddings) if the tokenizer has none."""
    tokenizer = (
        processor
        if hasattr(processor, "padding_side")
        else getattr(processor, "tokenizer", None)
    )
    if tokenizer is None or not hasattr(tokenizer, "padding_side"):
        raise ValueError(f"Cannot configure padding: {type(processor).__name__} has no padding_side.")

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is not None:
        return

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    for obj in (model, getattr(model, "language_model", None)):
        resize_fn = getattr(obj, "resize_token_embeddings", None)
        if callable(resize_fn):
            resize_fn(len(tokenizer))
            return


def load_hf_model_and_processor_or_tokenizer(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    device_map: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "auto",
    skip_processor: bool = False,
):
    """Load a model in eval mode with its processor (tokenizer for InternVL2 and Cambrian).

    Args:
        model_name_or_path: Hub id or local path.
        cache_dir: model cache (default: ``HF_HUB_CACHE``).
        device_map: device map of ``from_pretrained``.
        dtype: parameter dtype (Cambrian always loads in fp16).
        attn_implementation: requested attention implementation, resolved by
            ``resolve_attn_implementation_for_model`` ("auto": flash-attention for InternVL2, sdpa otherwise).
        skip_processor: return the model only.

    Returns:
        ``(model, processor)``, or the model alone with ``skip_processor``.
    """
    if cache_dir is None:
        cache_dir = os.environ.get("HF_HUB_CACHE")
    if cache_dir:
        cache_dir = os.path.expanduser(cache_dir)

    attn_implementation = resolve_attn_implementation_for_model(
        model_name_or_path,
        attn_implementation,
    )
    model_name_lower = model_name_or_path.lower()

    if "cambrian" in model_name_lower:
        model, tokenizer = load_cambrian_pretrained(
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        return model if skip_processor else (model, tokenizer)

    if "internvl" in model_name_lower:
        subfolder = None
        code_repo = None
        # pre-trained weights, e.g. "OpenGVLab/InternVL2-Pretrain-Models/InternVL2-8B-Pretrain": the code
        # comes from the fine-tuned model of the same size (OpenGVLab/InternVL2-8B)
        if "pretrain-models" in model_name_lower:
            parts = model_name_or_path.split("/")
            if len(parts) >= 3:
                repo_id = "/".join(parts[:2])
                subfolder = "/".join(parts[2:])
                size_match = re.search(r"(\d+B)", subfolder, re.IGNORECASE)
                if size_match:
                    code_repo = f"OpenGVLab/InternVL2-{size_match.group(1).upper()}"
                model_name_or_path = repo_id

        return _load_internvl2_model_and_tokenizer(
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir,
            device_map=device_map,
            dtype=dtype,
            use_flash_attn=(attn_implementation == "flash_attention_2"),
            skip_processor=skip_processor,
            subfolder=subfolder,
            code_repo=code_repo,
        )

    ProcessorClass, ModelClass = _get_vl_model_classes(model_name_or_path)
    processor_kwargs = {
        "pretrained_model_name_or_path": model_name_or_path,
        "cache_dir": cache_dir,
    }
    if "onevision" in model_name_lower:
        processor_kwargs["trust_remote_code"] = True

    if not skip_processor:
        print(f"Loading processor: {model_name_or_path}")
        try:
            processor = ProcessorClass.from_pretrained(**processor_kwargs, use_fast=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load the processor of {model_name_or_path} (cache {cache_dir}): {e}"
            ) from e

    print(f"Loading model: {model_name_or_path}")
    try:
        model = ModelClass.from_pretrained(
            **processor_kwargs,
            device_map=device_map,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            dtype=dtype,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load the model {model_name_or_path} (cache {cache_dir}): {e}"
        ) from e

    if skip_processor:
        return model.eval()

    _configure_padding(processor, model)
    return model.eval(), processor
