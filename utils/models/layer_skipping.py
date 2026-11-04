"""Layer skipping: replace decoder layers with identity modules (``ablation_mode="layer"``).

Used by the text-only interventions (scripts/interventions/text.py) through the model patches
(``utils/models/<family>/patch_<family>.py``). The identity module returns what the replaced layer's forward
returns: a plain tensor (transformers >= 4.48 decoder layers) or a tuple (the remote / vendored
code of InternVL2, OneVision-1.5 and Cambrian).
"""

from __future__ import annotations

import torch
from torch import nn


class _IdentityDecoderLayer(nn.Module):
    """Decoder layer that returns its input hidden states unchanged and writes nothing to the cache."""

    def __init__(self, layer_idx: int, returns_tensor: bool):
        super().__init__()
        self._returns_tensor = returns_tensor
        self.layer_idx = layer_idx

    @staticmethod
    def _is_cache_object(obj):
        """True for a transformers Cache (DynamicCache, ...)."""
        return hasattr(obj, "update") and hasattr(obj, "get_seq_length")

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        if self._returns_tensor:
            return hidden_states

        outputs = (hidden_states,)
        if kwargs.get("output_attentions", False):
            outputs += (None,)
        if kwargs.get("use_cache", False):
            past_key_value = kwargs.get("past_key_value")
            if past_key_value is None:
                past_key_value = kwargs.get("past_key_values")
            if past_key_value is not None and self._is_cache_object(past_key_value):
                # shared Cache object: pass it through; the skipped layer's slot stays empty
                outputs += (past_key_value,)
            else:
                # legacy per-layer cache tuples: an empty entry
                outputs += (None,)
        return outputs


def _layer_returns_tensor(layer) -> bool:
    """True if the layer's forward is annotated ``-> torch.Tensor`` (transformers >= 4.48)."""
    forward_fn = getattr(type(layer), "forward", None)
    if forward_fn is None:
        return False
    return getattr(forward_fn, "__annotations__", {}).get("return") is torch.Tensor


def get_decoder_layers(model, model_type: str) -> nn.ModuleList:
    """The decoder-layer ``nn.ModuleList`` of a "llava", "internvl", "onevision" or "cambrian" model."""
    if model_type == "internvl":
        return model.language_model.model.layers  # InternVLChatModel -> InternLM2ForCausalLM -> model
    if model_type == "onevision":
        return model.model.language_model.layers
    if model_type == "llava":
        return model.language_model.layers  # LlavaForConditionalGeneration.language_model is the LlamaModel
    if model_type == "cambrian":
        return model.model.layers
    raise ValueError(f"Unknown model_type {model_type!r}.")


_ORIGINALS_ATTR = "_apply_layer_skip_originals"


def _originals_for(layers: nn.ModuleList) -> dict:
    """Original (non-identity) layers of ``layers`` by index, recorded on first use.

    Stored with ``object.__setattr__`` so that the dict is not registered as a submodule.
    """
    originals = getattr(layers, _ORIGINALS_ATTR, None)
    if originals is None:
        originals = {}
        object.__setattr__(layers, _ORIGINALS_ATTR, originals)
    for i, layer in enumerate(layers):
        if i not in originals and not isinstance(layer, _IdentityDecoderLayer):
            originals[i] = layer
    return originals


def apply_layer_skip(model, layers_to_skip: list[int], model_type: str) -> list[int]:
    """Make exactly the layers in ``layers_to_skip`` identity modules, in place.

    Out-of-range indices are dropped. Layers skipped by a previous call and not in ``layers_to_skip``
    are restored, so that a sweep can reuse one model instance. The remaining layers keep their
    ``layer_idx`` (their cache slots do not move). Returns the sorted list of skipped indices.
    """
    layers = get_decoder_layers(model, model_type)
    num_layers = len(layers)
    originals = _originals_for(layers)

    target = sorted({i for i in (layers_to_skip or []) if 0 <= i < num_layers})
    target_set = set(target)

    restored = []
    for idx in range(num_layers):
        if isinstance(layers[idx], _IdentityDecoderLayer) and idx not in target_set and idx in originals:
            layers[idx] = originals[idx]
            restored.append(idx)
    restored_msg = f" Restored layers {restored[0]}-{restored[-1]}." if restored else ""

    if not target:
        print(f"[apply_layer_skip] No layers skipped (all {num_layers} layers used).{restored_msg}")
        return target

    # return convention of a real layer (after the restore pass every slot outside target is real)
    sample_idx = next(
        (i for i in range(num_layers) if not isinstance(layers[i], _IdentityDecoderLayer)), None
    )
    returns_tensor = _layer_returns_tensor(layers[sample_idx]) if sample_idx is not None else False

    for idx in target:
        if not isinstance(layers[idx], _IdentityDecoderLayer):
            layers[idx] = _IdentityDecoderLayer(layer_idx=idx, returns_tensor=returns_tensor)

    print(
        f"[apply_layer_skip] Layers {target[0]}-{target[-1]} replaced with identity modules "
        f"(of {num_layers}).{restored_msg}"
    )
    return target
