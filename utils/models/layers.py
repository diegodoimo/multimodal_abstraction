"""Locate the decoder layers of a VLM's language model and probe their output size and dtype."""

from collections import defaultdict

import torch

# Families whose decoder-layer prefix is fixed instead of auto-detected.
LAYER_PREFIXES = {
    "internvl": "language_model.model.layers",
    "cambrian": "model.layers",
}


def _detect_lm_layer_prefix(model, n_layer):
    """named_modules prefix of the decoder layers, first match among
    model.language_model.layers (OneVision), language_model.layers (LLaVA-1.5),
    language_model.model.layers (InternVL2) and model.layers (Cambrian)."""
    names = {name for name, _ in model.named_modules()}
    candidates = [
        "model.language_model.layers",
        "language_model.layers",
        "language_model.model.layers",
        "model.layers",
    ]
    for cand in candidates:
        if f"{cand}.0" in names and f"{cand}.{n_layer - 1}" in names:
            return cand
    raise RuntimeError(f"No decoder layers found for n_layer={n_layer}; tried {candidates}.")


def get_target_layers(model, n_layer, family=None):
    """{layer index: module name} of every decoder layer (residual stream at the block output)."""
    layer_prefix = LAYER_PREFIXES.get(family) or _detect_lm_layer_prefix(model, n_layer)
    names = {name for name, _ in model.named_modules()}
    target_layers = {i: f"{layer_prefix}.{i}" for i in range(n_layer)}
    for target_layer in target_layers.values():
        assert target_layer in names, target_layer
    return target_layers


def to_cuda(batch):
    """Move the tensors of a batch, and the tensors inside list values, to the GPU."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            v = v.to("cuda")
        elif isinstance(v, list):
            v = [t.to("cuda") if isinstance(t, torch.Tensor) else t for t in v]
        out[k] = v
    return out


def get_embdims(model, dataloader, target_layers, forward_fn=None):
    """Hidden size and dtype of each target layer, from one forward pass on the first batch."""
    embdims = defaultdict(lambda: None)
    dtypes = defaultdict(lambda: None)

    def get_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            embdims[name] = output.shape[-1]
            dtypes[name] = output.dtype

        return hook_fn

    handles = [
        module.register_forward_hook(get_hook(name))
        for name, module in model.named_modules()
        if name in target_layers
    ]
    batch = to_cuda(next(iter(dataloader)))
    if forward_fn is not None:
        forward_fn(batch)
    else:
        model(**batch, use_cache=False)
    for handle in handles:
        handle.remove()

    assert len(embdims) == len(target_layers)
    return embdims, dtypes
