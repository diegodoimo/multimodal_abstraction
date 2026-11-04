"""Choose InternVL2's attention implementation when the model is loaded by code we do not control.

lmms-eval's InternVL2 wrapper (interventions) calls ``AutoModel.from_pretrained`` without
``attn_implementation`` or ``use_flash_attn``; the ablation patch (patch_internvl.py) needs eager
attention for its 4D additive mask. InternVLChatModel.__init__ picks the attention class of the
language model from ``use_flash_attn``, so both arguments are injected. With "flash_attention_2"
(linguistic probing, COCO retrieval) the remote code falls back to eager when flash-attn is not
installed.
"""

from contextlib import contextmanager

from transformers import AutoModel


@contextmanager
def force_internvl_attn_implementation(attn_implementation):
    """Inject ``attn_implementation`` and the matching ``use_flash_attn`` into every
    ``AutoModel.from_pretrained`` call inside the block; restored on exit. None: no change.

    Example::

        with force_internvl_attn_implementation("eager"):
            lm = InternVL2(pretrained=model_path, device_map="auto")
    """
    if attn_implementation is None:
        yield
        return

    original_from_pretrained = AutoModel.from_pretrained

    def wrapper(*args, **kwargs):
        kwargs["attn_implementation"] = attn_implementation
        if attn_implementation == "eager":
            kwargs["use_flash_attn"] = False
        elif attn_implementation == "flash_attention_2":
            kwargs["use_flash_attn"] = True
        return original_from_pretrained(*args, **kwargs)

    AutoModel.from_pretrained = wrapper
    try:
        yield
    finally:
        AutoModel.from_pretrained = original_from_pretrained
