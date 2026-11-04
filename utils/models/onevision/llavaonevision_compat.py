"""Compatibility of the LLaVA-OneVision-1.5 remote code with transformers 4.57.

The checkpoints' ``modeling_llavaonevision1_5.py`` imports ``flash_attn_varlen_func`` from
``transformers.modeling_flash_attention_utils`` when flash-attn is available; transformers no longer
exports it there (it lives in the ``flash_attn`` package). Upstream fix, not yet in the Hub checkpoints:
https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5/pull/33
"""

import logging

logger = logging.getLogger(__name__)


def patch_flash_attn_varlen_import() -> None:
    """Put ``flash_attn_varlen_func`` back into ``transformers.modeling_flash_attention_utils``.

    Call before loading the remote code. Idempotent; a no-op when the symbol is present or flash-attn is
    not available (the remote import is then skipped as well).
    """
    try:
        import transformers.modeling_flash_attention_utils as _fa_utils
    except ImportError:
        return

    if hasattr(_fa_utils, "flash_attn_varlen_func"):
        return

    try:
        from transformers.modeling_flash_attention_utils import is_flash_attn_available
    except ImportError:
        return

    if not is_flash_attn_available():
        return

    try:
        from flash_attn import flash_attn_varlen_func

        _fa_utils.flash_attn_varlen_func = flash_attn_varlen_func
        logger.info("Added flash_attn_varlen_func to transformers.modeling_flash_attention_utils.")
    except ImportError:
        logger.warning("flash_attn could not be imported; the OneVision-1.5 remote code will fail to load.")
