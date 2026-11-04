"""lm-evaluation-harness HFLM wrapper for Cambrian models (text-only interventions)."""

from typing import Optional

import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import configure_pad_token, stop_sequences_criteria

from utils.models.cambrian.cambrian_bridge import load_cambrian_pretrained
from utils.models.cambrian.cambrian_utils import (
    get_cambrian_device,
    resolve_cambrian_conv_mode,
)


class CambrianHFLM(HFLM):
    """lm-eval ``HFLM`` around a Cambrian model loaded by ``load_cambrian_pretrained``.

    The loader installs the compatibility patch of ``patch_cambrian.py``; layer skipping is configured
    by the caller with ``patch_cambrian.configure_layer_skip``. Prompts use Cambrian's own
    conversation template, without image placeholder.
    """

    def __init__(
        self,
        pretrained: str,
        device: str = "cuda",
        device_map: str = "auto",
        batch_size: int = 1,
        max_gen_toks: int = 256,
        attn_implementation: str = "sdpa",
        cache_dir: Optional[str] = None,
        conv_mode: Optional[str] = None,
        **kwargs,
    ):
        base_model, tokenizer = load_cambrian_pretrained(
            pretrained,
            cache_dir=cache_dir,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )

        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        # Bypass HFLM's __init__: the model is already loaded.
        super(HFLM, self).__init__()

        self._model = base_model
        self.tokenizer = tokenizer
        self._device = (
            get_cambrian_device(base_model)
            if device in (None, "cuda")
            else torch.device(device)
        )
        self.backend = "causal"

        # CambrianConfig subclasses LlamaConfig: max_position_embeddings is on the top-level config.
        self._max_length = getattr(
            base_model.config,
            "max_position_embeddings",
            getattr(
                getattr(base_model.config, "text_config", None),
                "max_position_embeddings",
                2048,
            ),
        )

        self.truncation = False
        self.logits_cache = False
        self._rank = 0
        self._world_size = 1
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = batch_size
        self.batch_size_per_gpu = batch_size

        self._config = base_model.config
        self.mixed_precision_dtype = None
        self.softmax_dtype = torch.float32
        self.add_bos_token = False
        self.custom_prefix_token_id = tokenizer.bos_token_id
        self._max_gen_toks = max_gen_toks if max_gen_toks is not None else 256
        # the concrete model class, so that HFLM's introspection accepts the model
        self.AUTO_MODEL_CLASS = type(base_model)
        self.revision = "main"
        self.pretrained = pretrained
        self.peft = None
        self.delta = None
        self.think_end_token = None
        self.chat_template_args = {}

        self._conv_mode = resolve_cambrian_conv_mode(
            model=base_model,
            model_name_or_path=pretrained,
            explicit_conv_mode=conv_mode,
        )

        from cambrian.conversation import conv_templates

        self._system_prompt = conv_templates[self._conv_mode].system

        self.tokenizer = configure_pad_token(self.tokenizer, model_config=self.config)

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Build the prompt with Cambrian's conversation template for ``conv_mode``."""
        from cambrian.conversation import conv_templates

        conv = conv_templates[self._conv_mode].copy()
        # Conversation.get_prompt requires a non-empty first user message; lm-eval passes a
        # system-only history for some zero-shot likelihood tasks (e.g. winogrande), so an empty
        # or missing user turn is replaced by a single space.
        sentinel = " "
        has_user_msg = False
        for msg in chat_history:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "system":
                conv.system = content
            elif role == "user":
                conv.append_message(conv.roles[0], content or sentinel)
                has_user_msg = True
            elif role == "assistant":
                conv.append_message(conv.roles[1], content)

        if not has_user_msg:
            conv.append_message(conv.roles[0], sentinel)

        if add_generation_prompt:
            conv.append_message(conv.roles[1], None)

        return conv.get_prompt()

    # The patched CambrianLlamaForCausalLM.forward / generate run text-only inputs (images=None)
    # through the language model and its lm_head.

    def _model_call(
        self,
        inps: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self._model(input_ids=inps, use_cache=False).logits

    def _model_generate(self, context, max_length, stop, **kwargs):
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )

        kwargs["temperature"] = kwargs.get("temperature", 0.0)
        do_sample = kwargs.get("do_sample")
        if kwargs.get("temperature") == 0.0 and do_sample is None:
            kwargs["do_sample"] = False
        if kwargs.get("do_sample") is False and kwargs.get("temperature") == 0.0:
            kwargs.pop("temperature", None)

        # the patched generate takes ``inputs`` (not ``input_ids``); images=None is the text-only path
        return self._model.generate(
            inputs=context,
            images=None,
            image_sizes=None,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **kwargs,
        )
