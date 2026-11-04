"""lm-evaluation-harness wrapper for LLaVA-1.5 (``LlavaForConditionalGeneration``), text-only prompts.

Used by scripts/interventions/text.py; the layer skipping is applied externally with
``utils.models.llava.patch_llava.patch``.
"""

import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import stop_sequences_criteria
from transformers import AutoProcessor, LlavaForConditionalGeneration

from utils.data.constants import LLAVA_CHAT_TEMPLATE


class LlavaHFLM(HFLM):
    """``HFLM`` around the full LLaVA model (bf16), which runs text-only when ``pixel_values`` is None."""

    def __init__(
        self,
        pretrained: str,
        device: str = "cuda",
        batch_size: int = 1,
        max_gen_toks: int = 256,
    ):
        # LLaVA-1.5's chat template has no system turn: "USER: <question> ASSISTANT:"
        self._system_prompt = None

        base_model = LlavaForConditionalGeneration.from_pretrained(
            pretrained, torch_dtype=torch.bfloat16, device_map=device
        ).eval()
        print(f"Loaded {pretrained}")

        processor = AutoProcessor.from_pretrained(pretrained)
        tokenizer = processor.tokenizer
        tokenizer.chat_template = LLAVA_CHAT_TEMPLATE
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # HFLM.__init__ would load the model again: initialise the base class only
        super(HFLM, self).__init__()

        self._model = base_model
        self.tokenizer = tokenizer
        self._device = torch.device(device if isinstance(device, str) else "cuda")
        self.backend = "causal"
        self._max_length = base_model.config.text_config.max_position_embeddings
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
        self._max_gen_toks = max_gen_toks
        self.AUTO_MODEL_CLASS = LlavaForConditionalGeneration
        self.revision = "main"
        self.pretrained = pretrained
        self.peft = None
        self.delta = None
        self.think_end_token = None
        self.chat_template_args = {}

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    def apply_chat_template(self, chat_history, add_generation_prompt: bool = True) -> str:
        """Render lm-eval's chat history with the LLaVA template.

        LLAVA_CHAT_TEMPLATE expects typed content (``[{"type": "text", "text": ...}]``); lm-eval passes
        plain strings, which the template would silently drop, so they are wrapped first.
        """
        normalized = []
        for msg in chat_history:
            content = msg.get("content", "")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            normalized.append({"role": msg["role"], "content": content})
        return self.tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    # HFLM's own _model_call asserts an AutoModelForCausalLM / AutoModelForVision2Seq class
    def _model_call(
        self,
        inps: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self._model(input_ids=inps).logits

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

        return self._model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **kwargs,
        )
