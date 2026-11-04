"""lm-evaluation-harness wrapper for LLaVA-OneVision-1.5 (remote code), text-only prompts.

Used by scripts/interventions/text.py; the layer skipping is applied externally with
``utils.models.onevision.patch_onevision.patch``.
"""

import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import stop_sequences_criteria
from transformers import AutoModelForCausalLM, AutoProcessor

from utils.models.onevision.llavaonevision_compat import patch_flash_attn_varlen_import


class OnevisionHFLM(HFLM):
    """``HFLM`` around the full OneVision-1.5 model (sdpa, checkpoint dtype), text-only when ``pixel_values`` is None."""

    def __init__(
        self,
        pretrained: str,
        device: str = "cuda",
        batch_size: int = 1,
        max_gen_toks: int = 64,
    ):
        self._system_prompt = "You are a helpful assistant."  # the default system turn of the chat template

        patch_flash_attn_varlen_import()  # the remote code imports it from transformers at load time
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True,
            attn_implementation="sdpa",
        ).eval()
        print(f"Loaded {pretrained}")

        processor = AutoProcessor.from_pretrained(
            pretrained, trust_remote_code=True, max_pixels=1605632, min_pixels=256 * 28 * 28
        )
        tokenizer = processor.tokenizer
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
        self.AUTO_MODEL_CLASS = AutoModelForCausalLM
        self.revision = "main"
        self.pretrained = pretrained
        self.peft = None
        self.delta = None
        self.think_end_token = None
        self.chat_template_args = {}

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

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

        return self._model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **kwargs,
        )
