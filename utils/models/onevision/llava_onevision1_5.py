"""lmms-eval wrapper for LLaVA-OneVision-1.5 (benchmarks, scripts/finetuning/benchmarks.py).

The checkpoint's remote code (AutoModelForCausalLM, trust_remote_code) with sdpa attention, prompts from
``processor.apply_chat_template`` and the processor's own pixel budget, greedy generation; one process.
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from utils.models.onevision.llavaonevision_compat import patch_flash_attn_varlen_import

eval_logger = logging.getLogger("eval_logger")

DEFAULT_GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)


@register_model("llava_onevision1_5_custom")
class LlavaOneVision1_5Custom(lmms):
    """LLaVA-OneVision-1.5 wrapper for lmms-eval."""

    def __init__(
        self,
        pretrained: str = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        device: Optional[str] = "cuda",
        device_map: str = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        patch_flash_attn_varlen_import()  # before from_pretrained

        self.batch_size_per_gpu = int(batch_size)
        self._device = torch.device(device)
        self.device_map = device_map

        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=self.device_map,
            attn_implementation="sdpa",
        )
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        self._model = AutoModelForCausalLM.from_pretrained(pretrained, **load_kwargs).eval()

        self._processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True)
        if hasattr(self._processor, "tokenizer"):
            self._processor.tokenizer.padding_side = "left"  # batch generation

        self.model_name = pretrained.rstrip("/").split("/")[-1]
        self._rank = 0
        self._world_size = 1

    @property
    def model(self):
        return self._model

    @property
    def processor(self):
        return self._processor

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def eot_token_id(self):
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        return tokenizer.eos_token_id

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: list[str] = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            # Parse generation kwargs
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
            else:
                until = []

            for k, v in DEFAULT_GEN_KWARGS.items():
                if k not in gen_kwargs:
                    gen_kwargs[k] = v

            # Remove unsupported kwargs
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)
            gen_kwargs.pop("image_aspect_ratio", None)

            max_new_tokens = gen_kwargs.pop("max_new_tokens", 1024)
            gen_kwargs["max_new_tokens"] = max_new_tokens

            # Get visuals
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            if not isinstance(visuals, list):
                visuals = [visuals]

            # Build conversation
            content_parts: list[dict] = []
            images: list[Image.Image] = []

            for visual in visuals:
                if isinstance(visual, Image.Image):
                    images.append(visual.convert("RGB"))
                    content_parts.append({"type": "image"})

            content_parts.append({"type": "text", "text": contexts})

            messages = [{"role": "user", "content": content_parts}]

            # Apply chat template
            prompt = self._processor.apply_chat_template(
                messages, add_generation_prompt=True
            )

            # Process inputs
            processor_kwargs = dict(
                text=prompt,
                return_tensors="pt",
                padding=True,
            )
            if images:
                processor_kwargs["images"] = images

            inputs = self._processor(**processor_kwargs)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                output_ids = self._model.generate(**inputs, **gen_kwargs)

            # Decode only generated tokens
            generated_ids = output_ids[:, input_len:]
            tokenizer = getattr(self._processor, "tokenizer", self._processor)
            text_output = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            # Apply until terminators
            for term in until:
                if term:
                    text_output = text_output.split(term)[0]

            res.append(text_output)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(
        self, requests: List[Instance]
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError(
            "loglikelihood not implemented for LlavaOneVision1_5Custom"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError(
            "Multi-round generation not implemented for LlavaOneVision1_5Custom"
        )
