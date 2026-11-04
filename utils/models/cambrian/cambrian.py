"""lmms-eval wrapper for Cambrian, used by scripts/finetuning/benchmarks.py.

The model is loaded by ``load_cambrian_pretrained`` (vendored Cambrian code with the compatibility
patch) instead of lmms-eval's own Cambrian wrapper, whose ``loglikelihood`` calls a Qwen tokenizer
method. Prompts use Cambrian's conversation template with the image placeholder before the question.
cambrian_interventions.py subclasses it for the interventions.
"""

from typing import Any, List, Optional, Tuple, Union

import torch
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from PIL import Image
from tqdm import tqdm

from utils.models.cambrian.cambrian_bridge import load_cambrian_pretrained
from utils.models.cambrian.cambrian_utils import resolve_cambrian_conv_mode


class CambrianCustom(lmms):
    """Cambrian for lmms-eval, single process: sdpa attention, KV cache during generation."""

    # flash-attention fails on padded multimodal batches ("cu_seqlens_q must have shape (batch_size + 1)")
    attn_implementation = "sdpa"
    generate_use_cache = True

    def __init__(
        self,
        pretrained: str = "nyu-visionx/cambrian-8b",
        device: Optional[str] = "cuda",
        device_map: str = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        model, tokenizer = load_cambrian_pretrained(
            pretrained,
            cache_dir=cache_dir,
            device_map=device_map,
            attn_implementation=self.attn_implementation,
        )

        self._model = model
        self._tokenizer = tokenizer
        self.image_processor = getattr(tokenizer, "image_processor", None)
        self._model.eval()

        self.conv_mode = resolve_cambrian_conv_mode(
            model=model, model_name_or_path=pretrained
        )
        self.model_name = pretrained.rstrip("/").split("/")[-1]
        self.batch_size_per_gpu = int(batch_size)
        self._device = torch.device(device)
        self._rank = 0
        self._world_size = 1

    @property
    def model(self):
        return self._model

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
        return self._tokenizer.eos_token_id

    def _query_and_image(self, question: str, visuals: list) -> tuple[str, Any, Any]:
        """The user turn (image placeholder + question) and the fp16 tensors of the first image, if any."""
        if not visuals:
            return question, None, None

        from cambrian.constants import (
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IMAGE_TOKEN,
        )
        from cambrian.mm_utils import process_images

        image = visuals[0]
        if isinstance(image, Image.Image):
            image = image.convert("RGB")

        if getattr(self._model.config, "mm_use_im_start_end", False):
            img_prefix = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}"
        else:
            img_prefix = DEFAULT_IMAGE_TOKEN

        image_tensor = process_images([image], self.image_processor, self._model.config)
        if isinstance(image_tensor, list):
            image_tensor = [t.to(self._device, dtype=torch.float16) for t in image_tensor]
        else:
            image_tensor = image_tensor.to(self._device, dtype=torch.float16)
        return f"{img_prefix}\n{question}", image_tensor, [image.size]

    def _prompt(self, query: str, answer: Optional[str] = None) -> str:
        """Conversation-template prompt; ``answer=None`` ends with the assistant header."""
        from cambrian.conversation import conv_templates

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], answer)
        return conv.get_prompt()

    def _tokenize(self, prompt: str) -> torch.Tensor:
        """Token ids (1, T) with the image placeholder as IMAGE_TOKEN_INDEX, on CPU."""
        from cambrian.constants import IMAGE_TOKEN_INDEX
        from cambrian.mm_utils import tokenizer_image_token, tokenizer_image_token_llama3

        tokenize_fn = (
            tokenizer_image_token_llama3
            if self.conv_mode in {"llama_3", "llama_v3"}
            else tokenizer_image_token
        )
        ids = tokenize_fn(
            prompt, self._tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: list[str] = []

        def _collate(x):
            toks = self._tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="Model Responding"
        )
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [
                v
                for doc_vis in doc_to_visual
                for v in doc_vis(self.task_dict[task][split][doc_id[0]])
            ]

            gen_kwargs = {**all_gen_kwargs[0]}
            until = [self._tokenizer.decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
            until.append("<|eot_id|>")

            max_new_tokens = gen_kwargs.pop("max_new_tokens", 1024)
            temperature = gen_kwargs.pop("temperature", 0)
            num_beams = gen_kwargs.pop("num_beams", 1)

            qs, image_tensor, image_sizes = self._query_and_image(contexts[0], visuals)
            input_ids = self._tokenize(self._prompt(qs)).to(self._device)

            generate_kwargs: dict[str, Any] = {
                "inputs": input_ids,
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "num_beams": num_beams,
                "use_cache": self.generate_use_cache,
            }
            if temperature > 0:
                generate_kwargs["temperature"] = temperature
            if image_tensor is not None:
                generate_kwargs["images"] = image_tensor
                generate_kwargs["image_sizes"] = image_sizes

            with torch.inference_mode():
                output_ids = self._model.generate(**generate_kwargs)

            text_output = self._tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )[0].strip()

            for term in until:
                if term:
                    text_output = text_output.split(term)[0]

            res.append(text_output)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res: list[tuple[float, bool]] = []
        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="Model Responding"
        )

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            continuation = (
                doc_to_target
                if isinstance(doc_to_target, str)
                else doc_to_target(self.task_dict[task][split][doc_id])
            )
            visuals = list(doc_to_visual(self.task_dict[task][split][doc_id]))

            qs, image_tensor, image_sizes = self._query_and_image(contexts, visuals)

            # full conversation (continuation as the answer) and context only (assistant header)
            full_ids = self._tokenize(self._prompt(qs, continuation)).to(self._device)
            ctx_len = self._tokenize(self._prompt(qs)).shape[-1]

            # loss on the continuation tokens only
            labels = full_ids.clone()
            labels[:, :ctx_len] = -100

            forward_kwargs: dict[str, Any] = {
                "input_ids": full_ids,
                "labels": labels,
                "attention_mask": torch.ones_like(full_ids),
                "use_cache": False,
            }
            if image_tensor is not None:
                forward_kwargs["images"] = image_tensor
                forward_kwargs["image_sizes"] = image_sizes

            with torch.inference_mode():
                outputs = self._model(**forward_kwargs)

            loss = outputs.loss
            logits = outputs.logits
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = full_ids[:, ctx_len:]
            greedy_tokens = greedy_tokens[:, ctx_len - 1 : full_ids.shape[1] - 1]
            max_equal = (greedy_tokens == cont_toks).all()

            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for CambrianCustom.")
