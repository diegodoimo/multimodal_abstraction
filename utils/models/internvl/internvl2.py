"""lmms-eval wrapper for InternVL2 (InternVL2Custom), used by scripts/finetuning/benchmarks.py.

Adapted from lmms-eval's InternVL2 wrapper: the model is loaded with
utils.models.loading.load_hf_model_and_processor_or_tokenizer (the loader of every other experiment),
images are cut into up to 12 tiles plus a thumbnail, and answers are generated with the model's
own ``chat``. Single process, batch size 1, image tasks only.
"""

import gc
import logging
import math
from datetime import timedelta
from typing import List, Tuple

import torch
import torchvision.transforms as T
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from utils.models.internvl.internvl2_compat import patch_internvl_model_post_instantiation
from utils.models.loading import load_hf_model_and_processor_or_tokenizer

eval_logger = logging.getLogger("eval_logger")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)


def build_transform(input_size):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    """Cut the image into the tile grid closest to its aspect ratio (InternVL's dynamic tiling)."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def split_model(model_name):
    """Device map that keeps the vision model, the embeddings, the output head and the last LLM layer
    on GPU 0 and spreads the other LLM layers over the visible GPUs (all on GPU 0 with one GPU)."""
    device_map = {}
    world_size = torch.cuda.device_count()
    # pretrained variants (e.g. InternVL2-8B-Pretrain) share the architecture of the fine-tuned model
    model_name = model_name.replace("-Pretrain", "")
    num_layers = {
        "InternVL2_5-1B": 24,
        "InternVL2_5-2B": 24,
        "InternVL2_5-4B": 36,
        "InternVL2_5-8B": 32,
        "InternVL2_5-26B": 48,
        "InternVL2_5-38B": 64,
        "InternVL2_5-78B": 80,
        "InternVL2-1B": 24,
        "InternVL2-2B": 24,
        "InternVL2-4B": 32,
        "InternVL2-8B": 32,
        "InternVL2-26B": 48,
        "InternVL2-40B": 60,
        "InternVL2-Llama3-76B": 80,
    }[model_name]
    # The first GPU also holds the vision model: count it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


@register_model("internvl2_custom")
class InternVL2Custom(lmms):
    def __init__(
        self,
        pretrained: str,
        device: str = "cuda:0",
        batch_size: str = "1",
        cache_dir: str = None,
    ):
        super().__init__()

        self.path = pretrained

        batch_size = int(batch_size)
        assert batch_size == 1, (
            f"Batch size should be 1 for InternVL2, but got {batch_size}."
        )
        self.batch_size_per_gpu = batch_size

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        self.accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self._device = torch.device(device)
        self.device_map = split_model(pretrained.split("/")[-1])

        eval_logger.info("Loading InternVL2 with utils.models.loading")
        # "sdpa" resolves to use_flash_attn=False, i.e. eager attention: InternVL2's remote code has
        # no sdpa (REPORT.md, open issue 8).
        self._model, self._tokenizer = load_hf_model_and_processor_or_tokenizer(
            model_name_or_path=self.path,
            cache_dir=cache_dir,
            device_map=self.device_map,
            attn_implementation="sdpa",
            skip_processor=False,
        )
        patch_internvl_model_post_instantiation(self._model)

        eval_logger.info(f"Using single device: {self._device}")
        self.model.to(self._device)

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self.accelerator.unwrap_model(self._model)

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="Model Responding"
        )

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            # keep only the generation arguments of DEFAULT_GEN_KWARGS (greedy decoding)
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            for k, v in DEFAULT_GEN_KWARGS.items():
                if k not in gen_kwargs:
                    gen_kwargs[k] = v

            pop_keys = []
            for k, v in gen_kwargs.items():
                if k not in DEFAULT_GEN_KWARGS:
                    pop_keys.append(k)

            for k in pop_keys:
                gen_kwargs.pop(k)

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            if visuals:
                visuals = [
                    load_image(visual).to(torch.bfloat16).cuda()
                    for visual in visuals
                ]
                pixel_values = torch.cat(visuals, dim=0)
                num_patches_list = [visual.size(0) for visual in visuals]
                image_tokens = ["<image>"] * len(visuals)
                image_tokens = " ".join(image_tokens)
                contexts = image_tokens + "\n" + contexts
            else:
                pixel_values = None
                num_patches_list = None

            try:
                response, history = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    contexts,
                    gen_kwargs,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True,
                )
            except torch.OutOfMemoryError:
                eval_logger.warning(
                    f"OOM on task={task} doc_id={doc_id}; skipping sample."
                )
                del pixel_values, visuals
                gc.collect()
                torch.cuda.empty_cache()
                response = ""

            res.append(response)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not implemented for InternVL2")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError(
            "generate_until_multi_round is not implemented for InternVL2"
        )
