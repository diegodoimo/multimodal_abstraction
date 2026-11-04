"""Per-layer mean activations of a VLM on the LLaVA detail_23k COCO subset (COCO retrieval appendix).

  --image_only : the prompt holds only the image; mean over the image tokens
                 -> <out_dir>/<model>/coco_image_only/layers/l{layer}res2_imgavg.pt
  --text_only  : the prompt holds only the detailed caption; mean over the caption tokens (all
                 tokens except the special ones) -> <out_dir>/<model>/coco_text_only/layers/l{layer}res2_textavg.pt
Each file holds the residual stream at the output of decoder block {layer}, one row per sample.
scripts/probing/retrieval.py reads them.

Example (one job per model and prompt type; slurm/coco_retrieval.sbatch runs both):
  python scripts/probing/extract_coco_embeddings.py --model_name_or_path lbasile/llava_7b \
      --image_dir /path/to/mscoco/train2014 --out_dir representations/coco --target_layer res2 --image_only
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root
import argparse
import os

import torch

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

from utils.retrieval.cambrian_pooling import (
    IMAGE_TOKEN_SENTINEL as CAMBRIAN_IMAGE_SENTINEL,
    CambrianImageDataset,
    cambrian_collator,
    make_cambrian_pool_info_provider,
    wrap_cambrian_for_pooling,
)
from utils.retrieval.datasets import (
    DataCollatorForVisionLM,
    DetailImageDataset,
    InternVLImageDataset,
    TextOnlyDataset,
    load_and_validate_detail23k,
)
from utils.retrieval.extraction import mean_representations
from utils.models.internvl.internvl2_compat import patch_internvl_model_post_instantiation
from utils.models.internvl.load_helpers import force_internvl_attn_implementation
from utils.models.cambrian.cambrian_utils import load_cambrian
from utils.models.cambrian.patch_cambrian import install_compat as install_cambrian_compat
from utils.models.onevision.llavaonevision_compat import patch_flash_attn_varlen_import
from utils.models.layers import get_target_layers

patch_flash_attn_varlen_import()  # OneVision-8B remote code

NUM_WORKERS = 6

# Output folder per model (the same names as in slurm/coco_retrieval.sbatch); other models use
# the last component of their path.
MODEL_DIRS = {
    "llava-hf/llava-1.5-7b-hf": "llava_7b_hf",
    "llava-hf/llava-1.5-13b-hf": "llava_13b_hf",
    "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct": "onevision_4b_hf",
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct": "onevision_8b_hf",
    "OpenGVLab/InternVL2-8B": "internvl_8b_hf",
    "nyu-visionx/cambrian-8b": "cambrian_8b",
    "lbasile/llava_13b": "llava_13b",
    "lbasile/llava_7b": "llava_7b",
}

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True, help="Root of the representation cache.")
    parser.add_argument(
        "--target_layer", type=str, default="res2", choices=["res2"],
        help="Residual stream at the output of each decoder block.",
    )
    parser.add_argument(
        "--detail_json", type=str, default=None,
        help="Path to LLaVA's detail_23k.json; downloaded from liuhaotian/LLaVA-Instruct-150K when omitted.",
    )
    parser.add_argument(
        "--image_dir", type=str, default="",
        help="Directory with the COCO train2014 images (COCO_train2014_*.jpg).",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image_only", action="store_true", help="Prompts with the image only.")
    mode.add_argument("--text_only", action="store_true", help="Prompts with the detailed caption only.")
    return parser.parse_args()


def model_family(model_name_or_path):
    name = model_name_or_path.lower()
    for family in ("cambrian", "internvl", "onevision", "llava"):
        if family in name:
            return family
    raise ValueError(f"Unsupported model: {model_name_or_path}")


def image_token_id(family, tokenizer):
    """Id of the image-token positions averaged in the image prompts (Cambrian: the sentinel of the
    synthetic input_ids, see utils/retrieval/cambrian_pooling.py)."""
    if family == "internvl":
        return tokenizer.convert_tokens_to_ids(InternVLImageDataset.IMG_CONTEXT_TOKEN)
    if family == "onevision":
        return tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if family == "cambrian":
        return CAMBRIAN_IMAGE_SENTINEL
    return tokenizer.convert_tokens_to_ids("<image>")  # llava


def main():
    args = parse_args()
    if not args.image_dir and not args.text_only:
        raise ValueError("--image_dir is required with --image_only")
    if args.detail_json is None:
        from huggingface_hub import hf_hub_download
        args.detail_json = hf_hub_download("liuhaotian/LLaVA-Instruct-150K", "detail_23k.json", repo_type="dataset")
    family = model_family(args.model_name_or_path)

    # processor: the Hugging Face processor (LLaVA-1.5, OneVision) or the tokenizer (InternVL2, Cambrian).
    if family == "cambrian":
        model, tokenizer, image_processor, conv_mode = load_cambrian(args.model_name_or_path)
        processor = tokenizer
    elif family == "internvl":
        processor = tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    else:
        processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        tokenizer = processor.tokenizer

    records = load_and_validate_detail23k(args.detail_json, args.image_dir, validate_images=not args.text_only)
    print(f"{len(records)} detail_23k records")
    if args.text_only:
        dataset_tag = "coco_text_only"
        dataset = TextOnlyDataset(
            records, processor, family=family, cambrian_conv_mode=conv_mode if family == "cambrian" else None
        )
    else:
        dataset_tag = "coco_image_only"
        if family == "cambrian":
            dataset = CambrianImageDataset(records, tokenizer, image_processor, model.config, conv_mode)
        elif family == "internvl":
            # (image_size // patch_size) ** 2 * downsample_ratio ** 2 tokens per tile, as in InternVLChatModel
            config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
            num_image_token = int(
                (config.force_image_size // config.vision_config.patch_size) ** 2 * (config.downsample_ratio**2)
            )
            dataset = InternVLImageDataset(records, tokenizer, num_image_token=num_image_token)
        else:
            dataset = DetailImageDataset(records, processor)
    print(f"Prompt: {dataset[0]['text_prompts']!r}")

    if family == "cambrian" and not args.text_only:
        collate_fn = cambrian_collator
    else:
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        collate_fn = DataCollatorForVisionLM(pad_token_id=pad_token_id)
    # batch size 1: the number of image tokens differs between images
    dataloader = DataLoader(
        dataset, shuffle=False, collate_fn=collate_fn, batch_size=1, num_workers=NUM_WORKERS, pin_memory=True
    )

    from_pretrained_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
    if family == "cambrian":
        install_cambrian_compat(model)  # forward compatible with current transformers
        if not args.text_only:
            wrap_cambrian_for_pooling(model)  # after install_compat, which replaces the wrapped method
    elif family == "internvl":
        with force_internvl_attn_implementation("flash_attention_2"):
            model = AutoModel.from_pretrained(args.model_name_or_path, **from_pretrained_kwargs).to("cuda")
        patch_internvl_model_post_instantiation(model)
        if not args.text_only:
            # The forward puts the vision features at the positions of this token.
            model.img_context_token_id = tokenizer.convert_tokens_to_ids(InternVLImageDataset.IMG_CONTEXT_TOKEN)
    else:
        if family == "onevision":
            # sdpa, as for every model except InternVL2 (utils.models.loading.resolve_attn_implementation_for_model);
            # eager attention would run out of memory on OneVision's vision tower.
            from_pretrained_kwargs["attn_implementation"] = "sdpa"
        model = AutoModel.from_pretrained(args.model_name_or_path, **from_pretrained_kwargs).to("cuda")
    model = model.eval()

    if family == "cambrian":
        n_layer = model.config.num_hidden_layers
    else:
        n_layer = model.language_model.config.num_hidden_layers
    target_layers = get_target_layers(model, n_layer, family)
    nsamples = len(dataset)
    print(f"{nsamples} samples, {len(target_layers)} layers")

    model_name = MODEL_DIRS.get(args.model_name_or_path, args.model_name_or_path.rstrip("/").split("/")[-1])
    layers_dir = f"{args.out_dir}/{model_name}/{dataset_tag}/layers"
    os.makedirs(layers_dir, exist_ok=True)

    # Cambrian with images: pool over the expanded sequence (see utils/retrieval/cambrian_pooling.py).
    pool_info_provider = (
        make_cambrian_pool_info_provider(model) if family == "cambrian" and not args.text_only else None
    )
    # Text only: InternVL2 and Cambrian run the language model alone (their wrappers expect images).
    forward_fn = None
    if args.text_only and family == "internvl":
        forward_fn = lambda batch: model.language_model(**batch, use_cache=False)
    elif args.text_only and family == "cambrian":
        forward_fn = lambda batch: model.model(**batch, use_cache=False)

    means = mean_representations(
        model=model,
        dataloader=dataloader,
        target_layers=target_layers,
        nsamples=nsamples,
        image_token_id=None if args.text_only else image_token_id(family, tokenizer),
        exclude_token_ids=list(tokenizer.all_special_ids) if args.text_only else None,
        forward_fn=forward_fn,
        pool_info_provider=pool_info_provider,
    )
    suffix = "textavg" if args.text_only else "imgavg"
    for layer_id, layer_name in target_layers.items():
        torch.save(means[layer_name], f"{layers_dir}/l{layer_id}{args.target_layer}_{suffix}.pt")
    print(f"Saved {layers_dir}")


if __name__ == "__main__":
    main()
