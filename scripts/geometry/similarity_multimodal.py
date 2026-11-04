"""Layer-by-layer similarity between text-only and multimodal representations.

For a VQA dataset that pairs every image with a caption conveying the same
content, the script builds two prompt sets (image + question, caption +
question), extracts the last-token residual stream at the output of every
decoder layer for both, and compares layer i of the text prompts with layer j
of the multimodal prompts with linear CKA and the neighborhood overlap (k = 30); rows are text
layers, columns image layers. The information imbalance planes are computed from the same cache by
scripts/geometry/imbalance_planes.py.

The measures use the first --n-samples prompts (default 3,000), the same points as the imbalance planes
of scripts/geometry/imbalance_planes.py; the cache keeps every extracted prompt (up to 10,000).

Outputs (read by figures/fig4_geometry.py):
  <results-dir>/<dataset_prefix>/<model>_<measure>.json
      {"matrix": [[...]], "metadata": {...}}   rows = text layers, cols = image layers
Representations are cached under
  <representations-dir>/representations/<model>/<dataset_prefix>_{txt,img}/output_layer/last/unique_images/representations.safetensors
so that the measures can be recomputed without a GPU model (--force-metrics-only).

Example:
  python scripts/geometry/similarity_multimodal.py \
      --model-name-or-path lmms-lab/LLaVA-OneVision-1.5-4B-Instruct \
      --dataset-path-or-name l11p/VQAv2-COCO-restval-ICQ --caption-source human --dataset-split split01 \
      --results-dir results/similarity --representations-dir representations --save-representations
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root

import argparse
import gc
import json
import os

import torch
from safetensors.torch import load_file, save_file

from utils.representations.residual_stream import residual_stream_tracer
from utils.representations.metrics import compute_cross_layers_residual_stream_similarities
from utils.data.dataset import get_dataloader
from utils.models.loading import load_hf_model_and_processor_or_tokenizer
from utils.seed import seed_all
from utils.models.cambrian.cambrian_utils import is_cambrian_model
from utils.data.dataset import MULTIPLE_CHOICE_BENCHMARKS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
ATTN_IMPLEMENTATION = "auto" if DEVICE.type == "cuda" else "sdpa"   # flash-attention for InternVL2, sdpa otherwise
RESIDUAL_STREAM_TYPE = "output_layer"
POOLING_METHOD = "last"          # last-token representation for both prompt types
SAMPLE_SET_TAG = "unique_images"  # multiple-choice datasets are deduplicated by image

SIMILARITY_MEASURES_DEFAULT = [
    "linear_cka",
    "neighborhood_overlap",
]


def load_representation_from_cache(path: str, force: bool) -> torch.Tensor | None:
    if force or not os.path.exists(path):
        return None
    print(f"  Found cached representation: {path}")
    loaded = load_file(path)
    tensor = loaded.get("residual_stream")
    if tensor is None or tensor.ndim < 2 or tensor.shape[0] == 0:
        print("  Invalid cache, re-extracting.")
        return None
    print(f"  Loaded. Shape: {tuple(tensor.shape)}")
    return tensor


def extract_representation(dl, model, processor) -> torch.Tensor:
    return residual_stream_tracer(
        model=model,
        processed_dataloader=dl,
        processor=processor,
    )


def save_raw_representation(tensor: torch.Tensor, path: str, metadata: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({"residual_stream": tensor.cpu()}, path, metadata={k: str(v) for k, v in metadata.items()})
    print(f"  Saved representations: {path}")


def similarity_path(results_dir: str, dataset_prefix: str, model_name: str, measure: str) -> str:
    return os.path.join(results_dir, dataset_prefix, f"{model_name}_{measure}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=str, required=True, help="Where the similarity JSON files are written.")
    parser.add_argument("--representations-dir", type=str, required=True, help="Cache root for the extracted representations.")
    parser.add_argument("--dataset-path-or-name", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guide-text", type=str, default="Answer the question using a single word or phrase.")
    parser.add_argument("--n-samples", type=int, default=3000,
                        help="Prompts used by the measures (the first ones of the cache), as in imbalance_planes.py.")
    parser.add_argument("--caption-source", type=str, default="synthetic", choices=["human", "synthetic"],
                        help="Captions replacing the image in the text prompts.")
    parser.add_argument("--open-ended", action="store_true",
                        help="Format multiple-choice datasets (MMBench, SEED) as open-ended questions.")
    parser.add_argument("--dataset-split", type=str, default=None,
                        help="Precomputed split for COCO-QA / VQAv2 (e.g. split01, see data/splits).")
    parser.add_argument("--similarity-measures", nargs="+", default=SIMILARITY_MEASURES_DEFAULT)
    parser.add_argument("--save-representations", action="store_true")
    parser.add_argument("--attn-implementation", default=ATTN_IMPLEMENTATION,
                        help="HF attention backend for the extraction (default: %(default)s; Cambrian always uses sdpa unless eager is requested)")
    parser.add_argument("--force", action="store_true", help="Re-extract representations and recompute all measures.")
    parser.add_argument("--force-metrics-only", action="store_true", help="Recompute measures from cached representations.")
    args = parser.parse_args()

    seed_all(args.seed)
    results_dir = os.path.expanduser(args.results_dir)
    dataset_slug = args.dataset_path_or_name.split("/")[-1].replace("-", "_").lower()
    is_mc = any(b in dataset_slug for b in MULTIPLE_CHOICE_BENCHMARKS)
    if args.dataset_split and is_mc:
        parser.error("--dataset-split is only valid for open-ended VQA datasets.")
    if args.dataset_split:
        dataset_slug = f"{dataset_slug}_{args.dataset_split}"
    open_ended_tag = "open_ended" if (is_mc and args.open_ended) else ""
    dataset_prefix = (
        f"{dataset_slug}_{open_ended_tag}_{args.caption_source}" if open_ended_tag
        else f"{dataset_slug}_{args.caption_source}"
    )
    model_name = args.model_name_or_path.rstrip("/").split("/")[-1]
    model_cache_dir = os.getenv("HF_HUB_CACHE")
    datasets_cache_dir = os.getenv("HF_DATASETS_CACHE")

    # Which measures still have to be computed.
    if args.force or args.force_metrics_only:
        missing = list(args.similarity_measures)
    else:
        missing = [m for m in args.similarity_measures
                   if not os.path.exists(similarity_path(results_dir, dataset_prefix, model_name, m))]
    if not missing:
        print(f"All measures already computed for {model_name} / {dataset_prefix}.")
        return

    def repr_path(dataset_n: str) -> str:
        return os.path.join(os.path.expanduser(args.representations_dir), "representations", model_name,
                            dataset_n, RESIDUAL_STREAM_TYPE, POOLING_METHOD, SAMPLE_SET_TAG, "representations.safetensors")

    repr_path_txt = repr_path(f"{dataset_prefix}_txt")
    repr_path_img = repr_path(f"{dataset_prefix}_img")
    res_txt = load_representation_from_cache(repr_path_txt, args.force)
    res_img = load_representation_from_cache(repr_path_img, args.force)

    if res_txt is None or res_img is None:
        print(f"\nLoading model: {args.model_name_or_path}")
        model, processor = load_hf_model_and_processor_or_tokenizer(
            model_name_or_path=args.model_name_or_path, cache_dir=model_cache_dir,
            dtype=MODEL_DTYPE, attn_implementation=args.attn_implementation, device_map=DEVICE,
        )
        dl_args = {
            "dataset_path_or_name": args.dataset_path_or_name,
            "seed": args.seed, "processor": processor, "batch_size": args.batch_size,
            "cache_dir": datasets_cache_dir, "guide_text": args.guide_text,
            "split": "train", "caption_source": args.caption_source,
        }
        if is_mc:
            dl_args["unique_images"] = True
        if open_ended_tag:
            dl_args["open_ended"] = True
        if args.dataset_split:
            dl_args["dataset_split"] = args.dataset_split
        if "intern" in model.__class__.__name__.lower() or is_cambrian_model(model=model):
            dl_args["model"] = model
        if is_cambrian_model(model=model):
            dl_args["num_workers"] = 0

        base_meta = {"model": model_name, "type": RESIDUAL_STREAM_TYPE, "pooling_method": POOLING_METHOD,
                     "caption_source": args.caption_source, "open_ended": str(bool(open_ended_tag))}
        if res_txt is None:
            print(f"  Extracting text representations ({dataset_prefix}_txt)...")
            res_txt = extract_representation(get_dataloader(texts_qa=True, **dl_args), model, processor)
            if args.save_representations:
                save_raw_representation(res_txt, repr_path_txt, {**base_meta, "dataset": f"{dataset_prefix}_txt"})
        if res_img is None:
            print(f"  Extracting image representations ({dataset_prefix}_img)...")
            res_img = extract_representation(get_dataloader(images_qa=True, **dl_args), model, processor)
            if args.save_representations:
                save_raw_representation(res_img, repr_path_img, {**base_meta, "dataset": f"{dataset_prefix}_img"})
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    n_points = min(args.n_samples, res_txt.shape[1], res_img.shape[1])
    res_txt, res_img = res_txt[:, :n_points], res_img[:, :n_points]
    print(f"Computing {missing} for {model_name} / {dataset_prefix} on {n_points} prompts "
          "(rows: text layers, cols: image layers)")
    for measure in missing:
        matrix = compute_cross_layers_residual_stream_similarities(
            res_txt, res_img, similarity_measures=[measure], maxk=30)[measure]
        out = similarity_path(results_dir, dataset_prefix, model_name, measure)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"matrix": matrix.cpu().tolist(),
                       "metadata": {"model_id": model_name, "metric": measure, "dataset": dataset_prefix,
                                    "row_modality": "text", "col_modality": "image",
                                    "pooling_method": POOLING_METHOD, "variant_tag": SAMPLE_SET_TAG,
                                    "n_points": n_points}}, f)
        print(f"  Saved {measure}: {out}")


if __name__ == "__main__":
    main()
