"""Text-vs-image residual-stream similarity for Llama-3.2-Vision (Mllama), e.g. the 90B model (appendix).

The Llama-3.2-Vision counterpart of scripts/geometry/similarity_multimodal.py: for each dataset spec,
the last-token hidden state of every decoder layer is extracted for the caption + question and the
image + question prompts (the model is loaded once, on all visible GPUs), and the similarity measures
between text layer i and image layer j are computed.

Outputs (read by figures/fig4_geometry.py):
  <results-dir>/<dataset_prefix>/<model>_<measure>.json   {"matrix": [[...]], "metadata": {...}}, rows = text layers
Representations are cached under
  <representations-dir>/<model>/<dataset_prefix>_{txt,img}/output_layer/last/<unique_images|repeated_images>/representations.safetensors
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root
import argparse
import json
import gc
import os
from functools import partial
from typing import Any

import torch
from dotenv import load_dotenv
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, MllamaForConditionalGeneration

from utils.representations.metrics import compute_cross_layers_residual_stream_similarities
from utils.seed import seed_all
from utils.data.dataset import (
    MULTIPLE_CHOICE_BENCHMARKS,
    MultipleChoiceDatasetBenchmark,
    OpenVQADataset,
)

load_dotenv()

MODEL_DTYPE = torch.bfloat16
# flash_attention_2 fails with transformers 4.57.6 (MllamaTextSelfAttention has no is_causal attribute)
ATTN_IMPLEMENTATION = "sdpa"
INPUT_DEVICE = "cuda:0"
RESIDUAL_STREAM_TYPE = "output_layer"
POOLING_METHOD = "last"

SIMILARITY_MEASURES_DEFAULT = [
    "linear_cka",
    "neighborhood_overlap",
    "information_imbalance",
    "information_imbalance_reversed",
]


def parse_dataset_spec(spec: str) -> dict[str, Any]:
    """Parse a pipe-delimited dataset run spec.

    Format: 'dataset_path|caption_source|dataset_split|downsample_size|repeated_images|open_ended'
    Example: 'l11p/MMBench_EN-dev-captioned|synthetic|||false|true'
    """
    parts = spec.split("|")
    if len(parts) != 6:
        raise ValueError(f"Dataset spec must have exactly 6 pipe-separated fields, got {len(parts)}: {spec!r}")
    dataset_path, caption_source, dataset_split, downsample_size_s, repeated_images_s, open_ended_s = parts
    return {
        "dataset_path_or_name": dataset_path,
        "caption_source": caption_source if caption_source else "synthetic",
        "dataset_split": dataset_split if dataset_split else None,
        "downsample_size": int(downsample_size_s) if downsample_size_s else None,
        "repeated_images": repeated_images_s.strip().lower() == "true",
        "open_ended": open_ended_s.strip().lower() == "true",
    }


def collate_llama_vision_txt(batch: list[dict], processor: AutoProcessor, guide_text: str) -> dict:
    """Caption + question prompts (text representations)."""
    prompts = [
        processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "text", "text": f"{item['question']}\n{guide_text}\n".strip()}
            ]}],
            add_generation_prompt=True,
        )
        for item in batch
    ]
    return processor(text=prompts, return_tensors="pt", padding=True)


def collate_llama_vision_img(batch: list[dict], processor: AutoProcessor, guide_text: str) -> dict:
    """Image + question prompts (image representations).

    MllamaProcessor requires a nested list [[img_per_item0], [img_per_item1], ...]; a flat list fails
    for batch_size > 1.
    """
    images = [[item["image"]] for item in batch]
    prompts = [
        processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"{item['question']}\n{guide_text}\n".strip()},
            ]}],
            add_generation_prompt=True,
        )
        for item in batch
    ]
    return processor(images=images, text=prompts, add_special_tokens=False, return_tensors="pt", padding=True)


def build_dataloader(
    dataset_path_or_name: str,
    cache_dir: str | None,
    processor: AutoProcessor,
    guide_text: str,
    batch_size: int,
    seed: int,
    downsample_size: int | None,
    caption_source: str | None,
    unique_images: bool,
    open_ended: bool,
    dataset_split: str | None,
    txt_mode: bool,
) -> DataLoader:
    """DataLoader of the text (``txt_mode``) or image prompts of one dataset."""
    is_mc = any(b in dataset_path_or_name.lower() for b in MULTIPLE_CHOICE_BENCHMARKS)
    if is_mc:
        dataset = MultipleChoiceDatasetBenchmark(
            dataset_path_or_name=dataset_path_or_name,
            cache_dir=cache_dir,
            split="train",
            texts_qa=txt_mode,
            images_qa=not txt_mode,
            downsample_size=downsample_size,
            seed=seed,
            unique_images=unique_images,
            open_ended=open_ended,
        )
    else:
        dataset = OpenVQADataset(
            dataset_path_or_name=dataset_path_or_name,
            cache_dir=cache_dir,
            split="train",
            downsample_size=downsample_size,
            seed=seed,
            images_qa=not txt_mode,
            texts_qa=txt_mode,
            caption_source=caption_source,
            dataset_split=dataset_split,
        )

    collate_fn = partial(
        collate_llama_vision_txt if txt_mode else collate_llama_vision_img,
        processor=processor,
        guide_text=guide_text,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # required for multi-GPU pipeline parallelism
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=False,
    )


def extract_last_token_representations(
    model: MllamaForConditionalGeneration,
    dataloader: DataLoader,
) -> torch.Tensor:
    """Last-token hidden state of every decoder layer, (num_layers, N, hidden_size) on the CPU.

    hidden_states[0] (the token embeddings) is skipped. With left padding the last real token is
    always at position -1.
    """
    all_layers: list[list[torch.Tensor]] | None = None
    model.eval()

    dataset_size = len(dataloader.dataset)
    samples_processed = 0
    update_count = 0
    update_every = max(1, int(0.1 * dataset_size))
    progress_bar = tqdm(total=dataset_size, desc="Extracting", unit="sample")

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(INPUT_DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            current_batch_size = inputs["input_ids"].shape[0]
            outputs = model(**inputs, output_hidden_states=True)
            if outputs.hidden_states is None:
                raise RuntimeError("The model returned no hidden states (output_hidden_states=True).")

            decoder_hidden_states = outputs.hidden_states[1:]
            if all_layers is None:
                all_layers = [[] for _ in decoder_hidden_states]
            for layer_idx, hs in enumerate(decoder_hidden_states):
                # (batch, seq_len, hidden): move to the CPU at once, each layer lives on its own GPU
                all_layers[layer_idx].append(hs[:, -1, :].cpu())

            samples_processed += current_batch_size
            update_count += current_batch_size
            if update_count % update_every == 0 or samples_processed == dataset_size:
                progress_bar.update(update_count)
                torch.cuda.empty_cache()
                gc.collect()
                update_count = 0

    progress_bar.close()
    if all_layers is None:
        raise RuntimeError("The dataloader is empty")
    return torch.stack([torch.cat(x, dim=0) for x in all_layers])


def build_similarity_filename(model_name: str, measure_name: str) -> str:
    return f"{model_name}_{measure_name}.json"


def load_representation_from_cache(path: str, force: bool) -> torch.Tensor | None:
    if force or not os.path.exists(path):
        return None
    print(f"  Found cached representation: {path}")
    try:
        loaded = load_file(path)
        if "residual_stream" in loaded:
            tensor = loaded["residual_stream"]
            if tensor.ndim >= 2 and tensor.shape[0] > 0 and tensor.shape[1] > 0:
                print(f"  Loaded. Shape: {tuple(tensor.shape)}")
                return tensor[:]
            print(f"  Invalid tensor shape {tuple(tensor.shape)}, re-extracting.")
        else:
            print("  No 'residual_stream' tensor, re-extracting.")
    except (OSError, IOError, ValueError, KeyError) as e:
        print(f"  Cannot read the cache ({type(e).__name__}: {e}), re-extracting.")
    return None


def save_raw_representation(tensor: torch.Tensor, path: str, metadata: dict, force: bool) -> None:
    if not force and os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        save_file({"residual_stream": tensor.cpu()}, path, metadata={k: str(v) for k, v in metadata.items()})
        print(f"  Saved representations: {path}")
    except Exception as e:
        print(f"  Error saving {path}: {type(e).__name__}: {e}")


def find_missing_measures(
    similarity_measures: list[str],
    similarities_dir: str,
    model_name: str,
    force_recompute: bool,
) -> list[str]:
    if force_recompute:
        return list(similarity_measures)
    return [
        measure for measure in similarity_measures
        if not os.path.exists(os.path.join(similarities_dir, build_similarity_filename(model_name, measure)))
    ]


def run_pair_analysis(
    res_a: torch.Tensor,
    res_b: torch.Tensor,
    modality_a: str,
    modality_b: str,
    similarities_dir: str,
    model_name: str,
    missing_measures: list[str],
) -> None:
    """Compute and save the similarity measures of a (text, image) representation pair."""
    sim_metadata = {
        "task": "vqa",
        "row_modality": modality_a,
        "col_modality": modality_b,
        "description": f"Rows represent {modality_a} layers, Columns represent {modality_b} layers",
        "pooling_method": POOLING_METHOD,
        "model_name": model_name,
        "residual_stream_type": RESIDUAL_STREAM_TYPE,
    }
    for measure_name in missing_measures:
        try:
            measure_val = compute_cross_layers_residual_stream_similarities(
                res_a, res_b, similarity_measures=[measure_name], maxk=30,
            )[measure_name]
            save_path = os.path.join(similarities_dir, build_similarity_filename(model_name, measure_name))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump({"matrix": measure_val.cpu().tolist(),
                           "metadata": {**{k: str(v) for k, v in sim_metadata.items()},
                                        "metric": measure_name, "model_id": model_name}}, f)
            print(f"  Saved {measure_name} {tuple(measure_val.shape)}: {save_path}")
        except Exception as e:
            print(f"  Error computing {measure_name}, skipped: {type(e).__name__}: {e}")


def process_dataset(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    ensure_model: callable,
    datasets_cache_dir: str | None,
    model_name: str,
) -> None:
    """Extract the representations and compute the similarity measures for one dataset spec."""
    dataset_path_or_name = cfg["dataset_path_or_name"]
    caption_source = cfg["caption_source"]
    dataset_split = cfg["dataset_split"]
    open_ended = cfg["open_ended"]
    downsample_size = None if dataset_split else cfg["downsample_size"]  # a fixed split is used whole

    dataset_slug = dataset_path_or_name.split("/")[-1].replace("-", "_").lower()
    is_mc = any(b in dataset_slug for b in MULTIPLE_CHOICE_BENCHMARKS)
    if dataset_split and is_mc:
        raise ValueError(f"dataset_split is only valid for open-ended VQA datasets, not {dataset_path_or_name}")

    open_ended_tag = "open_ended" if (is_mc and open_ended) else ""
    sample_set_tag = "repeated_images" if (is_mc and cfg["repeated_images"]) else "unique_images"
    unique_images = is_mc and not cfg["repeated_images"]

    dataset_prefix = (
        f"{dataset_slug}_{open_ended_tag}_{caption_source}" if open_ended_tag else f"{dataset_slug}_{caption_source}"
    )
    if dataset_split:
        dataset_prefix = f"{dataset_prefix}_{dataset_split}"

    similarities_dir = os.path.join(os.path.expanduser(args.results_dir), dataset_prefix)
    os.makedirs(similarities_dir, exist_ok=True)

    def get_repr_path(dataset_n: str) -> str:
        repr_dir = os.path.join(
            os.path.expanduser(args.representations_dir),
            model_name,
            dataset_n,
            RESIDUAL_STREAM_TYPE,
            POOLING_METHOD,
            sample_set_tag,
        )
        os.makedirs(repr_dir, exist_ok=True)
        return os.path.join(repr_dir, "representations.safetensors")

    print(f"\nDataset: {dataset_path_or_name} ({dataset_prefix})")
    missing_measures = find_missing_measures(
        args.similarity_measures, similarities_dir, model_name, args.force or args.force_metrics_only,
    )
    if not missing_measures:
        print("  All measures already computed, skipping.")
        return
    print(f"  Measures to compute: {missing_measures}")

    repr_path_txt = get_repr_path(f"{dataset_prefix}_txt")
    repr_path_img = get_repr_path(f"{dataset_prefix}_img")
    res_txt = load_representation_from_cache(repr_path_txt, args.force)
    res_img = load_representation_from_cache(repr_path_img, args.force)
    txt_from_cache = res_txt is not None
    img_from_cache = res_img is not None

    dl_kwargs = dict(
        dataset_path_or_name=dataset_path_or_name,
        cache_dir=datasets_cache_dir,
        guide_text=args.guide_text,
        batch_size=args.batch_size,
        seed=args.seed,
        downsample_size=downsample_size,
        caption_source=caption_source,
        unique_images=unique_images,
        open_ended=open_ended,
        dataset_split=dataset_split,
    )
    if not txt_from_cache or not img_from_cache:
        model, processor = ensure_model()
        if not txt_from_cache:
            print(f"  Extracting text representations ({dataset_prefix}_txt)")
            res_txt = extract_last_token_representations(
                model, build_dataloader(**dl_kwargs, processor=processor, txt_mode=True)
            )
        if not img_from_cache:
            print(f"  Extracting image representations ({dataset_prefix}_img)")
            res_img = extract_last_token_representations(
                model, build_dataloader(**dl_kwargs, processor=processor, txt_mode=False)
            )
        gc.collect()
        torch.cuda.empty_cache()

    base_meta = {
        "model": model_name,
        "type": RESIDUAL_STREAM_TYPE,
        "pooling_method": POOLING_METHOD,
        "caption_source": caption_source,
        "open_ended": str(bool(open_ended_tag)),
        "repeated_images": str(sample_set_tag == "repeated_images"),
    }
    if not txt_from_cache and args.save_representations:
        save_raw_representation(
            res_txt, repr_path_txt, {**base_meta, "dataset": f"{dataset_prefix}_txt", "modality": "text"}, args.force,
        )
    if not img_from_cache and args.save_representations:
        save_raw_representation(
            res_img, repr_path_img, {**base_meta, "dataset": f"{dataset_prefix}_img", "modality": "image"}, args.force,
        )

    run_pair_analysis(
        res_txt, res_img,
        modality_a="text", modality_b="image",
        similarities_dir=similarities_dir,
        model_name=model_name,
        missing_measures=missing_measures,
    )

    del res_txt, res_img
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Text-vs-image residual stream similarity for Llama-3.2-Vision. "
            "Loads the model once and processes all dataset specs in sequence."
        )
    )
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--representations-dir", type=str, required=True,
                        help="Cache root for the extracted representations.")
    parser.add_argument(
        "--dataset-runs",
        nargs="+",
        required=True,
        help=(
            "One or more dataset run specs: "
            "'dataset_path|caption_source|dataset_split|downsample_size|repeated_images|open_ended'. "
            "Example: 'l11p/GQA-val-balanced-noyesno-captioned|synthetic|||false|true'"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, required=True, help="Path or Hugging Face ID of the model.")
    parser.add_argument("--guide-text", type=str, default="Answer the question using a single word or phrase.")
    parser.add_argument("--similarity-measures", nargs="+", default=SIMILARITY_MEASURES_DEFAULT)
    parser.add_argument("--force", action="store_true", help="Re-extract the representations and recompute all measures.")
    parser.add_argument("--force-metrics-only", action="store_true",
                        help="Recompute the measures from the cached representations.")
    parser.add_argument("--save-representations", action="store_true", help="Cache the extracted representations.")
    args = parser.parse_args()

    if args.force and args.force_metrics_only:
        parser.error("--force and --force-metrics-only are mutually exclusive.")
    try:
        dataset_configs = [parse_dataset_spec(s) for s in args.dataset_runs]
    except ValueError as e:
        parser.error(str(e))

    seed_all(args.seed)
    datasets_cache_dir = os.getenv("HF_DATASETS_CACHE")
    model_path = os.path.expanduser(args.model_path)
    model_name = model_path.rstrip("/").split("/")[-1]

    # The model is loaded on first use, once for all datasets.
    model: MllamaForConditionalGeneration | None = None
    processor: AutoProcessor | None = None

    def ensure_model() -> tuple[MllamaForConditionalGeneration, AutoProcessor]:
        nonlocal model, processor
        if model is not None:
            return model, processor
        print(f"Loading model: {model_path} (bf16, device_map=auto, attention {ATTN_IMPLEMENTATION})")
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            dtype=MODEL_DTYPE,
            device_map="auto",
            attn_implementation=ATTN_IMPLEMENTATION,
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(model_path)
        processor.tokenizer.padding_side = "left"
        return model, processor

    os.makedirs(os.path.expanduser(args.results_dir), exist_ok=True)
    for cfg in dataset_configs:
        try:
            process_dataset(cfg, args, ensure_model, datasets_cache_dir, model_name)
        except Exception as e:
            print(f"Dataset {cfg['dataset_path_or_name']} failed, continuing: {type(e).__name__}: {e}")

    if model is not None:
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    print("\nAll datasets complete.")


if __name__ == "__main__":
    main()
