"""Per-layer label overlap of last-token representations on image-classification MCQ prompts.

Each image-classification dataset is restricted to 10 classes (random classes for
*_filter modes, the 10 most populated ones for *_top_filter), balanced, and turned into a
10-option multiple-choice prompt whose letter/option assignment changes per sample.
The last-token residual stream of every layer is extracted and the label overlap
(k nearest neighbours sharing the class, k = 30 % of the class size, DADApy) is computed.

Outputs (read by figures/fig3_probing.py):
  <results-dir>/<dataset>/<split>/<model>/label_overlap_<prompt_mode>_n<N>_cf<fraction>.json
      {"label_overlap": [... one value per layer ...], "metadata": {...}}
  <results-dir>/<dataset>/<split>/<model>/run_metrics.jsonl   (exact-match accuracy of the model)
  <results-dir>/<dataset>/<split>/<model>/prompts_<prompt_mode>_n<N>.json   (--save-prompts)
Representations are cached under <representations-dir>/representations/<model>/<dataset>/output_layer/last/<tag>/.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root
import argparse
import gc
import json
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from utils.representations.residual_stream import residual_stream_tracer
from utils.models.loading import load_hf_model_and_processor_or_tokenizer
from utils.seed import seed_all
from utils.data.collators import (
    _build_sample_signature,
    _cambrian_collate_fn,
    _internvl_collate_fn,
    _is_cambrian,
    _is_internvl,
    _llava_collate_fn,
)
from utils.data.dataset import MCQ_FILTER_MODES, SEED, ClassificationDataset
from utils.representations.metrics import classify_per_layer_label_overlap

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
ATTN_IMPLEMENTATION = "auto" if DEVICE.type == "cuda" else "sdpa"   # flash-attention for InternVL2, sdpa otherwise
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_BASE = REPO_ROOT / "results" / "label_overlap"
REPRESENTATIONS_BASE = REPO_ROOT / "representations"
RESIDUAL_STREAM_TYPE = "output_layer"
METRIC_SHUFFLE_SEED = SEED
SUPPORTED_METRICS = ("label_overlap", "accuracy")
LABEL_MODE = "class_name"   # written to the outputs: the labels are the class names

# Split of each dataset (overrides --split); keys are substrings of the dataset name, DTD's name must match exactly.
DATASET_SPLITS = {
    "mini-imagenet": "validation",
    "food101": "validation",
    "sun397": "test",
    "caltech101": "train",
    "oxford_flowers102": "test",
    "places365": "train",
}


def get_representations_and_labels(
    model_name: str,
    dataset_name: str,
    split: str,
    guide_text: str,
    tokens_pooling_method: str,
    representation_cache_path: Path,
    force_recompute_representations: bool,
    save_representations: bool,
    sample_set_tag: str,
    prompt_mode: str,
    n_choices: int,
    metrics_only: bool = False,
    save_prompts_path: Path | None = None,
    collect_predictions: bool = False,
) -> tuple[torch.Tensor, np.ndarray, list[list[str]], list[str]]:
    """(representations, labels, per-layer next-token predictions, correct option letters).

    The representations are read from the cache when its metadata match this run; the predictions
    are recorded only during an extraction (empty list otherwise).
    """
    dataset = ClassificationDataset(
        dataset_name=dataset_name,
        split=split,
        prompt_mode=prompt_mode,
        seed=SEED,
        n_choices=n_choices,
    )
    labels = np.array(dataset.labels)
    sample_ids = list(dataset.sample_ids)
    sample_signature = _build_sample_signature(sample_ids=sample_ids, labels=labels)
    prompt_signature = dataset.prompt_signature
    option_letters = list(dataset.option_letters)

    if representation_cache_path.exists() and not force_recompute_representations:
        print(f"Loading cached representations: {representation_cache_path}")
        cached_representations: torch.Tensor | None = None
        with safe_open(str(representation_cache_path), framework="pt") as handle:
            metadata = handle.metadata() or {}
            if "residual_stream" in handle.keys():
                cached_representations = handle.get_tensor("residual_stream")

        if cached_representations is not None:
            cache_prompt_signature = metadata.get("prompt_signature", "")
            is_valid_cache = (
                cached_representations.shape[1] == len(labels)
                and metadata.get("num_samples", "") == str(len(labels))
                and metadata.get("sample_signature", "") == sample_signature
                and metadata.get("tokens_pooling_method", "") == tokens_pooling_method
                and metadata.get("split", "") == split
                and metadata.get("sample_set_tag", "") == sample_set_tag
                and cache_prompt_signature in ("", prompt_signature)   # a cache without prompt signature is accepted
            )
            if is_valid_cache:
                if save_prompts_path is not None:
                    print("Representations loaded from cache: prompts not saved (use --force-recompute-representations).")
                return cached_representations, labels, [], option_letters

            if metrics_only:
                raise RuntimeError(
                    f"--metrics-only: the cached representations at {representation_cache_path} "
                    "do not match this run (shape, metadata or signatures)."
                )
            print("Cached representations do not match this run; recomputing.")

    prompt_log: list[str] = []
    _layer_token_log: list[list[str]] = []

    model, processor = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=model_name,
        dtype=MODEL_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map=DEVICE,
    )

    if _is_internvl(model_name):
        collate_fn = partial(
            _internvl_collate_fn, model=model, tokenizer=processor, guide_text=guide_text, prompt_log=prompt_log
        )
    elif _is_cambrian(model_name):
        collate_fn = partial(
            _cambrian_collate_fn, model=model, tokenizer=processor, guide_text=guide_text, prompt_log=prompt_log
        )
    else:  # LLaVA-1.5, OneVision
        collate_fn = partial(_llava_collate_fn, processor=processor, guide_text=guide_text, prompt_log=prompt_log)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    need_predictions = save_prompts_path is not None or collect_predictions
    representations = residual_stream_tracer(
        model=model,
        processed_dataloader=dataloader,
        processor=processor,
        return_dtype=torch.float32,
        layer_token_log=_layer_token_log if need_predictions else None,
    )

    if save_representations:
        os.makedirs(representation_cache_path.parent, exist_ok=True)
        save_file(
            {"residual_stream": representations.cpu().to(torch.float32)},
            str(representation_cache_path),
            metadata={
                "model_name": model_name,
                "dataset_name": dataset_name,
                "split": split,
                "tokens_pooling_method": tokens_pooling_method,
                "num_samples": str(len(labels)),
                "sample_signature": sample_signature,
                "prompt_signature": prompt_signature,
                "sample_set_tag": sample_set_tag,
            },
        )
        print(f"Saved representations cache: {representation_cache_path}")

    if save_prompts_path is not None and prompt_log:
        records = [
            {
                "idx": i,
                "sample_id": sample_ids[i],
                "label": str(labels[i]),
                "class_name": str(labels[i]),
                "option_letter": option_letters[i],
                "formatted_prompt": prompt,
                "next_tokens_per_layer": _layer_token_log[i] if i < len(_layer_token_log) else [],
            }
            for i, prompt in enumerate(prompt_log)
        ]
        save_prompts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_prompts_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"Saved prompts: {save_prompts_path}")

    result = representations.cpu().to(torch.float32)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return result, labels, _layer_token_log, option_letters


def _compute_accuracy(
    layer_predictions: list[list[str]],
    option_letters: list[str | None],
) -> tuple[float, int, int]:
    """Exact-match accuracy at the last transformer layer (logit-lens prediction)."""
    valid = [
        (preds[-1], gt)
        for preds, gt in zip(layer_predictions, option_letters)
        if gt and preds
    ]
    if not valid:
        return 0.0, 0, 0
    hits = sum(p.strip().lstrip("▁").upper() == gt.upper() for p, gt in valid)   # "▁": SentencePiece word start
    return hits / len(valid), hits, len(valid)


def _effective_dataset_split(dataset_name: str, split: str) -> str:
    lower_name = dataset_name.lower()
    if lower_name.split("/")[-1] == "dtd":
        return "train"
    for key, dataset_split in DATASET_SPLITS.items():
        if key in lower_name:
            return dataset_split
    return split


@torch.no_grad()
def _metric_permutation(num_samples: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(METRIC_SHUFFLE_SEED)
    return torch.randperm(num_samples, generator=generator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="lbasile/llava_7b")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument(
        "--split", type=str, default="test",
        help="Dataset split; the supported datasets use a fixed split (DATASET_SPLITS), which overrides it.",
    )
    parser.add_argument(
        "--guide-text",
        type=str,
        default="Answer with the option's letter from the given choices directly.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(RESULTS_BASE),
        help="Root directory for the label-overlap results (default: results/label_overlap of this repository).",
    )
    parser.add_argument("--representations-dir", type=str, default=str(REPRESENTATIONS_BASE))
    parser.add_argument("--tokens-pooling-method", type=str, default="last", choices=["last"])
    parser.add_argument("--save-representations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-recompute-representations", action="store_true")
    parser.add_argument(
        "--force-recompute-metrics",
        action="store_true",
        help="Overwrite existing metric files (by default a metric whose output file exists is skipped).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["label_overlap", "accuracy"],
        choices=list(SUPPORTED_METRICS),
    )
    parser.add_argument(
        "--save-prompts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save the formatted prompts and the per-layer next-token predictions to prompts_<prompt_mode>_n<N>.json.",
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        required=True,
        choices=list(MCQ_FILTER_MODES),
        help=(
            "mcq_nocat_filter (mini-ImageNet) and generic_mcq_filter (the other datasets): n-choices random "
            "classes; generic_mcq_top_filter: the n-choices most populated classes, balanced."
        ),
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Compute the metrics from the representation cache; fail if there is no valid cache.",
    )
    parser.add_argument(
        "--class-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of the class size used as k by the label overlap; adds a _cf<value> suffix to "
            "its output file. Default: the default of compute_label_overlap (0.3), no suffix."
        ),
    )
    parser.add_argument(
        "--n-choices",
        type=int,
        default=10,
        help="Number of classes kept and of options in the prompts; _n<N> suffix of the output files.",
    )
    return parser.parse_args()


def _output_dir(results_base: Path, dataset_name: str, split: str, model_name: str) -> Path:
    dataset_slug = dataset_name.split("/")[-1].replace("-", "_").lower()
    out_dir = results_base / dataset_slug / split / Path(model_name).name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _append_run_metric(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"Appended {record['metric']} ({record['value']:.4f}): {path}")


def _build_representation_cache_path(
    *,
    representations_base: Path,
    dataset_name: str,
    model_name: str,
    pooling_method: str,
    sample_set_tag: str,
) -> Path:
    dataset_slug = dataset_name.split("/")[-1].replace("-", "_").lower()
    model_slug = model_name.split("/")[-1]
    out_dir = (
        representations_base
        / "representations"
        / model_slug
        / dataset_slug
        / RESIDUAL_STREAM_TYPE
        / pooling_method
        / sample_set_tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "representations.safetensors"


def _save_label_overlap(
    *,
    out_path: Path,
    layer_results: dict[int, dict[str, float]],
    num_layers: int,
    args: argparse.Namespace,
    num_samples: int,
    label_classes: np.ndarray,
) -> None:
    metadata = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "pooling_method": args.tokens_pooling_method,
        "guide_text": args.guide_text,
        "num_samples": str(num_samples),
        "classes": ", ".join(str(c) for c in label_classes),
        "metric": "label_overlap",
        "metric_keys": "label_overlap",
        "prompt_mode": args.prompt_mode,
        "label_mode": LABEL_MODE,
        "n_choices": str(args.n_choices),
    }
    if args.class_fraction is not None:
        metadata["class_fraction"] = str(args.class_fraction)
    values = [float(layer_results[i]["label_overlap"]) for i in range(num_layers)]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"label_overlap": values, "metadata": metadata}, f, indent=2)
    print(f"Saved label_overlap: {out_path}")


def main() -> None:
    args = parse_args()
    effective_split = _effective_dataset_split(args.dataset_name, args.split)
    if effective_split != args.split:
        print(f"Using the dataset's split '{effective_split}' instead of '{args.split}'")
        args.split = effective_split
    results_base = Path(args.results_dir).expanduser().resolve()
    representations_base = Path(args.representations_dir).expanduser().resolve()

    output_suffix = f"_{args.prompt_mode}_n{args.n_choices}"
    sample_set_tag = f"unique_images{output_suffix}"
    label_overlap_suffix = f"_cf{args.class_fraction}" if args.class_fraction is not None else ""

    representation_cache_path = _build_representation_cache_path(
        representations_base=representations_base,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        pooling_method=args.tokens_pooling_method,
        sample_set_tag=sample_set_tag,
    )
    if args.metrics_only and not representation_cache_path.exists():
        raise FileNotFoundError(f"--metrics-only: no cached representations at {representation_cache_path}")

    out_dir = _output_dir(results_base, args.dataset_name, args.split, args.model_name)
    prompts_save_path = out_dir / f"prompts{output_suffix}.json" if args.save_prompts else None

    representations, labels, layer_predictions, option_letters = get_representations_and_labels(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        split=args.split,
        guide_text=args.guide_text,
        tokens_pooling_method=args.tokens_pooling_method,
        representation_cache_path=representation_cache_path,
        force_recompute_representations=args.force_recompute_representations,
        save_representations=args.save_representations,
        sample_set_tag=sample_set_tag,
        prompt_mode=args.prompt_mode,
        n_choices=args.n_choices,
        metrics_only=args.metrics_only,
        save_prompts_path=prompts_save_path,
        collect_predictions="accuracy" in args.metrics,
    )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(np.asarray(labels))
    num_layers, num_samples, _ = representations.shape
    print(f"{num_samples} samples, {label_encoder.classes_.size} classes, {num_layers} layers")

    if "label_overlap" in args.metrics:
        out_path = out_dir / f"label_overlap{output_suffix}{label_overlap_suffix}.json"
        if out_path.exists() and not args.force_recompute_metrics:
            print(f"Skipping label_overlap: {out_path} exists")
        else:
            permutation = _metric_permutation(num_samples)
            lo_kwargs = {"class_fraction": args.class_fraction} if args.class_fraction is not None else {}
            layer_results = classify_per_layer_label_overlap(
                representations[:, permutation, :], y[permutation.numpy()], **lo_kwargs
            )
            _save_label_overlap(
                out_path=out_path,
                layer_results=layer_results,
                num_layers=num_layers,
                args=args,
                num_samples=num_samples,
                label_classes=label_encoder.classes_,
            )

    if "accuracy" in args.metrics:
        if not layer_predictions:
            print("Accuracy not computed: next-token predictions are recorded only when the representations "
                  "are extracted (use --force-recompute-representations).")
        elif len(layer_predictions) != len(option_letters):
            print(f"Accuracy not computed: {len(layer_predictions)} predictions for {len(option_letters)} samples.")
        else:
            acc, n_correct, n_total = _compute_accuracy(layer_predictions, option_letters)
            record = {
                "metric": "accuracy",
                "value": float(acc),
                "model_name": args.model_name,
                "model_slug": Path(args.model_name).name,
                "dataset_name": args.dataset_name,
                "dataset_slug": args.dataset_name.split("/")[-1].replace("-", "_").lower(),
                "split": args.split,
                "pooling_method": args.tokens_pooling_method,
                "prompt_mode": args.prompt_mode,
                "label_mode": LABEL_MODE,
                "subsample_size": None,
                "num_samples": int(n_total),
                "experiment_suffix": output_suffix,
                "n_choices": int(args.n_choices),
                "n_correct": int(n_correct),
                "n_total": int(n_total),
            }
            _append_run_metric(out_dir / "run_metrics.jsonl", record)


if __name__ == "__main__":
    seed_all(SEED)
    main()
