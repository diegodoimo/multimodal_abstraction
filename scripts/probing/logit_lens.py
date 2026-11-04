"""Per-layer logit-lens accuracy and linear-probe accuracy on VQA benchmarks (Fig. 3 right and appendix).

For each dataset of DATASETS, 2,000 image + question prompts are run through the model and the
last-token residual stream of every layer is extracted. The logit lens decodes it with the final norm
and the LM head: multiple-choice datasets score the most likely answer option, open-ended datasets
count a hit when the first token of the answer (lower-case or capitalized) is among the top 10 tokens.
The linear probe (multiple-choice datasets) is a logistic regression per layer on an 80/20 split.

Outputs (read by figures/fig3_probing.py):
  <results-dir>/<model>_<dataset>_images_qa_{logit_lens,linear_probe}.safetensors   {"accuracies": per-layer array}
Existing output files are skipped.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root
import argparse
import gc
import os

import numpy as np
import torch
from dotenv import load_dotenv
from safetensors.numpy import save_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from utils.representations.residual_stream import residual_stream_tracer
from utils.models.cambrian.cambrian_utils import is_cambrian_model
from utils.data.constants import (
    MMBENCH_ANSWER_CHOICES,
    MME_ANSWER_CHOICES,
    POPE_ANSWER_CHOICES,
    SQA_ANSWER_CHOICES,
)

from utils.data.dataset import get_dataloader
from utils.models.loading import load_hf_model_and_processor_or_tokenizer
from utils.seed import seed_all

load_dotenv()

MODEL_DTYPE = torch.bfloat16
ATTN_IMPLEMENTATION = "sdpa"
TOP_K_LOGITS = 10        # open-ended logit lens: hit if the answer's first token is among the top k
N_SAMPLES = 2000         # samples per dataset
SUBSET = "images_qa"     # image + question prompts (part of the output file names)

SHORT_ANSWER = "Answer the question using a single word or phrase."
OPTION_LETTER = "Answer with the option's letter from the given choices directly"

# "choices": answer options of the multiple-choice datasets (logit lens restricted to their tokens;
# these datasets also get a linear probe). The open-ended datasets have none.
DATASETS = {
    "l11p/COCOQA-restval-ICQA": {"guide_text": SHORT_ANSWER, "split": "train", "remove_yes_no": True},
    "l11p/VQAv2-COCO-restval-ICQ": {"guide_text": SHORT_ANSWER, "split": "train", "remove_yes_no": True},
    "l11p/MME-test-captioned": {"guide_text": SHORT_ANSWER, "split": "train", "choices": MME_ANSWER_CHOICES},
    "l11p/MMBench_EN-dev-captioned": {
        "guide_text": OPTION_LETTER, "split": "train", "choices": MMBENCH_ANSWER_CHOICES,
        "filter_mmbench_4_choices": True,
    },
    "lmms-lab/POPE": {"guide_text": SHORT_ANSWER, "split": "test", "choices": POPE_ANSWER_CHOICES},
    "l11p/GQA-val-balanced-noyesno-captioned": {"guide_text": SHORT_ANSWER, "split": "train"},  # no yes/no answers
    "lmms-lab/ScienceQA": {"guide_text": OPTION_LETTER, "split": "test", "choices": SQA_ANSWER_CHOICES},
    "vikhyatk/CountBenchQA": {"guide_text": "Answer with just a number.", "split": "test"},
}


def apply_rmsnorm_residual(
    x: torch.Tensor,  # (n, t, ..., d)
    second_moment: torch.Tensor,  # (n, t)
    weight: torch.Tensor,  # (d,)
    variance_eps: float,
):
    """RMSNorm of x with the given second moment, cast to the weight's dtype before scaling (as LlamaRMSNorm)."""
    assert second_moment.dim() == 2, f"second moments must have 2 dimensions (n, t). Found {second_moment.shape}"
    assert weight.dim() == 1, f"weight must have 1 dimension (d,). Found {weight.shape}"
    assert x.size(0) == second_moment.size(0), f"n dimension mismatch. {x.size(0)=} {second_moment.size(0)=}"
    assert x.size(-1) == weight.size(0), f"d dimension mismatch. {x.size(-1)=} {weight.size(0)=}"

    num_unit_dims = x.ndim - 3
    second_moment = second_moment.view(
        second_moment.shape[0], second_moment.shape[1], *[1] * num_unit_dims, 1
    )  # (n, t, ..., 1)
    weight = weight.view(*[1] * (num_unit_dims + 1), -1)  # (1, ..., d)

    x = x * torch.rsqrt(second_moment + variance_eps)
    if x.dtype != weight.dtype:
        x = x.to(weight.dtype)
    return weight * x


def get_target_token_ids(tokenizer, choices):
    """First token id of each answer option."""
    return [tokenizer.encode(c, add_special_tokens=False)[0] for c in choices]


def evaluate_logit_lens(model, tokenizer, residual_stream, labels, choices=None):
    """Per-layer logit-lens accuracy: option accuracy if ``choices`` is given, top-k hit rate otherwise."""
    num_layers, num_samples, _ = residual_stream.shape
    layer_hits = torch.zeros(num_layers, device="cpu")
    all_ground_truths = []
    all_predictions = [[] for _ in range(num_layers)]

    target_ids_tensor = None
    if choices is not None:
        target_ids = get_target_token_ids(tokenizer, choices)
        target_ids_tensor = torch.tensor(target_ids, device=model.device)
        print(f"  Answer options {choices} -> token ids {target_ids}")

    for i in tqdm(range(num_samples), desc="Logit lens"):
        ground_truth = labels[i].strip()
        sample_res = residual_stream[:, i, :].to(model.device).to(MODEL_DTYPE).unsqueeze(0)  # (1, layers, d)

        with torch.no_grad():
            # second moment of the last layer's residual, used for every layer
            second_moment = sample_res[:, -1].to(torch.float32).pow(2).mean(dim=-1).unsqueeze(1)

            if "llava" in model.__class__.__name__.lower():
                normalized_res = (
                    apply_rmsnorm_residual(
                        sample_res.to(torch.float32),
                        second_moment,
                        model.model.language_model.norm.weight,
                        model.model.language_model.norm.variance_epsilon,
                    )
                    .squeeze()
                    .to(MODEL_DTYPE)
                )
                logits = model.lm_head(normalized_res)  # (layers, vocab)
            elif "intern" in model.__class__.__name__.lower():
                normalized_res = (
                    apply_rmsnorm_residual(
                        sample_res.to(torch.float32),
                        second_moment,
                        model.language_model.model.norm.weight,
                        model.language_model.model.norm.variance_epsilon,
                    )
                    .squeeze()
                    .to(MODEL_DTYPE)
                )
                logits = model.language_model.output(normalized_res)
            elif is_cambrian_model(model=model):
                normalized_res = (
                    apply_rmsnorm_residual(
                        sample_res.to(torch.float32),
                        second_moment,
                        model.model.norm.weight,
                        model.model.norm.variance_epsilon,
                    )
                    .squeeze()
                    .to(MODEL_DTYPE)
                )
                logits = model.lm_head(normalized_res.to(model.lm_head.weight.dtype))
            else:
                raise ValueError(f"Unsupported model for the logit lens: {model.__class__.__name__}")

            if choices is not None:
                pred_indices = torch.argmax(logits[:, target_ids_tensor], dim=-1)  # (layers,)
                if ground_truth in choices:
                    all_ground_truths.append(choices.index(ground_truth))
                    for l_idx in range(num_layers):
                        all_predictions[l_idx].append(pred_indices[l_idx].item())
            else:
                _, topk_ids = logits.topk(TOP_K_LOGITS, dim=-1)  # (layers, k)
                # first token of the lower-case and of the capitalized answer
                target_token_ids = []
                for variant in (ground_truth.lower(), ground_truth.capitalize()):
                    ids = tokenizer.encode(variant, add_special_tokens=False)
                    if len(ids) > 0:
                        target_token_ids.append(ids[0])
                target_token_ids = list(set(target_token_ids))
                if target_token_ids:
                    hits = torch.zeros(topk_ids.shape[0], device="cpu")
                    for tid in target_token_ids:
                        hits = torch.maximum(hits, (topk_ids == tid).any(dim=-1).float().cpu())
                    layer_hits += hits

    if choices is None:
        return (layer_hits / num_samples).numpy()
    if not all_ground_truths:
        print("  No answer matches the answer options; accuracies set to 0.")
        return np.zeros(num_layers)
    return np.array([accuracy_score(all_ground_truths, all_predictions[l_idx]) for l_idx in range(num_layers)])


def evaluate_linear_probe(residual_stream, labels, seed):
    """Per-layer test accuracy of a logistic regression on a stratified 80/20 split."""
    y = LabelEncoder().fit_transform(labels)
    res_stream_np = residual_stream.float().cpu().numpy()
    accuracies = []
    for layer_idx in range(res_stream_np.shape[0]):
        X_train, X_test, y_train, y_test = train_test_split(
            res_stream_np[layer_idx], y, test_size=0.2, random_state=seed, stratify=y
        )
        clf = LogisticRegression(max_iter=5000, random_state=seed)
        clf.fit(X_train, y_train)
        accuracies.append(accuracy_score(y_test, clf.predict(X_test)))
    return np.array(accuracies)


def main():
    parser = argparse.ArgumentParser(description="Logit lens and linear probes")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = os.path.expanduser(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    seed_all(args.seed)

    print(f"Loading model: {args.model_name_or_path}")
    model, processor = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=args.model_name_or_path,
        cache_dir=os.path.expanduser(os.getenv("HF_HUB_CACHE")),
        dtype=MODEL_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map="auto",
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    model_name_slug = args.model_name_or_path.split("/")[-1]
    datasets_cache_dir = os.path.expanduser(os.getenv("HF_DATASETS_CACHE"))

    for dataset_name, config in DATASETS.items():
        dataset_slug = dataset_name.split("/")[-1].replace("-", "_").lower()
        print(f"\nDataset: {dataset_name}")
        choices = config.get("choices")
        eval_methods = ["linear_probe", "logit_lens"] if choices is not None else ["logit_lens"]

        def result_path(method):
            return os.path.join(results_dir, f"{model_name_slug}_{dataset_slug}_{SUBSET}_{method}.safetensors")

        methods_to_run = [m for m in eval_methods if not os.path.exists(result_path(m))]
        if not methods_to_run:
            print("  All results exist; skipping.")
            continue

        dl_args = {
            "dataset_path_or_name": dataset_name,
            "seed": args.seed,
            "processor": processor,
            "batch_size": args.batch_size,
            "cache_dir": datasets_cache_dir,
            "answer_letters_with_processed_batch": True,
            "split": config["split"],
            "guide_text": config["guide_text"],
            "downsample_size": N_SAMPLES,
            "remove_yes_no": config.get("remove_yes_no", False),
            "filter_mmbench_4_choices": config.get("filter_mmbench_4_choices", False),
            "images_qa": True,
        }
        if "intern" in model.__class__.__name__.lower() or is_cambrian_model(model=model):
            dl_args["model"] = model
        if is_cambrian_model(model=model):
            dl_args["num_workers"] = 0

        dataloader = get_dataloader(**dl_args)
        all_labels = [label for batch in dataloader for label in batch[-1]]  # answers: last element of each batch

        residual_stream = residual_stream_tracer(
            model=model,
            processor=processor,
            processed_dataloader=dataloader,
        )

        for eval_method in methods_to_run:
            if eval_method == "logit_lens":
                accuracies = evaluate_logit_lens(model, tokenizer, residual_stream, all_labels, choices)
            else:
                accuracies = evaluate_linear_probe(residual_stream, all_labels, args.seed)
            save_file({"accuracies": accuracies}, result_path(eval_method))
            print(f"  Saved {eval_method}: {result_path(eval_method)}")

        del residual_stream
        gc.collect()
        torch.cuda.empty_cache()

    print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()
