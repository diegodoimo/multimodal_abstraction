"""SentEval linguistic probing (appendix): logistic-regression probes on the residual stream of every layer.

Each sentence is rendered as one user turn with the model's chat template (no assistant header) and
represented per layer by text_avg, the mean over the non-special tokens. Probes are trained on --n_rep
disjoint class-balanced training subsets and scored on the test split.

Output: <out_dir>/<model>/<target_layer>/<pool>/<task>_accuracy_logistic.json,
{layer module name: [balanced accuracy per training subset]}.
Data: bash scripts/probing/download_senteval.sh (SOMO, CoordInv -> data/probing).
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root
import argparse
import gc
import json
import os
from collections import defaultdict
from functools import partial

import numpy as np
import torch

from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from utils.models.onevision.llavaonevision_compat import patch_flash_attn_varlen_import
from utils.linguistic_probing.activations import extract_activations
from utils.models.layers import get_embdims, get_target_layers
from utils.linguistic_probing.probes import fit_sklearn_no_cv
from utils.linguistic_probing.senteval_data import DataCollatorForCausalLM, load_senteval_data

patch_flash_attn_varlen_import()  # OneVision-8B remote code

PAPER_TASKS = ["CoordInv", "SOMO"]
POOLS = ["text_avg"]  # the pool read by figures/app_linguistic_probing.py
NUM_WORKERS = 6

# Output folder per model; other models use the last component of their path.
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
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(_Path(__file__).resolve().parents[2] / "data" / "probing"),
        help="Directory with the SentEval probing files.",
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/linguistic_probing")
    parser.add_argument("--subset_size", type=int, default=5000, help="Sentences per training subset.")
    parser.add_argument("--n_rep", type=int, default=3, help="Number of training subsets (one probe each).")
    parser.add_argument(
        "--task_name", type=str, default="paper", choices=["paper"] + PAPER_TASKS,
        help="One task, or 'paper' for both.",
    )
    parser.add_argument(
        "--target_layer", type=str, default="res2", choices=["res2"],
        help="Residual stream at the output of each decoder block.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size of the forward passes.")
    return parser.parse_args()


def detect_family(model_name_or_path):
    name = model_name_or_path.lower()
    for family in ("internvl", "cambrian", "onevision", "llava"):
        if family in name:
            return family
    raise ValueError(f"Unsupported model: {model_name_or_path}")


def load_model(model_name_or_path, family):
    """(model, tokenizer, render) where render(text) is the chat-templated user turn."""
    print(f"Loading {model_name_or_path}")
    if family == "cambrian":
        from utils.models.cambrian.patch_cambrian import install_compat as install_cambrian_compat
        from utils.models.cambrian.cambrian_utils import load_cambrian

        model, tokenizer, _, conv_mode = load_cambrian(model_name_or_path)
        install_cambrian_compat(model)  # forward compatible with current transformers
        from cambrian.conversation import conv_templates  # importable once the model is loaded

        def render(text):
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], text)
            return conv.get_prompt()

        return model, tokenizer, render

    if family == "internvl":
        from utils.models.internvl.internvl2_compat import patch_internvl_model_post_instantiation
        from utils.models.internvl.internvl_conversations import get_conv_template
        from utils.models.internvl.load_helpers import force_internvl_attn_implementation

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        with force_internvl_attn_implementation("flash_attention_2"):
            model = AutoModel.from_pretrained(
                model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            ).to("cuda")
        patch_internvl_model_post_instantiation(model)

        def render(text):
            tmpl = get_conv_template("internlm2-chat")
            tmpl.append_message(tmpl.roles[0], text)
            return tmpl.get_prompt()

        return model, tokenizer, render

    # LLaVA-1.5 and OneVision: the chat template is on the processor.
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
    if family == "onevision":
        # sdpa, as for every model except InternVL2 (utils.models.loading.resolve_attn_implementation_for_model)
        kwargs["attn_implementation"] = "sdpa"
    model = AutoModel.from_pretrained(model_name_or_path, **kwargs).to("cuda")

    def render(text):
        return processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": text}]}],
            add_generation_prompt=False,
            tokenize=False,
        )

    return model, processor.tokenizer, render


def make_forward_fn(model, family):
    """InternVL2 and Cambrian: run the language model alone (their wrappers expect image inputs)."""
    if family == "internvl":
        return lambda batch: model.language_model(**batch, use_cache=False)
    if family == "cambrian":
        return lambda batch: model.model(**batch, use_cache=False)
    return None


def tokenize(inputs, tokenizer, render):
    encoded = tokenizer([render(text) for text in inputs["text"]], add_special_tokens=False)
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


def get_activations(dataset, tokenize_fn, model, target_layer_names, pad_token_id, batch_size,
                    forward_fn, special_token_ids):
    """{pool: {layer module name: (n_sentences, hidden) tensor}} for POOLS."""
    dataset = dataset.map(tokenize_fn, batched=True)
    loader = DataLoader(
        dataset,
        shuffle=False,
        collate_fn=DataCollatorForCausalLM(pad_token_id=pad_token_id),  # left padding
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    embdims, dtypes = get_embdims(model, loader, target_layer_names, forward_fn=forward_fn)
    extractor = extract_activations(
        model=model,
        dataloader=loader,
        target_layers=target_layer_names,
        embdims=embdims,
        dtypes=dtypes,
        special_token_ids=special_token_ids,
        forward_fn=forward_fn,
    )
    extractor.extract(loader)
    return {"text_avg": extractor.text_avg}


def get_train_subsets(dataset, subset_size=5000, n_rep=5):
    """n_rep disjoint class-stratified subsets of about subset_size sentences each (seed 42)."""
    assert subset_size < 8000
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(dataset["label"]):
        label_to_indices[label].append(idx)

    rng = np.random.default_rng(seed=42)
    for label in label_to_indices:
        rng.shuffle(label_to_indices[label])

    n = len(dataset) // subset_size
    assert n_rep <= n, f"n_rep ({n_rep}) must be <= {n}"

    label_chunks = {}
    for label, indices in label_to_indices.items():
        if len(indices) < n:
            raise ValueError(f"Label {label} has fewer samples than subsets")
        label_chunks[label] = np.array_split(indices, n)

    subsets = []
    for i in range(n_rep):
        subset_indices = []
        for label in label_chunks:
            subset_indices.extend(label_chunks[label][i])
        subsets.append(dataset.select(subset_indices))
    return [subset.shuffle(seed=42) for subset in subsets]


def main(args):
    family = detect_family(args.model_name_or_path)
    model, tokenizer, render = load_model(args.model_name_or_path, family)
    model = model.eval()
    if family == "cambrian":
        n_layer = model.config.num_hidden_layers
    else:
        n_layer = model.language_model.config.num_hidden_layers

    target_layers = get_target_layers(model, n_layer, family)
    target_layer_names = list(target_layers.values())
    forward_fn = make_forward_fn(model, family)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    # Tokens excluded from the pools (BOS / EOS / PAD and chat-template tokens such as <|im_end|>).
    special_token_ids = list(tokenizer.all_special_ids)

    model_name = MODEL_DIRS.get(args.model_name_or_path, args.model_name_or_path.rstrip("/").split("/")[-1])
    dirpath_base = os.path.join(args.out_dir, model_name, args.target_layer)
    os.makedirs(dirpath_base, exist_ok=True)
    task_names = PAPER_TASKS if args.task_name == "paper" else [args.task_name]
    tokenize_fn = partial(tokenize, tokenizer=tokenizer, render=render)
    extract = partial(
        get_activations,
        tokenize_fn=tokenize_fn,
        model=model,
        target_layer_names=target_layer_names,
        pad_token_id=pad_token_id,
        batch_size=args.batch_size,
        forward_fn=forward_fn,
        special_token_ids=special_token_ids,
    )

    for task_name in task_names:
        print(f"Task {task_name}")
        train_dataset, test_dataset = load_senteval_data(task_name, args.data_path)
        train_subsets = get_train_subsets(train_dataset, subset_size=args.subset_size, n_rep=args.n_rep)

        test_pools = {
            pool: {layer: tensor.to(torch.float64).numpy() for layer, tensor in layers.items()}
            for pool, layers in extract(test_dataset).items()
        }
        y_test = np.array(test_dataset["label"])
        accuracy = {pool: {layer: [] for layer in test_pools[pool]} for pool in POOLS}

        for i, train_subset in enumerate(train_subsets):
            train_pools = extract(train_subset)
            y_train = np.array(train_subset["label"])
            for pool in POOLS:
                X_train = train_pools[pool]
                scores = Parallel(n_jobs=8)(
                    delayed(fit_sklearn_no_cv)(X_train[layer].to(torch.float64).numpy(), y_train, X_test, y_test)
                    for layer, X_test in test_pools[pool].items()
                )
                for layer, acc in zip(test_pools[pool], scores):
                    accuracy[pool][layer].append(acc)
            del train_pools
            gc.collect()
            means = ", ".join(
                f"{pool} {np.mean([np.mean(a) for a in accuracy[pool].values()]):.4f}" for pool in POOLS
            )
            print(f"subset {i + 1}/{args.n_rep}: mean accuracy over layers {means}")

        for pool in POOLS:
            pool_dir = os.path.join(dirpath_base, pool)
            os.makedirs(pool_dir, exist_ok=True)
            with open(os.path.join(pool_dir, f"{task_name}_accuracy_logistic.json"), "w") as f:
                json.dump(accuracy[pool], f, indent=2)


if __name__ == "__main__":
    main(parse_args())
