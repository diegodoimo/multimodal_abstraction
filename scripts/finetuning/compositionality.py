"""ARO and SugarCrepe compositionality benchmarks (Fig. 5 table, Fig. 6 and the radar appendix).

Each sample is a multiple-choice question (which caption describes the image, options in an order
shuffled with a fixed seed per sample); the answer is the option letter with the highest logit at the
last prompt position, without generation. ARO: VG-Relation, VG-Attribution, COCO-Order and
Flickr30k-Order, at most 5,000 samples each; SugarCrepe: the seven categories in full.
--no-image evaluates the same prompts with the image tokens masked at every layer (image-reliance
baseline).

Models: LLaVA-1.5 and pixtral, LLaVA-OneVision-1.5, InternVL2, Cambrian.
Data: bash scripts/finetuning/download_aro.sh and bash scripts/finetuning/download_sugarcrepe.sh;
the ARO dataset code is vendored in third_party/vision-language-models-are-bows.
Launcher: slurm/compositionality.sbatch.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root
import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import nltk
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.seed import seed_all
from utils.models.cambrian.cambrian_utils import (
    is_cambrian_model,
    prepare_cambrian_batch_inputs,
)
from utils.models.internvl.internvl_utils import (
    IMG_CONTEXT_TOKEN,
    apply_chat_template as apply_internvl_chat_template,
    get_combined_embeddings,
    get_pixel_values,
)
from utils.models.loading import (
    load_hf_model_and_processor_or_tokenizer,
    resolve_attn_implementation_for_model,
)
# The image-blind evaluation (--no-image) masks the image tokens at every layer with the
# ablation patches of the intervention experiments; the *_eval patches of the benchmark path
# are driven by the lmms-eval generation wrappers and do not work from a plain forward call.
from utils.models.llava.patch_llava import patch as patch_llava
from utils.models.internvl.patch_internvl import patch as patch_internvl
from utils.models.onevision.patch_onevision import patch as patch_onevision
from utils.models.cambrian.patch_cambrian import patch as patch_cambrian

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_BACKEND_REGISTRY = [
    ("internvl", "internvl2"),
    ("onevision", "llava_onevision1_5"),
    ("cambrian", "cambrian"),
    ("llava", "llava_hf"),
    ("pixtral", "llava_hf"),
]

RANDOM_SEED = 42
ARO_SOURCE_ROOT = str(PROJECT_ROOT / "third_party" / "vision-language-models-are-bows")
ARO_SUBTASKS = ("flickr30k_order", "coco_order", "vg_relation", "vg_attribution")
ARO_SAMPLE_LIMIT = 5000  # per subtask (Flickr30k-Order has 4,885 samples)

# SugarCrepe categories (github.com/RAIVNLab/sugar-crepe; images: COCO val2017) and their labels
# in the column order of the SugarCrepe paper (Table 3).
SUGARCREPE_CATEGORIES = (
    "add_obj",
    "add_att",
    "replace_obj",
    "replace_att",
    "replace_rel",
    "swap_obj",
    "swap_att",
)
SUGARCREPE_LABELS: dict[str, str] = {
    "add_obj": "Add-Obj",
    "add_att": "Add-Att",
    "replace_obj": "Rep-Obj",
    "replace_att": "Rep-Att",
    "replace_rel": "Rep-Rel",
    "swap_obj": "Swap-Obj",
    "swap_att": "Swap-Att",
}
SUGARCREPE_GROUPS: dict[str, tuple[str, ...]] = {
    "Add": ("add_obj", "add_att"),
    "Replace": ("replace_obj", "replace_att", "replace_rel"),
    "Swap": ("swap_obj", "swap_att"),
}

# Two options (ARO VG subtasks, SugarCrepe) or five (ARO Order subtasks).
MCQ_BINARY_PROMPT = (
    "Which caption better describes this image?\n"
    "A) {caption_a}\n"
    "B) {caption_b}\n"
    "Answer with the option's letter from the given choices directly."
)
MCQ_5WAY_PROMPT = (
    "Which caption best describes this image?\n"
    "A) {caption_a}\n"
    "B) {caption_b}\n"
    "C) {caption_c}\n"
    "D) {caption_d}\n"
    "E) {caption_e}\n"
    "Answer with the option's letter from the given choices directly."
)


def infer_backend(model_name: str) -> str:
    name_lower = model_name.lower()
    for pattern, backend in MODEL_BACKEND_REGISTRY:
        if pattern in name_lower:
            return backend
    raise ValueError(
        f"Cannot infer backend for '{model_name}'. "
        f"Known patterns: {[p for p, _ in MODEL_BACKEND_REGISTRY]}"
    )


def _num_language_layers(model, backend: str) -> int:
    if backend == "cambrian":
        return model.config.num_hidden_layers
    return model.language_model.config.num_hidden_layers


def apply_no_image_patch(model, backend: str) -> None:
    """Mask the image tokens at every layer (default "image" mode of the ablation patches)."""
    num_layers = _num_language_layers(model, backend)
    layers = range(num_layers)
    if backend == "llava_onevision1_5":
        patch_onevision(model, layers_to_skip=layers)
    elif backend == "llava_hf":
        patch_llava(model, layers_to_skip=layers)
    elif backend == "internvl2":
        patch_internvl(model, layers_to_skip=layers)
    elif backend == "cambrian":
        patch_cambrian(model, layers_to_skip=layers)


def build_mcq_prompt(captions: list[str]) -> str:
    if len(captions) == 2:
        return MCQ_BINARY_PROMPT.format(caption_a=captions[0], caption_b=captions[1])
    if len(captions) == 5:
        return MCQ_5WAY_PROMPT.format(
            caption_a=captions[0],
            caption_b=captions[1],
            caption_c=captions[2],
            caption_d=captions[3],
            caption_e=captions[4],
        )
    raise ValueError(f"Expected 2 or 5 captions, got {len(captions)}")


def _shuffle_captions(
    captions: list[str],
    correct_idx: int,
    idx: int,
) -> tuple[list[str], int, list[int]]:
    """Shuffle the options with seed RANDOM_SEED + idx.

    Returns (shuffled_captions, new_correct_idx, shuffle_order), where shuffle_order[i] is the
    original index of the caption at position i.
    """
    rng = random.Random(RANDOM_SEED + idx)
    order = list(range(len(captions)))
    rng.shuffle(order)
    shuffled = [captions[i] for i in order]
    new_correct = order.index(correct_idx)
    return shuffled, new_correct, order


# =============================================================================
# CONSTRAINED-LOGITS MCQ
# =============================================================================


def is_internvl_model(model: torch.nn.Module) -> bool:
    return "intern" in model.__class__.__name__.lower()


def _get_answer_token_ids(tokenizer: Any, num_choices: int) -> list[int]:
    """Token ids of the letters A, B, ... (single tokens in every supported tokenizer)."""
    ids = []
    for i in range(num_choices):
        letter = chr(ord("A") + i)
        ids.append(tokenizer.encode(letter, add_special_tokens=False)[0])
    return ids


def _pick_answer_from_logits(
    logits: torch.Tensor,
    answer_token_ids: list[int],
) -> tuple[int, str]:
    """(index, letter) of the answer token with the highest logit; ``logits`` has shape (vocab,)."""
    answer_logits = logits[answer_token_ids]
    predicted = int(answer_logits.argmax().item())
    letter = chr(ord("A") + predicted)
    return predicted, letter


def _get_mcq_logits_llava(
    model: torch.nn.Module,
    processor: Any,
    image: Image.Image,
    question: str,
    answer_token_ids: list[int],
) -> tuple[int, str, str]:
    """LLaVA-1.5, pixtral and OneVision: the processor's chat template."""
    conv: list = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ]
    prompt_text = processor.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=prompt_text, return_tensors="pt", padding=False, images=image)

    model_dtype = next(model.parameters()).dtype
    moved = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            continue
        v = v.to(model.device)
        if v.is_floating_point():
            v = v.to(model_dtype)
        moved[k] = v

    with torch.no_grad():
        outputs = model(**moved)

    last_logits = outputs.logits[0, -1, :]
    predicted_idx, letter = _pick_answer_from_logits(last_logits, answer_token_ids)
    return predicted_idx, letter, prompt_text


def _get_mcq_logits_internvl(
    model: torch.nn.Module,
    tokenizer: Any,
    image: Image.Image,
    question: str,
    answer_token_ids: list[int],
) -> tuple[int, str, str]:
    """InternVL2: model.conv_template, up to 12 tiles, forward of the language model on the
    embeddings with the vision features in place."""
    device = model.device
    pixel_values, num_patches_list = get_pixel_values(image)
    pixel_values = pixel_values.to(device)

    query = apply_internvl_chat_template(
        model=model,
        text=[question],
        num_patches_list=num_patches_list,
    )[0]
    # saved prompt without the image placeholder tokens
    prompt_text = query.replace(IMG_CONTEXT_TOKEN, "").replace("</img>", "").replace("<img>", "<image>")

    input_ids = tokenizer(query, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    img_ctx_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    inputs_embeds = get_combined_embeddings(
        model, input_ids, pixel_values, img_context_token_id=img_ctx_id
    )

    # image span (<img> ... </img>) masked by the --no-image patch (patch_internvl)
    image_limits = []
    for i in range(input_ids.shape[0]):
        indices = torch.where(input_ids[i] == img_ctx_id)[0]
        image_limits.append((indices[0].item() - 1, indices[-1].item() + 1))
    model.language_model.model.image_limits = image_limits

    with torch.no_grad():
        outputs = model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

    last_logits = outputs.logits[0, -1, :]
    predicted_idx, letter = _pick_answer_from_logits(last_logits, answer_token_ids)
    return predicted_idx, letter, prompt_text


def _get_mcq_logits_cambrian(
    model: torch.nn.Module,
    tokenizer: Any,
    image: Image.Image,
    question: str,
    answer_token_ids: list[int],
) -> tuple[int, str, str]:
    """Cambrian: the logits of the first token generated by the (patched) generate."""
    prepared = prepare_cambrian_batch_inputs(
        model=model,
        tokenizer=tokenizer,
        questions=[question],
        images=[image],
    )
    prompt_text = prepared["prompts"][0]

    input_ids = prepared["tokenized"]["input_ids"]
    attention_mask = prepared["tokenized"]["attention_mask"]
    gen_kwargs: dict[str, Any] = {
        "inputs": input_ids,
        "attention_mask": attention_mask,
    }
    if prepared["image_tensors"] is not None:
        gen_kwargs["images"] = prepared["image_tensors"]
        gen_kwargs["image_sizes"] = prepared["image_sizes"]

    # Go through the (patched) generate: it expands the image placeholder, records the
    # image-token span the image-blind patch masks (--no-image) and stashes the vision
    # features for the SVA layers; a plain forward would skip the span bookkeeping.
    # The raw logits of the first generated token are the logits at the last input position.
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.inference_mode():
        generated = model.generate(
            **gen_kwargs,
            max_new_tokens=1,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
            pad_token_id=pad_id,
        )

    last_logits = generated.logits[0][0]
    predicted_idx, letter = _pick_answer_from_logits(last_logits, answer_token_ids)
    return predicted_idx, letter, prompt_text


def get_mcq_logits(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    image: Image.Image,
    question: str,
    answer_token_ids: list[int],
) -> tuple[int, str, str]:
    """Returns (predicted_index, predicted_letter, prompt_text)."""
    if is_cambrian_model(model):
        return _get_mcq_logits_cambrian(
            model, processor_or_tokenizer, image, question, answer_token_ids
        )
    if is_internvl_model(model):
        return _get_mcq_logits_internvl(
            model, processor_or_tokenizer, image, question, answer_token_ids
        )
    return _get_mcq_logits_llava(
        model, processor_or_tokenizer, image, question, answer_token_ids
    )


# =============================================================================
# DATASETS
# =============================================================================


def _ensure_aro_on_sys_path() -> None:
    """Make the vendored ARO dataset code (dataset_zoo) importable."""
    if not os.path.isdir(os.path.join(ARO_SOURCE_ROOT, "dataset_zoo")):
        raise ImportError(f"ARO dataset code not found in {ARO_SOURCE_ROOT}")
    if ARO_SOURCE_ROOT not in sys.path:
        sys.path.insert(0, ARO_SOURCE_ROOT)


def _ensure_nltk_punkt_tab() -> None:
    """The ARO Order text shuffling uses nltk.word_tokenize (punkt_tab)."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def _subsample_indices(n: int, limit: Optional[int]) -> list[int]:
    if limit and limit < n:
        rng = random.Random(RANDOM_SEED)
        return sorted(rng.sample(range(n), limit))
    return list(range(n))


class AROVGDataset(Dataset):
    """ARO VG-Relation or VG-Attribution; the true caption is option 1."""

    def __init__(
        self,
        subtask: str,
        aro_data_root: str,
        limit: Optional[int] = None,
    ):
        if subtask not in ("vg_relation", "vg_attribution"):
            raise ValueError(f"Invalid ARO VG subtask: {subtask}")

        self.subtask = subtask
        _ensure_aro_on_sys_path()
        data_path = os.path.expanduser(aro_data_root)

        from dataset_zoo import VG_Relation, VG_Attribution

        DatasetClass = VG_Relation if subtask == "vg_relation" else VG_Attribution
        self.dataset = DatasetClass(
            image_preprocess=None, download=True, root_dir=data_path
        )
        self.indices = _subsample_indices(len(self.dataset), limit)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[self.indices[idx]]
        image = item["image_options"][0]
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        return {
            "image": image,
            "caption_options": item["caption_options"],
            "correct_index": 1,
        }


class AROOrderDataset(Dataset):
    """ARO COCO-Order or Flickr30k-Order (test split); the original caption is option 0."""

    def __init__(
        self,
        subtask: str,
        aro_data_root: str,
        limit: Optional[int] = None,
    ):
        if subtask not in ("coco_order", "flickr30k_order"):
            raise ValueError(f"Invalid ARO Order subtask: {subtask}")

        self.subtask = subtask
        _ensure_aro_on_sys_path()
        _ensure_nltk_punkt_tab()
        data_path = os.path.expanduser(aro_data_root)

        from dataset_zoo import COCO_Order, Flickr30k_Order

        DatasetClass = COCO_Order if subtask == "coco_order" else Flickr30k_Order
        self.dataset = DatasetClass(
            image_preprocess=None, split="test", download=True, root_dir=data_path
        )
        self.indices = _subsample_indices(len(self.dataset), limit)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[self.indices[idx]]
        image = item["image_options"][0]
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        return {
            "image": image,
            "caption_options": item["caption_options"],
            "correct_index": 0,
        }


def load_aro_dataset(
    subtask: str,
    aro_data_root: str,
    limit: Optional[int] = None,
) -> Dataset:
    if subtask in ("vg_relation", "vg_attribution"):
        return AROVGDataset(subtask, aro_data_root, limit)
    return AROOrderDataset(subtask, aro_data_root, limit)


_ARO_ORDER_IMAGE_DIRS: dict[str, tuple[str, str]] = {
    "coco_order": ("val2014", "COCO val2014"),
    "flickr30k_order": ("flickr30k-images", "Flickr30k"),
}


def validate_aro_image_dirs(subtasks: tuple[str, ...], aro_data_root: str) -> None:
    """Fail before loading the model if the images of the ARO Order subtasks are missing."""
    aro_path = os.path.expanduser(aro_data_root)
    missing = []
    for st in subtasks:
        if st in _ARO_ORDER_IMAGE_DIRS:
            subdir, name = _ARO_ORDER_IMAGE_DIRS[st]
            if not os.path.isdir(os.path.join(aro_path, subdir)):
                missing.append(f"{st}: {name} images not found in {os.path.join(aro_path, subdir)}")
    if missing:
        raise FileNotFoundError(
            "Missing ARO images (download with bash scripts/finetuning/download_aro.sh):\n  "
            + "\n  ".join(missing)
        )


def load_sugarcrepe_data(data_root: str, category: str) -> list[dict]:
    """Samples (filename, caption, negative_caption) of one SugarCrepe category."""
    json_path = os.path.join(data_root, "data", f"{category}.json")
    with open(json_path) as f:
        raw = json.load(f)
    return list(raw.values())


def validate_sugarcrepe_data(data_root: str) -> None:
    """Fail before loading the model if SugarCrepe annotations or images are missing."""
    data_dir = os.path.join(data_root, "data")
    img_dir = os.path.join(data_root, "val2017")
    missing = [
        c
        for c in SUGARCREPE_CATEGORIES
        if not os.path.isfile(os.path.join(data_dir, f"{c}.json"))
    ]
    if missing:
        raise FileNotFoundError(
            f"SugarCrepe annotations missing in {data_dir}: {missing} "
            "(download with bash scripts/finetuning/download_sugarcrepe.sh)"
        )
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(
            f"COCO val2017 images not found in {img_dir} "
            "(download with bash scripts/finetuning/download_sugarcrepe.sh)"
        )


# =============================================================================
# EVALUATION
# =============================================================================


def _answer_tokenizer(processor_or_tokenizer: Any) -> Any:
    if hasattr(processor_or_tokenizer, "tokenizer"):
        return processor_or_tokenizer.tokenizer
    return processor_or_tokenizer


def evaluate_aro_mcq_logits(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    subtask: str,
    aro_data_root: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Accuracy on one ARO subtask; samples whose image file is missing are skipped."""
    dataset = load_aro_dataset(subtask, aro_data_root, limit)
    all_answer_ids = _get_answer_token_ids(_answer_tokenizer(processor_or_tokenizer), 5)

    correct = 0
    total = 0
    per_sample: list = []

    n_samples = len(dataset)
    update_every = max(1, int(0.1 * n_samples))
    progress_bar = tqdm(
        total=n_samples, desc=f"ARO {subtask} (mcq_logits)", unit="sample"
    )
    update_count = 0

    skipped = 0
    for idx in range(n_samples):
        try:
            item = dataset[idx]
        except (FileNotFoundError, OSError):
            skipped += 1
            update_count += 1
            if update_count >= update_every or idx == n_samples - 1:
                progress_bar.update(update_count)
                update_count = 0
            continue

        image = item["image"]
        captions = item["caption_options"]
        answer_token_ids = all_answer_ids[: len(captions)]

        shuffled, correct_idx, shuffle_order = _shuffle_captions(
            captions, item["correct_index"], idx
        )
        prompt = build_mcq_prompt(shuffled)

        try:
            predicted_idx, predicted_token, prompt_text = get_mcq_logits(
                model, processor_or_tokenizer, image, prompt, answer_token_ids
            )
        except Exception as e:
            print(f"Sample {idx} failed: {e}")
            predicted_idx = None
            predicted_token = ""
            prompt_text = prompt

        is_correct = predicted_idx == correct_idx
        if is_correct:
            correct += 1
        total += 1

        per_sample.append(
            {
                "idx": idx,
                "prompt": prompt_text,
                "predicted_token": predicted_token,
                "predicted": predicted_idx,
                "correct": correct_idx,
                "is_correct": is_correct,
                "shuffle_order": shuffle_order,
            }
        )

        update_count += 1
        if update_count >= update_every or idx == n_samples - 1:
            progress_bar.update(update_count)
            update_count = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    progress_bar.close()
    if skipped:
        print(f"Skipped {skipped}/{n_samples} samples (missing images)")

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "num_correct": correct,
        "num_total": total,
        "num_skipped": skipped,
        "per_sample": per_sample,
    }


def evaluate_sugarcrepe_mcq_logits(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    data_root: str,
    category: str,
) -> Dict[str, Any]:
    """Accuracy on one SugarCrepe category (true caption against its negative)."""
    samples = load_sugarcrepe_data(data_root, category)
    img_dir = os.path.join(data_root, "val2017")
    answer_token_ids = _get_answer_token_ids(_answer_tokenizer(processor_or_tokenizer), 2)

    correct = 0
    total = 0
    per_sample: list = []

    n_samples = len(samples)
    update_every = max(1, int(0.1 * n_samples))
    progress_bar = tqdm(
        total=n_samples, desc=f"SugarCrepe {category} (mcq_logits)", unit="sample"
    )
    update_count = 0

    for idx, sample in enumerate(samples):
        image = Image.open(os.path.join(img_dir, sample["filename"])).convert("RGB")

        options, correct_idx, shuffle_order = _shuffle_captions(
            [sample["caption"], sample["negative_caption"]], 0, idx
        )
        prompt = build_mcq_prompt(options)

        try:
            predicted_idx, predicted_token, prompt_text = get_mcq_logits(
                model, processor_or_tokenizer, image, prompt, answer_token_ids
            )
        except Exception as e:
            print(f"Sample {idx} failed: {e}")
            predicted_idx = None
            predicted_token = ""
            prompt_text = prompt

        is_correct = predicted_idx == correct_idx
        if is_correct:
            correct += 1
        total += 1

        per_sample.append(
            {
                "idx": idx,
                "filename": sample["filename"],
                "prompt": prompt_text,
                "predicted_token": predicted_token,
                "predicted": predicted_idx,
                "correct": correct_idx,
                "is_correct": is_correct,
                "shuffle_order": shuffle_order,
            }
        )

        update_count += 1
        if update_count >= update_every or idx == n_samples - 1:
            progress_bar.update(update_count)
            update_count = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    progress_bar.close()

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "num_correct": correct,
        "num_total": total,
        "per_sample": per_sample,
    }


# =============================================================================
# RESULTS
# =============================================================================


def format_elapsed_time(seconds: float) -> str:
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s" if m > 0 else f"{seconds:.1f}s"


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    model_identifier: str,
    benchmark: str,
    elapsed_sec: float,
) -> None:
    """Append the metrics to <benchmark>_results.jsonl and write the per-sample responses."""
    os.makedirs(output_dir, exist_ok=True)

    per_sample = results.pop("per_sample", [])
    summary = {
        "experiment_name": f"single/{model_identifier}",
        "eval_method": "mcq_logits",
        **results,
        "elapsed_time": format_elapsed_time(elapsed_sec),
    }

    jsonl_path = os.path.join(output_dir, f"{benchmark}_results.jsonl")
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"Metrics: {jsonl_path}")

    json_path = os.path.join(
        output_dir, f"{benchmark}_{model_identifier}_mcq_logits_responses.json"
    )
    with open(json_path, "w") as f:
        json.dump(per_sample, f, indent=2, default=str)
    print(f"Samples: {json_path}")


def run_aro_evaluation(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    args: argparse.Namespace,
    model_identifier: str,
) -> None:
    """The four ARO subtasks; results in <output-dir>/aro/."""
    seed_all(42)
    print("ARO")

    all_accs: dict[str, float] = {}

    for subtask in ARO_SUBTASKS:
        print(f"\n{subtask} (sample limit {ARO_SAMPLE_LIMIT})")
        output_dir = os.path.join(args.output_dir, "aro")
        start = time.time()

        try:
            results = evaluate_aro_mcq_logits(
                model,
                processor_or_tokenizer,
                subtask,
                args.aro_data_root,
                limit=ARO_SAMPLE_LIMIT,
            )
            elapsed = time.time() - start
            print(f"{subtask}: {results['accuracy']:.4f} ({format_elapsed_time(elapsed)})")
            save_results(results, output_dir, model_identifier, f"aro_{subtask}", elapsed)
            all_accs[subtask] = results["accuracy"]
        except Exception as e:
            print(f"{subtask} failed: {e}")

    if all_accs:
        mean_acc = sum(all_accs.values()) / len(all_accs)
        print(f"\nARO mean: {mean_acc:.4f}")


def run_sugarcrepe_evaluation(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    args: argparse.Namespace,
    model_identifier: str,
) -> None:
    """The seven SugarCrepe categories; results in <output-dir>/sugarcrepe/: one response file per
    category and one aggregate line (per category, per type, mean) in sugarcrepe_results.jsonl."""
    seed_all(42)
    print("SugarCrepe")

    output_dir = os.path.join(args.output_dir, "sugarcrepe")
    os.makedirs(output_dir, exist_ok=True)

    method_start = time.time()

    category_accs: dict[str, float] = {}
    total_samples = 0

    for category in SUGARCREPE_CATEGORIES:
        start = time.time()
        try:
            results = evaluate_sugarcrepe_mcq_logits(
                model,
                processor_or_tokenizer,
                args.sugarcrepe_data_root,
                category,
            )
            elapsed = time.time() - start
            acc = results["accuracy"]
            category_accs[category] = acc
            total_samples += results["num_total"]

            print(
                f"{SUGARCREPE_LABELS[category]}: {acc * 100:.1f}% "
                f"({results['num_correct']}/{results['num_total']}, "
                f"{format_elapsed_time(elapsed)})"
            )

            per_sample = results.pop("per_sample", [])
            resp_path = os.path.join(
                output_dir,
                f"sugarcrepe_{category}_{model_identifier}_mcq_logits_responses.json",
            )
            with open(resp_path, "w") as f:
                json.dump(per_sample, f, indent=2, default=str)
            print(f"Samples: {resp_path}")

        except Exception as e:
            print(f"{category} failed: {e}")
            traceback.print_exc()

    if not category_accs:
        return

    per_category = {
        SUGARCREPE_LABELS[c]: category_accs[c]
        for c in SUGARCREPE_CATEGORIES
        if c in category_accs
    }
    per_type: dict[str, float] = {}
    for grp, cats in SUGARCREPE_GROUPS.items():
        done = [c for c in cats if c in category_accs]
        if done:
            per_type[grp] = sum(category_accs[c] for c in done) / len(done)
    mean_acc = sum(per_category.values()) / len(per_category)
    total_elapsed = time.time() - method_start

    col_order = [SUGARCREPE_LABELS[c] for c in SUGARCREPE_CATEGORIES if c in category_accs]
    header = "  ".join(f"{lbl:>7}" for lbl in col_order)
    values = "  ".join(f"{per_category[lbl] * 100:>7.1f}" for lbl in col_order)
    grp_header = "  ".join(f"{g:>7}" for g in per_type)
    grp_values = "  ".join(f"{v * 100:>7.1f}" for v in per_type.values())
    print("\nSugarCrepe accuracy (%)")
    print(f"  {header}  |  {grp_header}  |    Mean")
    print(f"  {values}  |  {grp_values}  |  {mean_acc * 100:>6.2f}")

    aggregate = {
        "experiment_name": f"single/{model_identifier}",
        "eval_method": "mcq_logits",
        "per_category": per_category,
        "per_type": per_type,
        "mean": mean_acc,
        "num_total": total_samples,
        "elapsed_time": format_elapsed_time(total_elapsed),
    }
    jsonl_path = os.path.join(output_dir, "sugarcrepe_results.jsonl")
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(aggregate) + "\n")
    print(f"Aggregate: {jsonl_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a VLM on the ARO and SugarCrepe compositionality benchmarks",
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        choices=["aro", "sugarcrepe", "all"],
    )
    parser.add_argument(
        "--aro-data-root",
        type=str,
        required=True,
        help="ARO annotations and images (scripts/finetuning/download_aro.sh)",
    )
    parser.add_argument(
        "--sugarcrepe-data-root",
        type=str,
        required=True,
        help="SugarCrepe data/*.json and val2017/ (scripts/finetuning/download_sugarcrepe.sh)",
    )
    parser.add_argument(
        "--eval-method",
        type=str,
        default="mcq_logits",
        choices=["mcq_logits"],
        help="logits of the option letters at the last prompt position (the only method)",
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="mask the image tokens at every layer (image-blind evaluation)",
    )
    args = parser.parse_args()
    # flash-attention for InternVL2 (eager without flash-attn), sdpa otherwise; the ablation patches
    # of --no-image edit a 4D additive attention mask and need eager attention
    if args.no_image:
        attn_implementation = "eager"
    else:
        attn_implementation = resolve_attn_implementation_for_model(args.model_name_or_path, "auto")

    print(f"Model: {args.model_name_or_path}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Attention: {attn_implementation}")
    print(f"Image: {'masked at every layer' if args.no_image else 'yes'}")

    if args.benchmark in ("aro", "all"):
        validate_aro_image_dirs(ARO_SUBTASKS, args.aro_data_root)
    if args.benchmark in ("sugarcrepe", "all"):
        validate_sugarcrepe_data(args.sugarcrepe_data_root)

    model, processor_or_tokenizer = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=args.model_name_or_path,
        cache_dir=os.environ.get("HF_HUB_CACHE"),
        attn_implementation=attn_implementation,
    )
    model.eval()
    model_identifier = args.model_name_or_path.replace("/", "_")
    print(f"Loaded {model.__class__.__name__}")

    if args.no_image:
        apply_no_image_patch(model, infer_backend(args.model_name_or_path))

    os.makedirs(args.output_dir, exist_ok=True)
    overall_start = time.time()

    if args.benchmark in ("aro", "all"):
        try:
            run_aro_evaluation(model, processor_or_tokenizer, args, model_identifier)
        except Exception as e:
            print(f"ARO evaluation failed: {e}")
            traceback.print_exc()

    if args.benchmark in ("sugarcrepe", "all"):
        try:
            run_sugarcrepe_evaluation(
                model, processor_or_tokenizer, args, model_identifier
            )
        except Exception as e:
            print(f"SugarCrepe evaluation failed: {e}")
            traceback.print_exc()

    print(f"\nDone in {format_elapsed_time(time.time() - overall_start)}; results in {args.output_dir}")


if __name__ == "__main__":
    main()
