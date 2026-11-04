"""Datasets and dataloaders of the representation experiments.

* ``get_dataloader``: VQA prompts (image + question, or caption + question) tokenized with the model's
  chat template, for the similarity measures (scripts/geometry/similarity_multimodal.py) and the logit
  lens (scripts/probing/logit_lens.py).
* ``MultipleChoiceDatasetBenchmark`` (MMBench, MME, POPE, ScienceQA, SEED-Bench) and ``OpenVQADataset``
  (VQAv2, COCO-QA, GQA, CountBench): the samples behind ``get_dataloader``, also used directly by
  scripts/geometry/similarity_llama_vision.py.
* ``ClassificationDataset``: N-option multiple-choice prompts on image-classification datasets, for the
  label overlap (scripts/probing/label_overlap.py).
"""

import base64
import csv
import hashlib
import json
import os
import random
from collections import Counter
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
import numpy as np
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from utils.models.cambrian.cambrian_utils import is_cambrian_model, prepare_cambrian_batch_inputs
from utils.data.constants import MMBENCH_ANSWER_CHOICES, SQA_ANSWER_CHOICES
from utils.models.internvl.internvl_utils import apply_chat_template as apply_internvl_chat_template
from utils.models.internvl.internvl_utils import get_pixel_values
from utils.data.mcq import build_mcq_prompt
from utils.models.loading import ProcessorType
from utils.data.prompts import format_user_turn_text
from utils.data.subsample import stratified_indices

SEED: int = 42

# Substrings of the dataset names handled by MultipleChoiceDatasetBenchmark.
MULTIPLE_CHOICE_BENCHMARKS = ["scienceqa", "mmbench", "mme", "pope", "seed"]

# Answer instructions built into MME's questions, removed in this order.
_MME_INSTRUCTIONS = (
    " Please answer yes or no.",
    "\nAnswer the question using a single word or phrase.",
    "\nAnswer the question with Yes or No.",
    " Answer:",
    " Answer this question briefly[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n",
    "[UNUSED_TOKEN_146]user\n",
)

_CAPTION_INTRO = "Below is a description of an image. Use it to answer the question.\n\n"


class MultipleChoiceDatasetBenchmark(Dataset):
    """Multiple-choice benchmark (MMBench, MME, POPE, ScienceQA, SEED-Bench) as VQA samples.

    With ``texts_qa`` the image is replaced by its caption in the prompt; with ``images_qa`` the sample
    carries the image. ``open_ended`` leaves the options out of the prompt. ``unique_images`` keeps one
    seeded random question per image (``imageHash``).
    """

    def __init__(
        self,
        dataset_path_or_name: str,
        cache_dir: Optional[str],
        split: str,
        texts_qa: bool = False,
        images_qa: bool = False,
        downsample_size: Optional[int] = None,
        seed: int = SEED,
        filter_mmbench_4_choices: bool = False,
        unique_images: bool = False,
        open_ended: bool = False,
    ):
        self.dataset_path_or_name = dataset_path_or_name.lower()
        if not any(ds in self.dataset_path_or_name for ds in MULTIPLE_CHOICE_BENCHMARKS):
            raise ValueError(
                f"Dataset {dataset_path_or_name} is not supported; supported: {', '.join(MULTIPLE_CHOICE_BENCHMARKS)}"
            )
        if texts_qa and images_qa:
            raise ValueError("texts_qa and images_qa cannot be both True.")
        self.texts_qa = texts_qa
        self.images_qa = images_qa
        self.open_ended = open_ended

        config_name = "ScienceQA-IMG" if "scienceqa" in self.dataset_path_or_name else None  # image subset
        dataset = load_dataset(dataset_path_or_name, name=config_name, split=split, cache_dir=cache_dir)

        if unique_images and "imageHash" in dataset.features:
            groups: dict[str, list[int]] = {}
            for idx, image_hash in enumerate(dataset["imageHash"]):
                if image_hash is not None:
                    groups.setdefault(image_hash, []).append(idx)
            rng = random.Random(seed)
            chosen = sorted(rng.choice(indices) for indices in groups.values())  # original order
            before = len(dataset)
            dataset = dataset.select(chosen)
            print(f"Unique-image filter: {before} -> {len(dataset)} samples")

        if filter_mmbench_4_choices and "mmbench" in self.dataset_path_or_name:

            def has_4_choices(example):
                values = [example.get(choice, None) for choice in MMBENCH_ANSWER_CHOICES]
                return sum(v is not None and str(v).lower() != "nan" for v in values) == 4

            before = len(dataset)
            dataset = dataset.filter(has_4_choices)
            print(f"MMBench 4-choice filter: {before} -> {len(dataset)} samples")

        if downsample_size and downsample_size < len(dataset):
            print(f"Downsampling dataset from {len(dataset)} to {downsample_size} samples")
            dataset = dataset.shuffle(seed=seed).select(range(downsample_size))

        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def _choices(self, question_data: dict) -> Optional[list]:
        if "scienceqa" in self.dataset_path_or_name:
            return question_data["choices"]
        if "mmbench" in self.dataset_path_or_name:
            keys = MMBENCH_ANSWER_CHOICES
        else:  # SEED-Bench
            keys = ["choice_a", "choice_b", "choice_c", "choice_d"]
        return [question_data[k] for k in keys if question_data.get(k, None) is not None]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        question_data = self.dataset[idx]
        name = self.dataset_path_or_name
        is_yes_no = "mme" in name or "pope" in name

        answer = question_data["answer"]
        if "scienceqa" in name:
            answer = SQA_ANSWER_CHOICES[answer]
        elif "pope" in name:
            answer = {"yes": "Yes", "no": "No"}.get(str(answer).lower(), answer)

        prompt = ""
        if self.texts_qa:
            caption = question_data.get("synthetic-caption") or question_data.get("caption")
            if not caption:
                raise ValueError("No caption found")
            prompt += f"{_CAPTION_INTRO}{caption}\n\n"

        context_text = question_data.get("hint", None)  # MMBench, ScienceQA
        if context_text is not None and context_text not in ("", "nan"):
            if "scienceqa" in name:
                prompt += "Context: "
            prompt += f"{context_text} "

        question_text = question_data["question"]
        if "mme" in name:
            for instruction in _MME_INSTRUCTIONS:
                question_text = question_text.replace(instruction, "")
        prompt += f"{question_text}"

        if not self.open_ended:
            if "mmbench" in name:
                prompt += "\nThere are several options:\n"
            elif not is_yes_no:
                prompt += "\n"
            if not is_yes_no:
                choices = self._choices(question_data)
                if choices is not None:
                    separator = ": " if "scienceqa" in name else ". "
                    last_choice = choices[-1]
                    for letter_ord, choice_text in zip(range(ord("A"), ord("Z") + 1), choices):
                        prompt += f"{chr(letter_ord)}{separator}{choice_text}"
                        prompt += "\n" if choice_text != last_choice else ""

        sample = {"question": prompt, "answer_letter": answer}
        if self.images_qa:
            image = question_data.get("image", None)
            if isinstance(image, str) and "mmbench" in name:  # MMBench stores base64-encoded images
                image = Image.open(BytesIO(base64.b64decode(image))).convert("RGB")
            if isinstance(image, Image.Image):
                image = image.convert("RGB")
            sample["image"] = image
        return sample


def _resolve_split_path(dataset_path_or_name: str, split_name: str) -> str:
    """Path of a fixed split, data/splits/<dataset_slug>/<split_name>.json in the repository."""
    project_root = Path(__file__).resolve().parents[2]  # repository root
    slug = dataset_path_or_name.split("/")[-1].replace("-", "_").lower()
    path = os.path.join(project_root, "data", "splits", slug, f"{split_name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    return path


class OpenVQADataset(Dataset):
    """Open-ended VQA dataset (VQAv2, COCO-QA, GQA, CountBench) as VQA samples.

    Datasets with several question-answer pairs per image (``questions_and_answers``: VQAv2, COCO-QA)
    use one pair per image, chosen with a seeded shuffle that avoids repeating a pair. With ``texts_qa``
    the image is replaced by its captions: ``caption_source`` 'synthetic' (one generated caption) or
    'human' (the list of COCO captions). ``dataset_split`` selects the rows of a fixed split
    (data/splits) instead of ``downsample_size``; the original row order is kept in every case, so
    that the image and text versions of a dataset contain the same samples in the same order.
    """

    def __init__(
        self,
        dataset_path_or_name: str,
        cache_dir: Optional[str] = None,
        split: str = "train",
        downsample_size: Optional[int] = None,
        seed: int = SEED,
        images_qa: bool = False,
        texts_qa: bool = False,
        remove_yes_no: bool = False,
        caption_source: Optional[str] = None,
        dataset_split: Optional[str] = None,
    ):
        if images_qa and texts_qa:
            raise ValueError("images_qa and texts_qa cannot be both True.")

        self.images_qa = images_qa
        self.texts_qa = texts_qa
        self.caption_source = caption_source
        self.dataset_path_or_name = dataset_path_or_name
        self.rng = np.random.RandomState(seed=seed)

        dataset = load_dataset(dataset_path_or_name, split=split, cache_dir=cache_dir)
        dataset = dataset.add_column("idx", list(range(len(dataset))))
        self.has_qa_list = "questions_and_answers" in dataset.features

        if remove_yes_no:  # COCO-QA, VQAv2: drop the yes/no pairs, then the images left without pairs

            def drop_yes_no(example):
                example["questions_and_answers"] = [
                    qa for qa in example["questions_and_answers"]
                    if qa["answer"].strip().lower() not in ["yes", "no"]
                ]
                return example

            num_proc = max(1, (os.cpu_count() or 2) // 2)
            dataset = dataset.map(drop_yes_no, num_proc=num_proc, desc="Filtering Yes/No answers")
            before = len(dataset)
            dataset = dataset.filter(lambda x: len(x["questions_and_answers"]) > 0, desc="Removing empty samples")
            print(f"Yes/No filter: {before} -> {len(dataset)} samples")

        if dataset_split is not None:
            split_path = _resolve_split_path(dataset_path_or_name, dataset_split)
            with open(split_path) as f:
                split_indices = json.load(f)["indices"]
            before = len(dataset)
            dataset = dataset.select(split_indices)
            print(f"Dataset split '{dataset_split}': {before} -> {len(dataset)} samples ({split_path})")
        elif downsample_size and downsample_size < len(dataset):
            print(f"Downsampling dataset from {len(dataset)} to {downsample_size} samples")
            dataset = dataset.shuffle(seed=seed).select(range(downsample_size))

        if images_qa:
            cols_to_remove = [c for c in ["captions", "caption", "synthetic-caption"] if c in dataset.column_names]
            if cols_to_remove:
                dataset = dataset.remove_columns(cols_to_remove)
        elif texts_qa:
            dataset = dataset.remove_columns("image")
            unused = ["captions"] if caption_source == "synthetic" else ["synthetic-caption"]
            cols_to_remove = [c for c in unused if c in dataset.column_names]
            if cols_to_remove:
                dataset = dataset.remove_columns(cols_to_remove)

        self.dataset = dataset.sort("idx")

        if self.has_qa_list:
            self._precompute_qa_selection()

    @staticmethod
    def _hash_qa(qa_dict: dict) -> str:
        """SHA-256 of the lower-cased, stripped question-answer pair."""
        question = qa_dict["question"].strip().lower()
        answer = qa_dict["answer"].strip().lower()
        return hashlib.sha256(f"{question}|||{answer}".encode("utf-8")).hexdigest()

    def _precompute_qa_selection(self) -> None:
        """For each sample, the first pair of a seeded shuffle not used by an earlier sample."""
        seen_hashes = set()
        self._selected_qa_indices = {}
        for sample_idx in range(len(self.dataset)):
            questions_and_answers = self.dataset[sample_idx]["questions_and_answers"]
            shuffled_indices = list(range(len(questions_and_answers)))
            self.rng.shuffle(shuffled_indices)
            selected_idx = None
            for shuffled_idx in shuffled_indices:
                hash_qa = self._hash_qa(questions_and_answers[shuffled_idx])
                if hash_qa not in seen_hashes:
                    seen_hashes.add(hash_qa)
                    selected_idx = shuffled_idx
                    break
            if selected_idx is None:  # every pair already used: take the first one
                selected_idx = shuffled_indices[0]
            self._selected_qa_indices[sample_idx] = selected_idx

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        question_data = self.dataset[idx]

        if self.has_qa_list:
            qa = question_data["questions_and_answers"][self._selected_qa_indices[idx]]
            q_text, a_text = qa["question"], qa["answer"]
        else:  # GQA, CountBench
            q_text = question_data["question"]
            if "countbench" in self.dataset_path_or_name.lower():
                a_text = str(question_data["number"])
            else:
                a_text = question_data["answer"]

        prompt = ""
        if self.texts_qa:
            if self.caption_source == "synthetic":
                prompt += f"{_CAPTION_INTRO}{question_data['synthetic-caption']}\n\n"
            elif "captions" in question_data:
                captions = question_data["captions"]
                prompt += f"Below are {len(captions)} descriptions of an image. Use them to answer the question.\n\n"
                prompt += "\n".join(f"{i + 1}: {caption}" for i, caption in enumerate(captions)) + "\n\n"
            else:
                raise ValueError("No captions found in the dataset")

        prompt += f"{q_text}"
        if "cocoqa" in self.dataset_path_or_name.lower():
            prompt += "?"

        sample = {"question": prompt, "answer_letter": a_text}
        if self.images_qa:
            sample["image"] = question_data.get("image", None)
        return sample


def _is_llava_family_processor(processor: Any) -> bool:
    """True for LLaVA-1.5 processors and for OneVision's ``Qwen2_5_VLProcessor`` (no "llava" in its name)."""
    proc_name = processor.__class__.__name__.lower()
    tokenizer = getattr(processor, "tokenizer", None)
    tokenizer_name = tokenizer.__class__.__name__.lower() if tokenizer is not None else ""
    llava_markers = ("llava", "qwen2_5_vl", "qwen2vl")
    return any(marker in proc_name or marker in tokenizer_name for marker in llava_markers)


def _build_llava_prompts(
    questions: List[str],
    images: List[Optional[Image.Image]],
    processor: Any,
    guide_text: str = "",
) -> List[str]:
    """Chat-template strings for LLaVA-1.5 / OneVision, one user turn per question.

    No system turn is added: LLaVA-1.5's template has none (the prompt starts with "USER:", as in
    lmms-eval), and LLaVA-OneVision's template adds its own default.
    """
    if not images:
        images = [None] * len(questions)
    conversations = []
    for question, image in zip(questions, images):
        content: List[Dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": format_user_turn_text(question, guide_text)})
        conversations.append([{"role": "user", "content": content}])
    return processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)


def collate_vqa_batch(
    batch: List[dict],
    processor: ProcessorType,
    guide_text: str,
    answer_letters_with_processed_batch: bool,
    model=None,
) -> tuple:
    """Tokenize a batch of VQA samples with the model's chat template.

    Returns (inputs,) for LLaVA-1.5 / OneVision; (tokenized, pixel_values, prompts) or, without images,
    (tokenized, prompts) for InternVL2; (tokenized, image_tensors, image_sizes, prompts) or
    (tokenized, prompts) for Cambrian. With ``answer_letters_with_processed_batch`` the list of
    answers is appended.
    """
    questions = [item["question"] for item in batch]
    answer_letters = [item["answer_letter"] for item in batch]
    images = [item.get("image", None) for item in batch]
    has_images = any(image is not None for image in images)
    padding = len(questions) > 1

    if _is_llava_family_processor(processor):
        processor_kwargs = {
            "text": _build_llava_prompts(questions=questions, images=images, processor=processor, guide_text=guide_text),
            "return_tensors": "pt",
            "padding": padding,
        }
        if has_images:
            processor_kwargs["images"] = images
        outputs = (processor(**processor_kwargs),)
    elif model is not None and "intern" in model.__class__.__name__.lower():
        if has_images:
            pixel_values, num_patches_list = get_pixel_values(images)
            queries = apply_internvl_chat_template(
                model=model, text=questions, num_patches_list=num_patches_list, guide_text=guide_text
            )
        else:
            pixel_values = None
            queries = apply_internvl_chat_template(model=model, text=questions, guide_text=guide_text)
        tokenized = processor(text=queries, return_tensors="pt", padding=padding)
        outputs = (tokenized, pixel_values, queries) if pixel_values is not None else (tokenized, queries)
    elif model is not None and is_cambrian_model(model=model):
        prepared = prepare_cambrian_batch_inputs(
            model=model,
            tokenizer=processor,
            questions=questions,
            images=images if has_images else None,
            guide_text=guide_text,
        )
        if prepared["image_tensors"] is not None:
            outputs = (prepared["tokenized"], prepared["image_tensors"], prepared["image_sizes"], prepared["prompts"])
        else:
            outputs = (prepared["tokenized"], prepared["prompts"])
    else:
        raise ValueError(f"Unsupported processor {processor.__class__.__name__} for the VQA dataloader")

    if answer_letters_with_processed_batch:
        return outputs + (answer_letters,)
    return outputs


def get_dataloader(
    dataset_path_or_name: str,
    split: str,
    processor: ProcessorType,
    guide_text: str,
    cache_dir: Optional[str] = None,
    texts_qa: bool = False,
    images_qa: bool = False,
    batch_size: int = 1,
    num_workers: int = 4,
    downsample_size: Optional[int] = None,
    seed: int = SEED,
    answer_letters_with_processed_batch: bool = False,
    caption_source: Optional[str] = None,
    remove_yes_no: bool = False,
    filter_mmbench_4_choices: bool = False,
    model=None,
    unique_images: bool = False,
    open_ended: bool = False,
    dataset_split: Optional[str] = None,
) -> DataLoader:
    """DataLoader of tokenized VQA prompts (see ``collate_vqa_batch`` for the batch format).

    Multiple-choice benchmarks (``MULTIPLE_CHOICE_BENCHMARKS``) are read by
    ``MultipleChoiceDatasetBenchmark``, open-ended VQA datasets by ``OpenVQADataset``; the arguments
    are those of the two classes. ``model`` is needed for InternVL2 and Cambrian.
    """
    name = dataset_path_or_name.lower()
    if any(benchmark in name for benchmark in MULTIPLE_CHOICE_BENCHMARKS):
        if cache_dir is None:
            raise ValueError("cache_dir is required for benchmark datasets")
        dataset = MultipleChoiceDatasetBenchmark(
            dataset_path_or_name=dataset_path_or_name,
            cache_dir=cache_dir,
            split=split,
            texts_qa=texts_qa,
            images_qa=images_qa,
            downsample_size=downsample_size,
            seed=seed,
            filter_mmbench_4_choices=filter_mmbench_4_choices,
            unique_images=unique_images,
            open_ended=open_ended,
        )
    elif any(ds in name for ds in ("vqav2", "cocoqa", "gqa", "countbench")):
        dataset = OpenVQADataset(
            dataset_path_or_name=dataset_path_or_name,
            cache_dir=cache_dir,
            split=split,
            downsample_size=downsample_size,
            seed=seed,
            images_qa=images_qa,
            texts_qa=texts_qa,
            remove_yes_no=remove_yes_no,
            caption_source=caption_source,
            dataset_split=dataset_split,
        )
    else:
        raise ValueError(f"Dataset {dataset_path_or_name} is not supported.")

    collate_fn = partial(
        collate_vqa_batch,
        processor=processor,
        guide_text=guide_text,
        answer_letters_with_processed_batch=answer_letters_with_processed_batch,
        model=model,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=not (model is not None and is_cambrian_model(model=model)),
    )


# ---------------------------------------------------------------------------
# Image classification as multiple choice (label overlap)
# ---------------------------------------------------------------------------

# mcq_nocat_filter: mini-ImageNet, n_choices random classes. generic_mcq_filter: the other datasets,
# n_choices random classes; generic_mcq_top_filter: the n_choices most populated classes, balanced.
MCQ_FILTER_MODES = ("mcq_nocat_filter", "generic_mcq_filter", "generic_mcq_top_filter")

# Question of each generic classification dataset (keys are substrings of the dataset name).
_GENERIC_MCQ_QUESTIONS: dict[str, str] = {
    "food101":           "What food is shown in this image?",
    "sun397":            "What scene is shown in this image?",
    "caltech101":        "What object is shown in this image?",
    "oxford_flowers102": "What flower is shown in this image?",
    "places365":         "What place or scene is shown in this image?",
    "dtd":               "What texture is shown in this image?",
}

_MCQ_CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "mini-imagenet" / "mini-imagenet_mcq_categorization.csv"
)


def _detect_label_col(column_names: list[str]) -> str:
    for name in ("label", "labels"):
        if name in column_names:
            return name
    raise ValueError(f"No label column found in dataset. Columns: {column_names}")


def _load_mcq_catalog(path: Path) -> dict[str, str]:
    """wnid -> class name, from the mini-ImageNet class catalog (CSV with columns wnid, class_name)."""
    with path.open(newline="", encoding="utf-8") as f:
        return {row["wnid"]: row["class_name"] for row in csv.DictReader(f)}


def _load_places365(dataset_name: str, split: str) -> datasets.Dataset:
    """Load torch-uncertainty/Places365 (a bare zip of <class>/<image>.jpg, 36,500 images).

    `datasets` >= 4.1 mislabels this repository when it is loaded by name; extracting the zip and
    loading the files as an image folder gives the 365-class ClassLabel of `datasets` 4.0.
    """
    import zipfile
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(dataset_name, "places365.zip", repo_type="dataset")
    cache_root = Path(os.environ.get("HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets"))
    extracted = cache_root / "places365_imagefolder"
    if not (extracted / ".done").exists():
        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extracted)
        (extracted / ".done").touch()
    # an explicit glob: with data_dir, recent versions take the class folders whose name contains
    # "train" (train_interior, train_station-platform) as the train split
    return load_dataset("imagefolder", data_files={split: f"{extracted}/**/*.jpg"}, split=split)


class ClassificationDataset(Dataset):
    """Image-classification dataset restricted to ``n_choices`` classes, as N-option MCQ prompts.

    Each sample asks "What <x> is shown in this image?" with the selected classes as options, in an
    order seeded by the correct class (by its wnid for mini-ImageNet); the position of the correct
    option cycles through A, B, ... within each class. ``labels`` are the class names, and
    ``option_letters`` the letters of the correct options; ``sample_ids`` and ``prompt_signature``
    identify the samples and the prompts for the representation cache.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        prompt_mode: str,
        seed: int = SEED,
        n_choices: int = 10,
    ) -> None:
        self.dataset_name_lower = dataset_name.lower()
        self.n_choices = n_choices
        self.is_mini_imagenet = "mini-imagenet" in self.dataset_name_lower
        if self.is_mini_imagenet:
            modes = ("mcq_nocat_filter",)
        elif any(ds in self.dataset_name_lower for ds in _GENERIC_MCQ_QUESTIONS):
            modes = ("generic_mcq_filter", "generic_mcq_top_filter")
        else:
            raise ValueError(
                f"Unsupported dataset for the classification pipeline: {dataset_name!r}. Supported: "
                f"mini-imagenet, {', '.join(_GENERIC_MCQ_QUESTIONS)}."
            )
        if prompt_mode not in modes:
            raise ValueError(f"prompt_mode {prompt_mode!r} does not apply to {dataset_name!r}; use {' or '.join(modes)}")

        if "places365" in self.dataset_name_lower:
            dataset = _load_places365(dataset_name, split)
        else:
            dataset = load_dataset(dataset_name, split=split)

        if self.is_mini_imagenet:
            label_feature = dataset.features["label"]
            self._class_names: list[str] | None = label_feature.names if hasattr(label_feature, "names") else None
            self._wnid_to_class = _load_mcq_catalog(_MCQ_CATALOG_PATH)
            self._question = "What is shown in this image?"
            all_classes = sorted(set(self._wnid_to_class.values()))
            self._selected = set(random.Random(seed).sample(all_classes, n_choices))
            dataset = dataset.filter(lambda x: self._extract_label(x) in self._selected)
        else:
            self._label_col = _detect_label_col(dataset.column_names)
            label_feature = dataset.features[self._label_col]
            self._class_names_list: list[str] = (
                label_feature.names if hasattr(label_feature, "names")
                else [str(i) for i in range(label_feature.num_classes)]
            )
            self._question = next(q for k, q in _GENERIC_MCQ_QUESTIONS.items() if k in self.dataset_name_lower)
            dataset = self._select_generic_classes(dataset, dataset_name, prompt_mode, seed)
        self.dataset = dataset

        # The loops below read labels only: decoding the images (ZIP-backed on CephFS) would be slow.
        has_image = "image" in self.dataset.column_names
        if has_image:
            self.dataset = self.dataset.cast_column("image", HFImage(decode=False))

        self.labels = [self._extract_label(instance) for instance in self.dataset]
        self._within_class_idx = self._build_within_class_indices(self.labels)
        self.option_letters = [chr(ord("A") + i % n_choices) for i in self._within_class_idx]
        self.sample_ids = [
            self._extract_sample_id(index=idx, instance=instance) for idx, instance in enumerate(self.dataset)
        ]
        prompts = [self._build_prompt(instance, idx) for idx, instance in enumerate(self.dataset)]
        self.prompt_signature = hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest()

        if has_image:
            self.dataset = self.dataset.cast_column("image", HFImage(decode=True))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, str | Image.Image]:
        instance = self.dataset[idx]
        image = instance["image"]
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        return {"question": self._build_prompt(instance, idx), "image": image, "label": self.labels[idx]}

    @staticmethod
    def _build_within_class_indices(labels: list[str]) -> list[int]:
        class_seen: dict[str, int] = {}
        within_class_idx: list[int] = []
        for label in labels:
            n = class_seen.get(label, 0)
            within_class_idx.append(n)
            class_seen[label] = n + 1
        return within_class_idx

    def _labels_no_decode(self, dataset: datasets.Dataset) -> list[str]:
        if "image" in dataset.column_names:
            dataset = dataset.cast_column("image", HFImage(decode=False))
        return [self._extract_label(instance) for instance in dataset]

    @staticmethod
    def _filter_no_decode(dataset: datasets.Dataset, fn: Any, *, num_proc: int = 8) -> datasets.Dataset:
        """Filter without decoding the image column (label-only filters)."""
        has_image = "image" in dataset.column_names
        if has_image:
            dataset = dataset.cast_column("image", HFImage(decode=False))
        result = dataset.filter(fn, num_proc=num_proc)
        if has_image:
            result = result.cast_column("image", HFImage(decode=True))
        return result

    def _select_generic_classes(
        self, dataset: datasets.Dataset, dataset_name: str, prompt_mode: str, seed: int
    ) -> datasets.Dataset:
        """Keep ``n_choices`` classes: seeded random ones (generic_mcq_filter), or the most populated
        ones, subsampled (seeded, stratified) to the size of the smallest of them (generic_mcq_top_filter)."""
        n_choices = self.n_choices
        if prompt_mode == "generic_mcq_filter":
            all_classes = sorted(self._class_names_list)
            if len(all_classes) < n_choices:
                raise ValueError(
                    f"generic_mcq_filter requires at least n_choices={n_choices} classes "
                    f"in {dataset_name!r}, but only {len(all_classes)} found."
                )
            self._selected = set(random.Random(seed).sample(all_classes, n_choices))
            return self._filter_no_decode(dataset, lambda x: self._extract_label(x) in self._selected)

        counts = Counter(self._labels_no_decode(dataset))
        top_n = sorted(counts, key=lambda class_name: (-counts[class_name], class_name))[:n_choices]
        if len(top_n) < n_choices:
            raise ValueError(
                f"generic_mcq_top_filter requires at least n_choices={n_choices} classes "
                f"in {dataset_name!r}, but only {len(top_n)} found."
            )
        min_count = min(counts[class_name] for class_name in top_n)
        self._selected = set(top_n)
        dataset = self._filter_no_decode(dataset, lambda x: self._extract_label(x) in self._selected)
        keep = stratified_indices(self._labels_no_decode(dataset), n_choices * min_count, seed=seed)
        return dataset.select(keep)

    def _wnid(self, instance: dict) -> str:
        label = instance["label"]
        return self._class_names[label] if self._class_names is not None else str(label)

    def _extract_label(self, instance: dict) -> str:
        if self.is_mini_imagenet:
            return self._wnid_to_class[self._wnid(instance)]
        return self._class_names_list[instance[self._label_col]]

    def _extract_sample_id(self, index: int, instance: dict) -> str:
        if "id" in instance:
            return str(instance["id"])
        if "pid" in instance:
            return str(instance["pid"])
        question = str(instance.get("question", ""))
        label = self._extract_label(instance)
        return f"{self.dataset_name_lower}:{index}:{label}:{question}"

    def _build_prompt(self, instance: dict, idx: int) -> str:
        correct = self._extract_label(instance)
        return build_mcq_prompt(
            question=self._question,
            correct=correct,
            pool=[c for c in sorted(self._selected) if c != correct],
            n_choices=self.n_choices,
            correct_pos=self._within_class_idx[idx] % self.n_choices,
            shuffle_seed=self._wnid(instance) if self.is_mini_imagenet else correct,
        )
