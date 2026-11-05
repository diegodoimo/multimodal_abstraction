"""Run one lmms-eval benchmark for one model (Fig. 5 table and sliding-window plot, Fig. 6 and the radar appendix).

The lmms-eval wrapper is chosen from the model name. --no-image evaluates the same prompts with the
image tokens masked at every layer (image-reliance baseline, utils/models/<family>/patch_<family>_eval.py).
Writes <output-dir>/<basename of output-dir>_results.jsonl (metrics; the launcher names the
directory after the task) and <output-dir>/<model>_responses.json (per-sample outputs).
Launcher: slurm/benchmarks.sbatch. MMBench needs the OPENAI_* judge variables (README).
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root
import argparse
import json
import os
import re
import sys
import time

from lmms_eval.evaluator import simple_evaluate

from utils.models.llava.patch_llava_eval import patch as patch_llava
from utils.models.internvl.patch_internvl_eval import patch as patch_internvl
from utils.models.onevision.patch_onevision_eval import patch as patch_onevision
from utils.models.cambrian.patch_cambrian_eval import patch as patch_cambrian
from utils.evaluation.llm_judge import ensure_llm_judge_env, install_parallel_mmbench_judge
from utils.models.cambrian.patch_cambrian import install_compat as install_cambrian_compat
from lmms_eval.models.chat.llava_hf import LlavaHf
from utils.models.internvl.internvl2 import InternVL2Custom
from utils.models.onevision.llava_onevision1_5 import LlavaOneVision1_5Custom
from utils.models.cambrian.cambrian import CambrianCustom

RANDOM_SEED = NUMPY_RANDOM_SEED = TORCH_RANDOM_SEED = FEWSHOT_RANDOM_SEED = 42

# Samples per benchmark (the paper's cap). NaturalBench always runs in full: its aggregator needs
# all 1,900 x 4 question/image combinations.
SAMPLE_LIMIT = 5000

LLM_JUDGE_TASKS = {"mmbench_en_dev", "mmbench_cn_dev"}

# Tasks restricted to the samples that reference at most one image.
SINGLE_IMAGE_FILTER_TASKS = {"mmmu_val", "mmmu_pro_standard"}

# Model name -> backend (first match wins: more specific patterns first).
MODEL_BACKEND_REGISTRY = [
    ("internvl", "internvl2"),
    ("onevision", "llava_onevision1_5"),
    ("cambrian", "cambrian"),
    ("llava", "llava_hf"),
    ("pixtral", "llava_hf"),
]


def configure_lmms_datasets_cache() -> None:
    """Create the lmms-eval datasets cache (LMMS_EVAL_DATASETS_CACHE, set in slurm/env.sh)."""
    cache = os.environ.get("LMMS_EVAL_DATASETS_CACHE", "").strip()
    if not cache:
        raise RuntimeError(
            "LMMS_EVAL_DATASETS_CACHE is not set; set it to a writable directory (slurm/env.sh)."
        )
    cache = os.path.expanduser(os.path.expandvars(cache))
    os.makedirs(cache, exist_ok=True)
    print(f"LMMS_EVAL_DATASETS_CACHE: {cache}")


def _count_image_tokens(doc) -> int:
    prompt = doc.get("question", "") or ""
    opts = doc.get("options")
    if opts:
        prompt = f"{prompt} {opts}"
    return len(set(re.findall(r"<image \d+>", prompt)))


def install_single_image_filter() -> None:
    """Drop the samples of SINGLE_IMAGE_FILTER_TASKS that reference more than one image."""
    from lmms_eval.api.task import ConfigurableTask

    if getattr(ConfigurableTask, "_single_image_filter_installed", False):
        return

    original_download = ConfigurableTask.download

    def patched_download(self, dataset_kwargs=None):
        original_download(self, dataset_kwargs)
        task_name = getattr(self.config, "task", None)
        if task_name in SINGLE_IMAGE_FILTER_TASKS:
            split = self.config.test_split or self.config.validation_split
            before = len(self.dataset[split])
            keep_fn = lambda d: _count_image_tokens(d) <= 1
            self.dataset[split] = self.dataset[split].filter(keep_fn)
            if hasattr(self, "dataset_no_image") and split in self.dataset_no_image:
                self.dataset_no_image[split] = self.dataset_no_image[split].filter(keep_fn)
            after = len(self.dataset[split])
            print(f"{task_name}: kept {after} of {before} samples (single image)")

    ConfigurableTask.download = patched_download
    ConfigurableTask._single_image_filter_installed = True


def infer_backend(model_name: str) -> str:
    """Infer the lmms-eval backend from the model name."""
    name_lower = model_name.lower()
    for pattern, backend in MODEL_BACKEND_REGISTRY:
        if pattern in name_lower:
            return backend
    raise ValueError(
        f"Cannot infer backend for '{model_name}'. "
        f"Known patterns: {[p for p, _ in MODEL_BACKEND_REGISTRY]}"
    )


def get_num_layers(model_instance, backend: str) -> int:
    """Number of decoder layers of the language model."""
    if backend == "llava_hf":
        return model_instance.model.language_model.config.num_hidden_layers
    elif backend == "internvl2":
        return model_instance._model.language_model.config.num_hidden_layers
    elif backend == "cambrian":
        return model_instance.model.config.num_hidden_layers
    elif backend == "llava_onevision1_5":
        return model_instance.model.language_model.config.num_hidden_layers
    raise ValueError(f"Unknown backend: {backend}")


def validate_task_requirements(task_name: str) -> None:
    """Check the LLM-judge credentials of the judged tasks (MMBench) and install the parallel judge."""
    if task_name in LLM_JUDGE_TASKS:
        ensure_llm_judge_env()  # OPENAI_API_KEY / OPENAI_API_URL / MODEL_VERSION from the environment
        install_parallel_mmbench_judge()  # concurrent judge calls (same scores, much faster)
        if not os.environ.get("OPENAI_API_KEY"):
            print(f"Task '{task_name}' needs an LLM judge: OPENAI_API_KEY is not set.")
            sys.exit(1)

        api_url = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1")
        model_version = os.environ.get("MODEL_VERSION", "gpt-4o-2024-11-20")
        print(f"LLM judge: {model_version} at {api_url}")


def format_elapsed_time(seconds: float) -> str:
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s" if m > 0 else f"{seconds:.1f}s"


def save_metrics(
    model_identifier: str,
    output_dir: str,
    raw_results: dict,
    elapsed_time_sec: float,
) -> None:
    """Append the metrics of one task to <output_dir>/<basename(output_dir)>_results.jsonl."""
    filename = f"{os.path.basename(output_dir)}_results"
    results = {"model": model_identifier, **raw_results}
    results["elapsed_time"] = format_elapsed_time(elapsed_time_sec)

    with open(os.path.join(output_dir, f"{filename}.jsonl"), "a") as f:
        f.write(json.dumps(results) + "\n")


def save_responses(
    model_identifier: str,
    output_dir: str,
    samples: dict,
    task_name: str,
) -> None:
    """Save the per-sample model responses to <output_dir>/<model_identifier>_responses.json."""
    task_keys = [k for k in samples if k.startswith(task_name)]
    output = (
        samples[task_keys[0]]
        if len(task_keys) == 1
        else {k: samples[k] for k in task_keys}
    )

    with open(os.path.join(output_dir, f"{model_identifier}_responses.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)


def run_evaluation(args: argparse.Namespace) -> None:
    """Evaluate one model on one task and save metrics and responses."""
    model_name = args.model_name_or_path
    backend = args.backend
    model_identifier = model_name.replace("/", "_")
    cache_dir = os.environ.get("HF_HUB_CACHE")

    validate_task_requirements(args.task_name)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Model: {model_name} (backend: {backend})")
    start_time = time.time()

    limit = args.limit if args.limit is not None else SAMPLE_LIMIT
    if args.task_name == "naturalbench":
        limit = None
    if limit:
        print(f"Sample limit: {limit}")

    if backend == "llava_hf":
        model_instance = LlavaHf(
            pretrained=model_name,
            device_map="auto",
            batch_size=args.batch_size,
        )
    elif backend == "internvl2":
        model_instance = InternVL2Custom(
            pretrained=model_name,
            device="cuda:0",
            batch_size=str(args.batch_size),
            cache_dir=cache_dir,
        )
    elif backend == "llava_onevision1_5":
        model_instance = LlavaOneVision1_5Custom(
            pretrained=model_name,
            device="cuda:0",
            device_map="auto",
            batch_size=args.batch_size,
            cache_dir=cache_dir,
        )
    elif backend == "cambrian":
        model_instance = CambrianCustom(
            pretrained=model_name,
            device="cuda:0",
            device_map="auto",
            batch_size=args.batch_size,
            cache_dir=cache_dir,
        )
        # the vendored Cambrian forward predates transformers 4.45; same shim as the interventions
        install_cambrian_compat(model_instance.model)

    if args.no_image:
        num_layers = get_num_layers(model_instance, backend)
        if backend == "llava_onevision1_5":
            patch_onevision(model_instance.model, layers_to_skip=range(num_layers))
        elif backend == "llava_hf":
            patch_llava(model_instance.model, layers_to_skip=range(num_layers))
        elif backend == "internvl2":
            patch_internvl(model_instance._model, layers_to_skip=range(num_layers))
        elif backend == "cambrian":
            patch_cambrian(model_instance._model, layers_to_skip=range(num_layers))

    results = simple_evaluate(
        model=model_instance,
        tasks=[args.task_name],
        batch_size=args.batch_size,
        log_samples=True,
        limit=limit,
        cli_args=argparse.Namespace(output_path=args.output_dir, process_with_media=True),
        random_seed=RANDOM_SEED,
        numpy_random_seed=NUMPY_RANDOM_SEED,
        torch_random_seed=TORCH_RANDOM_SEED,
        fewshot_random_seed=FEWSHOT_RANDOM_SEED,
    )
    elapsed_sec = time.time() - start_time
    print(f"{model_name}: done in {format_elapsed_time(elapsed_sec)}")

    if results and "results" in results:
        for res_metrics in results["results"].values():
            if not res_metrics:
                continue
            if len(res_metrics) == 1 and " " in res_metrics:
                continue
            metric_keys = [
                k
                for k in res_metrics
                if k != "alias" and k != " " and not k.endswith("_stderr,none")
            ]
            if not metric_keys:
                continue

            save_metrics(
                output_dir=args.output_dir,
                model_identifier=model_identifier,
                raw_results=res_metrics,
                elapsed_time_sec=elapsed_sec,
            )
    else:
        print(f"No results for {args.task_name}")

    if results and "samples" in results:
        save_responses(
            model_identifier=model_identifier,
            output_dir=args.output_dir,
            samples=results["samples"],
            task_name=args.task_name,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a vision-language model on one lmms-eval task"
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--task-name", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None, help=f"samples per task (default {SAMPLE_LIMIT})")
    parser.add_argument("--no-image", action="store_true", help="mask the image tokens at every layer")

    args = parser.parse_args()
    configure_lmms_datasets_cache()
    install_single_image_filter()
    args.backend = infer_backend(args.model_name_or_path)

    run_evaluation(args)
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
