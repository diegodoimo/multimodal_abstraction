"""Layer skipping on text-only benchmarks with lm-evaluation-harness (appendix "Interventions: all
datasets (text benchmarks)"; launched by slurm/interventions_text.sbatch), and the text benchmarks of the
full model (``--full_model``; appendix "Text benchmarks after localized fine-tuning", slurm/text_benchmarks.sbatch).

For every window of a sweep, the decoder layers in the window are replaced by identity modules
(``ablation_mode="layer"`` of utils/models/<family>/patch_<family>.py) and the model is evaluated with its chat template
and the system turn of that template. Sweeps, in steps of ``--interval``:
``--evaluate_full_progressive_ablation`` (layers [0, w), w = 0, ..., L) and ``--evaluate_full_sliding_window``
(windows of ``--window_size`` layers starting at 0, interval, ...); ``--full_model`` evaluates only the
unmodified model (the w = 0 point of a progressive sweep).

Output: ``<output_dir>/<model>[_sw<width>]/<task>/<task>_L<start>_W<width>_<n>shot.json``.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root
import argparse
import json
import os
import time
import warnings
from collections import defaultdict
from typing import List, Optional

import torch
from lm_eval import evaluator

from utils.models.cambrian.cambrian_lm_harness import CambrianHFLM
from utils.models.cambrian.patch_cambrian import (
    configure_layer_skip as configure_cambrian_layer_skip,
)
from utils.models.cambrian.patch_cambrian import (
    install_compat as install_cambrian_compat,
)
from utils.models.internvl.internvl_lm_harness import InternVLHFLM
from utils.models.internvl.patch_internvl import patch as patch_internvl
from utils.models.llava.llava_lm_harness import LlavaHFLM
from utils.models.llava.patch_llava import patch as patch_llava
from utils.models.onevision.onevision_lm_harness import OnevisionHFLM
from utils.models.onevision.patch_onevision import patch as patch_onevision

warnings.filterwarnings("ignore")

# num_fewshot per task (other tasks: the task's default for the evaluation, 0 in the file name)
TASK_NUM_FEWSHOT = {"mmlu": 0}

BATCH_SIZE = 8
MAX_GEN_TOKS = 128  # generate_until tasks only


def evaluate_with_skipped_layers(
    model_path: str,
    tasks: List[str],
    output_dir: str,
    window_size: Optional[int],
    limit: Optional[int],
    evaluate_full_sliding_window: bool,
    evaluate_full_progressive_ablation: bool,
    interval: int,
    full_model: bool = False,
):
    """Evaluate ``tasks`` for every window of the selected sweep (module docstring) and save the results."""
    wrapper_kwargs = dict(
        pretrained=model_path, device="cuda", batch_size=BATCH_SIZE, max_gen_toks=MAX_GEN_TOKS
    )
    if "internvl" in model_path.lower():
        lm = InternVLHFLM(**wrapper_kwargs)
        num_layers = lm._model.language_model.model.config.num_hidden_layers

        def configure_skip(layers):
            patch_internvl(lm._model, layers_to_skip=layers, ablation_mode="layer")
    elif "onevision" in model_path.lower():
        lm = OnevisionHFLM(**wrapper_kwargs)
        num_layers = lm._model.config.text_config.num_hidden_layers

        def configure_skip(layers):
            patch_onevision(lm._model, layers_to_skip=layers, ablation_mode="layer")
    elif "cambrian" in model_path.lower():
        lm = CambrianHFLM(**wrapper_kwargs)
        # the vendored Cambrian forward needs the compatibility patch, also for the full model
        install_cambrian_compat(lm._model)
        num_layers = lm._model.config.num_hidden_layers

        def configure_skip(layers):
            configure_cambrian_layer_skip(
                model=lm._model, layers_to_skip=layers, ablation_mode="layer"
            )
    else:  # LLaVA-1.5
        lm = LlavaHFLM(**wrapper_kwargs)
        num_layers = lm._model.config.text_config.num_hidden_layers

        def configure_skip(layers):
            patch_llava(lm._model, layers_to_skip=layers, ablation_mode="layer")

    print(f"{model_path}: {num_layers} decoder layers")

    if full_model:
        windows = [[]]
    elif evaluate_full_sliding_window:
        if window_size is None or window_size <= 0:
            raise ValueError("--evaluate_full_sliding_window needs --window_size > 0")
        windows = [
            list(range(s, s + window_size))
            for s in range(0, num_layers - window_size + 2, interval)
        ]
    else:  # evaluate_full_progressive_ablation; w = 0 is the full model
        windows = [list(range(0, w)) for w in range(0, num_layers + 1, interval)]

    # one simple_evaluate call per num_fewshot setting (None: the task's default)
    tasks_by_fewshot = defaultdict(list)
    for task in tasks:
        tasks_by_fewshot[TASK_NUM_FEWSHOT.get(task)].append(task)

    # the system turn of each model's own template (none for LLaVA-1.5)
    system_prompt = lm._system_prompt or None

    model_name = model_path.split("/")[-1]
    if evaluate_full_sliding_window:
        model_name += f"_sw{window_size}"

    for layers_to_skip in windows:
        configure_skip(layers_to_skip)

        if not layers_to_skip:
            cur_start, cur_window = 0, 0
            print("Window: none (full model)")
        else:
            cur_start = layers_to_skip[0]
            cur_window = len(layers_to_skip)
            print(f"Window: layers {layers_to_skip[0]}-{layers_to_skip[-1]} skipped")

        all_task_results = {}
        for num_fewshot, task_group in tasks_by_fewshot.items():
            print(f"Tasks {task_group}, num_fewshot={num_fewshot}")
            group_results = evaluator.simple_evaluate(
                model=lm,
                tasks=task_group,
                batch_size=BATCH_SIZE,
                device="cuda",
                num_fewshot=num_fewshot,
                limit=limit,
                apply_chat_template=True,
                system_instruction=system_prompt,
            )
            all_task_results.update(group_results.get("results", {}))

        for task in tasks:
            task_results = {
                k: v
                for k, v in all_task_results.items()
                if k == task or k.startswith(f"{task}_")
            }
            if not task_results:
                continue
            num_fewshot = TASK_NUM_FEWSHOT.get(task, 0)
            task_output_dir = os.path.join(output_dir, model_name, task)
            os.makedirs(task_output_dir, exist_ok=True)
            output_path = os.path.join(
                task_output_dir, f"{task}_L{cur_start:02d}_W{cur_window}_{num_fewshot}shot.json"
            )
            with open(output_path, "w") as f:
                serializable = {
                    "config": {
                        "model_path": model_path,
                        "start_layer": cur_start,
                        "window_size": cur_window,
                        "task": task,
                        "num_fewshot": num_fewshot,
                    },
                    "results": task_results,
                }
                json.dump(serializable, f, indent=2, default=str)
            print(f"Saved {output_path}")

        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer skipping on text-only benchmarks (lm-evaluation-harness)."
    )
    parser.add_argument("--model_path", type=str, default="lbasile/llava_7b", help="Hub id or local path")
    parser.add_argument("--tasks", type=str, nargs="+", default=["mmlu"], help="lm-eval tasks")
    parser.add_argument(
        "--output_dir", type=str, default="results/interventions/text", help="output directory"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="samples per task (default: all)"
    )
    sweep = parser.add_mutually_exclusive_group(required=True)
    sweep.add_argument(
        "--evaluate_full_progressive_ablation",
        action="store_true",
        help="skip layers [0, w) for w = 0, interval, ..., num_layers",
    )
    sweep.add_argument(
        "--evaluate_full_sliding_window",
        action="store_true",
        help="skip windows of --window_size layers starting at 0, interval, ...",
    )
    sweep.add_argument(
        "--full_model",
        action="store_true",
        help="evaluate only the unmodified model (no layer skipped)",
    )
    parser.add_argument(
        "--window_size", type=int, default=None, help="window width of --evaluate_full_sliding_window"
    )
    parser.add_argument(
        "--interval", type=int, default=1, help="step of the sweeps (default: 1)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    start = time.time()
    evaluate_with_skipped_layers(
        model_path=args.model_path,
        tasks=args.tasks,
        output_dir=args.output_dir,
        window_size=args.window_size,
        limit=args.limit,
        evaluate_full_sliding_window=args.evaluate_full_sliding_window,
        evaluate_full_progressive_ablation=args.evaluate_full_progressive_ablation,
        interval=args.interval,
        full_model=args.full_model,
    )
    print(f"Total time: {(time.time() - start) / 60:.2f} min")
