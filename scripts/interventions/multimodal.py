"""Layer interventions on multimodal benchmarks with lmms-eval (Fig. 2, appendix multimodal and Llama-90B
interventions; launched by slurm/interventions_multimodal.sbatch and slurm/llama90b.sbatch).

For every window of decoder layers of a sweep, the model is patched (utils/models/<family>/patch_<family>.py) and
evaluated on the lmms-eval tasks. Ablation modes:

* ``image``: in the window, the text cannot attend to the image tokens, and the image hidden states are
  reset to their value at the start of the window when it ends (layer skip of the image tokens);
* ``cross_modal``: the same attention mask without the reset (attention knockout); for
  Llama-3.2-Vision, the cross-attention layers in the window are skipped.

Sweeps, in steps of ``--interval``: ``--evaluate_full_progressive_ablation`` (layers [0, w), w = 0, ..., L),
``--evaluate_full_reverse_progressive_ablation`` (layers [L - w, L)), ``--evaluate_full_sliding_window``
(windows of ``--window_size`` layers starting at 0, interval, ...); ``--evaluate_complementary_intervals``
(Llama only: cross-attention kept only inside, then removed inside, layers [``--start_layer``, ``--end_layer``)).

Output: ``<output_dir>/<model>_<mode>[_sw<width>|_reverse|_win<start>_<end>]/<task>/<task>_L<start>_W<width>.json``
(``<task>_{keep,remove}.json`` for the complementary intervals). Windows whose files all exist are skipped.
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
import warnings

import lmms_eval.utils as _lmms_utils
import torch
from lmms_eval import evaluator
from lmms_eval.models.chat.llava_hf import LlavaHf
from lmms_eval.models.chat.llava_onevision1_5 import Llava_OneVision1_5
from lmms_eval.models.simple.internvl2 import InternVL2
from lmms_eval.models.simple.llama_vision import LlamaVision

from utils.models.cambrian.cambrian_interventions import CambrianCustom
from utils.models.cambrian.patch_cambrian import (
    configure_layer_skip as configure_cambrian_layer_skip,
)
from utils.models.cambrian.patch_cambrian import (
    install_compat as install_cambrian_compat,
)
from utils.models.internvl.internvl2_compat import (
    patch_internvl_model_post_instantiation,
)
from utils.models.internvl.load_helpers import force_internvl_attn_implementation
from utils.models.internvl.patch_internvl import patch as patch_internvl
from utils.models.llama.patch_llama import patch as patch_llama
from utils.models.llava.patch_llava import patch as patch_llava
from utils.evaluation.llm_judge import ensure_llm_judge_env, install_parallel_mmbench_judge
from utils.evaluation.lmms_common import onevision_pixel_budget, use_internvl_tiles
from utils.models.onevision.llavaonevision_compat import patch_flash_attn_varlen_import
from utils.models.onevision.patch_onevision import patch as patch_onevision

warnings.filterwarnings("ignore")
from loguru import logger as eval_logger

# Tasks whose aggregate metric needs the full dataset: lmms-eval applies one --limit to all tasks, so
# --limit is dropped for the whole run when one of them is present.
TASKS_REQUIRE_FULL = {"naturalbench"}

# Tasks scored by the LLM judge (OPENAI_API_KEY)
LLM_JUDGE_TASKS = {
    "mmvet",
    "mmbench_en_dev",
    "mmbench_cn_dev",
    "mmbench_en_test",
    "mmbench_cn_test",
    "llava_in_the_wild",
}

RANDOM_SEED = NUMPY_RANDOM_SEED = TORCH_RANDOM_SEED = FEWSHOT_RANDOM_SEED = 42


# =============================================================================
# Llama-3.2-Vision: short-answer prompts and answer extraction
#
# Llama-3.2-Vision ignores the "single word or phrase" instruction of the stock vqav2 / gqa / mme
# prompts and answers in sentences ("The walls are white.", "No, it is not."), which the exact-match
# scorers count as wrong. For Llama models only, the task YAMLs' !function loader is intercepted to
# use stronger prompts (vqav2, gqa, mme) and to extract the short answer before scoring (vqav2, mme
# through process_results; gqa, which has no process_results, through lm.generate_until).
# =============================================================================

_VQA_PROMPT_TAIL = (
    "\nAnswer with a single word or short phrase only. "
    "Do not write a full sentence. Do not repeat the question."
)
_MME_PROMPT_TAIL = '\nAnswer with only "Yes" or "No". Do not add anything else.'

_VERBOSE_PREFIX_RE = re.compile(
    r"^(?:the|this|that|it|they|there|these|those)\s+(?:\S+\s+)*?"
    r"(?:is|are|was|were)\s+",
    re.IGNORECASE,
)
_YN_PREFIX_RE = re.compile(r"^(yes|no)\b", re.IGNORECASE)
# "Right side" -> "right"; directional word + "side" only, so that answers like "tank top" are kept
_TRAILING_SIDE_RE = re.compile(
    r"^(left|right|top|bottom|middle|front|back|center|upper|lower)\s+side$",
    re.IGNORECASE,
)
# "To the right" / "On the floor" -> "right" / "floor" (only when the rest has at most 3 words)
_LEADING_PREP_RE = re.compile(
    r"^(?:to|on|in|at|by|with|of)\s+(?:the|a|an)?\s*",
    re.IGNORECASE,
)
# "A woman" / "The beach" -> "woman" / "beach" (no GQA answer starts with an article; VQAv2 strips them)
_LEADING_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)


def _extract_short_answer(resp: str) -> str:
    """Short answer of a sentence-style response (rules above)."""
    s = resp.strip().rstrip(".").strip()
    if not s:
        return s
    # "No, it does not." -> "No"
    yn = _YN_PREFIX_RE.match(s)
    if yn:
        return s[: yn.end()]
    # "The X is Y." -> "Y"
    m = _VERBOSE_PREFIX_RE.match(s)
    if m:
        s = s[m.end() :].strip().rstrip(".").strip()
    # before the article / preposition rules, so that the direction word is kept
    m = _TRAILING_SIDE_RE.match(s)
    if m:
        return m.group(1)
    m = _LEADING_PREP_RE.match(s)
    if m:
        rest = s[m.end() :].strip().rstrip(".").strip()
        if rest and len(rest.split()) <= 3:
            s = rest
    m = _LEADING_ARTICLE_RE.match(s)
    if m:
        rest = s[m.end() :].strip().rstrip(".").strip()
        if rest:
            s = rest
    # "Bedroom/office" -> "Bedroom": both scorers strip "/" as punctuation, so the joined form
    # could match neither alternative; keep the first one
    if "/" in s:
        first = s.split("/", 1)[0].strip().rstrip(".").strip()
        if first:
            s = first
    return s


def _patched_vqav2_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return f"{doc['question']}{_VQA_PROMPT_TAIL}"


_orig_vqav2_process_results_val = None


def _patched_vqav2_process_results_val(doc, result):
    extracted = [_extract_short_answer(r) if isinstance(r, str) else r for r in result]
    return _orig_vqav2_process_results_val(doc, extracted)


def _patched_gqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return f"{doc['question']}{_VQA_PROMPT_TAIL}"


def _patched_mme_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return f"{doc['question'].strip()}{_MME_PROMPT_TAIL}"


_orig_mme_process_results = None


def _patched_mme_process_results(doc, results):
    extracted = [_extract_short_answer(r) if isinstance(r, str) else r for r in results]
    return _orig_mme_process_results(doc, extracted)


_orig_import_function = _lmms_utils.import_function


def _import_function_with_overrides(loader, node):
    fn = _orig_import_function(loader, node)
    global _orig_vqav2_process_results_val, _orig_mme_process_results
    name = getattr(fn, "__name__", "")
    if name == "vqav2_doc_to_text":
        return _patched_vqav2_doc_to_text
    if name == "vqav2_process_results_val":
        _orig_vqav2_process_results_val = fn
        return _patched_vqav2_process_results_val
    if name == "gqa_doc_to_text":
        return _patched_gqa_doc_to_text
    if name == "mme_doc_to_text":
        return _patched_mme_doc_to_text
    if name == "mme_process_results":
        _orig_mme_process_results = fn
        return _patched_mme_process_results
    return fn


# tasks without a process_results hook: the answer is extracted from lm.generate_until's output
_TASKS_NEEDING_GENERATE_UNTIL_EXTRACTION = {"gqa"}


def _install_response_extractor(lm, active_tasks: set[str]) -> None:
    """Extract the short answer of the responses to ``active_tasks`` in ``lm.generate_until``."""
    if not active_tasks:
        return
    original = lm.generate_until

    def patched(requests):
        responses = original(requests)
        out = []
        for req, resp in zip(requests, responses):
            tname = getattr(req, "task_name", None)
            if tname in active_tasks and isinstance(resp, str):
                resp = _extract_short_answer(resp)
            out.append(resp)
        return out

    lm.generate_until = patched


def _enable_short_answer_overrides() -> None:
    """Use the overrides above when the task YAMLs are loaded (call before simple_evaluate)."""
    _lmms_utils.import_function = _import_function_with_overrides
    print("Llama: short-answer prompts and answer extraction for vqav2, gqa and mme.")


def attach_system_instruction(lm, system_text: str) -> None:
    """Prepend a system message to every prompt of lmms-eval's ``LlamaVision``.

    Its ``generate_until`` builds user-only messages and ignores simple_evaluate's
    ``system_instruction``, so ``processor.apply_chat_template`` is wrapped instead.
    """
    original_apply = lm.processor.apply_chat_template

    def patched_apply(messages, **kwargs):
        if isinstance(messages, list) and any(
            isinstance(m, dict) and m.get("role") == "system" for m in messages
        ):
            return original_apply(messages, **kwargs)
        new_messages = [{"role": "system", "content": system_text}] + list(messages)
        return original_apply(new_messages, **kwargs)

    lm.processor.apply_chat_template = patched_apply
    print(f"System instruction added to every prompt ({len(system_text)} characters).")


# =============================================================================
# Models
# =============================================================================


def get_num_layers(model, model_path: str) -> int:
    """Number of decoder layers of the language model."""
    mp = model_path.lower()
    if "llava" in mp and "onevision" not in mp:
        return model.config.text_config.num_hidden_layers
    if "internvl" in mp:
        return model.language_model.model.config.num_hidden_layers
    if "onevision" in mp:
        return model.config.text_config.num_hidden_layers
    if "cambrian" in mp:
        return model.config.num_hidden_layers
    if "llama" in mp:
        return model.config.text_config.num_hidden_layers  # MllamaForConditionalGeneration
    raise ValueError(f"Cannot determine the number of layers of {model_path}")


def load_model(model_path: str):
    """lmms-eval wrapper of the model, with eager attention where the patch edits the attention mask."""
    mp = model_path.lower()
    print(f"Loading {model_path}")
    if "llava" in mp and "onevision" not in mp:
        # the checkpoint's chat template, which has no system turn: "USER: <image>\n<question> ASSISTANT:"
        return LlavaHf(pretrained=model_path, device_map="auto", attn_implementation="eager")

    if "internvl" in mp:
        # the wrapper's template already carries InternVL2's system message: no system instruction
        use_internvl_tiles()  # 12 tiles per image, the InternVL2 default
        # lmms-eval's InternVL2 does not expose attn_implementation
        with force_internvl_attn_implementation("eager"):
            lm = InternVL2(pretrained=model_path, device_map="auto")
        # generate() for the remote InternLM2ForCausalLM on transformers >= 4.50
        patch_internvl_model_post_instantiation(lm._model)
        # the tokenizer has no pad token: use its EOS (batch size 1) instead of a warning at every call
        lm._model.language_model.generation_config.pad_token_id = lm._tokenizer.eos_token_id
        return lm

    if "onevision" in mp:
        patch_flash_attn_varlen_import()  # needed by the remote code
        return Llava_OneVision1_5(
            pretrained=model_path,
            device_map="auto",
            attn_implementation="eager",
            **onevision_pixel_budget(model_path),  # the processor's own pixel budget
        )

    if "cambrian" in mp:
        lm = CambrianCustom(pretrained=model_path, device_map="auto", attn_implementation="eager")
        install_cambrian_compat(lm.model)  # transformers 4.57 compatibility, also for the full model
        return lm

    if "llama" in mp:
        # local directory or Hub id; the cross-attention knockout skips whole layers, so the
        # default attention implementation is kept
        return LlamaVision(pretrained=model_path, device_map="auto")

    raise ValueError(f"Unsupported model: {model_path}")


def patch_model(lm, model_path: str, layers_to_skip: list[int], ablation_mode: str) -> None:
    """Apply the intervention to ``layers_to_skip`` (no layer: the full model)."""
    mp = model_path.lower()
    if "llava" in mp and "onevision" not in mp:
        patch_llava(model=lm.model, layers_to_skip=layers_to_skip, ablation_mode=ablation_mode)
    elif "internvl" in mp:
        patch_internvl(model=lm.model, layers_to_skip=layers_to_skip, ablation_mode=ablation_mode)
    elif "onevision" in mp:
        patch_onevision(model=lm.model, layers_to_skip=layers_to_skip, ablation_mode=ablation_mode)
    elif "cambrian" in mp:
        configure_cambrian_layer_skip(
            model=lm.model, layers_to_skip=layers_to_skip, ablation_mode=ablation_mode
        )
    elif "llama" in mp:
        patch_llama(model=lm.model, layers_to_skip=layers_to_skip)


def validate_task_requirements(tasks: list) -> None:
    """Set up the LLM judge (utils.evaluation.llm_judge) when a task needs it; exit if no key is available."""
    judge_tasks = [t for t in tasks if t in LLM_JUDGE_TASKS]
    if not judge_tasks:
        return
    if not ensure_llm_judge_env():
        eval_logger.error(f"OPENAI_API_KEY is not set; {judge_tasks} need the LLM judge.")
        sys.exit(1)
    eval_logger.info(
        f"LLM judge: {os.environ.get('MODEL_VERSION')} at {os.environ.get('OPENAI_API_URL')}"
    )
    install_parallel_mmbench_judge()  # concurrent judge calls, same scores


# =============================================================================
# Sweep
# =============================================================================


def evaluate_with_ablation(
    model_path: str,
    tasks: list[str],
    output_dir: str,
    ablation_mode: str,
    limit: int | None,
    max_new_tokens: int,
    window_size: int,
    interval: int,
    start_layer: int,
    end_layer: int | None,
    evaluate_full_sliding_window: bool,
    evaluate_full_progressive_ablation: bool,
    evaluate_full_reverse_progressive_ablation: bool,
    evaluate_complementary_intervals: bool,
    llama_system_instruction: str | None,
):
    """Evaluate ``tasks`` for every window of the selected sweep (module docstring) and save the results."""
    is_llama = "llama" in model_path.lower()
    if is_llama:
        _enable_short_answer_overrides()  # before simple_evaluate loads the task YAMLs

    lm = load_model(model_path)
    # lmms-eval's clean() deletes the model after a run: keep it for the next windows
    lm.clean = lambda: torch.cuda.empty_cache()
    if is_llama:
        _install_response_extractor(lm, _TASKS_NEEDING_GENERATE_UNTIL_EXTRACTION.intersection(tasks))
    if llama_system_instruction and is_llama:
        attach_system_instruction(lm, llama_system_instruction)

    num_layers = get_num_layers(lm._model, model_path)
    print(f"{model_path}: {num_layers} decoder layers")

    if evaluate_full_sliding_window:
        if window_size is None or window_size <= 0:
            raise ValueError("--evaluate_full_sliding_window needs --window_size > 0")
        windows = [
            list(range(s, s + window_size))
            for s in range(0, num_layers - window_size + 2, interval)
        ]
    elif evaluate_full_progressive_ablation:
        # w = 0 is the full model
        windows = [list(range(0, w)) for w in range(0, num_layers + 1, interval)]
    elif evaluate_full_reverse_progressive_ablation:
        windows = [list(range(num_layers - w, num_layers)) for w in range(0, num_layers + 1, interval)]
    else:  # evaluate_complementary_intervals
        # patch_llama skips only the cross-attention layers among the indices
        if not is_llama:
            raise ValueError("--evaluate_complementary_intervals is for Llama-3.2-Vision models only.")
        if end_layer is None:
            raise ValueError("--evaluate_complementary_intervals needs --end_layer.")
        if not 0 <= start_layer < end_layer <= num_layers:
            raise ValueError(
                f"Need 0 <= start_layer ({start_layer}) < end_layer ({end_layer}) <= {num_layers}."
            )
        windows = [
            list(range(0, start_layer)) + list(range(end_layer, num_layers)),  # keep: skip outside
            list(range(start_layer, end_layer)),  # remove: skip inside
        ]

    effective_limit = limit
    full_tasks = [t for t in tasks if t in TASKS_REQUIRE_FULL]
    if limit is not None and full_tasks:
        non_full = [t for t in tasks if t not in TASKS_REQUIRE_FULL]
        msg = f"--limit {limit} ignored: {full_tasks} need the full dataset"
        if non_full:
            msg += f", and lmms-eval applies one limit to all tasks ({non_full} run in full too)"
        eval_logger.warning(msg + ".")
        effective_limit = None

    for iter_idx, layers_to_skip in enumerate(windows):
        patch_model(lm, model_path, layers_to_skip, ablation_mode)

        if not layers_to_skip:
            cur_window = 0
            # the reverse sweep anchors its file names at L = num_layers: L{num_layers}_W0, L{num_layers}_W{w}, ...
            cur_start = num_layers if evaluate_full_reverse_progressive_ablation else None
            print("Window: none (full model)")
        else:
            cur_window = len(layers_to_skip)
            cur_start = (
                num_layers
                if evaluate_full_reverse_progressive_ablation
                else layers_to_skip[0]
            )
            if evaluate_complementary_intervals:
                action = "kept only" if iter_idx == 0 else "removed"
                print(f"Window: cross-attention {action} in layers [{start_layer}, {end_layer})")
            else:
                print(f"Window: layers {layers_to_skip[0]}-{layers_to_skip[-1]} ({ablation_mode})")

        model_name = model_path.split("/")[-1] + f"_{ablation_mode}"
        if evaluate_full_sliding_window:
            model_name += f"_sw{cur_window}"
        if evaluate_full_reverse_progressive_ablation:
            model_name += "_reverse"
        if evaluate_complementary_intervals:
            model_name += f"_win{start_layer}_{end_layer}"

        def result_filename(task):
            if evaluate_complementary_intervals:
                return f"{task}_{'keep' if iter_idx == 0 else 'remove'}.json"
            if cur_start is None:
                return f"{task}_L0_W0.json"
            return f"{task}_L{cur_start}_W{cur_window}.json"

        if all(
            os.path.isfile(os.path.join(output_dir, model_name, t, result_filename(t))) for t in tasks
        ):
            print(f"Results exist, skipping: {model_name}/{result_filename('*')}")
            continue

        start = time.time()
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            batch_size=1,
            device="cuda",
            limit=effective_limit,
            log_samples=False,
            # directory of the submission files some tasks write
            cli_args=argparse.Namespace(output_path="./tmp", process_with_media=True),
            random_seed=RANDOM_SEED,
            numpy_random_seed=NUMPY_RANDOM_SEED,
            torch_random_seed=TORCH_RANDOM_SEED,
            fewshot_random_seed=FEWSHOT_RANDOM_SEED,
            gen_kwargs=f"max_new_tokens={max_new_tokens}",
        )
        print(f"Window evaluated in {(time.time() - start) / 60:.1f} min")

        if results:
            all_results = results.get("results", {})
            for task in tasks:
                task_results = {
                    k: v
                    for k, v in all_results.items()
                    if k == task or k.startswith(f"{task}_")
                }
                if not task_results:
                    continue
                task_output_dir = os.path.join(output_dir, model_name, task)
                os.makedirs(task_output_dir, exist_ok=True)
                output_path = os.path.join(task_output_dir, result_filename(task))
                with open(output_path, "w") as f:
                    serializable = {
                        "config": {
                            "model_path": model_path,
                            "start_layer": cur_start,
                            "window_size": cur_window,
                            "task": task,
                        },
                        "results": task_results,
                    }
                    json.dump(serializable, f, indent=2, default=str)
                print(f"Saved {output_path}")

        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer interventions on multimodal benchmarks (lmms-eval)."
    )
    parser.add_argument("--model_path", type=str, default="lbasile/llava_7b", help="Hub id or local path")
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=["vqav2"], help="lmms-eval tasks"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/interventions/multimodal", help="output directory"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="samples per task (default: all)"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=32, help="maximum number of generated tokens"
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="image",
        choices=["image", "cross_modal"],
        help="image: layer skip of the image tokens; cross_modal: attention knockout (module docstring)",
    )
    sweep = parser.add_mutually_exclusive_group(required=True)
    sweep.add_argument(
        "--evaluate_full_progressive_ablation",
        action="store_true",
        help="skip layers [0, w) for w = 0, interval, ..., num_layers",
    )
    sweep.add_argument(
        "--evaluate_full_reverse_progressive_ablation",
        action="store_true",
        help="skip layers [num_layers - w, num_layers) for w = 0, interval, ..., num_layers",
    )
    sweep.add_argument(
        "--evaluate_full_sliding_window",
        action="store_true",
        help="skip windows of --window_size layers starting at 0, interval, ...",
    )
    sweep.add_argument(
        "--evaluate_complementary_intervals",
        action="store_true",
        help="Llama only: keep cross-attention only inside [--start_layer, --end_layer), then remove it there",
    )
    parser.add_argument(
        "--window_size", type=int, default=0, help="window width of --evaluate_full_sliding_window"
    )
    parser.add_argument(
        "--interval", type=int, default=1, help="step of the sweeps (default: 1)"
    )
    parser.add_argument(
        "--start_layer", type=int, default=0, help="first layer of --evaluate_complementary_intervals"
    )
    parser.add_argument(
        "--end_layer", type=int, default=None, help="end (exclusive) of --evaluate_complementary_intervals"
    )
    parser.add_argument(
        "--llama_system_instruction",
        type=str,
        default=None,
        help="Llama-3.2-Vision only: system message prepended to every prompt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_task_requirements(args.tasks)

    start = time.time()
    evaluate_with_ablation(
        model_path=args.model_path,
        tasks=args.tasks,
        output_dir=args.output_dir,
        ablation_mode=args.ablation_mode,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        window_size=args.window_size,
        interval=args.interval,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        evaluate_full_sliding_window=args.evaluate_full_sliding_window,
        evaluate_full_progressive_ablation=args.evaluate_full_progressive_ablation,
        evaluate_full_reverse_progressive_ablation=args.evaluate_full_reverse_progressive_ablation,
        evaluate_complementary_intervals=args.evaluate_complementary_intervals,
        llama_system_instruction=args.llama_system_instruction,
    )
    print(f"Total time: {(time.time() - start) / 60:.2f} min")
