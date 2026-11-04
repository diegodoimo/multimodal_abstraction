"""Figure 2 and the multimodal panels of the appendix "Interventions: additional results".

Reads results/interventions/multimodal/<model>_<mode>[_sw5]/<task>/<task>_L{l}_W{w}.json
(lmms-eval scores written by scripts/interventions/multimodal.py) and writes to
figures/out/interventions/multimodal/ (PNG copies under figures/out_png/):

  average_main.pdf       Fig. 2: LLaVA-7B, OneVision-4B, InternVL2-8B
  average_appendix.pdf   LLaVA-13B, OneVision-8B, Cambrian-8B (abstraction band shown)
  layer_skip_progressive.pdf / layer_skip_sliding.pdf         per-dataset curves, all models
  attn_knockout_progressive.pdf / attn_knockout_sliding.pdf

Performance is normalized so that 1 is the full model (L0_W0) and 0 is the model with the
intervention applied to all layers (L0_W<num_layers>).  Run: python figures/fig2_interventions.py
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from savefig import save

RESULTS = Path(__file__).resolve().parents[1] / "results" / "interventions" / "multimodal"

# model key -> (title, number of layers, results directory prefix, abstraction region in 1-based layers)
MODELS = {
    "llava_7b":     ("LLaVA-7B",     32, "llava_7b",                        (6, 16)),
    "llava_13b":    ("LLaVA-13B",    40, "llava_13b",                       (6, 16)),
    "cambrian_8b":  ("Cambrian-8B",  32, "cambrian-8b",                     (5, 16)),
    "onevision_4b": ("OneVision-4B", 36, "LLaVA-OneVision-1.5-4B-Instruct", (8, 25)),
    "onevision_8b": ("OneVision-8B", 36, "LLaVA-OneVision-1.5-8B-Instruct", (8, 25)),
    "internvl2_8b": ("InternVL2-8B", 32, "InternVL2-8B",                    (6, 20)),
}
MAIN_MODELS = ["llava_7b", "onevision_4b", "internvl2_8b"]
APPENDIX_MODELS = ["llava_13b", "onevision_8b", "cambrian_8b"]
ALL_MODELS = ["llava_7b", "llava_13b", "cambrian_8b", "onevision_4b", "onevision_8b", "internvl2_8b"]

# task -> (score key, legend label)
TASKS = {
    "gqa":            ("exact_match,none",         "GQA"),
    "mmbench_cn_dev": ("gpt_eval_score,none",      "MMBench-CN"),
    "mmbench_en_dev": ("gpt_eval_score,none",      "MMBench-EN"),
    "mme":            ("mme_perception_score,none", "MME"),
    "seedbench":      ("seed_image,none",          "SEEDBench"),
    "vqav2_val":      ("exact_match,none",         "VQAv2"),
}
TASK_COLORS = dict(zip(TASKS, ("#f94144", "#f8961e", "#f9c74f", "#90be6d", "#469583", "#557CAF")))
MODE_COLORS = {"image": "#991499", "cross_modal": "#007e5d"}
MODE_LABELS = {"image": "Skip image layer", "cross_modal": r"Knockout img$\rightarrow$text"}
SLIDING_WINDOW = 5
BAND_COLOR, BAND_ALPHA = "goldenrod", 0.18   # same as the information-imbalance planes

STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10, "figure.constrained_layout.use": True,
}


def read_score(path, task):
    """Score of one lmms-eval run, or None if the file is missing."""
    if not path.is_file():
        return None
    block = json.loads(path.read_text()).get("results", {}).get(task)
    return None if block is None else float(block[TASKS[task][0]])


def load_curves(model, mode, schedule):
    """{task: (x, normalized score)} for one model / intervention / schedule."""
    _, num_layers, prefix, _ = MODELS[model]
    suffix = f"_sw{SLIDING_WINDOW}" if schedule == "sliding" else ""
    run_dir = RESULTS / f"{prefix}_{mode}{suffix}"
    norm_dir = RESULTS / f"{prefix}_{mode}"       # L0_W0 (full model) and L0_W<L> (all layers) live here
    curves = {}
    for task in TASKS:
        full = read_score(norm_dir / task / f"{task}_L0_W0.json", task)
        none = read_score(norm_dir / task / f"{task}_L0_W{num_layers}.json", task)
        if full is None or none is None or not (run_dir / task).is_dir():
            continue
        points = {}
        for f in (run_dir / task).iterdir():
            if schedule == "sliding":
                m = re.fullmatch(rf"{task}_L(\d+)_W{SLIDING_WINDOW}\.json", f.name)
                x = None if m is None else int(m.group(1)) + SLIDING_WINDOW / 2
            else:
                m = re.fullmatch(rf"{task}_L0_W(\d+)\.json", f.name)
                x = None if m is None else int(m.group(1))
            if x is not None:
                points[x] = (read_score(f, task) - none) / (full - none)
        if model == "cambrian_8b" and mode == "image" and schedule == "progressive":
            # Cambrian was also evaluated at W = 1, 2, 3; as in the paper the W = 3 run is shown at x = 2
            # and the W = 1, 2 runs are not drawn.
            points = {(2 if x == 3 else x): v for x, v in points.items() if x not in (1, 2)}
        if points:
            xs = np.array(sorted(points))
            curves[task] = (xs, np.array([points[x] for x in xs]))
    # keep only the x positions shared by the tasks, so that averages are well defined
    if curves:
        common = sorted(set.intersection(*(set(x) for x, _ in curves.values())))
        curves = {t: (np.array(common), np.array([y[list(x).index(c)] for c in common])) for t, (x, y) in curves.items()}
    return curves


def average_curve(curves):
    x = next(iter(curves.values()))[0]
    return x, np.mean([y for _, y in curves.values()], axis=0)


def draw_band(ax, model):
    """Abstraction region; x is the number of layers counted from the input, so layer l (1-based) spans [l-1, l]."""
    lo, hi = MODELS[model][3]
    ax.axvspan(lo - 1, hi, color=BAND_COLOR, alpha=BAND_ALPHA, zorder=0, lw=0)


def style_axis(ax, model):
    title, num_layers = MODELS[model][:2]
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_xticks(np.arange(0, num_layers + 1, 4))
    ax.set_xlabel("Layers")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_average(models, filename, band):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        for ax, model in zip(axes, models):
            if band:
                draw_band(ax, model)
            for mode in ("image", "cross_modal"):
                for schedule, ls, marker in (("progressive", "-", "o"), ("sliding", (0, (5, 5)), "^")):
                    curves = load_curves(model, mode, schedule)
                    if curves:
                        x, y = average_curve(curves)
                        ax.plot(x, y, color=MODE_COLORS[mode], linestyle=ls, marker=marker, markersize=5, linewidth=2)
            style_axis(ax, model)
        axes[0].set_ylabel("Performance")
        handles = [Patch(facecolor=MODE_COLORS[m], label=MODE_LABELS[m]) for m in ("image", "cross_modal")]
        handles += [Line2D([0], [0], color="black", linestyle="-", marker="o", markersize=5, linewidth=2, label="Progressive"),
                    Line2D([0], [0], color="black", linestyle=(0, (4, 4)), marker="^", markersize=5, linewidth=2, label="Sliding window")]
        fig.legend(handles=handles, title="Intervention:", loc="lower center", bbox_to_anchor=(0.5, -0.18),
                   ncol=4, frameon=False, columnspacing=1.5)
        save(fig, filename)
        plt.close(fig)


def plot_tasks(mode, schedule, filename):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(11, 6))
        for ax, model in zip(axes.flat, ALL_MODELS):
            draw_band(ax, model)
            for task, (x, y) in load_curves(model, mode, schedule).items():
                ax.plot(x, y, color=TASK_COLORS[task], marker="o", markersize=5, linewidth=2)
            style_axis(ax, model)
        for ax in axes[:, 0]:
            ax.set_ylabel("Performance")
        handles = [Patch(facecolor=TASK_COLORS[t], label=TASKS[t][1]) for t in TASKS]
        fig.legend(handles=handles, title="Dataset:", loc="lower center", bbox_to_anchor=(0.5, -0.09),
                   ncol=len(TASKS), frameon=False)
        save(fig, filename)
        plt.close(fig)


if __name__ == "__main__":
    plot_average(MAIN_MODELS, "interventions/multimodal/average_main.pdf", band=False)
    plot_average(APPENDIX_MODELS, "interventions/multimodal/average_appendix.pdf", band=True)
    plot_tasks("image", "progressive", "interventions/multimodal/layer_skip_progressive.pdf")
    plot_tasks("image", "sliding", "interventions/multimodal/layer_skip_sliding.pdf")
    plot_tasks("cross_modal", "progressive", "interventions/multimodal/attn_knockout_progressive.pdf")
    plot_tasks("cross_modal", "sliding", "interventions/multimodal/attn_knockout_sliding.pdf")
