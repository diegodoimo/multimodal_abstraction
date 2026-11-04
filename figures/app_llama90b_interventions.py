"""Appendix "Extension to multimodal cross-attention integration": cross-attention ablations in Llama-3.2-90B-Vision.

Reads results/interventions/llama90b/Llama-3.2-90B-Vision-Instruct_cross_modal[_reverse]/<task>/<task>_L{l}_W{w}.json
(scripts/interventions/multimodal.py with --ablation_mode cross_modal on the Llama-Vision cross-attention
layers, progressively from the input or from the output) and writes
figures/out/llama90b/ablations_cross_modal.pdf: average normalized performance (left) and per-dataset curves
(center: from the input, right: from the output).
Run: python figures/app_llama90b_interventions.py
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from savefig import save

RESULTS = Path(__file__).resolve().parents[1] / "results" / "interventions" / "llama90b"
MODEL = "Llama-3.2-90B-Vision-Instruct"
NUM_LAYERS = 100
TASKS = {
    "gqa":            ("exact_match,none",         "GQA"),
    "mmbench_cn_dev": ("gpt_eval_score,none",      "MMBench-CN"),
    "mmbench_en_dev": ("gpt_eval_score,none",      "MMBench-EN"),
    "mme":            ("mme_perception_score,none", "MME"),
    "seedbench":      ("seed_image,none",          "SEEDBench"),
    "vqav2_val":      ("exact_match,none",         "VQAv2"),
}
TASK_COLORS = dict(zip(TASKS, ("#f94144", "#f8961e", "#f9c74f", "#90be6d", "#469583", "#557CAF")))
VARIANTS = {"input": ("cross_modal", "#007e5d", "Ablation from input"),
            "output": ("cross_modal_reverse", "#f8961e", "Ablation from output")}
STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10, "figure.constrained_layout.use": True,
}


def read_score(path, task):
    if not path.is_file():
        return None
    block = json.loads(path.read_text()).get("results", {}).get(task)
    return None if block is None else float(block[TASKS[task][0]])


def load_curves(variant):
    """{task: (number of ablated layers, normalized score)}; files are L0_W{w} (input) or L100_W{w} (output)."""
    run_dir = RESULTS / f"{MODEL}_{VARIANTS[variant][0]}"
    anchor = 0 if variant == "input" else NUM_LAYERS
    curves = {}
    for task in TASKS:
        full = read_score(run_dir / task / f"{task}_L{anchor}_W0.json", task)
        none = read_score(run_dir / task / f"{task}_L{anchor}_W{NUM_LAYERS}.json", task)
        if full is None or none is None or not (run_dir / task).is_dir():
            continue
        points = {}
        for f in (run_dir / task).iterdir():
            m = re.fullmatch(rf"{task}_L{anchor}_W(\d+)\.json", f.name)
            if m:
                points[int(m.group(1))] = (read_score(f, task) - none) / (full - none)
        xs = np.array(sorted(points))
        curves[task] = (xs, np.array([points[x] for x in xs]))
    if curves:   # keep the x positions shared by all tasks
        common = sorted(set.intersection(*(set(x) for x, _ in curves.values())))
        curves = {t: (np.array(common), np.array([y[list(x).index(c)] for c in common])) for t, (x, y) in curves.items()}
    return curves


def style(ax, title):
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_xticks(np.arange(0, NUM_LAYERS + 1, 10))
    ax.set_xlabel("Layers")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


if __name__ == "__main__":
    with plt.rc_context(STYLE):
        fig, (ax_avg, ax_in, ax_out) = plt.subplots(1, 3, figsize=(12, 3.6))
        for variant, ax in (("input", ax_in), ("output", ax_out)):
            curves = load_curves(variant)
            _, color, label = VARIANTS[variant]
            if curves:
                x = next(iter(curves.values()))[0]
                x_plot = x if variant == "input" else NUM_LAYERS - x   # position of the ablation front
                ax_avg.plot(x_plot, np.mean([y for _, y in curves.values()], axis=0), color=color, marker="o", markersize=5, linewidth=2)
                for task, (xs, ys) in curves.items():
                    ax.plot(xs, ys, color=TASK_COLORS[task], marker="o", markersize=4, linewidth=1.5)
            style(ax, label)
            if variant == "output":
                ax.invert_xaxis()
        style(ax_avg, "Average")
        ax_avg.set_ylabel("Performance")
        fig.legend(handles=[Line2D([0], [0], color=VARIANTS[v][1], marker="o", markersize=5, linewidth=2, label=VARIANTS[v][2]) for v in VARIANTS],
                   loc="lower center", bbox_to_anchor=(0.18, -0.12), ncol=1, frameon=False)
        fig.legend(handles=[Line2D([0], [0], color=TASK_COLORS[t], marker="o", markersize=5, linewidth=2, label=TASKS[t][1]) for t in TASKS],
                   title="Datasets", loc="lower center", bbox_to_anchor=(0.68, -0.14), ncol=3, frameon=False)
        save(fig, "llama90b/ablations_cross_modal.pdf")
        plt.close(fig)
