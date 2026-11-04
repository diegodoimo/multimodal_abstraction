"""Appendix "Interventions: all datasets (text benchmarks)": layer skipping on text-only tasks.

Reads results/interventions/text/<model>[_sw4]/<task>/<task>_L{ll}_W{w}_0shot.json
(lm-eval-harness scores written by scripts/interventions/text.py) and writes to
figures/out/interventions/text/ (PNG copies under figures/out_png/):

  layer_skip_progressive.pdf   skip layers [0, w)
  layer_skip_sliding.pdf       skip a window of 4 layers

Performance is normalized so that 1 is the full model and 0 is the model with all layers skipped.
Run: python figures/app_interventions_text.py
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from savefig import save

RESULTS = Path(__file__).resolve().parents[1] / "results" / "interventions" / "text"

MODELS = {
    "llava_7b":     ("LLaVA-7B",     32, "llava_7b",                        (6, 16)),
    "llava_13b":    ("LLaVA-13B",    40, "llava_13b",                       (6, 16)),
    "cambrian_8b":  ("Cambrian-8B",  32, "cambrian-8b",                     (5, 16)),
    "onevision_4b": ("OneVision-4B", 36, "LLaVA-OneVision-1.5-4B-Instruct", (8, 25)),
    "onevision_8b": ("OneVision-8B", 36, "LLaVA-OneVision-1.5-8B-Instruct", (8, 25)),
    "internvl2_8b": ("InternVL2-8B", 32, "InternVL2-8B",                    (6, 20)),
}
TASKS = {
    "arc_easy": "ARC-Easy", "commonsense_qa": "CommonsenseQA", "hellaswag": "HellaSwag",
    "mmlu": "MMLU", "piqa": "PIQA", "winogrande": "WinoGrande",
}
TASK_COLORS = dict(zip(TASKS, ("#f94144", "#f8961e", "#f9c74f", "#90be6d", "#469583", "#557CAF")))
SLIDING_WINDOW = 4
BAND_COLOR, BAND_ALPHA = "goldenrod", 0.18

STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10, "figure.constrained_layout.use": True,
}


def read_score(path, task):
    if not path.is_file():
        return None
    return float(json.loads(path.read_text())["results"][task]["acc,none"])


def load_curves(model, schedule):
    _, num_layers, prefix, _ = MODELS[model]
    run_dir = RESULTS / (f"{prefix}_sw{SLIDING_WINDOW}" if schedule == "sliding" else prefix)
    norm_dir = RESULTS / prefix
    curves = {}
    for task in TASKS:
        full = read_score(norm_dir / task / f"{task}_L00_W0_0shot.json", task)
        none = read_score(norm_dir / task / f"{task}_L00_W{num_layers}_0shot.json", task)
        if full is None or none is None or not (run_dir / task).is_dir():
            continue
        points = {}
        for f in (run_dir / task).iterdir():
            if schedule == "sliding":
                m = re.fullmatch(rf"{task}_L(\d+)_W{SLIDING_WINDOW}_0shot\.json", f.name)
                x = None if m is None else int(m.group(1)) + SLIDING_WINDOW / 2
            else:
                m = re.fullmatch(rf"{task}_L0+_W(\d+)_0shot\.json", f.name)
                x = None if m is None else int(m.group(1))
            if x is not None:
                points[x] = (read_score(f, task) - none) / (full - none)
        if points:
            xs = np.array(sorted(points))
            curves[task] = (xs, np.array([points[x] for x in xs]))
    return curves


def plot(schedule, filename):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(11, 6))
        for ax, model in zip(axes.flat, MODELS):
            title, num_layers, _, (lo, hi) = MODELS[model]
            ax.axvspan(lo - 1, hi, color=BAND_COLOR, alpha=BAND_ALPHA, zorder=0, lw=0)
            for task, (x, y) in load_curves(model, schedule).items():
                ax.plot(x, y, color=TASK_COLORS[task], marker="o", markersize=5, linewidth=2)
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
            ax.axhline(0.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
            ax.set_xticks(np.arange(0, num_layers + 1, 4))
            ax.set_xlabel("Layers")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        for ax in axes[:, 0]:
            ax.set_ylabel("Performance")
        handles = [Patch(facecolor=TASK_COLORS[t], label=TASKS[t]) for t in TASKS]
        fig.legend(handles=handles, title="Dataset:", loc="lower center", bbox_to_anchor=(0.5, -0.09),
                   ncol=len(TASKS), frameon=False)
        save(fig, filename)
        plt.close(fig)


if __name__ == "__main__":
    plot("progressive", "interventions/text/layer_skip_progressive.pdf")
    plot("sliding", "interventions/text/layer_skip_sliding.pdf")
