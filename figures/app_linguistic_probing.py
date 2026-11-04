"""Appendix "Probing semantic linguistic properties of hidden layers": SentEval probes.

Reads results/linguistic_probing/<model>/res2/text_avg/{CoordInv,SOMO}_accuracy_logistic.json
(scripts/probing/linguistic_probing.py: logistic-regression accuracy per layer, several training subsets)
and writes figures/out/probing/linguistic_probing.pdf, one panel per model, abstraction region in yellow.
Run: python figures/app_linguistic_probing.py
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from savefig import save

RESULTS = Path(__file__).resolve().parents[1] / "results" / "linguistic_probing"

MODELS = {
    "llava_7b":     ("LLaVA-7B",     "llava_7b",        (6, 16)),
    "llava_13b":    ("LLaVA-13B",    "llava_13b",       (6, 16)),
    "cambrian_8b":  ("Cambrian-8B",  "cambrian_8b",     (5, 16)),
    "onevision_4b": ("OneVision-4B", "onevision_4b_hf", (8, 25)),
    "onevision_8b": ("OneVision-8B", "onevision_8b_hf", (8, 25)),
    "internvl2_8b": ("InternVL2-8B", "internvl_8b_hf",  (6, 20)),
}
TASKS = {"CoordInv": "#f94144", "SOMO": "#f8961e"}
EMBEDDING, POOLING = "res2", "text_avg"   # block-output residual stream, mean over the sentence tokens
BAND_COLOR, BAND_ALPHA = "goldenrod", 0.18
STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10, "figure.constrained_layout.use": True,
}


def load_curve(model, task):
    """(layer indices, mean accuracy over the training subsets) or None."""
    path = RESULTS / MODELS[model][1] / EMBEDDING / POOLING / f"{task}_accuracy_logistic.json"
    if not path.is_file():
        print(f"missing {path}")
        return None
    points = []
    for key, values in json.loads(path.read_text()).items():
        m = re.search(r"\.layers\.(\d+)", key)
        if m and values:
            points.append((int(m.group(1)), float(np.mean(values))))
    points.sort()
    return np.array([p[0] for p in points]), np.array([p[1] for p in points])


if __name__ == "__main__":
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, model in zip(axes.flat, MODELS):
            label, _, (lo, hi) = MODELS[model]
            ax.axvspan(lo - 1.5, hi - 0.5, color=BAND_COLOR, alpha=BAND_ALPHA, zorder=0, lw=0)  # layer l at index l-1
            max_layer = 0
            for task, color in TASKS.items():
                curve = load_curve(model, task)
                if curve is not None:
                    x, y = curve
                    ax.plot(x[1::2], y[1::2], color=color, linewidth=2, marker="o", markersize=5)
                    max_layer = max(max_layer, int(x.max()))
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.9)
            ax.set_xticks(np.arange(0, max_layer + 1, 4))
            ax.set_xlim(-1, max_layer + 1)
            ax.set_ylim(0.45, 1.0)
            ax.set_title(label)
            ax.set_xlabel("Layer")
            ax.grid(True, alpha=0.3)
        for ax in axes[:, 0]:
            ax.set_ylabel("Accuracy")
        fig.legend(handles=[Patch(facecolor=c, label=t) for t, c in TASKS.items()],
                   loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
        save(fig, "probing/linguistic_probing.pdf")
        plt.close(fig)
