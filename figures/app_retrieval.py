"""Appendix "Multimodal retrieval metrics": COCO image <-> caption retrieval across layers.

Reads results/retrieval/<model>/retrieval_imgavg_vs_textavg.json (scripts/probing/retrieval.py) and writes
figures/out/probing/retrieval_coco.pdf: mean percentile rank (average of the image->text and text->image
directions) under Euclidean and cosine ranking, one panel per model, abstraction region in yellow.
Run: python figures/app_retrieval.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from savefig import save

RESULTS = Path(__file__).resolve().parents[1] / "results" / "retrieval"

# model key -> (label, results slug, abstraction region (1-based layers))
MODELS = {
    "llava_7b":     ("LLaVA-7B",     "llava_7b",        (6, 16)),
    "llava_13b":    ("LLaVA-13B",    "llava_13b",       (6, 16)),
    "cambrian_8b":  ("Cambrian-8B",  "cambrian_8b",     (5, 16)),
    "onevision_4b": ("OneVision-4B", "onevision_4b_hf", (8, 25)),
    "onevision_8b": ("OneVision-8B", "onevision_8b_hf", (8, 25)),
    "internvl2_8b": ("InternVL2-8B", "internvl_8b_hf",  (6, 20)),
}
METRICS = {"euclidean": "#f94144", "cosine": "#457b9d"}
PART = "res2"                     # residual stream at the output of each decoder block
DIRECTIONS = ("imgavg_to_textavg", "textavg_to_imgavg")
BAND_COLOR, BAND_ALPHA = "goldenrod", 0.18
STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10, "figure.constrained_layout.use": True,
}


def load_curve(model, metric):
    """(layers, mean percentile rank averaged over the two retrieval directions) or None."""
    path = RESULTS / MODELS[model][1] / "retrieval_imgavg_vs_textavg.json"
    if not path.is_file():
        print(f"missing {path}")
        return None
    data = json.loads(path.read_text())["results"][metric][PART]
    layers = sorted(int(k) for k in data[DIRECTIONS[0]])
    values = np.mean([[data[d][str(l)]["mean_percentile_rank"] for l in layers] for d in DIRECTIONS], axis=0)
    return np.array(layers), values


if __name__ == "__main__":
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, model in zip(axes.flat, MODELS):
            label, _, (lo, hi) = MODELS[model]
            ax.axvspan(lo - 1.5, hi - 0.5, color=BAND_COLOR, alpha=BAND_ALPHA, zorder=0, lw=0)  # layer l at index l-1
            for metric, color in METRICS.items():
                curve = load_curve(model, metric)
                if curve is not None:
                    x, y = curve
                    ax.plot(x[1::2], y[1::2], color=color, linewidth=2, marker="o", markersize=5)
            ax.set_title(label)
            ax.set_xlabel("Layer")
            ax.grid(True, alpha=0.3)
        for ax in axes[:, 0]:
            ax.set_ylabel("Mean percentile rank")
        fig.legend(handles=[Patch(facecolor=c, label=m) for m, c in METRICS.items()],
                   loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
        save(fig, "probing/retrieval_coco.pdf")
        plt.close(fig)
