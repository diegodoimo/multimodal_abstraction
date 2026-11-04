"""Figure 3 and the appendix "Probing the semantic content of intermediate layers".

Reads
  results/label_overlap/<dataset>/<split>/<model>/label_overlap_<mode>_n10_cf0.3.json  (scripts/probing/label_overlap.py)
  results/logit_lens/<model>_<dataset>_images_qa_{logit_lens,linear_probe}.safetensors   (scripts/probing/logit_lens.py)
and writes to figures/out/probing/ (PNG copies under figures/out_png/):

  label_overlap_and_logit_lens.pdf  Fig. 3: label overlap (left) and chance-adjusted MCQ accuracy of the logit
                                    lens (solid) and of a linear probe (dashed) for LLaVA-7B, OneVision-4B,
                                    InternVL2-8B
  label_overlap_average.pdf         average label overlap of the six models vs normalized depth
  label_overlap_by_dataset.pdf      per-dataset label overlap, one panel per model
  linear_probe_mc.pdf, logit_lens_mc.pdf, logit_lens_oe.pdf   per-dataset probing accuracies, one panel per model

In the per-model appendix panels the yellow band is the abstraction region of the model; the average
label-overlap figure has no band.  Run: python figures/fig3_probing.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from safetensors.numpy import load_file

from savefig import save

ROOT = Path(__file__).resolve().parents[1] / "results"

# model key -> (label, results slug, number of layers, abstraction region (1-based layers), color)
MODELS = {
    "llava_7b":     ("LLaVA-7B",     "llava_7b",                        32, (6, 16), "#E09F3E"),
    "llava_13b":    ("LLaVA-13B",    "llava_13b",                       40, (6, 16), "#E9CF3A"),
    "cambrian_8b":  ("Cambrian-8B",  "cambrian-8b",                     32, (5, 16), "#9e2a2b"),
    "onevision_4b": ("OneVision-4B", "LLaVA-OneVision-1.5-4B-Instruct", 36, (8, 25), "#457b9d"),
    "onevision_8b": ("OneVision-8B", "LLaVA-OneVision-1.5-8B-Instruct", 36, (8, 25), "#335c67"),
    "internvl2_8b": ("InternVL2-8B", "InternVL2-8B",                    32, (6, 20), "#99a88c"),
}
MAIN_MODELS = ["llava_7b", "onevision_4b", "internvl2_8b"]

# label-overlap datasets: slug, split, prompt mode, label
LO_DATASETS = [
    ("mini_imagenet", "validation", "mcq_nocat_filter", "mini-ImageNet"),
    ("food101", "validation", "generic_mcq_filter", "Food101"),
    ("sun397", "test", "generic_mcq_filter", "SUN397"),
    ("caltech101", "train", "generic_mcq_top_filter", "Caltech101"),
    ("dtd", "train", "generic_mcq_filter", "DTD"),
    ("oxford_flowers102", "test", "generic_mcq_top_filter", "Flowers102"),
    ("places365", "train", "generic_mcq_filter", "Places365"),
]
# logit-lens datasets: slug, label, chance level used for the chance-adjusted accuracy
MC_DATASETS = [("scienceqa", "ScienceQA", 0.2), ("mme_test_captioned", "MME", 0.5),
               ("mmbench_en_dev_captioned", "MMBench", 0.25), ("pope", "POPE", 0.5)]
OE_DATASETS = [("gqa_val_balanced_noyesno_captioned", "GQA", 0.0), ("cocoqa_restval_icqa", "COCOQA", 0.0),
               ("vqav2_coco_restval_icq", "VQAv2", 0.0), ("countbenchqa", "CountBench", 0.0)]

PALETTE = ("#f94144", "#f8961e", "#f9c74f", "#90be6d", "#469583", "#557CAF", "#0077B6")
OE_PALETTE = ("#469583", "#557CAF", "#0077B6", "#9e2a2b")
BAND_COLOR, BAND_ALPHA = "goldenrod", 0.18
STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10, "figure.constrained_layout.use": True,
}


# ----------------------------------------------------------------------------- data
def label_overlap_curves(model):
    """{dataset label: per-layer label overlap} for one model (None if a dataset is missing)."""
    slug = MODELS[model][1]
    curves = {}
    for ds, split, mode, label in LO_DATASETS:
        path = ROOT / "label_overlap" / ds / split / slug / f"label_overlap_{mode}_n10_cf0.3.json"
        if not path.is_file():
            print(f"missing {path}")
            return None
        curves[label] = np.asarray(json.loads(path.read_text())["label_overlap"], dtype=float)
    return curves


def probe_accuracy(model, dataset, chance, method):
    """Chance-adjusted per-layer accuracy, (acc - chance) / (max(acc) - chance), or None."""
    path = ROOT / "logit_lens" / f"{MODELS[model][1]}_{dataset}_images_qa_{method}.safetensors"
    if not path.is_file():
        return None
    acc = load_file(str(path))["accuracies"].astype(float)
    top = acc.max()
    return (acc - chance) / ((top - chance) if top > chance else 1.0)


def mean_probe_accuracy(model, datasets, method):
    curves = [c for ds, _, chance in datasets if (c := probe_accuracy(model, ds, chance, method)) is not None]
    if not curves:
        return None
    n = min(len(c) for c in curves)
    return np.mean([c[:n] for c in curves], axis=0)


def sampled(values):
    """Every other layer (odd layer indices), as in the paper."""
    return values[1::2]


def depth(values):
    return np.linspace(0.0, 1.0, len(values))


# ----------------------------------------------------------------------------- plotting
def draw_band_layers(ax, model):
    """Band on a 0-based layer-index axis: layer l (1-based) is at index l - 1."""
    lo, hi = MODELS[model][3]
    ax.axvspan(lo - 1.5, hi - 0.5, color=BAND_COLOR, alpha=BAND_ALPHA, zorder=0, lw=0)


def plot_main():
    with plt.rc_context(STYLE):
        fig, (ax_lo, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.0))
        for model in MAIN_MODELS:
            label, _, _, _, color = MODELS[model]
            curves = label_overlap_curves(model)
            if curves is not None:
                mean = sampled(np.mean(list(curves.values()), axis=0))
                ax_lo.plot(depth(mean), mean, color=color, marker="o", markersize=6, linewidth=2.3, label=label)
            ll = mean_probe_accuracy(model, MC_DATASETS, "logit_lens")
            lp = mean_probe_accuracy(model, MC_DATASETS, "linear_probe")
            if ll is not None:
                ax_pr.plot(depth(sampled(ll)), sampled(ll), color=color, marker="o", markersize=6, linewidth=2.3)
            if lp is not None:
                ax_pr.plot(depth(sampled(lp)), sampled(lp), color=color, marker="s", markersize=6, linewidth=2.3, linestyle="--")
        for ax, ylabel in ((ax_lo, "Label overlap"), (ax_pr, "Accuracy")):
            ax.set_ylim(-0.11, 1.05)
            ax.set_xlabel("Normalized depth")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        ax_pr.legend(handles=[Line2D([0], [0], color="black", marker="o", markersize=5, linewidth=2, label="Logit Lens"),
                              Line2D([0], [0], color="black", marker="s", markersize=5, linewidth=2, linestyle="--", label="Linear Probe")],
                     loc="lower right", frameon=False)
        fig.legend(handles=[Patch(facecolor=MODELS[m][4], label=MODELS[m][0]) for m in MAIN_MODELS],
                   loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
        save(fig, "probing/label_overlap_and_logit_lens.pdf")
        plt.close(fig)


def plot_label_overlap_average():
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.0, 4.4))
        for model in MODELS:
            curves = label_overlap_curves(model)
            if curves is None:
                continue
            mean = sampled(np.mean(list(curves.values()), axis=0))
            ax.plot(depth(mean), mean, color=MODELS[model][4], marker="o", markersize=5, linewidth=2)
        ax.set_xlabel("Normalized depth")
        ax.set_ylabel("Label overlap")
        ax.set_ylim(0.0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(handles=[Patch(facecolor=MODELS[m][4], label=MODELS[m][0]) for m in MODELS],
                  loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
        save(fig, "probing/label_overlap_average.pdf")
        plt.close(fig)


def plot_label_overlap_by_dataset():
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, model in zip(axes.flat, MODELS):
            draw_band_layers(ax, model)
            curves = label_overlap_curves(model) or {}
            for color, values in zip(PALETTE, curves.values()):
                ax.plot(np.arange(len(values))[1::2], sampled(values), color=color, marker="o", markersize=5, linewidth=2)
            ax.set_title(MODELS[model][0])
            ax.set_xlabel("Layer")
            ax.set_ylim(0.0, 1.1)
            ax.grid(True, alpha=0.3)
        for ax in axes[:, 0]:
            ax.set_ylabel("Label overlap")
        fig.legend(handles=[Patch(facecolor=c, label=d[3]) for c, d in zip(PALETTE, LO_DATASETS)],
                   loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False)
        save(fig, "probing/label_overlap_by_dataset.pdf")
        plt.close(fig)


def plot_probe_grid(datasets, method, palette, ylim, filename):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, model in zip(axes.flat, MODELS):
            draw_band_layers(ax, model)
            for color, (ds, _, chance) in zip(palette, datasets):
                acc = probe_accuracy(model, ds, chance, method)
                if acc is not None:
                    ax.plot(np.arange(len(acc))[1::2], sampled(acc), color=color, marker="o", markersize=5, linewidth=2)
            n = MODELS[model][2]
            ax.set_xticks(range(0, n + 1, 4 if n <= 36 else 8))
            ax.set_ylim(*ylim)
            ax.set_title(MODELS[model][0])
            ax.set_xlabel("Layers")
            ax.grid(True, alpha=0.3)
        for ax in axes[:, 0]:
            ax.set_ylabel("Accuracy")
        fig.legend(handles=[Patch(facecolor=c, label=d[1]) for c, d in zip(palette, datasets)],
                   loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False)
        save(fig, filename)
        plt.close(fig)


if __name__ == "__main__":
    plot_main()
    plot_label_overlap_average()
    plot_label_overlap_by_dataset()
    plot_probe_grid(MC_DATASETS, "linear_probe", PALETTE, (-0.37, 1.05), "probing/linear_probe_mc.pdf")
    plot_probe_grid(MC_DATASETS, "logit_lens", PALETTE, (-0.30, 1.05), "probing/logit_lens_mc.pdf")
    plot_probe_grid(OE_DATASETS, "logit_lens", OE_PALETTE, (-0.05, 1.05), "probing/logit_lens_oe.pdf")
