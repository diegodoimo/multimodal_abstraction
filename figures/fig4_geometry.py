"""Figure 4 and the geometry appendices: similarity between text-only and multimodal representations.

Reads results/imbalance_planes/<dataset>/<model>_{txt_img,img_txt}.json (scripts/geometry/imbalance_planes.py)
for the imbalance planes and results/similarity/<dataset>/<model>_<metric>.json
(scripts/geometry/similarity_multimodal.py) for linear CKA and neighborhood overlap, all computed on the
first 3,000 prompts of each dataset (matrix rows = text-prompt layers, columns = multimodal-prompt
layers; averaged over the five VQA datasets), and writes to figures/out/:

  geometry/information_imbalance/<model>.pdf  Fig. 4 (OneVision-4B, InternVL2-8B) and the appendix planes:
                                      green squares text -> image imbalance, purple circles image -> text;
                                      larger markers = lower imbalance (thresholds in the legend)
  geometry/linear_cka/<model>.pdf     the same plane for linear CKA, drawn like the imbalance plane
  geometry/neighborhood_overlap/<model>.pdf   the same for neighborhood overlap (k = 30)
  llama90b/information_imbalance.pdf  Llama-3.2-90B-Vision-Instruct (results/similarity/llama90b)

The CKA and overlap planes use the imbalance plane's marker areas, band and layout, with larger markers
for more similar layers; their three thresholds are chosen so that each marker size covers the same share
of the cells as in the model's imbalance plane (pre-trained and fine-tuned pooled, both directions),
rounded to 0.005.

The yellow band is the abstraction region of each model.  Run: python figures/fig4_geometry.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from savefig import save

RESULTS = Path(__file__).resolve().parents[1] / "results" / "similarity"

# key -> (title, fine-tuned id, pre-trained id, abstraction region (0-based layer indices, the layers of the
#         localized fine-tuning), imbalance thresholds, base marker area of the planes)
MODELS = {
    "llava_7b":     ("LLaVA-7B",     "llava_7b",  "llava-v1.5-7b-pretrained-lbasile",  (6, 16),  (0.025, 0.050, 0.070), 80),
    "llava_13b":    ("LLaVA-13B",    "llava_13b", "llava-v1.5-13b-pretrained-lbasile", (6, 16),  (0.020, 0.040, 0.060), 45),
    "onevision_4b": ("OneVision-4B", "LLaVA-OneVision-1.5-4B-Instruct", "LLaVA-OneVision-1.5-4B-Base", (8, 25), (0.040, 0.065, 0.090), 60),
    "onevision_8b": ("OneVision-8B", "LLaVA-OneVision-1.5-8B-Instruct", "LLaVA-OneVision-1.5-8B-Base", (8, 25), (0.060, 0.090, 0.120), 60),
    "internvl2_8b": ("InternVL2-8B", "InternVL2-8B", "InternVL2-8B-Pretrain",            (6, 20),  (0.020, 0.040, 0.060), 60),
    "cambrian_8b":  ("Cambrian-8B",  "cambrian-8b",  "cambrian-8b-pretrained",           (5, 16),  (0.025, 0.050, 0.070), 80),
}
PLANES = RESULTS.parent / "imbalance_planes"   # scripts/geometry/imbalance_planes.py: rows = text layers, cols = image layers
DATASETS = (
    "vqav2_coco_restval_icq_split01_human",
    "cocoqa_restval_icqa_split01_human",
    "gqa_val_balanced_noyesno_captioned_synthetic",
    "mmbench_en_dev_captioned_open_ended_synthetic",
    "seed_bench_images_captioned_open_ended_synthetic",
)
# Llama-3.2-90B-Vision-Instruct: 100 decoder layers, 20 of which are cross-attention layers that
# are absent from the text-only forward pass (80 text layers).
LLAMA = {
    "id": "Llama-3.2-90B-Vision-Instruct", "title": "Llama-3.2-90B-Vision-Instruct",
    "datasets": ("gqa_val_balanced_noyesno_captioned_synthetic", "mmbench_en_dev_captioned_open_ended_synthetic",
                 "vqav2_coco_restval_icq_human_split01"),
    "cross_attention_layers": (3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58, 63, 68, 73, 78, 83, 88, 93, 98),
    "thresholds": (0.04, 0.05, 0.06),
}

BAND_COLOR = "goldenrod"
T2I = dict(marker="s", edgecolors="#228833", facecolors="none", linewidths=0.85)   # Llama plane: text -> image
I2T = dict(marker="o", edgecolors="#7b2d8b", facecolors="none", linewidths=0.85)   # Llama plane: image -> text
PLANE_T2I = dict(marker="s", color="#007e5d", zorder=2)   # model planes: text -> image
PLANE_I2T = dict(marker="o", color="#991499", zorder=2)   # model planes: image -> text
# CKA and overlap planes: one matrix each, drawn with the imbalance planes' markers; dark colours give a
# contrast of 8 to 10 on the band
PLANE_SYMMETRIC = {"linear_cka": dict(marker="o", color="#08306b", zorder=2),
                   "neighborhood_overlap": dict(marker="o", color="#7f0000", zorder=2)}
SYMMETRIC_NAME = {"linear_cka": ("linear CKA", "CKA"), "neighborhood_overlap": (r"neighborhood overlap $\chi$ (k = 30)", r"$\chi$")}
MARKER_SCALE = 0.63
SIZE_FACTORS = (1.0, 0.25, 0.05)   # marker areas for the three buckets, relative to one grid cell

STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
}


def load_matrix(dataset, model_id, metric, root=RESULTS):
    """Matrix with rows = multimodal (image) layers and columns = text layers."""
    path = root / dataset / f"{model_id}_{metric}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    m = np.asarray(payload["matrix"], dtype=float)
    meta = payload.get("metadata", {})
    if meta.get("row_modality", "text") == "text":   # stored as rows = text, cols = image
        m = m.T
    return m


def average_matrix(model_id, metric, datasets=DATASETS, root=RESULTS):
    mats = [m for d in datasets if (m := load_matrix(d, model_id, metric, root)) is not None]
    if not mats:
        return None
    if len(mats) < len(datasets):
        print(f"warning: {model_id}/{metric}: only {len(mats)}/{len(datasets)} datasets found")
    return np.mean(mats, axis=0)


def marker_areas(n_layers, cell_in):
    """Marker areas of the three buckets for a panel of side cell_in inches."""
    s_max = (cell_in * 72 / n_layers) ** 2 * MARKER_SCALE
    return [s_max * f for f in SIZE_FACTORS]


def scatter_buckets(ax, matrix, buckets, style, areas):
    """One scatter call per bucket; buckets are (lo, hi) half-open intervals on the matrix values."""
    n_img, n_txt = matrix.shape
    img_idx, txt_idx = np.indices((n_img, n_txt))
    for (lo, hi), area in zip(buckets, areas):
        mask = (matrix >= lo) & (matrix < hi)
        if mask.any():
            ax.scatter(txt_idx[mask] + 1, img_idx[mask] + 1, s=area, zorder=3, **style)


def setup_axis(ax, n_txt, n_img, tick_step):
    ax.set_xlim(0.5, n_txt + 0.5)
    ax.set_ylim(0.5, n_img + 0.5)
    ax.set_xticks(range(1, n_txt + 1, tick_step))
    ax.set_yticks(range(1, n_img + 1, tick_step))
    ax.grid(True, linewidth=0.4, color="gray", alpha=0.5)
    ax.set_xlabel("Layer (text input)")


def size_legend(areas, labels):
    return [Line2D([], [], linestyle="None", marker="o", color="black", markerfacecolor="none",
                   markersize=np.sqrt(a), label=l) for a, l in zip(areas, labels)]


def average_plane(model_id, direction):
    """Mean over the five datasets of the k = 1 imbalance matrices (rows = text layers), or None."""
    mats = []
    for ds in DATASETS:
        path = PLANES / ds / f"{model_id}_{direction}.json"
        if not path.is_file():
            return None
        mats.append(np.array(json.loads(path.read_text())["matrix"], dtype=float))
    return np.mean(mats, axis=0)


def scatter_plane(ax, matrix, buckets, base_area, style):
    """Markers at (text layer, image layer) of a matrix with rows = text layers; buckets are half-open
    (lo, hi) intervals from the most to the least similar, drawn with areas base, base / 5, base / 25.

    A marker sits at the 0-based layer index, as in the paper's plotting script (imbalance_tueplots.py):
    the band is then the 0-based indices of the localized fine-tuning layers ([6, 16] for LLaVA). The
    first layer (index 0) falls outside the axis limits, as in the paper; no plane has a marker there."""
    n_txt, n_img = matrix.shape
    txt_idx, img_idx = np.indices((n_txt, n_img))
    for (lo, hi), area in zip(buckets, (base_area, base_area / 5, base_area / 25)):
        mask = (matrix >= lo) & (matrix < hi)
        if mask.any():
            ax.scatter(txt_idx[mask], img_idx[mask], s=area, marker=style["marker"], facecolors="none",
                       edgecolors=style["color"], linewidths=0.8, zorder=style["zorder"])


def draw_planes(model, layers, top_handles, size_labels, filename):
    """Pre-trained | fine-tuned panels; layers[tag] = [(matrix with rows = text layers, buckets, style), ...]."""
    title, _, _, band, _, base_area = MODELS[model]
    n = layers["fine"][0][0].shape[0]
    lo, hi = band[0] - 0.5, band[1] + 0.5
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 5.2))
        for ax, tag, panel_title in zip(axes, ("pre", "fine"), ("Pretrained", "Finetuned")):
            for matrix, buckets, style in layers[tag]:
                scatter_plane(ax, matrix, buckets, base_area, style)
            ax.axvspan(lo, hi, alpha=0.10, color=BAND_COLOR, zorder=0)
            ax.axhspan(lo, hi, alpha=0.10, color=BAND_COLOR, zorder=0)
            ax.add_patch(Rectangle((lo, lo), hi - lo, hi - lo, alpha=0.20, facecolor=BAND_COLOR, edgecolor="none", zorder=0))
            ticks = np.arange(0, n, 4) + 1
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xlim(0.4, n + 1)
            ax.set_ylim(0.4, n + 1)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linewidth=0.5, alpha=0.5)
            ax.set_title(panel_title, fontsize=16)
            ax.set_xlabel("Layer (text input)", fontsize=15)
        axes[0].set_ylabel("Layer (multimodal input)", fontsize=15)
        fig.suptitle(title, fontsize=22, y=1.0)
        kw = dict(color="w", markerfacecolor="none", markeredgecolor="black", marker="o", linewidth=0.8)
        sizes = [Line2D([0], [0], markersize=np.sqrt(area), label=label, **kw)
                 for area, label in zip((base_area, base_area / 5, base_area / 25), size_labels)]
        leg = fig.legend(handles=top_handles, fontsize=15, loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, columnspacing=1.5)
        fig.add_artist(leg)
        fig.legend(handles=sizes, fontsize=15, loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False, columnspacing=1.0)
        fig.tight_layout()
        fig.subplots_adjust(top=0.93)
        save(fig, filename, dpi=300)
        plt.close(fig)


def imbalance_data(model):
    """{"pre": (text -> image, image -> text), "fine": (...)} averaged over the datasets, or None."""
    _, fine_id, pre_id = MODELS[model][:3]
    data = {}
    for tag, mid in (("pre", pre_id), ("fine", fine_id)):
        t2i, i2t = average_plane(mid, "txt_img"), average_plane(mid, "img_txt")
        if t2i is None or i2t is None:
            print(f"skipping {model}: missing imbalance data for {mid}")
            return None
        data[tag] = (t2i, i2t)
    return data


def plot_imbalance(model, filename):
    """Two panels (pre-trained | fine-tuned) of the information-imbalance plane of one model."""
    thr = MODELS[model][4]
    data = imbalance_data(model)
    if data is None:
        return
    buckets = [(0.0, thr[0]), (thr[0], thr[1]), (thr[1], thr[2])]
    kw = dict(color="w", markerfacecolor="none", linewidth=0)
    direction = [Line2D([0], [0], marker="s", markeredgecolor=PLANE_T2I["color"], markersize=8, label="text → image", **kw),
                 Line2D([0], [0], marker="o", markeredgecolor=PLANE_I2T["color"], markersize=8, label="image → text", **kw)]
    labels = [rf"$\Delta$ < {thr[0]:.3f}", rf"{thr[0]:.3f} ≤ $\Delta$ < {thr[1]:.3f}", rf"{thr[1]:.3f} ≤ $\Delta$ < {thr[2]:.3f}"]
    layers = {tag: [(data[tag][0], buckets, PLANE_T2I), (data[tag][1], buckets, PLANE_I2T)] for tag in data}
    draw_planes(model, layers, direction, labels, filename)


def matched_thresholds(values, imbalance_values, thr):
    """Thresholds t1 > t2 > t3 on a similarity (larger = more similar) such that values >= t_k cover the
    same share of cells as imbalance values < thr[k]; rounded to 0.005."""
    shares = [np.mean(imbalance_values < x) for x in thr]
    out = []
    for share in shares:
        t = np.quantile(values, 1 - share)
        t = round(t / 0.005) * 0.005
        if out and t >= out[-1]:
            t = out[-1] - 0.005
        out.append(t)
    return out


def plot_symmetric(model, metric, filename):
    """Pre-trained | fine-tuned panels of linear CKA or neighborhood overlap, drawn like the imbalance plane."""
    _, fine_id, pre_id, _, thr, _ = MODELS[model]
    imb = imbalance_data(model)
    data = {}
    for tag, mid in (("pre", pre_id), ("fine", fine_id)):
        m = average_matrix(mid, metric)
        if m is None or imb is None:
            print(f"skipping {model}/{metric}: missing data for {mid}")
            return
        data[tag] = m.T   # rows = text layers, as the imbalance planes
    pooled = np.concatenate([data["pre"].ravel(), data["fine"].ravel()])
    pooled_imb = np.concatenate([a.ravel() for tag in imb for a in imb[tag]])
    t1, t2, t3 = matched_thresholds(pooled, pooled_imb, thr)
    buckets = [(t1, np.inf), (t2, t1), (t3, t2)]
    style = PLANE_SYMMETRIC[metric]
    long_name, short = SYMMETRIC_NAME[metric]
    top = [Line2D([0], [0], marker=style["marker"], color="w", markerfacecolor="none", markeredgecolor=style["color"],
                  markersize=8, linewidth=0, label=long_name)]
    labels = [f"{short} ≥ {t1:.3f}", f"{t2:.3f} ≤ {short} < {t1:.3f}", f"{t3:.3f} ≤ {short} < {t2:.3f}"]
    draw_planes(model, {tag: [(data[tag], buckets, style)] for tag in data}, top, labels, filename)


def plot_llama():
    root = RESULTS / "llama90b"
    t2i = average_matrix(LLAMA["id"], "information_imbalance", LLAMA["datasets"], root)
    i2t = average_matrix(LLAMA["id"], "information_imbalance_reversed", LLAMA["datasets"], root)
    if t2i is None or i2t is None:
        print("skipping Llama-90B plane: results/similarity/llama90b not found")
        return

    def expand_text_axis(m):   # insert a copy of the previous column at each cross-attention layer
        cols, src = [], 0
        for layer in range(m.shape[1] + len(LLAMA["cross_attention_layers"])):
            if layer in LLAMA["cross_attention_layers"]:
                cols.append(cols[-1])
            else:
                cols.append(m[:, src])
                src += 1
        return np.stack(cols, axis=1)

    t2i, i2t = expand_text_axis(t2i), expand_text_axis(i2t)
    thr = LLAMA["thresholds"]
    buckets = [(0.0, thr[0]), (thr[0], thr[1]), (thr[1], thr[2])]
    labels = [rf"$\Delta$ < {thr[0]:.2f}", rf"{thr[0]:.2f} $\leq \Delta$ < {thr[1]:.2f}", rf"{thr[1]:.2f} $\leq \Delta$ < {thr[2]:.2f}"]
    n_img, n_txt = t2i.shape
    areas = marker_areas(n_img, cell_in=5.5)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        setup_axis(ax, n_txt, n_img, tick_step=10)
        scatter_buckets(ax, t2i, buckets, T2I, areas)
        scatter_buckets(ax, i2t, buckets, I2T, areas)
        ax.set_ylabel("Layer (multimodal input)")
        ax.set_aspect("equal", adjustable="box")
        fig.suptitle(LLAMA["title"], fontsize=18)
        kw = dict(linestyle="None", markerfacecolor="none", markersize=8)
        direction = [Line2D([], [], color="#228833", marker="s", label=r"text $\rightarrow$ image", **kw),
                     Line2D([], [], color="#7b2d8b", marker="o", label=r"image $\rightarrow$ text", **kw)]
        fig.legend(handles=direction, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
        fig.legend(handles=size_legend(areas, labels), loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)
        save(fig, "llama90b/information_imbalance.pdf")
        plt.close(fig)


if __name__ == "__main__":
    for model in MODELS:
        name = model.replace("onevision", "llava_ov")   # file names of the paper
        plot_imbalance(model, f"geometry/information_imbalance/{name}.pdf")
        plot_symmetric(model, "linear_cka", f"geometry/linear_cka/{name}.pdf")
        plot_symmetric(model, "neighborhood_overlap", f"geometry/neighborhood_overlap/{name}.pdf")
    plot_llama()
