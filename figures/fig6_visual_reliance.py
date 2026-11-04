"""Figure 6 and the Llama-90B appendix panel: localized fine-tuning advantage vs image reliance.

Benchmark scores are read from results/benchmarks/<model>/<task>/<task>_results.jsonl (scripts/finetuning/benchmarks.py)
and results/benchmarks/<model>/compositionality/*.jsonl (scripts/finetuning/compositionality.py).

  x: image reliance of a benchmark = (score with image - score without image) / score with image,
     averaged over four reference VLMs (InternVL2-8B, OneVision-8B, Cambrian-8B, Pixtral-12B);
     models scoring below half of the best reference model on a benchmark are left out of that average.
  y: (score of the model fine-tuned on the abstraction layers - score of the model fine-tuned on the
     complementary layers) / score of the model fine-tuned on the abstraction layers.

Writes figures/out/finetuning/visual_reliance_correlation.pdf (LLaVA-7B [6,16] vs its complement, OneVision-4B
[8,25] vs its complement) and figures/out/llama90b/visual_reliance_correlation.pdf, where the "fine-tuned" models
are replaced by Llama-3.2-90B-Vision with cross-attention kept only inside / only outside layers [10, 65)
(results/interventions/llama90b/Llama-3.2-90B-Vision-Instruct_cross_modal_win10_65/<task>/<task>_{keep,remove}.json).
Run: python figures/fig6_visual_reliance.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from matplotlib.lines import Line2D
from scipy import stats

from savefig import save

ROOT = Path(__file__).resolve().parents[1] / "results"

# benchmark -> (results directory under results/benchmarks/<model>/, score key, scale, label, category),
# in the order of the radar plot
BENCHMARKS = {
    "POPE":         ("pope",              "pope_accuracy,none",          1,    "POPE",         "mc"),
    "ScienceQA":    ("scienceqa_img",     "exact_match,none",            1,    "SQA",          "mc"),
    "GQA":          ("gqa",               "exact_match,none",            1,    "GQA",          "oe"),
    "MME":          ("mme",               "mme_total",                   2800, "MME",          "mc"),
    "ARO":          ("compositionality/aro",        "accuracy", 1,       "ARO",          "comp"),
    "SugarCrepe":   ("compositionality/sugarcrepe", "mean",     1,       "SugarCrepe",   "comp"),
    "Flickr":       ("flickr30k",         "flickr_CIDEr,none",           1,    "Flickr",       "cap"),
    "COCO":         ("coco2017_cap_val",  "coco_CIDEr,none",             1,    "COCO",         "cap"),
    "CountBench":   ("countbench",        "acc,none",                    1,    "CountBench",   "oe"),
    "RealWorldQA":  ("realworldqa",       "exact_match,flexible-extract", 1,   "RWQA",         "mc"),
    "MMMU-Pro":     ("mmmu_pro_standard", "mmmu_acc,none",               1,    "MMMU-Pro-S",   "mc"),
    "MMStar":       ("mmstar",            "average,none",                1,    "MMStar",       "mc"),
    "ChartQA":      ("chartqa",           "relaxed_overall,none",        1,    "ChartQA",      "oe"),
    "MMMU":         ("mmmu_val",          "mmmu_acc,none",               1,    "MMMU",         "mc"),
    "Naturalbench": ("naturalbench",      "naturalbench_G_ACC,none",     1,    "Naturalbench", "mc"),
    "VQAv2":        ("vqav2_val",         "exact_match,none",            1,    "VQAv2",        "oe"),
    "MMBench-CN":   ("mmbench_cn_dev",    "gpt_eval_score,none",         100,  "MMB-CN",       "mc"),
    "MMBench-EN":   ("mmbench_en_dev",    "gpt_eval_score,none",         100,  "MMB-EN",       "mc"),
    "SEEDBench":    ("seedbench",         "seed_all,none",               1,    "SEED",         "mc"),
    "CVBench":      ("cv_bench",          "cv_bench_acc,none",           1,    "CVBench",      "mc"),
}
REFERENCE_MODELS = ["InternVL2-8B", "LLaVA-OneVision-1.5-8B-Instruct", "cambrian-8b", "pixtral-12b"]
PAIRS = {   # title -> (abstraction-layer fine-tuning, complementary fine-tuning)
    "LLaVA-7B":     ("llava_7b_6_17", "llava_7b_6_17_complementary"),
    "OneVision-4B": ("LLaVA-OneVision-1.5-4B-8-26", "LLaVA-OneVision-1.5-4B-8-26-comp"),
}
EXCLUDED = {"cap"}   # captioning is impossible without the image
CATEGORY = {"comp": ("Compositional", "#e76f51", "^"), "oe": ("Open-ended", "#e9c46a", "o"), "mc": ("Multiple choice", "#669bbc", "s")}

LLAMA_DIR = ROOT / "interventions" / "llama90b" / "Llama-3.2-90B-Vision-Instruct_cross_modal_win10_65"
LLAMA_TASKS = {  # benchmark -> (task directory, score key)
    "ChartQA": ("chartqa", "relaxed_overall,none"), "CountBench": ("countbench", "acc,none"),
    "CVBench": ("cv_bench", "cv_bench_acc,none"), "GQA": ("gqa", "exact_match,none"),
    "MMBench-CN": ("mmbench_cn_dev", "gpt_eval_score,none"), "MMBench-EN": ("mmbench_en_dev", "gpt_eval_score,none"),
    "MME": ("mme", "mme_perception_score,none"), "MMMU": ("mmmu_val", "mmmu_acc,none"),
    "Naturalbench": ("naturalbench", "naturalbench_G_ACC,none"), "ScienceQA": ("scienceqa_img", "exact_match,none"),
    "SEEDBench": ("seedbench", "seed_image,none"), "VQAv2": ("vqav2_val", "exact_match,none"),
}
STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
    "legend.fontsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
}


# ----------------------------------------------------------------------------- scores
def _last_record(path):
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    return json.loads(lines[-1]) if lines else None


def score(model, benchmark):
    """Benchmark score of a model in [0, 1], or None if not evaluated."""
    subdir, key, scale, _, _ = BENCHMARKS[benchmark]
    bench_dir = ROOT / "benchmarks" / model / subdir
    if not bench_dir.is_dir():
        return None
    files = sorted(bench_dir.glob("*results*.jsonl"))
    if not files:
        return None
    if benchmark == "ARO":            # mean over the four ARO sub-tasks, one file each
        vals = [r[key] for f in files if (r := _last_record(f)) and key in r]
        return float(np.mean(vals)) / scale if vals else None
    rec = _last_record(files[0])
    if rec is None:
        return None
    if benchmark == "MME":
        return (rec.get("mme_perception_score,none", 0) + rec.get("mme_cognition_score,none", 0)) / scale
    return rec[key] / scale if key in rec else None


def image_reliance():
    """Benchmark -> image reliance averaged over the reference models."""
    out = {}
    for b in BENCHMARKS:
        with_img = np.array([score(m, b) for m in REFERENCE_MODELS], dtype=float)
        no_img = np.array([score(m + "_noimage", b) for m in REFERENCE_MODELS], dtype=float)
        ok = np.isfinite(with_img) & np.isfinite(no_img)
        if not ok.any():
            continue
        with_img, no_img = with_img[ok], no_img[ok]
        keep = with_img >= 0.5 * with_img.max()        # drop reference models that fail the benchmark
        rel = (with_img[keep] - no_img[keep]) / with_img[keep]
        out[b] = float(rel[np.isfinite(rel)].mean())
    return out


def localized_advantage(loc_model, comp_model):
    out = {}
    for b in BENCHMARKS:
        s_loc, s_comp = score(loc_model, b), score(comp_model, b)
        if s_loc and s_comp is not None:
            out[b] = (s_loc - s_comp) / s_loc
    return out


def llama_advantage():
    out = {}
    for b, (task, key) in LLAMA_TASKS.items():
        keep, remove = LLAMA_DIR / task / f"{task}_keep.json", LLAMA_DIR / task / f"{task}_remove.json"
        if keep.is_file() and remove.is_file():
            k = json.loads(keep.read_text())["results"][task][key]
            r = json.loads(remove.read_text())["results"][task][key]
            out[b] = (k - r) / k
    return out


# ----------------------------------------------------------------------------- plotting
def correlation_panel(ax, reliance, advantage, title):
    bench = [b for b in BENCHMARKS if BENCHMARKS[b][4] not in EXCLUDED and b in reliance and b in advantage]
    if len(bench) < 3:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    x = np.array([reliance[b] for b in bench])
    y = np.array([advantage[b] for b in bench])
    rho, p = stats.spearmanr(x, y, alternative="greater")
    for b, xi, yi in zip(bench, x, y):
        _, color, marker = CATEGORY[BENCHMARKS[b][4]]
        ax.scatter(xi, yi, s=40, color=color, marker=marker, edgecolor="white", linewidth=0.6, zorder=3)
    # labels moved apart with adjustText (1.3), as in the paper's figure, with grey leader lines where they had to
    # move; the markers are obstacles too, and a fixed number of iterations keeps the layout reproducible
    texts = [ax.text(xi, yi, BENCHMARKS[b][3], fontsize=10) for b, xi, yi in zip(bench, x, y)]
    adjust_text(texts, x=x, y=y, ax=ax, force_text=(0.9, 1.1), force_static=(0.3, 0.5), force_pull=(0.02, 0.02),
                expand=(1.2, 1.4), iter_lim=1000, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5, lw=0.5))
    xpad, ypad = (x.max() - x.min()) * 0.08, (y.max() - y.min()) * 0.08
    ax.set_xlim(x.min() - xpad, x.max() + xpad)
    ax.set_ylim(y.min() - ypad, y.max() + ypad)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.3)
    ax.text(0.96, 0.04, f"Spearman\n$\\varrho={rho:.2f}$\n$p={p:.3f}$", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="lightgray", alpha=0.9))
    ax.set_xlabel("Image reliance")
    ax.set_ylabel("Localized ft advantage")
    ax.set_title(title)
    ax.grid(True, alpha=0.15)


def category_legend(fig, anchor):
    handles = [Line2D([0], [0], marker="s", linestyle="", color="none", label="Task type:")]
    handles += [Line2D([0], [0], marker=m, linestyle="", color=c, markeredgecolor="white", markersize=9, label=n)
                for n, c, m in CATEGORY.values()]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=anchor, frameon=False, handletextpad=0.4)


if __name__ == "__main__":
    reliance = image_reliance()
    print("image reliance:", {b: round(v, 3) for b, v in reliance.items()})
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, (title, (loc, comp)) in zip(axes, PAIRS.items()):
            correlation_panel(ax, reliance, localized_advantage(loc, comp), title)
        category_legend(fig, (0.5, -0.06))
        fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.18, wspace=0.22)
        save(fig, "finetuning/visual_reliance_correlation.pdf")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        correlation_panel(ax, reliance, llama_advantage(), "Llama-3.2-Vision-90B")
        category_legend(fig, (0.5, -0.04))
        fig.subplots_adjust(left=0.1, right=0.98, top=0.92, bottom=0.2)
        save(fig, "llama90b/visual_reliance_correlation.pdf")
        plt.close(fig)
