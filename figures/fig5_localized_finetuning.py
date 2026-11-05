"""Figure 5 (table of localized fine-tuning results), the sliding-window plot and the appendix radar plot.

Benchmark scores are read from results/benchmarks/<model>/... (see fig6_visual_reliance.py for the layout).
For each fine-tuning variant the score is divided by the score of the fully fine-tuned model:

  * prints the LaTeX table of Fig. 5 (averages over multiple-choice VQA, open-ended VQA,
    compositional and captioning benchmarks),
  * writes figures/out/finetuning/radar_localized_ft.pdf (per-benchmark normalized scores of the
    abstraction-layer and complementary fine-tuning runs of LLaVA-7B and OneVision-4B),
  * writes figures/out/finetuning/sliding_window.pdf (overall normalized score of the runs that fine-tune
    8 consecutive layers, moved by 4 layers at a time, against the normalized depth of the window centre;
    dashed lines bound the abstraction region of each model).

Run: python figures/fig5_localized_finetuning.py
"""

from math import pi

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from fig3_probing import MODELS as PROBING_MODELS, STYLE as PROBING_STYLE
from fig6_visual_reliance import BENCHMARKS, STYLE, score
from savefig import save

# title -> (fully fine-tuned reference, abstraction-layer fine-tuning, complementary fine-tuning, layer labels).
# The references are the models fine-tuned on all layers with the same recipe: LLaVA-1.5-7B itself
# (665k mixture) and lbasile/LLaVA-OneVision-1.5-4B (OneVision-1.5-4B on the LLaVA-NeXT 780k mixture).
MODELS = {
    "LLaVA-7B":     ("llava_7b", "llava_7b_6_17", "llava_7b_6_17_complementary", ("$[6, 16]$", "$[6, 16]^C$")),
    "OneVision-4B": ("LLaVA-OneVision-1.5-4B", "LLaVA-OneVision-1.5-4B-8-26", "LLaVA-OneVision-1.5-4B-8-26-comp", ("$[8, 25]$", "$[8, 25]^C$")),
}
CATEGORIES = {"mc": "MC VQA", "oe": "OE VQA", "comp": "Compositional", "cap": "Captioning"}

# title -> (fully fine-tuned reference, run name from first and one-past-last trained layer, fig3_probing model key).
# Each run fine-tunes WINDOW consecutive layers (lbasile/llava_7b_0_8, lbasile/llava_7b_4_12, ...).
SLIDING = {
    "LLaVA-7B":     ("llava_7b", "llava_7b_{}_{}", "llava_7b"),
    "OneVision-4B": ("LLaVA-OneVision-1.5-4B", "LLaVA-OneVision-1.5-4B-{}-{}", "onevision_4b"),
}
WINDOW, STRIDE = 8, 4


def normalized_scores(model, base):
    """Benchmark -> score(model) / score(base) for the benchmarks evaluated on both."""
    out = {}
    for b in BENCHMARKS:
        s, s0 = score(model, b), score(base, b)
        if s is not None and s0:
            out[b] = s / s0
    return out


def print_table():
    rows = {}
    for title, (base, loc, comp, labels) in MODELS.items():
        for model, label in zip((loc, comp), labels):
            norm = normalized_scores(model, base)
            cells = []
            for cat in CATEGORIES:
                vals = [v for b, v in norm.items() if BENCHMARKS[b][4] == cat]
                cells.append(f"{100 * sum(vals) / len(vals):.1f}" if vals else "-")
            cells.append(f"{100 * sum(norm.values()) / len(norm):.1f}" if norm else "-")
            rows[(title, label)] = cells
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{t}}}" for t in MODELS) + r" \\")
    print("Trained layers & " + " & ".join(l for _, _, _, labels in MODELS.values() for l in labels) + r" \\")
    print(r"\midrule")
    for i, cat in enumerate(list(CATEGORIES.values()) + ["Overall"]):
        if cat == "Overall":
            print(r"\midrule")
        print(f"{cat} $\\uparrow$ & " + " & ".join(rows[(t, l)][i] for t, (_, _, _, labels) in MODELS.items() for l in labels) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def radar():
    with plt.rc_context({**STYLE, "xtick.labelsize": 11, "ytick.labelsize": 8}):
        fig = plt.figure(figsize=(15, 8))
        for i, (title, (base, loc, comp, _)) in enumerate(MODELS.items()):
            ax = fig.add_subplot(1, 2, i + 1, projection="polar")
            ax.set_theta_zero_location("N")   # first benchmark at the top, clockwise
            ax.set_theta_direction(-1)
            bench = [b for b in BENCHMARKS if score(base, b)]
            n = len(bench)
            if n == 0:
                ax.set_title(f"{title} (no data)")
                continue
            angles = [k / n * 2 * pi for k in range(n)] + [0.0]
            for model, label, color, ls in ((base, "Base", "black", "--"), (loc, "Localized", "teal", "-"), (comp, "Complementary", "maroon", "-")):
                norm = normalized_scores(model, base)
                if not norm:
                    continue
                values = [norm.get(b, 0.0) for b in bench] + [norm.get(bench[0], 0.0)]
                ax.plot(angles, values, linewidth=1.5, linestyle=ls, color=color, label=label, alpha=0.85 if ls == "--" else 1)
                if ls == "-":
                    ax.fill(angles, values, alpha=0.06, color=color)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([BENCHMARKS[b][3] for b in bench])
            ax.tick_params(axis="x", pad=22)
            ax.set_yticks([0.8, 0.9, 1.0, 1.1])
            ax.set_ylim(0.75, 1.15)
            ax.set_title(title, pad=25)
            ax.grid(True, alpha=0.3)
        handles, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.02))
        fig.subplots_adjust(left=0.06, right=0.94, top=0.85, bottom=0.18, wspace=0.6)
        save(fig, "finetuning/radar_localized_ft.pdf")
        plt.close(fig)


def sliding_window():
    with plt.rc_context(PROBING_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        for title, (base, run, key) in SLIDING.items():
            _, _, n, (lo, hi), color = PROBING_MODELS[key]
            centers, overall = [], []
            for s in range(0, n - WINDOW + 1, STRIDE):
                norm = normalized_scores(run.format(s, s + WINDOW), base)
                if norm:
                    centers.append((s + WINDOW / 2) / n)
                    overall.append(100 * sum(norm.values()) / len(norm))
            ax.plot(centers, overall, color=color, marker="o", markersize=5, linewidth=2)
            # the abstraction region is numbered like the fine-tuned layers (llava_7b_6_17 trains [6, 16])
            for x in (lo, hi + 1):
                ax.axvline(x / n, color=color, linestyle="--", linewidth=1.0, zorder=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(84, 100)
        ax.set_xlabel("Normalized depth")
        ax.set_ylabel("Normalized performance (%)")
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)
        handles = [Patch(facecolor=PROBING_MODELS[key][4], label=title) for title, (_, _, key) in SLIDING.items()]
        handles.append(Line2D([0], [0], color="0.4", linestyle="--", linewidth=1.0, label="Abstraction phase boundary"))
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False,
                  columnspacing=1.0, handlelength=1.6)
        save(fig, "finetuning/sliding_window.pdf")
        plt.close(fig)


if __name__ == "__main__":
    print_table()
    radar()
    sliding_window()
