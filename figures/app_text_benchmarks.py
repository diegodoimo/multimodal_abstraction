"""Appendix tables of the text-only benchmarks (LaTeX on stdout).

  tab:text_benchmarks     the six models of the paper; the full-model point of the text-intervention sweeps
                          (results/interventions/text/<model>/<task>/<task>_L00_W0_0shot.json)
  tab:text_benchmarks_ft  localized fine-tuning: LLaVA-7B and OneVision-4B fine-tuned on all layers, on the
                          abstraction layers and on their complement
                          (results/text_benchmarks/<model>/<task>/<task>_L00_W0_0shot.json, slurm/text_benchmarks.sbatch)

Both use the same evaluation: lm-evaluation-harness, zero-shot, full evaluation sets, each model's chat
template with the system turn of that template. Run: python figures/app_text_benchmarks.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results"

# row label -> (lm-eval task, score key, metric label)
TASKS = {
    "ARC-Easy": ("arc_easy", "acc_norm,none", r"acc\_norm"),
    "CommonsenseQA": ("commonsense_qa", "acc,none", "acc"),
    "HellaSwag": ("hellaswag", "acc_norm,none", r"acc\_norm"),
    "MMLU": ("mmlu", "acc,none", "acc"),
    "PIQA": ("piqa", "acc_norm,none", r"acc\_norm"),
    "Winogrande": ("winogrande", "acc,none", "acc"),
}
# tab:text_benchmarks: column label -> results/interventions/text/<model>
PAPER_MODELS = {
    "LLaVA-7B": "llava_7b", "LLaVA-13B": "llava_13b", "Cambrian-8B": "cambrian-8b", "InternVL2-8B": "InternVL2-8B",
    "LLaVA-OV-1.5-4B": "LLaVA-OneVision-1.5-4B-Instruct", "LLaVA-OV-1.5-8B": "LLaVA-OneVision-1.5-8B-Instruct",
}
# tab:text_benchmarks_ft: model -> [(trained layers, results/text_benchmarks/<model>)], fully fine-tuned first
FT_MODELS = {
    "LLaVA-7B": [("all", "llava_7b"), ("$[6, 16]$", "llava_7b_6_17"), ("$[6, 16]^C$", "llava_7b_6_17_complementary")],
    "OneVision-4B": [("all", "LLaVA-OneVision-1.5-4B"), ("$[8, 25]$", "LLaVA-OneVision-1.5-4B-8-26"),
                     ("$[8, 25]^C$", "LLaVA-OneVision-1.5-4B-8-26-comp")],
}


def score(root, model, row):
    """Score in % of the full model, or None if the file is missing."""
    task, key, _ = TASKS[row]
    path = root / model / task / f"{task}_L00_W0_0shot.json"
    if not path.is_file():
        return None
    return 100 * json.loads(path.read_text())["results"][task][key]


def cell(v):
    return "--" if v is None else f"{v:.2f}"


def paper_table():
    root = ROOT / "interventions" / "text"
    print(r"\begin{tabular}{ll|" + "c" * len(PAPER_MODELS) + "}")
    print(r"\toprule")
    print("Task & Metric & " + " & ".join(PAPER_MODELS) + r" \\")
    print(r"\midrule")
    for row, (_, _, metric) in TASKS.items():
        print(f"{row} & {metric} & " + " & ".join(cell(score(root, m, row)) for m in PAPER_MODELS.values()) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def ft_table():
    root = ROOT / "text_benchmarks"
    columns = [(title, label, model) for title, runs in FT_MODELS.items() for label, model in runs]
    scores = {model: {row: score(root, model, row) for row in TASKS} for _, _, model in columns}
    print(r"\begin{tabular}{ll|" + "|".join("ccc" for _ in FT_MODELS) + "}")
    print(r"\toprule")
    heads = [rf"\multicolumn{{3}}{{c{'|' if i < len(FT_MODELS) - 1 else ''}}}{{{t}}}" for i, t in enumerate(FT_MODELS)]
    print("Task & Metric & " + " & ".join(heads) + r" \\")
    print(" & Trained layers & " + " & ".join(label for _, label, _ in columns) + r" \\")
    print(r"\midrule")
    for row, (_, _, metric) in TASKS.items():
        print(f"{row} & {metric} & " + " & ".join(cell(scores[m][row]) for _, _, m in columns) + r" \\")
    print(r"\midrule")
    means, relative = [], []
    for title, label, model in columns:
        vals = [scores[model][r] for r in TASKS]
        base = scores[FT_MODELS[title][0][1]]
        complete = all(v is not None for v in vals) and all(base[r] for r in TASKS)
        means.append(cell(sum(vals) / len(vals)) if complete else "--")
        relative.append(f"{100 * sum(scores[model][r] / base[r] for r in TASKS) / len(TASKS):.1f}" if complete else "--")
    print("Mean & & " + " & ".join(means) + r" \\")
    print(r"Relative to all layers (\%) & & " + " & ".join(relative) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    print("% tab:text_benchmarks")
    paper_table()
    print("\n% tab:text_benchmarks_ft")
    ft_table()
