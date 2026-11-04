"""COCO image <-> caption retrieval from cached per-layer representations.

Row i of the image folder (mean over image-token positions, "imgavg") is the
positive for row i of the text folder (mean over caption tokens, "textavg").
For every layer saved by scripts/probing/extract_coco_embeddings.py the script ranks
all N candidates with Euclidean distance and with cosine similarity, in both
directions (image -> text and text -> image), and reports NDCG@k, mean rank,
mean percentile rank (= AUC, higher is better) and MRR.

Output (read by figures/app_retrieval.py):
  <out-dir>/<model>/retrieval_<suffix1>_vs_<suffix2>.json
      results[metric][part][direction][layer] -> {ndcg@k..., mean_rank, mean_percentile_rank, mrr}

Example:
  python scripts/probing/retrieval.py --model llava_7b --representations-root representations/coco \
      --suffix1 imgavg --suffix2 textavg --out-dir results/retrieval
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root

import argparse
import json
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F

RANKING_METRICS = ("euclidean", "cosine")
FOLDER1 = "coco_image_only"  # image prompts
FOLDER2 = "coco_text_only"   # caption prompts
PART_TYPES = ["res2"]        # residual stream at the output of each decoder block
KS = [10, 30, 100, 300, 1000]  # cutoffs of NDCG@k
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="Model folder under --representations-root, e.g. llava_7b.")
    parser.add_argument("--representations-root", type=str, default="representations/coco",
                        help="Root with <model>/<folder>/layers/l{layer}{part}_{suffix}.pt files.")
    parser.add_argument("--suffix1", type=str, default="imgavg", help=f"File suffix in {FOLDER1} (pooling over image tokens).")
    parser.add_argument("--suffix2", type=str, default="textavg", help=f"File suffix in {FOLDER2} (pooling over caption tokens).")
    parser.add_argument("--out-dir", type=str, default="results/retrieval")
    return parser.parse_args()


def layer_file(folder: str, layer_id: int, part: str, suffix: str) -> Path:
    return Path(folder) / "layers" / f"l{layer_id}{part}_{suffix}.pt"


def available_layers(folder: str, part: str, suffix: str) -> list[int]:
    pattern = re.compile(rf"^l(\d+){re.escape(part)}_{re.escape(suffix)}\.pt$")
    layers_dir = Path(folder) / "layers"
    if not layers_dir.is_dir():
        return []
    return sorted(int(m.group(1)) for f in layers_dir.iterdir() if (m := pattern.match(f.name)))


@torch.inference_mode()
def retrieval_metrics_paired(emb_q: torch.Tensor, emb_d: torch.Tensor, ks, metric: str, device: str):
    """Single-relevance retrieval: the positive of query i is document i."""
    assert emb_q.shape == emb_d.shape, f"shape mismatch: {emb_q.shape} vs {emb_d.shape}"
    n = emb_q.shape[0]
    emb_q = emb_q.float().to(device)
    emb_d = emb_d.float().to(device)
    diag = torch.arange(n, device=device)
    if metric == "euclidean":
        scores = torch.cdist(emb_q, emb_d)                     # lower is better
        rank = 1 + (scores < scores[diag, diag].unsqueeze(1)).sum(dim=1).long()
    else:
        emb_q = F.normalize(emb_q, p=2, dim=1)
        emb_d = F.normalize(emb_d, p=2, dim=1)
        scores = emb_q @ emb_d.T                                # higher is better
        rank = 1 + (scores > scores[diag, diag].unsqueeze(1)).sum(dim=1).long()
    rank_f = rank.float()
    log_term = 1.0 / torch.log2(rank_f + 1.0)
    out = {str(k): torch.where(rank <= k, log_term, torch.zeros_like(log_term)).mean().item() for k in ks}
    out["mean_rank"] = rank_f.mean().item()
    out["mean_percentile_rank"] = (1.0 - (rank_f - 1.0) / (n - 1)).mean().item()
    out["mrr"] = (1.0 / rank_f).mean().item()
    return out


def main():
    args = parse_args()
    folder1 = os.path.join(args.representations_root, args.model, FOLDER1)
    folder2 = os.path.join(args.representations_root, args.model, FOLDER2)
    d1, d2 = f"{args.suffix1}_to_{args.suffix2}", f"{args.suffix2}_to_{args.suffix1}"
    results = {metric: {} for metric in RANKING_METRICS}
    for part in PART_TYPES:
        layers = sorted(set(available_layers(folder1, part, args.suffix1)) & set(available_layers(folder2, part, args.suffix2)))
        for layer_id in layers:
            e1 = torch.load(layer_file(folder1, layer_id, part, args.suffix1), map_location="cpu")
            e2 = torch.load(layer_file(folder2, layer_id, part, args.suffix2), map_location="cpu")
            line = f"[layer {layer_id}, part {part}]"
            for metric in RANKING_METRICS:
                m12 = retrieval_metrics_paired(e1, e2, KS, metric, DEVICE)
                m21 = retrieval_metrics_paired(e2, e1, KS, metric, DEVICE)
                results[metric].setdefault(part, {}).setdefault(d1, {})[str(layer_id)] = m12
                results[metric][part].setdefault(d2, {})[str(layer_id)] = m21
                line += f"  [{metric}] mpr {d1}={m12['mean_percentile_rank']:.4f} {d2}={m21['mean_percentile_rank']:.4f}"
            print(line)
    if not any(results[m] for m in RANKING_METRICS):
        raise SystemExit(f"No layer files found under {folder1} / {folder2}.")
    out_path = os.path.join(args.out_dir, args.model, f"retrieval_{args.suffix1}_vs_{args.suffix2}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "model": args.model, "folder1": folder1, "folder2": folder2,
        "suffix1": args.suffix1, "suffix2": args.suffix2, "ks": KS,
        "metrics": list(RANKING_METRICS), "part_types_evaluated": PART_TYPES,
        "description": "results[metric][part][direction][layer]: ndcg@k (keys = k), mean_rank, "
                       "mean_percentile_rank (in [0,1], higher better), mrr.",
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
