"""Information imbalance between text-only and multimodal residual streams, layer by layer
(Fig. 4 and the appendix imbalance planes).

For every pair (text layer i, image layer j) the imbalance Delta(i -> j) is the mean rank, in the
image-layer space, of each point's first nearest neighbour in the text-layer space, normalised by
N / 2 (k = 1); Delta(j -> i) likewise. Representations are the last-token residual streams cached
by scripts/geometry/similarity_multimodal.py --save-representations
(<representations-dir>/representations/<model>/<dataset_prefix>_{txt,img}/output_layer/last/
unique_images/representations.safetensors); the first --n-samples prompts of each dataset are used.
Writes <results-dir>/<dataset_prefix>/<model>_{txt_img,img_txt}.json, each holding
{"matrix": rows = text layers, cols = image layers, "metadata": {...}}.
"""
import argparse
import json
import os

import numpy as np
import torch
from safetensors import safe_open

DATASETS = (
    "vqav2_coco_restval_icq_split01_human",
    "cocoqa_restval_icqa_split01_human",
    "gqa_val_balanced_noyesno_captioned_synthetic",
    "mmbench_en_dev_captioned_open_ended_synthetic",
    "seed_bench_images_captioned_open_ended_synthetic",
)


def load_layers(path: str, n_samples: int) -> torch.Tensor:
    """(layers, min(N, n_samples), dim) tensor from a cached representations file."""
    with safe_open(path, framework="pt") as f:
        key = list(f.keys())[0]
        tensor = f.get_tensor(key)
    return tensor[:, :n_samples]


@torch.inference_mode()
def imbalance_first_nn(x1: torch.Tensor, x2: torch.Tensor) -> tuple[float, float]:
    """(Delta(1 -> 2), Delta(2 -> 1)): rank in one space of the first neighbour in the other, averaged
    over the points and divided by N / 2 (a rank of N / 2 is the expectation for unrelated spaces)."""
    x1, x2 = x1.float().cuda(), x2.float().cuda()
    n = x1.shape[0]
    d1, d2 = torch.cdist(x1, x1), torch.cdist(x2, x2)
    rows = torch.arange(n, device=d1.device)
    nn1 = torch.topk(d1, k=2, dim=1, largest=False).indices[:, 1]   # first neighbour (index 0 = self)
    ranks_1_in_2 = (d2 < d2[rows, nn1].unsqueeze(1)).sum(dim=1).double()
    nn2 = torch.topk(d2, k=2, dim=1, largest=False).indices[:, 1]
    ranks_2_in_1 = (d1 < d1[rows, nn2].unsqueeze(1)).sum(dim=1).double()
    return (2.0 * ranks_1_in_2.mean() / n).item(), (2.0 * ranks_2_in_1.mean() / n).item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--representations-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), help="dataset prefixes of the cache")
    parser.add_argument("--n-samples", type=int, default=3000, help="prompts per dataset (the first ones)")
    parser.add_argument("--force", action="store_true", help="recompute existing results")
    args = parser.parse_args()

    model = os.path.basename(args.model_name_or_path.rstrip("/"))
    for prefix in args.datasets:
        out_dir = os.path.join(args.results_dir, prefix)
        outputs = {d: os.path.join(out_dir, f"{model}_{d}.json") for d in ("txt_img", "img_txt")}
        if not args.force and all(os.path.isfile(p) for p in outputs.values()):
            print(f"{model} / {prefix}: already computed")
            continue
        cache = os.path.join(args.representations_dir, "representations", model)
        txt = load_layers(os.path.join(cache, f"{prefix}_txt", "output_layer", "last", "unique_images", "representations.safetensors"), args.n_samples)
        img = load_layers(os.path.join(cache, f"{prefix}_img", "output_layer", "last", "unique_images", "representations.safetensors"), args.n_samples)
        n_layers, n_points = txt.shape[0], min(txt.shape[1], img.shape[1])
        print(f"{model} / {prefix}: {n_layers} layers, {n_points} prompts")
        txt_img, img_txt = np.zeros((n_layers, n_layers)), np.zeros((n_layers, n_layers))
        for i in range(n_layers):
            for j in range(n_layers):
                txt_img[i, j], img_txt[i, j] = imbalance_first_nn(txt[i, :n_points], img[j, :n_points])
        os.makedirs(out_dir, exist_ok=True)
        meta = {"model": model, "dataset_prefix": prefix, "n_points": n_points, "k": 1,
                "row_modality": "text", "col_modality": "image"}
        for direction, matrix in (("txt_img", txt_img), ("img_txt", img_txt)):
            with open(outputs[direction], "w") as f:
                json.dump({"matrix": matrix.tolist(), "metadata": {**meta, "direction": direction}}, f)
            print(f"  saved {outputs[direction]}")


if __name__ == "__main__":
    main()
