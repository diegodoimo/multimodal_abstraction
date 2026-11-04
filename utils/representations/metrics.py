"""Representation metrics: cross-layer similarity of two residual streams (linear CKA, neighborhood
overlap, information imbalance) and per-layer label overlap."""

import time
import warnings
from typing import Dict, List

import numpy as np
import torch
from dadapy.data import Data
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIMILARITY_MEASURES = (
    "linear_cka",
    "neighborhood_overlap",
    "information_imbalance",
    "information_imbalance_reversed",
)


def _as_metric_tensor(data: torch.Tensor) -> torch.Tensor:
    """``data`` as a contiguous float32 tensor on DEVICE."""
    return data.detach().to(device=DEVICE, dtype=torch.float32, non_blocking=True).contiguous()


def _progress(total: int, measure: str) -> tqdm:
    """Progress bar over layer pairs, refreshed every 10%."""
    return tqdm(total=total, desc=f"Computing {measure}", unit="pair", miniters=max(1, total // 10))


# ---------------------------------------------------------------------------
# Linear CKA
# ---------------------------------------------------------------------------


def _center_gram_inplace(kernel: torch.Tensor) -> torch.Tensor:
    row_mean = kernel.mean(dim=1, keepdim=True)
    col_mean = kernel.mean(dim=0, keepdim=True)
    grand_mean = kernel.mean()
    kernel.sub_(row_mean)
    kernel.sub_(col_mean)
    kernel.add_(grand_mean)
    return kernel


def _linear_cka_states(network: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Centred Gram matrices X X^T of a stack of layers, flattened to (layers, N * N), and their norms."""
    num_layers, num_samples, _ = network.shape
    states = torch.empty(num_layers, num_samples * num_samples, device=DEVICE, dtype=torch.float32)
    norms = torch.empty(num_layers, device=DEVICE, dtype=torch.float32)
    for layer_idx, layer in enumerate(network):
        layer_tensor = _as_metric_tensor(layer)
        gram = _center_gram_inplace(layer_tensor @ layer_tensor.T)
        states[layer_idx].copy_(gram.reshape(-1))
        norms[layer_idx] = torch.sqrt(torch.sum(gram * gram))
        del layer_tensor, gram
    return states, norms


def _cka_memory_budgets(state_bytes: int) -> tuple[int, int]:
    """(cache budget, block budget) in bytes: 45% and 25% of the free GPU memory, else 8 and 2 GiB."""
    if DEVICE.type == "cuda":
        try:
            with torch.cuda.device(DEVICE):
                free_bytes, _ = torch.cuda.mem_get_info()
            return int(0.45 * free_bytes), max(int(0.25 * free_bytes), 3 * state_bytes)
        except RuntimeError:
            pass
    return 8 * (1024**3), 2 * (1024**3)


def _cross_layer_linear_cka(
    network1: torch.Tensor, network2: torch.Tensor, keep: torch.Tensor, total: int
) -> torch.Tensor:
    """Linear CKA of every pair of layers, CKA(i, j) = <K_i, K_j> / (|K_i| |K_j|) on centred Gram matrices.

    One Gram matrix ("state") takes 4 N^2 bytes (0.36 GiB at N = 9848). All states are cached when they fit
    in the cache budget; otherwise the states of one side are cached and the other side is streamed in
    blocks of up to 16 layers; otherwise both sides are streamed in blocks of up to 8 layers.
    """
    num_layers_1, num_samples, _ = network1.shape
    num_layers_2 = network2.shape[0]
    result = torch.zeros(num_layers_1, num_layers_2, dtype=torch.float64)

    with torch.no_grad():
        state_bytes = num_samples * num_samples * 4   # float32
        all_cache_bytes = (num_layers_1 + num_layers_2) * state_bytes
        cache_budget_bytes, block_budget_bytes = _cka_memory_budgets(state_bytes)

        # Block lengths along the rows (network1) and the columns (network2); one extra state is
        # reserved for the transient Gram matrix.
        row_block, col_block, cache_cols = num_layers_1, num_layers_2, False
        if all_cache_bytes + state_bytes > cache_budget_bytes:
            cache_rows = (
                num_layers_1 <= num_layers_2
                and (num_layers_1 + 2) * state_bytes <= cache_budget_bytes
            )
            cache_cols = not cache_rows and (num_layers_2 + 2) * state_bytes <= cache_budget_bytes
            if cache_rows or cache_cols:
                cached_bytes = (num_layers_1 if cache_rows else num_layers_2) * state_bytes
                remaining_bytes = max(cache_budget_bytes - cached_bytes, 2 * state_bytes)
                stream_block = max(1, min(16, int((remaining_bytes - state_bytes) // state_bytes)))
                if cache_rows:
                    col_block = stream_block
                else:
                    row_block = stream_block
                print(
                    f"    linear_cka: all Gram matrices need {all_cache_bytes / (1024**3):.2f} GiB; "
                    f"caching the {'row' if cache_rows else 'column'} layers, {stream_block}-layer blocks "
                    "for the others."
                )
            else:
                states_per_side = (block_budget_bytes - state_bytes) // (2 * state_bytes)
                row_block = col_block = max(1, min(8, int(states_per_side)))
                print(
                    f"    linear_cka: all Gram matrices need {all_cache_bytes / (1024**3):.2f} GiB; "
                    f"using {row_block}-layer row and column blocks."
                )

        cached_col_states = _linear_cka_states(network2) if cache_cols else None
        with _progress(total, "linear_cka") as pbar:
            for i0 in range(0, num_layers_1, row_block):
                i1 = min(i0 + row_block, num_layers_1)
                states1, norms1 = _linear_cka_states(network1[i0:i1])
                for j0 in range(0, num_layers_2, col_block):
                    j1 = min(j0 + col_block, num_layers_2)
                    if cached_col_states is not None:
                        states2, norms2 = cached_col_states
                    else:
                        states2, norms2 = _linear_cka_states(network2[j0:j1])
                    scores = (states1 @ states2.T) / (norms1[:, None] * norms2[None, :])
                    del states2, norms2
                    block_keep = keep[i0:i1, j0:j1]
                    checked = scores[block_keep.to(scores.device)]
                    if not torch.all((checked >= -1e-6) & (checked <= 1 + 1e-6)):
                        raise AssertionError("CKA value must be between 0 and 1.")
                    result[i0:i1, j0:j1][block_keep] = scores.to(device="cpu", dtype=torch.float64)[block_keep]
                    pbar.update(int(block_keep.sum()))
                    del scores, checked
                del states1, norms1
        del cached_col_states
    return result


# ---------------------------------------------------------------------------
# Neighborhood overlap
# ---------------------------------------------------------------------------


def _neighbor_indices(data: torch.Tensor, maxk: int) -> torch.Tensor:
    """Indices of the maxk Euclidean nearest neighbours of each row, self excluded."""
    return torch.topk(torch.cdist(data, data), maxk + 1, dim=1, largest=False).indices[:, 1:]


def _neighborhood_overlap(x_neighbors: torch.Tensor, y_neighbors: torch.Tensor, maxk: int) -> torch.Tensor:
    """Number of shared neighbours divided by maxk * N."""
    overlap = x_neighbors.unsqueeze(2).eq(y_neighbors.unsqueeze(1)).sum(dtype=torch.float64)
    return overlap / (maxk * x_neighbors.shape[0])


def _has_duplicate_rows(data: torch.Tensor) -> bool:
    """True if ``data`` has repeated rows or NaNs."""
    if torch.isnan(data).any().item():
        return True
    return torch.unique(data, dim=0).shape[0] != data.shape[0]


def _layer_neighbors(network: torch.Tensor, maxk: int) -> list[torch.Tensor] | None:
    """Neighbour indices of every layer, or None if a layer has duplicate or NaN rows."""
    neighbors = []
    for layer in network:
        layer_tensor = _as_metric_tensor(layer)
        if _has_duplicate_rows(layer_tensor):
            return None
        neighbors.append(_neighbor_indices(layer_tensor, maxk))
    return neighbors


def _neighborhood_overlap_without_duplicates(x: torch.Tensor, y: torch.Tensor, maxk: int) -> float:
    """Overlap after dropping the rows repeated in x or in y, with one mask so that x and y stay paired."""
    x_np = x.detach().cpu().to(torch.float32).numpy()
    y_np = y.detach().cpu().to(torch.float32).numpy()
    _, x_first = np.unique(x_np, axis=0, return_index=True)
    _, y_first = np.unique(y_np, axis=0, return_index=True)
    rows = np.intersect1d(x_first, y_first)
    x_t = torch.tensor(x_np[rows], dtype=torch.float32).to(DEVICE, non_blocking=True)
    y_t = torch.tensor(y_np[rows], dtype=torch.float32).to(DEVICE, non_blocking=True)
    with torch.no_grad():
        return _neighborhood_overlap(_neighbor_indices(x_t, maxk), _neighbor_indices(y_t, maxk), maxk).item()


def _cross_layer_neighborhood_overlap(
    network1: torch.Tensor, network2: torch.Tensor, keep: torch.Tensor, total: int, maxk: int
) -> torch.Tensor:
    """Neighborhood overlap of every pair of layers.

    The neighbours are computed once per layer when no layer has duplicate rows; otherwise each pair is
    recomputed without the rows that are duplicated in either layer.
    """
    num_layers_1, num_layers_2 = network1.shape[0], network2.shape[0]
    with torch.no_grad():
        neighbors1 = _layer_neighbors(network1, maxk)
        neighbors2 = _layer_neighbors(network2, maxk) if neighbors1 is not None else None
    if neighbors2 is None:
        neighbors1 = None
        print("    neighborhood_overlap: duplicate rows found, computing each pair without them.")

    result = torch.zeros(num_layers_1, num_layers_2, dtype=torch.float64)
    with torch.no_grad(), _progress(total, "neighborhood_overlap") as pbar:
        for i in range(num_layers_1):
            cols = keep[i].nonzero().flatten().tolist()
            if not cols:
                continue
            if neighbors2 is None:
                vals = [_neighborhood_overlap_without_duplicates(network1[i], network2[j], maxk) for j in cols]
                result[i, cols] = torch.tensor(vals, dtype=torch.float64)
            else:
                vals = [_neighborhood_overlap(neighbors1[i], neighbors2[j], maxk) for j in cols]
                result[i, cols] = torch.stack(vals).to(device="cpu", dtype=torch.float64)
            pbar.update(len(cols))
    return result


# ---------------------------------------------------------------------------
# Information imbalance
# ---------------------------------------------------------------------------


def _first_neighbor_indices(data: torch.Tensor) -> torch.Tensor:
    return torch.topk(torch.cdist(data, data), k=2, dim=1, largest=False).indices[:, 1]


def _information_imbalance(source_neighbor_indices: torch.Tensor, target_distances: torch.Tensor) -> torch.Tensor:
    """Mean rank, in the target space, of each point's first neighbour in the source space, times 2 / N."""
    n_samples = target_distances.shape[0]
    row_indices = torch.arange(n_samples, device=target_distances.device)
    dists_to_nn = target_distances[row_indices, source_neighbor_indices].unsqueeze(1)
    ranks = (target_distances < dists_to_nn).sum(dim=1).float()
    return 2 * ranks.mean() / n_samples


def _cross_layer_information_imbalance(
    source: torch.Tensor, target: torch.Tensor, keep: torch.Tensor, total: int, measure: str
) -> torch.Tensor:
    """M[s, t] = information imbalance from layer s of ``source`` to layer t of ``target``, where keep[s, t]."""
    num_source, num_target = source.shape[0], target.shape[0]
    result = torch.zeros(num_source, num_target, dtype=torch.float64)
    with torch.no_grad():
        source_neighbors = [_first_neighbor_indices(_as_metric_tensor(layer)) for layer in source]
        with _progress(total, measure) as pbar:
            for t in range(num_target):
                target_tensor = _as_metric_tensor(target[t])
                target_distances = torch.cdist(target_tensor, target_tensor)
                rows = keep[:, t].nonzero().flatten().tolist()
                if rows:
                    vals = [_information_imbalance(source_neighbors[s], target_distances) for s in rows]
                    result[rows, t] = torch.stack(vals).to(device="cpu", dtype=torch.float64)
                    pbar.update(len(rows))
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def compute_cross_layers_residual_stream_similarities(
    residual_stream_network1: torch.Tensor,
    residual_stream_network2: torch.Tensor,
    similarity_measures: List[str],
    lower_triangular_only: bool = False,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Compare every layer of network 1 with every layer of network 2 on the same samples.

    Args:
        residual_stream_network1: (num_layers_1, num_samples, hidden_size); the rows of the output.
        residual_stream_network2: (num_layers_2, num_samples, hidden_size); the columns.
        similarity_measures: names from SIMILARITY_MEASURES. "information_imbalance" is the imbalance
            from network 1 to network 2, "information_imbalance_reversed" from network 2 to network 1;
            neighborhood_overlap uses Euclidean distances.
        lower_triangular_only: compute only the pairs with column <= row; the others stay 0.
        **kwargs: maxk (int, default 30), the number of neighbours of neighborhood_overlap.

    Returns:
        {measure: float64 CPU tensor of shape (num_layers_1, num_layers_2)}.
    """
    maxk = kwargs.pop("maxk", 30)
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
    unknown = [m for m in similarity_measures if m not in SIMILARITY_MEASURES]
    if unknown:
        raise ValueError(f"Unknown similarity measures {unknown}; available: {SIMILARITY_MEASURES}")

    network1, network2 = residual_stream_network1, residual_stream_network2
    num_layers_1, num_samples, _ = network1.shape
    num_layers_2, num_samples_2, _ = network2.shape
    if num_samples != num_samples_2:
        raise ValueError("The number of samples of the two networks must be the same")

    keep = torch.ones(num_layers_1, num_layers_2, dtype=torch.bool)
    if lower_triangular_only:
        keep = keep.tril()
    total = int(keep.sum())

    results = {}
    for measure in similarity_measures:
        print(f"    - Computing {measure} matrix ({num_layers_1}x{num_layers_2})...")
        if total == 0:
            results[measure] = torch.zeros(num_layers_1, num_layers_2, dtype=torch.float64)
        elif measure == "linear_cka":
            results[measure] = _cross_layer_linear_cka(network1, network2, keep, total)
        elif measure == "neighborhood_overlap":
            results[measure] = _cross_layer_neighborhood_overlap(network1, network2, keep, total, maxk)
        elif measure == "information_imbalance":
            results[measure] = _cross_layer_information_imbalance(network1, network2, keep, total, measure)
        else:
            results[measure] = _cross_layer_information_imbalance(
                network2, network1, keep.T, total, measure
            ).T.contiguous()
    return results


# ---------------------------------------------------------------------------
# Label overlap
# ---------------------------------------------------------------------------


def compute_label_overlap(
    data: torch.Tensor,
    labels: np.ndarray,
    maxk: int = 30,
    class_fraction: float | None = 0.3,
) -> float:
    """Weighted DADApy label overlap of (N, D) ``data``, with the Euclidean neighbours computed by torch."""
    data_t = data.detach().to(device=DEVICE, dtype=torch.float32)
    labels_np = np.asarray(labels).reshape(-1).astype(int)

    n_samples = data_t.shape[0]
    if labels_np.shape[0] != n_samples:
        raise ValueError(f"Data has {n_samples} samples, labels has {labels_np.shape[0]} samples")
    if n_samples <= 1:
        return 0.0

    # Number of neighbours DADApy needs for this class_fraction; they are computed here and passed to
    # DADApy as a precomputed neighbour table.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, required_k = Data()._label_imbalance_helper(labels_np, maxk, class_fraction)
    effective_k = min(int(required_k), n_samples - 1)
    if effective_k <= 0:
        return 0.0

    with torch.no_grad():
        dist = torch.cdist(data_t, data_t)
        dist.fill_diagonal_(float("inf"))
        nn_dist, nn_idx = torch.topk(dist, effective_k, dim=1, largest=False)

    self_idx = torch.arange(n_samples, device=data_t.device).unsqueeze(1)
    self_dist = torch.zeros((n_samples, 1), dtype=nn_dist.dtype, device=data_t.device)
    dist_for_dadapy = torch.cat([self_dist, nn_dist], dim=1).cpu().numpy()
    idx_for_dadapy = torch.cat([self_idx, nn_idx], dim=1).cpu().numpy()

    d = Data(distances=(dist_for_dadapy, idx_for_dadapy), maxk=effective_k)
    overlap = d.return_label_overlap(labels_np, k=effective_k, class_fraction=class_fraction, weighted=True)
    return float(overlap)


def classify_per_layer_label_overlap(
    representations: torch.Tensor,
    y: np.ndarray,
    maxk: int = 30,
    class_fraction: float | None = None,
) -> dict[int, dict[str, float]]:
    """{layer: {"label_overlap": value}} for (layers, N, D) representations and N labels.

    class_fraction=None uses the default of compute_label_overlap (0.3).
    """
    start_time = time.time()
    num_layers = representations.shape[0]
    lo_kwargs = {} if class_fraction is None else {"class_fraction": class_fraction}
    layer_results: dict[int, dict[str, float]] = {}
    with tqdm(total=num_layers, desc="Metric label_overlap", unit="layer",
              miniters=max(1, num_layers // 10)) as pbar:
        for layer_idx in range(num_layers):
            overlap = compute_label_overlap(representations[layer_idx], y, maxk=maxk, **lo_kwargs)
            layer_results[layer_idx] = {"label_overlap": float(overlap)}
            pbar.update(1)
    print(f"label_overlap computed for {num_layers} layers in {time.time() - start_time:.0f} s.")
    return layer_results
