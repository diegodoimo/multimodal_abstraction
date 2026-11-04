"""Per-layer mean hidden state over selected prompt positions, collected with forward hooks (COCO retrieval).

Image prompts: mean over the image-token positions (input_ids == image_token_id).
Caption prompts: mean over the tokens that are not special tokens (exclude_token_ids).
"""

import time

import torch

from utils.models.layers import get_embdims, to_cuda


def _masked_mean(activations, mask):
    weights = mask.unsqueeze(-1).to(activations.dtype)
    return (activations * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


@torch.inference_mode()
def mean_representations(
    model,
    dataloader,
    target_layers: dict,
    nsamples: int,
    image_token_id=None,
    exclude_token_ids=None,
    forward_fn=None,
    pool_info_provider=None,
    print_every=1000,
):
    """{layer module name: (nsamples, hidden) tensor} of the mean hidden state over the selected positions.

    With image_token_id the image-token positions are averaged; otherwise every position whose id is not
    in exclude_token_ids. forward_fn overrides model(**batch, use_cache=False) (text-only InternVL2 and
    Cambrian run the language model alone). pool_info_provider(batch) -> {"mask", "input_ids"} replaces
    the attention mask and input ids when the model changes the sequence length (Cambrian expands its
    image placeholder inside the forward).
    """
    names = list(target_layers.values())
    start = time.time()
    embdims, dtypes = get_embdims(model, dataloader, names, forward_fn=forward_fn)
    out = {name: torch.zeros((nsamples, embdims[name]), dtype=dtypes[name]) for name in names}
    hidden = {}

    def hook(name):
        def fn(module, inputs, output):
            hidden[name] = (output[0] if isinstance(output, tuple) else output).cpu()
        return fn

    handles = [module.register_forward_hook(hook(name)) for name, module in model.named_modules() if name in names]
    n_done = 0
    batch_size, n_batches = dataloader.batch_size, len(dataloader)
    for i, batch in enumerate(dataloader):
        batch = to_cuda(batch)
        if forward_fn is not None:
            forward_fn(batch)
        else:
            model(**batch, use_cache=False)
        if pool_info_provider is not None:
            info = pool_info_provider(batch)
            valid, input_ids = info["mask"].cpu().bool(), info["input_ids"].cpu()
        else:
            valid, input_ids = (batch["attention_mask"] != 0).cpu(), batch["input_ids"].cpu()
        if image_token_id is not None:
            mask = (input_ids == image_token_id) & valid
        else:
            mask = valid & ~torch.isin(input_ids, torch.tensor(list(exclude_token_ids or []), dtype=input_ids.dtype))
        rows = slice(n_done, n_done + input_ids.shape[0])
        for name, activations in hidden.items():
            out[name][rows] = _masked_mean(activations, mask)
        n_done += input_ids.shape[0]
        if (i + 1) % (print_every // batch_size) == 0:
            print(f"batch {i + 1}/{n_batches}, {(time.time() - start) / 60:.1f} min")
    for h in handles:
        h.remove()
    print(f"Extracted {n_done} samples in {(time.time() - start) / 3600:.2f} h")
    return out
