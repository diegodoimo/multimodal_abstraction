"""Per-layer sentence representations for the linguistic probes, collected with forward hooks.

text_avg: per target layer, the mean over the real (non-pad) tokens that are not tokenizer special tokens.
"""

import time
from collections import defaultdict

import torch

from utils.models.layers import to_cuda


class extract_activations:
    def __init__(
        self,
        model,
        dataloader,
        target_layers,
        embdims,
        dtypes,
        special_token_ids,
        forward_fn=None,
        print_every=5000,
    ):
        self.model = model
        # Overrides model(**batch): InternVL2 and Cambrian run the language model alone.
        self.forward_fn = forward_fn
        self.print_every = print_every
        self.batch_size = dataloader.batch_size
        self.nbatches = len(dataloader)
        self.special_ids = torch.tensor(list(special_token_ids), dtype=torch.long)

        nsamples = len(dataloader.dataset)
        self.text_avg = {
            name: torch.zeros((nsamples, embdims[name]), dtype=dtypes[name])
            for name in target_layers
        }
        self.n_done = 0

        self.hidden_states_tmp = defaultdict(lambda: None)
        self.handles = [
            module.register_forward_hook(self._get_hook(name))
            for name, module in self.model.named_modules()
            if name in target_layers
        ]

    def _get_hook(self, name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.hidden_states_tmp[name] = output.cpu()

        return hook_fn

    def _update_pools(self, input_ids, mask):
        mask = mask.to("cpu")
        input_ids = input_ids.to("cpu")
        special_mask = torch.isin(input_ids, self.special_ids).to(mask.dtype)
        text_mask = mask * (1 - special_mask)  # real, non-special

        B = mask.shape[0]
        text_denom = text_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)

        rows = slice(self.n_done, self.n_done + B)
        for layer, activations in self.hidden_states_tmp.items():
            m = text_mask.unsqueeze(-1).to(activations.dtype)
            self.text_avg[layer][rows] = (activations * m).sum(dim=1) / text_denom.to(activations.dtype)
        self.n_done += B

    @torch.no_grad()
    def extract(self, dataloader):
        start = time.time()
        for i, batch in enumerate(dataloader):
            batch = to_cuda(batch)
            if self.forward_fn is not None:
                self.forward_fn(batch)
            else:
                self.model(**batch)
            self._update_pools(batch["input_ids"], batch["attention_mask"])

            if (i + 1) % (self.print_every // self.batch_size) == 0:
                print(f"batch {i + 1}/{self.nbatches}, {(time.time() - start) / 60:.1f} min")
        for handle in self.handles:  # the next extraction registers its own hooks
            handle.remove()
