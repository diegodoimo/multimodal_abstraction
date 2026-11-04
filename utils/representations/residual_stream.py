"""Last-token residual stream of a multimodal decoder at the output of every layer.

`residual_stream_tracer` runs the model over a dataloader with `output_hidden_states=True` and keeps,
for every decoder layer, the hidden state of the last prompt token (the loaders pad on the left), as a
tensor of shape (num_layers, num_samples, hidden_size); the embedding-layer output is not included.
It can also record, per sample and layer, the token that the final norm and LM head predict from that
hidden state (logit lens), which scripts/probing/label_overlap.py uses for the per-layer accuracy.

Batch formats (from the collate functions in utils/data/dataset.py and utils/data/collators.py):
LLaVA-1.5 / OneVision / pixtral: (inputs, ...); InternVL2: (tokenized, pixel_values, ...) with images,
(tokenized, ...) without; Cambrian: (tokenized, image_tensors, image_sizes, prompts) with images,
(tokenized, prompts) without.
"""

import gc
import time
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.feature_extraction_utils import BatchFeature

from utils.models.cambrian.cambrian_utils import is_cambrian_model
from utils.models.internvl.internvl_utils import IMG_CONTEXT_TOKEN, get_combined_embeddings


def _decoder_config(model):
    """Config of the language decoder (LLaVA / InternVL: model.language_model; Cambrian: model.model)."""
    for owner in (getattr(model, "language_model", None), getattr(model, "model", None), model):
        if owner is not None and hasattr(owner, "config"):
            return owner.config
    raise ValueError(f"Cannot find the decoder config of {model.__class__.__name__}")


def _lm_components(model):
    """(lm_head, final_norm) for the logit lens.

    LLaVA / Cambrian: lm_head on the top-level model, `language_model` is the bare backbone (with .norm).
    InternVL2: `language_model` is a full causal LM (its own lm_head and model.norm).
    """
    lm_head = getattr(model, "lm_head", None)
    lang = getattr(model, "language_model", None)
    if lm_head is None and lang is not None:
        lm_head = getattr(lang, "lm_head", None)
    backbone = lang if lang is not None else model
    inner = getattr(backbone, "model", backbone)
    norm = getattr(inner, "norm", None) or getattr(backbone, "norm", None)
    return lm_head, norm


def _model_inputs(model, batch, img_context_token_id):
    """Model inputs (on the model's device) for one batch of the dataloader."""
    name = model.__class__.__name__.lower()
    cambrian_images = cambrian_image_sizes = None
    if not isinstance(batch, (list, tuple)):
        inputs = batch
    elif "intern" in name and len(batch) > 1 and isinstance(batch[1], torch.Tensor):
        # InternVL2 with images: text embeddings with the vision features at the IMG_CONTEXT positions
        tokenized = batch[0]
        inputs = BatchFeature({
            "inputs_embeds": get_combined_embeddings(model, input_ids=tokenized["input_ids"], pixel_values=batch[1],
                                                     img_context_token_id=img_context_token_id),
            "attention_mask": tokenized["attention_mask"],
        })
    else:
        inputs = batch[0]
        if is_cambrian_model(model=model) and len(batch) >= 4 and isinstance(batch[1], (list, torch.Tensor)):
            cambrian_images, cambrian_image_sizes = batch[1], batch[2]
    inputs = inputs.to(model.device)
    if cambrian_images is not None:   # list of tensors: moved by hand
        if isinstance(cambrian_images, list):
            inputs["images"] = [t.to(device=model.device, dtype=torch.float16) for t in cambrian_images]
        else:
            inputs["images"] = cambrian_images.to(device=model.device, dtype=torch.float16)
        inputs["image_sizes"] = cambrian_image_sizes
    return inputs


def residual_stream_tracer(
    model,
    processed_dataloader: DataLoader,
    processor=None,
    return_dtype: torch.dtype = torch.bfloat16,
    layer_token_log: Optional[List[List[str]]] = None,
) -> torch.Tensor:
    """Last-token hidden state at the output of every decoder layer, for every sample of the dataloader.

    Args:
        model: Hugging Face multimodal decoder (LLaVA-1.5, OneVision-1.5, InternVL2, Cambrian, pixtral).
        processed_dataloader: dataloader of processed batches (see the module docstring).
        processor: processor or tokenizer; required for InternVL2 (IMG_CONTEXT token id) and for the logit lens.
        return_dtype: dtype of the returned tensor; activations are first cast to bfloat16.
        layer_token_log: if a list, one entry per sample is appended: the logit-lens token of every layer.

    Returns:
        Tensor of shape (num_layers, num_samples, hidden_size) on the CPU.
    """
    model.eval()
    config = _decoder_config(model)
    num_layers, num_samples = config.num_hidden_layers, len(processed_dataloader.dataset)
    if num_samples == 0:
        raise ValueError("The dataloader is empty")
    tokenizer = getattr(processor, "tokenizer", processor)
    is_internvl = "intern" in model.__class__.__name__.lower()
    img_context_token_id = None
    if is_internvl:
        if processor is None:
            raise ValueError("InternVL2 needs the processor (IMG_CONTEXT token id)")
        img_context_token_id = processor.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    lm_head = lm_norm = None
    if layer_token_log is not None:
        lm_head, lm_norm = _lm_components(model)
        if lm_head is None or tokenizer is None:
            raise ValueError(f"Logit lens: no LM head or tokenizer for {model.__class__.__name__}")

    residual_stream = torch.zeros(num_layers, num_samples, config.hidden_size, dtype=return_dtype)
    counter, since_cleanup, cleanup_every = 0, 0, max(1, num_samples // 10)
    start = time.time()
    progress = tqdm(total=num_samples, desc="Extracting residual stream", unit="sample")
    for batch in processed_dataloader:
        with torch.no_grad():
            inputs = _model_inputs(model, batch, img_context_token_id)
            batch_size = inputs["attention_mask"].shape[0]
            forward = model.language_model if is_internvl else model
            outputs = forward(**inputs, output_hidden_states=True, use_cache=False)
            hidden_states = [h.to(device=model.device, dtype=torch.bfloat16) for h in outputs.hidden_states[1:]]
            residual_stream[:, counter:counter + batch_size] = torch.stack(
                [h[:, -1, :].to(device="cpu", dtype=return_dtype) for h in hidden_states], dim=0)

            if layer_token_log is not None:
                lm_dtype = next(lm_head.parameters()).dtype
                for idx in range(batch_size):
                    tokens = []
                    for h in hidden_states:
                        x = h[idx, -1:, :].to(lm_dtype)
                        if lm_norm is not None:
                            x = lm_norm(x)
                        token_id = int(lm_head(x)[0].argmax().item())
                        tokens.append(tokenizer.decode([token_id], skip_special_tokens=False,
                                                       clean_up_tokenization_spaces=False))
                    layer_token_log.append(tokens)
            del outputs, hidden_states, inputs, batch

        counter += batch_size
        since_cleanup += batch_size
        progress.update(batch_size)
        if since_cleanup >= cleanup_every:
            torch.cuda.empty_cache()
            gc.collect()
            since_cleanup = 0
    progress.close()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Residual stream of {num_samples} samples extracted in {(time.time() - start) / 60:.1f} min")
    return residual_stream
