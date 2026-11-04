"""seed_all: seeds Python, NumPy, PyTorch and Hugging Face, and makes cuDNN deterministic."""

import random
import numpy as np
import torch
from transformers import set_seed as hf_set_seed


def seed_all(seed: int):
    """Set the random seed of Python, NumPy, PyTorch (CPU and all GPUs) and Hugging Face."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    hf_set_seed(seed)
