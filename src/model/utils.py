from math import sqrt
from typing import List, Tuple

import numpy as np
import torch
from scipy.io.wavfile import read  # type: ignore


def get_mask_from_lengths(lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len).to(device))
    mask = ids >= lengths.unsqueeze(1)
    return mask


def norm_emb_layer(emb: torch.nn.Embedding, n_symbols: int, embedding_dim: int) -> None:
    std = sqrt(2.0 / (n_symbols + embedding_dim))
    variance = sqrt(3.0) * std
    emb.weight.data.uniform_(-variance, variance)


def load_wav_to_torch(full_path: str) -> Tuple[torch.Tensor, int]:
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename: str, sep: str = "|") -> List[List[str]]:
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(sep) for line in f]
    return filepaths_and_text


def to_gpu(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
