import math
from typing import Optional, Tuple

import torch
from torch import nn


class Idomp(nn.Module):
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x


class LinearWithActivation(torch.nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        w_init_gain: str = "linear",
        activation: nn.Module = Idomp(),
    ):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear_layer(x))


class Conv1DNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        bias: bool = True,
        dropout_rate: float = 0.1,
        w_init_gain: str = "linear",
    ):
        super().__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size,),
            stride=(stride,),
            padding=padding,
            dilation=(dilation,),
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(out_channels)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        conv_signal = self.conv(signal)
        normed_signal = self.batch_norm(conv_signal)
        return self.dropout(self.relu(normed_signal))


class Conv2DNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: Optional[Tuple[int, int]] = None,
        dilation: int = 1,
        bias: bool = True,
        dropout_rate: float = 0.1,
        w_init_gain: str = "linear",
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=(dilation,),
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        conv_signal = self.conv(signal)
        normed_signal = self.batch_norm(conv_signal)
        return self.dropout(self.relu(normed_signal))


class PositionalEncoding(nn.Module):
    def __init__(self, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dimension = dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        position = torch.arange(x.shape[1]).unsqueeze(1).to(x.device)
        div_term = torch.exp(
            torch.arange(0, self.dimension, 2) * (-math.log(10000.0) / self.dimension)
        ).to(x.device)
        pe: torch.Tensor = torch.zeros(1, x.shape[1], self.dimension).to(x.device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        pe = torch.repeat_interleave(pe, x.shape[0], 0)

        x = torch.cat((x, pe[: x.shape[0]]), dim=-1)
        return self.dropout(x)
