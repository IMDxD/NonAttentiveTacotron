from dataclasses import dataclass
from typing import List


@dataclass
class DurationParams:

    lstm_layers: int
    lstm_hidden: int
    dropout: float


@dataclass
class RangeParams:

    lstm_layers: int
    lstm_hidden: int
    dropout: float


@dataclass
class GaussianUpsampleParams:

    duration_config: DurationParams
    range_config: RangeParams
    eps: float
    positional_dim: int
    teacher_forcing_ratio: float
    attention_dropout: float
    positional_dropout: float


@dataclass
class DecoderParams:

    prenet_layers: List[int]
    prenet_dropout: float
    decoder_rnn_dim: int
    decoder_num_layers: int
    teacher_forcing_ratio: float
    dropout: float


@dataclass
class EncoderParams:

    n_convolutions: int
    kernel_size: int
    conv_channel: int
    lstm_layers: int
    lstm_hidden: int
    dropout: float


@dataclass
class PostNetParams:

    embedding_dim: int
    n_convolutions: int
    kernel_size: int
    dropout: float


@dataclass
class ModelParams:

    encoder_config: EncoderParams
    attention_config: GaussianUpsampleParams
    decoder_config: DecoderParams
    postnet_config: PostNetParams
    mask_padding: bool
    phonem_embedding_dim: int
    speaker_embedding_dim: int
