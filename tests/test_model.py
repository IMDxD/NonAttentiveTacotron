from unittest.mock import MagicMock, patch

import pytest
import torch

from src.data_process import VCTKBatch
from src.model.config import (
    DecoderParams, DurationParams, EncoderParams, GaussianUpsampleParams,
    ModelParams, PostNetParams, RangeParams,
)
from src.model.non_attentive_tacotron import (
    Attention, Decoder, DurationPredictor, Encoder, NonAttentiveTacotron,
    Postnet, Prenet, RangePredictor,
)

DECODER_CONFIG = DecoderParams()
DURATION_CONFIG = DurationParams()
RANGE_CONFIG = RangeParams()
ENCODER_CONFIG = EncoderParams()
ATTENTION_CONFIG = GaussianUpsampleParams(
    duration_config=DURATION_CONFIG, range_config=RANGE_CONFIG
)
POSTNET_CONFIG = PostNetParams()
MODEL_CONFIG = ModelParams(
    encoder_config=ENCODER_CONFIG,
    attention_config=ATTENTION_CONFIG,
    decoder_config=DECODER_CONFIG,
    postnet_config=POSTNET_CONFIG,
)
N_PHONEMES = 100
N_SPEAKER = 4
N_MELS_DIM = 80
EMBEDDING_DIM = MODEL_CONFIG.phonem_embedding_dim + MODEL_CONFIG.speaker_embedding_dim
INPUT_PHONEMES = torch.randint(N_PHONEMES, size=(16, 50), dtype=torch.long)
INPUT_SPEAKERS = torch.randint(N_SPEAKER, size=(16,), dtype=torch.long)
PHONEM_EMB = torch.randn(16, 50, MODEL_CONFIG.phonem_embedding_dim, dtype=torch.float)
SPEAKER_EMB = torch.randn(16, MODEL_CONFIG.speaker_embedding_dim, dtype=torch.float)
EMBEDDING = torch.cat(
    [PHONEM_EMB, torch.repeat_interleave(SPEAKER_EMB.unsqueeze(1), 50, dim=1)], dim=-1
)
INPUT_LENGTH = torch.arange(35, 51, dtype=torch.long)
INPUT_DURATIONS = torch.randint(5, 10, size=(16, 50), dtype=torch.long)
for i, l in enumerate(INPUT_LENGTH):
    INPUT_DURATIONS[i, l:] = 0
DURATIONS_MAX = INPUT_DURATIONS.cumsum(dim=1).max(dim=1).values
INPUT_MELS = torch.randn(
    16, int(DURATIONS_MAX.max()), N_MELS_DIM, dtype=torch.float
)
for i, l in enumerate(DURATIONS_MAX):
    INPUT_MELS[i, l:, :] = 0
ATTENTION_OUT_DIM = EMBEDDING_DIM + ATTENTION_CONFIG.positional_dim
DECODER_RNN_OUT = torch.randn(16, 1, DECODER_CONFIG.decoder_rnn_dim)

MODEL_INPUT = VCTKBatch(
    phonemes=INPUT_PHONEMES,
    num_phonemes=INPUT_LENGTH,
    speaker_ids=INPUT_SPEAKERS,
    durations=INPUT_DURATIONS,
    mels=INPUT_MELS,
)
MODEL_INFERENCE_INPUT = (
    INPUT_PHONEMES,
    INPUT_LENGTH,
    INPUT_SPEAKERS,
    INPUT_MELS,
)
ATTENTION_OUT = torch.randn((16, int(DURATIONS_MAX.max()), ATTENTION_OUT_DIM))
for i, l in enumerate(DURATIONS_MAX):
    ATTENTION_OUT[i, l:, :] = 0


def test_encoder_layer() -> None:
    expected_shape = (16, 50, MODEL_CONFIG.encoder_config.lstm_hidden * 2)
    layer = Encoder(ModelParams.phonem_embedding_dim, config=ENCODER_CONFIG)
    out = layer(PHONEM_EMB.transpose(1, 2), INPUT_LENGTH)
    assert (
        out.shape == expected_shape
    ), f"Wrong shape, expected {expected_shape}, got: {out.shape}"
    for idx, length in enumerate(INPUT_LENGTH):
        assert (
            out[idx, length:] == 0
        ).all(), "All values of tensor higher sequence length must be zero"
        assert (out[idx, length - 1] != 0).any(), f"Wrong zero vector for id = {idx}"


def test_duration_layer() -> None:
    expected_shape = (16, 50, 1)
    layer = DurationPredictor(EMBEDDING_DIM, config=DURATION_CONFIG)
    zero_value = layer.projection.linear_layer.bias
    if zero_value is None:
        zero_value = 0
    out = layer(EMBEDDING, INPUT_LENGTH)
    assert (
        out.shape == expected_shape
    ), f"Wrong shape, expected {expected_shape}, got: {out.shape}"
    for idx, length in enumerate(INPUT_LENGTH):
        assert (
            out[idx, length:] == zero_value
        ).all(), "All values of tensor higher sequence length must be zero"
        assert (
            out[idx, length - 1] != zero_value
        ).any(), f"Wrong zero vector for id = {idx}"


def test_range_layer() -> None:
    expected_shape = (16, 50, 1)
    layer = RangePredictor(EMBEDDING_DIM, config=RANGE_CONFIG)
    zero_value = layer.projection.linear_layer.bias
    if zero_value is None:
        zero_value = 0
    out = layer(EMBEDDING, INPUT_DURATIONS.unsqueeze(2), INPUT_LENGTH)
    assert (
        out.shape == expected_shape
    ), f"Wrong shape, expected {expected_shape}, got: {out.shape}"


def test_attention_layer_forward() -> None:
    expected_shape_out = (
        16,
        DURATIONS_MAX.max().item(),
        EMBEDDING_DIM + ATTENTION_CONFIG.positional_dim,
    )
    expected_shape_dur = (16, 50)
    layer = Attention(
        EMBEDDING_DIM, config=ATTENTION_CONFIG
    )
    dur, out = layer(EMBEDDING, INPUT_LENGTH, INPUT_DURATIONS)
    assert (
        dur.shape == expected_shape_dur
    ), f"Wrong shape, expected {expected_shape_dur}, got: {dur.shape}"
    assert (
        out.shape == expected_shape_out
    ), f"Wrong shape, expected {expected_shape_out}, got: {out.shape}"


@patch(
    "src.model.non_attentive_tacotron.DurationPredictor.forward"
)
def test_attention_layer_inference(mock_duration: MagicMock) -> None:
    mock_duration.return_value = INPUT_DURATIONS.unsqueeze(2)
    expected_shape_out = (
        16,
        DURATIONS_MAX.max().item(),
        EMBEDDING_DIM + ATTENTION_CONFIG.positional_dim,
    )
    layer = Attention(
        EMBEDDING_DIM, config=ATTENTION_CONFIG
    )
    out = layer.inference(EMBEDDING, INPUT_LENGTH)
    assert (
        out.shape == expected_shape_out
    ), f"Wrong shape, expected {expected_shape_out}, got: {out.shape}"


def test_prenet_layer() -> None:
    expected_shape = (16, 1, DECODER_CONFIG.prenet_layers[-1])

    layer = Prenet(
        N_MELS_DIM,
        DECODER_CONFIG.prenet_layers,
        dropout=DECODER_CONFIG.prenet_dropout,
    )
    out = layer(INPUT_MELS[:, 0, :].unsqueeze(1))
    assert (
        out.shape == expected_shape
    ), f"Wrong shape, expected {expected_shape}, got: {out.shape}"


def test_decoder_layer_forward() -> None:
    expected_shape = (16, int(DURATIONS_MAX.max().long()), N_MELS_DIM)

    layer = Decoder(
        N_MELS_DIM,
        ATTENTION_OUT_DIM,
        config=DECODER_CONFIG,
    )
    out = layer(ATTENTION_OUT, INPUT_MELS)
    assert (
        out.shape == expected_shape
    ), f"Wrong shape, expected {expected_shape}, got: {out.shape}"


def test_decoder_layer_inference() -> None:
    expected_shape = (16, int(DURATIONS_MAX.max().long()), N_MELS_DIM)

    layer = Decoder(
        N_MELS_DIM,
        ATTENTION_OUT_DIM,
        config=DECODER_CONFIG,
    )
    out = layer.inference(ATTENTION_OUT)
    assert (
        out.shape == expected_shape
    ), f"Wrong shape, expected {expected_shape}, got: {out.shape}"


def test_postnet_layer() -> None:
    expected_shape = INPUT_MELS.transpose(1, 2).shape

    layer = Postnet(
        N_MELS_DIM,
        config=POSTNET_CONFIG,
    )
    out = layer(INPUT_MELS.transpose(1, 2))
    assert (
        out.shape == expected_shape
    ), f"Wrong shape, expected {expected_shape}, got: {out.shape}"


def test_model_forward() -> None:
    expected_mel_shape = INPUT_MELS.shape
    expected_duration_shape = (16, 50)

    model = NonAttentiveTacotron(
        N_PHONEMES, N_SPEAKER, N_MELS_DIM, config=MODEL_CONFIG
    )
    durations, mel_fixed, mel_predicted = model(MODEL_INPUT)
    assert (
        durations.shape == expected_duration_shape
    ), f"Wrong shape, expected {expected_duration_shape}, got: {durations.shape}"
    assert (
        mel_predicted.shape == expected_mel_shape
    ), f"Wrong shape, expected {expected_mel_shape}, got: {mel_predicted.shape}"
    assert (
        mel_fixed.shape == expected_mel_shape
    ), f"Wrong shape, expected {expected_mel_shape}, got: {mel_fixed.shape}"
    for idx, length in enumerate(DURATIONS_MAX):
        assert (
            mel_fixed[idx, length:] == 0
        ).all(), "All values of tensor higher sequence length must be zero"
        assert (
            mel_fixed[idx, length - 1] != 0
        ).any(), f"Wrong zero vector for id = {idx}"
        assert (
            mel_predicted[idx, length:] == 0
        ).all(), "All values of tensor higher sequence length must be zero"
        assert (
            mel_predicted[idx, length - 1] != 0
        ).any(), f"Wrong zero vector for id = {idx}"


@pytest.mark.skipif(torch.cuda.is_available() is False, reason="No cuda")
def test_model_forward_gpu() -> None:
    expected_mel_shape = INPUT_MELS.shape
    expected_duration_shape = (16, 50)

    model = NonAttentiveTacotron(
        N_PHONEMES, N_SPEAKER, N_MELS_DIM, config=MODEL_CONFIG
    ).to("cuda")
    gpu_input = VCTKBatch(
        phonemes=INPUT_PHONEMES.to("cuda"),
        num_phonemes=INPUT_LENGTH,
        speaker_ids=INPUT_SPEAKERS.to("cuda"),
        durations=INPUT_DURATIONS.to("cuda"),
        mels=INPUT_MELS.to("cuda")
    )
    durations, mel_fixed, mel_predicted = model(gpu_input)
    assert (
        durations.shape == expected_duration_shape
    ), f"Wrong shape, expected {expected_duration_shape}, got: {durations.shape}"
    assert (
        mel_predicted.shape == expected_mel_shape
    ), f"Wrong shape, expected {expected_mel_shape}, got: {mel_predicted.shape}"
    assert (
        mel_fixed.shape == expected_mel_shape
    ), f"Wrong shape, expected {expected_mel_shape}, got: {mel_fixed.shape}"
    for idx, length in enumerate(DURATIONS_MAX):
        assert (
            mel_fixed[idx, length:] == 0
        ).all(), "All values of tensor higher sequence length must be zero"
        assert (
            mel_fixed[idx, length - 1] != 0
        ).any(), f"Wrong zero vector for id = {idx}"
        assert (
            mel_predicted[idx, length:] == 0
        ).all(), "All values of tensor higher sequence length must be zero"
        assert (
            mel_predicted[idx, length - 1] != 0
        ).any(), f"Wrong zero vector for id = {idx}"


@patch(
    "src.model.non_attentive_tacotron.DurationPredictor.forward"
)
def test_model_inference(mock_duration: MagicMock) -> None:
    mock_duration.return_value = INPUT_DURATIONS.unsqueeze(2)
    expected_mel_shape = INPUT_MELS.shape

    model = NonAttentiveTacotron(
        N_PHONEMES, N_SPEAKER, N_MELS_DIM, config=MODEL_CONFIG
    )
    mel_predicted = model.inference(MODEL_INFERENCE_INPUT)
    assert (
        mel_predicted.shape == expected_mel_shape
    ), f"Wrong shape, expected {expected_mel_shape}, got: {mel_predicted.shape}"
