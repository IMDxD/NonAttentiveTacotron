import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import tgt
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data_process.config import VCTKDatasetParams

NUMBER = Union[int, float]
PHONES_TIER = "phones"
PAD_TOKEN = "<PAD>"


@dataclass
class VCTKSample:

    phonemes: List[int]
    num_phonemes: int
    speaker_id: int
    durations: np.ndarray
    mels: torch.Tensor


@dataclass
class VCTKInfo:

    text_path: Path
    mel_path: Path
    speaker_id: int
    phonemes_length: int


@dataclass
class VCTKBatch:

    phonemes: torch.Tensor
    num_phonemes: torch.Tensor
    speaker_ids: torch.Tensor
    durations: torch.Tensor
    mels: torch.Tensor


class VCTKDataset(Dataset[VCTKSample]):

    def __init__(
            self,
            sample_rate: int,
            hop_size: int,
            mels_mean: torch.Tensor,
            mels_std: torch.Tensor,
            phoneme_to_ids: Dict[str, int],
            data: List[VCTKInfo]
    ):
        self._phoneme_to_id = phoneme_to_ids
        self._dataset = data
        self._dataset.sort(key=lambda x: x.phonemes_length)
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.mels_mean = mels_mean
        self.mels_std = mels_std

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> VCTKSample:

        info = self._dataset[idx]
        text_grid = tgt.read_textgrid(info.text_path)
        phones_tier = text_grid.get_tier_by_name(PHONES_TIER)
        phoneme_ids = [self._phoneme_to_id[x.text] for x in phones_tier.get_copy_with_gaps_filled()]

        durations = np.array(
            [
                self.seconds_to_frame(x.duration())
                for x in phones_tier.get_copy_with_gaps_filled()
            ],
            dtype=np.float32
        )

        mels: torch.Tensor = torch.load(info.mel_path)
        mels = (mels - self.mels_mean) / self.mels_std

        pad_size = mels.shape[-1] - np.int64(durations.sum())
        if pad_size < 0:
            durations[-1] += pad_size
            assert durations[-1] >= 0
        if pad_size > 0:
            phoneme_ids.append(self._phoneme_to_id[PAD_TOKEN])
            np.append(durations, pad_size)

        return VCTKSample(
            phonemes=phoneme_ids,
            num_phonemes=len(phoneme_ids),
            speaker_id=info.speaker_id,
            mels=mels,
            durations=durations
        )

    def seconds_to_frame(self, seconds: float) -> float:
        return seconds * self.sample_rate / self.hop_size


class VCTKFactory:

    """Create VCTK Dataset

    Note:
        * All the speeches from speaker ``p315`` will be skipped due to the lack of the corresponding text files.
        * All the speeches from ``p280`` will be skipped for ``mic_id="mic2"`` due to the lack of the audio files.
        * Some of the speeches from speaker ``p362`` will be skipped due to the lack of  the audio files.
        * See Also: https://datashare.is.ed.ac.uk/handle/10283/3443
        * Make sure to put the files as the following structure:
            text
            ├── p225
            |   ├──p225_001.TextGrid
            |   ├──p225_002.TextGrid
            |   └──...
            └── pXXX
                ├──pXXX_YYY.TextGrid
                └──...
            mels
            ├── p225
            |   ├──p225_001.pkl
            |   ├──p225_002.pkl
            |   └──...
            └── pXXX
                ├──pXXX_YYY.pkl
                └──...
    """

    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        n_mels: int,
        config: VCTKDatasetParams,
        phonemes_to_id: Dict[str, int],
        speakers_to_id: Dict[str, int],
    ):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.n_mels = n_mels
        self._mels_dir = Path(config.mels_dir)
        self._text_dir = Path(config.text_dir)
        self._text_ext = config.text_ext
        self._mels_ext = config.mels_ext
        self.phoneme_to_id: Dict[str, int] = phonemes_to_id
        self.phoneme_to_id[PAD_TOKEN] = 0
        self.speaker_to_id: Dict[str, int] = speakers_to_id
        self._dataset: List[VCTKInfo] = self._build_dataset()
        self.mels_mean, self.mels_std = self._get_mean_and_std()

    @staticmethod
    def add_to_mapping(mapping: Dict[str, int], token: str) -> None:
        if token not in mapping:
            mapping[token] = len(mapping)

    def split_train_valid(
        self, test_fraction: float
    ) -> Tuple[VCTKDataset, VCTKDataset]:
        speakers_to_data_id: Dict[int, List[int]] = defaultdict(list)

        for i, sample in enumerate(self._dataset):
            speakers_to_data_id[sample.speaker_id].append(i)
        test_ids: List[int] = []
        for ids in speakers_to_data_id.values():
            test_size = int(len(ids) * test_fraction)
            if test_size > 0:
                test_indexes = random.choices(ids, k=test_size)
                test_ids.extend(test_indexes)

        train_data = []
        test_data = []
        for i in range(len(self._dataset)):
            if i in test_ids:
                test_data.append(self._dataset[i])
            else:
                train_data.append(self._dataset[i])
        train_dataset = VCTKDataset(
            sample_rate=self.sample_rate,
            hop_size=self.hop_size,
            mels_mean=self.mels_mean,
            mels_std=self.mels_std,
            phoneme_to_ids=self.phoneme_to_id,
            data=train_data
        )
        test_dataset = VCTKDataset(
            sample_rate=self.sample_rate,
            hop_size=self.hop_size,
            mels_mean=self.mels_mean,
            mels_std=self.mels_std,
            phoneme_to_ids=self.phoneme_to_id,
            data=test_data
        )
        return train_dataset, test_dataset

    def _build_dataset(self) -> List[VCTKInfo]:

        dataset: List[VCTKInfo] = []
        texts_set = {
            Path(x.parent.name) / x.stem
            for x in self._text_dir.rglob(f"*{self._text_ext}")
        }
        mels_set = {
            Path(x.parent.name) / x.stem
            for x in self._mels_dir.rglob(f"*{self._mels_ext}")
        }
        samples = list(mels_set & texts_set)
        for sample in tqdm(samples):
            tg_path = (self._text_dir / sample).with_suffix(self._text_ext)
            mels_path = (self._mels_dir / sample).with_suffix(self._mels_ext)

            text_grid = tgt.read_textgrid(tg_path)
            self.add_to_mapping(self.speaker_to_id, sample.parent.name)
            speaker_id = self.speaker_to_id[sample.parent.name]

            if PHONES_TIER in text_grid.get_tier_names():

                phones_tier = text_grid.get_tier_by_name(PHONES_TIER)
                phonemes = [x.text for x in phones_tier.get_copy_with_gaps_filled()]

                for phoneme in phonemes:
                    self.add_to_mapping(self.phoneme_to_id, phoneme)

                dataset.append(
                    VCTKInfo(
                        text_path=tg_path,
                        mel_path=mels_path,
                        phonemes_length=len(phonemes),
                        speaker_id=speaker_id
                    )
                )

        return dataset

    def _get_mean_and_std(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mel_sum = torch.zeros((self.n_mels), dtype=torch.float64)
        mel_squared_sum = torch.zeros((self.n_mels), dtype=torch.float64)

        print(torch.load(self._dataset[0].mel_path).shape)

        for info in tqdm(self._dataset, desc="Computing mels mean and std"):
            mels: torch.Tensor = torch.load(info.mel_path)
            mels_mean: torch.Tensor = mels.mean(dim=-1).squeeze(0)
            mel_sum += mels_mean
            mel_squared_sum += mels_mean.pow(2)

        mels_mean: torch.Tensor = mel_sum / len(self._dataset)
        mels_std: torch.Tensor = (
            mel_squared_sum - mel_sum * mel_sum / len(self._dataset)
        ) / len(self._dataset)

        return mels_mean.view(-1, 1), mels_std.view(-1, 1)


class VCTKCollate:
    """
    Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_per_step: int = 1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch: List[VCTKSample]) -> VCTKBatch:
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [{...}, {...}, ...]
        """
        # Right zero-pad all one-hot text sequences to max input length
        batch_size = len(batch)
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x.phonemes) for x in batch]),
            dim=0,
            descending=True,
        )
        max_input_len = int(input_lengths[0])

        input_speaker_ids = torch.LongTensor(
            [batch[i].speaker_id for i in ids_sorted_decreasing]
        )

        text_padded = torch.zeros((batch_size, max_input_len), dtype=torch.long)
        durations_padded = torch.zeros((batch_size, max_input_len), dtype=torch.float)
        for i, idx in enumerate(ids_sorted_decreasing):
            text = batch[idx].phonemes
            text_padded[i, : len(text)] = torch.LongTensor(text)
            durations = batch[idx].durations
            durations_padded[i, : len(durations)] = torch.FloatTensor(durations)

        num_mels = batch[0].mels.squeeze(0).size(0)
        max_target_len = max([x.mels.squeeze(0).size(1) for x in batch])

        # include mel padded and gate padded
        mel_padded = torch.zeros(
            (batch_size, num_mels, max_target_len), dtype=torch.float
        )
        for i, idx in enumerate(ids_sorted_decreasing):
            mel: torch.Tensor = batch[idx].mels.squeeze(0)
            mel_padded[i, :, : mel.shape[1]] = mel
        mel_padded = mel_padded.permute(0, 2, 1)

        return VCTKBatch(
            phonemes=text_padded,
            num_phonemes=input_lengths,
            speaker_ids=input_speaker_ids,
            durations=durations_padded,
            mels=mel_padded
        )
