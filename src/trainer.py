import json
import os
from pathlib import Path
from typing import Dict, OrderedDict, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.constants import (
    CHECKPOINT_DIR, FEATURE_MODEL_FILENAME, LOG_DIR, PHONEMES_FILENAME,
    SPEAKERS_FILENAME, MELS_MEAN_FILENAME, MELS_STD_FILENAME
)
from src.data_process import VCTKBatch, VCTKCollate, VCTKDataset, VCTKFactory
from src.model import NonAttentiveTacotron, NonAttentiveTacotronLoss
from src.train_config import TrainParams, load_config


class Trainer:

    OPTIMIZER_FILENAME = "optimizer.pth"
    SCHEDULER_FILENAME = "scheduler.pth"
    ITERATION_FILENAME = "iter.json"
    ITERATION_NAME = "iteration"
    EPOCH_NAME = "epoch"
    SAMPLE_SIZE = 10

    def __init__(self, config_path: Path):
        self.config: TrainParams = load_config(config_path)
        self.checkpoint_path = CHECKPOINT_DIR / self.config.checkpoint_name
        self.log_dir = LOG_DIR / self.config.checkpoint_name
        self.create_dirs()
        self.phonemes_to_id: Dict[str, int] = {}
        self.speakers_to_id: Dict[str, int] = {}
        self.device = torch.device(self.config.device)
        self.mels_weight = self.config.loss.mels_weight
        self.duration_weight = self.config.loss.duration_weight
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.start_epoch = 0
        self.iteration_step = 1
        self.upload_mapping()
        self.train_loader, self.valid_loader = self.prepare_loaders()

        self.feature_model = NonAttentiveTacotron(
            n_mel_channels=self.config.n_mels,
            n_phonems=len(self.phonemes_to_id),
            n_speakers=len(self.speakers_to_id),
            config=self.config.model,
        ).to(self.device)

        self.optimizer = Adam(
            self.feature_model.parameters(),
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.reg_weight,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            eps=self.config.optimizer.adam_epsilon,
        )
        self.scheduler = StepLR(
            optimizer=self.optimizer,
            step_size=self.config.scheduler.decay_steps,
            gamma=self.config.scheduler.decay_rate,
        )
        self.criterion = NonAttentiveTacotronLoss(
            sample_rate=self.config.sample_rate, hop_size=self.config.hop_size
        )
        self.adversatial_criterion = nn.NLLLoss()

        self.upload_checkpoints()

    def batch_to_device(self, batch: VCTKBatch) -> VCTKBatch:
        batch_on_device = VCTKBatch(
            phonemes=batch.phonemes.to(self.device).detach(),
            num_phonemes=batch.num_phonemes.detach(),
            speaker_ids=batch.speaker_ids.to(self.device).detach(),
            durations=batch.durations.to(self.device).detach(),
            mels=batch.mels.to(self.device).detach(),
        )
        return batch_on_device

    def create_dirs(self) -> None:
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def mapping_is_exist(self) -> bool:
        if not os.path.isfile(self.checkpoint_path / SPEAKERS_FILENAME):
            return False
        if not os.path.isfile(self.checkpoint_path / PHONEMES_FILENAME):
            return False
        return True

    def checkpoint_is_exist(self) -> bool:  # noqa: CFQ004
        if not os.path.isfile(self.checkpoint_path / FEATURE_MODEL_FILENAME):
            return False
        if not os.path.isfile(self.checkpoint_path / self.OPTIMIZER_FILENAME):
            return False
        if not os.path.isfile(self.checkpoint_path / self.SCHEDULER_FILENAME):
            return False
        if not os.path.isfile(self.checkpoint_path / self.ITERATION_FILENAME):
            return False
        else:
            with open(self.checkpoint_path / self.ITERATION_FILENAME) as f:
                iter_dict = json.load(f)
                if (
                    self.EPOCH_NAME not in iter_dict
                    or self.ITERATION_NAME not in iter_dict
                ):
                    return False
        return True

    def upload_mapping(self) -> None:
        if self.mapping_is_exist():
            with open(self.checkpoint_path / SPEAKERS_FILENAME) as f:
                self.speakers_to_id.update(json.load(f))
            with open(self.checkpoint_path / PHONEMES_FILENAME) as f:
                self.phonemes_to_id.update(json.load(f))

    def upload_checkpoints(self) -> None:
        if self.checkpoint_is_exist():
            feature_model: NonAttentiveTacotron = torch.load(
                self.checkpoint_path / FEATURE_MODEL_FILENAME, map_location=self.device
            )
            optimizer_state_dict: OrderedDict[str, torch.Tensor] = torch.load(
                self.checkpoint_path / self.OPTIMIZER_FILENAME, map_location=self.device
            )
            scheduler: StepLR = torch.load(
                self.checkpoint_path / self.SCHEDULER_FILENAME, map_location=self.device
            )
            with open(self.checkpoint_path / self.ITERATION_FILENAME) as f:
                iteration_dict: Dict[str, int] = json.load(f)
            self.feature_model = feature_model
            self.optimizer.load_state_dict(optimizer_state_dict)
            self.scheduler = scheduler
            self.start_epoch = iteration_dict[self.EPOCH_NAME]
            self.iteration_step = iteration_dict[self.ITERATION_NAME]

    def save_checkpoint(self, epoch: int, iteration: int) -> None:
        with open(self.checkpoint_path / SPEAKERS_FILENAME, "w") as f:
            json.dump(self.speakers_to_id, f)
        with open(self.checkpoint_path / PHONEMES_FILENAME, "w") as f:
            json.dump(self.phonemes_to_id, f)
        with open(self.checkpoint_path / self.ITERATION_FILENAME, "w") as f:
            json.dump({
                self.EPOCH_NAME: epoch,
                self.ITERATION_NAME: iteration
            }, f)
        torch.save(self.feature_model, self.checkpoint_path / FEATURE_MODEL_FILENAME)
        torch.save(self.optimizer.state_dict(), self.checkpoint_path / self.OPTIMIZER_FILENAME)
        torch.save(self.scheduler, self.checkpoint_path / self.SCHEDULER_FILENAME)
        torch.save(self.train_loader.dataset.mels_mean, self.checkpoint_path / MELS_MEAN_FILENAME)
        torch.save(self.train_loader.dataset.mels_std, self.checkpoint_path / MELS_STD_FILENAME)

    def prepare_loaders(self) -> Tuple[DataLoader[VCTKBatch], DataLoader[VCTKBatch]]:

        factory = VCTKFactory(
            sample_rate=self.config.sample_rate,
            hop_size=self.config.hop_size,
            n_mels=self.config.n_mels,
            config=self.config.data,
            phonemes_to_id=self.phonemes_to_id,
            speakers_to_id=self.speakers_to_id,
        )
        self.phonemes_to_id = factory.phoneme_to_id
        self.speakers_to_id = factory.speaker_to_id
        trainset, valset = factory.split_train_valid(self.config.test_size)
        collate_fn = VCTKCollate()

        train_loader = DataLoader(
            trainset,
            shuffle=False,
            batch_size=self.config.batch_size,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            valset,
            shuffle=False,
            batch_size=self.config.batch_size,
            collate_fn=collate_fn,
        )

        return train_loader, val_loader  # type: ignore

    def write_losses(
        self,
        tag: str,
        total_loss: float,
        prenet_loss: float,
        postnet_loss: float,
        durations_loss: float,
        global_step: int,
    ) -> None:
        self.writer.add_scalar(f"Loss/{tag}/total", total_loss, global_step=global_step)
        self.writer.add_scalar(
            f"Loss/{tag}/prenet", prenet_loss, global_step=global_step
        )
        self.writer.add_scalar(
            f"Loss/{tag}/postnet", postnet_loss, global_step=global_step
        )
        self.writer.add_scalar(
            f"Loss/{tag}/durations", durations_loss, global_step=global_step
        )

    def vocoder_inference(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = tensor.unsqueeze(0).to(self.device)
            y_g_hat = self.vocoder(x)
            audio = y_g_hat.squeeze()
        return audio

    def train(self) -> None:
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)

        self.feature_model.train()

        for epoch in range(self.start_epoch, self.config.epochs):
            for i, batch in enumerate(self.train_loader, start=self.iteration_step):
                global_step = epoch * len(self.train_loader) + i
                batch = self.batch_to_device(batch)
                self.optimizer.zero_grad()
                durations, mel_outputs_postnet, mel_outputs, style_emb, speaker_emb = self.feature_model(batch)
                
                loss_prenet, loss_postnet, loss_durations = self.criterion(
                    mel_outputs,
                    mel_outputs_postnet,
                    durations,
                    batch.durations,
                    batch.mels,
                )

                loss_mel = self.mels_weight * (loss_prenet + loss_postnet)
                loss_durations = self.duration_weight * loss_durations

                loss_full = loss_mel + loss_durations

                loss_full.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.feature_model.parameters(), self.config.grad_clip_thresh
                )

                self.optimizer.step()
                if (
                    self.config.scheduler.start_decay
                    <= global_step
                    <= self.config.scheduler.last_epoch
                ):
                    self.scheduler.step()

                if global_step % self.config.log_steps == 0:
                    self.write_losses(
                        "train",
                        loss_full,
                        loss_prenet,
                        loss_postnet,
                        loss_durations,
                        global_step,
                    )

                if global_step % self.config.iters_per_checkpoint == 0:
                    self.feature_model.eval()
                    self.validate(
                        global_step=global_step,
                    )
                    self.generate_samples(
                        global_step=global_step,
                    )
                    self.save_checkpoint(
                        epoch=epoch,
                        iteration=i,
                    )
                    self.feature_model.train()

            self.iteration_step = 1
        self.writer.close()

    def validate(self, global_step: int) -> None:
        with torch.no_grad():
            val_loss = 0.0
            val_loss_prenet = 0.0
            val_loss_postnet = 0.0
            val_loss_durations = 0.0
            for batch in self.valid_loader:
                batch = self.batch_to_device(batch)
                durations, mel_outputs_postnet, mel_outputs, _, _ = self.feature_model(batch)
                loss_prenet, loss_postnet, loss_durations = self.criterion(
                    mel_outputs,
                    mel_outputs_postnet,
                    durations,
                    batch.durations,
                    batch.mels,
                )
                loss_mel = self.mels_weight * (loss_prenet + loss_postnet)
                loss_durations = self.duration_weight * loss_durations

                loss = loss_mel + loss_durations
                val_loss += loss.item()
                val_loss_prenet += loss_prenet.item()
                val_loss_postnet += loss_postnet.item()
                val_loss_durations += loss_durations.item()

            val_loss = val_loss / len(self.valid_loader)
            val_loss_prenet = val_loss_prenet / len(self.valid_loader)
            val_loss_postnet = val_loss_postnet / len(self.valid_loader)
            val_loss_durations = val_loss_durations / len(self.valid_loader)
            self.write_losses(
                "valid",
                val_loss,
                val_loss_prenet,
                val_loss_postnet,
                val_loss_durations,
                global_step,
            )

    def generate_samples(
        self,
        global_step: int,
    ) -> None:
        valid_dataset: VCTKDataset = self.valid_loader.dataset  # type: ignore
        mels_mean = valid_dataset.mels_mean
        mels_std = valid_dataset.mels_std
        idx: np.ndarray = np.random.choice(
            np.arange(len(valid_dataset)), self.SAMPLE_SIZE, replace=False
        )
        for i in range(len(idx)):
            sample = valid_dataset[idx[i]]
            batch = (
                torch.LongTensor([sample.phonemes]).to(self.device),
                torch.LongTensor([sample.num_phonemes]),
                torch.LongTensor([sample.speaker_id]).to(self.device),
                torch.FloatTensor(sample.mels).to(self.device).permute(0, 2, 1),
            )
            output = self.feature_model.inference(batch)
            output = output.permute(0, 2, 1).squeeze(0)
            output = output * mels_std.to(self.device) + mels_mean.to(self.device)
            audio = self.vocoder_inference(output)
            self.writer.add_audio(
                f"Audio/Val/{i}",
                audio,
                sample_rate=self.config.sample_rate,
                global_step=global_step,
            )
