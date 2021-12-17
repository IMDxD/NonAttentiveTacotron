import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.constants import CHECKPOINT_DIR, LOG_DIR
from src.data_process import VCTKBatch, VctkCollate, VctkDataset, VCTKFactory
from src.data_process.constanst import MELS_MEAN, MELS_STD
from src.models import NonAttentiveTacotron, NonAttentiveTacotronLoss
from src.train_config import TrainParams, load_config

MODEL_NAME = "model.pth"
SAMPLE_SIZE = 10


def prepare_dataloaders(
    checkpoint: Path, config: TrainParams
) -> Tuple[DataLoader, DataLoader, int, int]:
    # Get data, data loaders and collate function ready
    phonemes_file = checkpoint / VCTKFactory.PHONEMES_JSON_NAME
    speakers_file = checkpoint / VCTKFactory.SPEAKER_JSON_NAME
    if os.path.isfile(phonemes_file):
        with open(phonemes_file, "r") as f:
            phonemes_to_id = json.load(f)
    else:
        phonemes_to_id = None
    if os.path.isfile(speakers_file):
        with open(speakers_file, "r") as f:
            speakers_to_id = json.load(f)
    else:
        speakers_to_id = None
    factory = VCTKFactory(
        sample_rate=config.sample_rate,
        hop_size=config.hop_size,
        config=config.data,
        phonemes_to_id=phonemes_to_id,
        speakers_to_id=speakers_to_id,
    )
    factory.save_mapping(checkpoint)
    trainset, valset = factory.split_train_valid(config.test_size)
    collate_fn = VctkCollate()

    train_loader = DataLoader(
        trainset,
        shuffle=False,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        valset,
        shuffle=False,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
    )

    return (
        train_loader,
        val_loader,
        len(factory.phoneme_to_id),
        len(factory.speaker_to_id),
    )


def load_model(
    config: TrainParams, n_phonemes: int, n_speakers: int
) -> NonAttentiveTacotron:
    model = NonAttentiveTacotron(
        n_mel_channels=config.n_mels,
        n_phonems=n_phonemes,
        n_speakers=n_speakers,
        device=torch.device(config.device),
        config=config.model,
    )
    return model


def load_checkpoint(
    checkpoint_path: Path,
    model: NonAttentiveTacotron,
    optimizer: Adam,
    scheduler: StepLR,
) -> [NonAttentiveTacotron, Adam, StepLR]:
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    scheduler.load_state_dict(checkpoint_dict['scheduler'])
    return model, optimizer, scheduler


def save_checkpoint(
    filepath: Path, model: NonAttentiveTacotron, optimizer: Adam, scheduler: StepLR
):
    torch.save(
        {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler,
        },
        filepath,
    )


def batch_to_device(batch: VCTKBatch, device: torch.device) -> VCTKBatch:
    batch_on_device = VCTKBatch(
        phonemes=batch.phonemes.to(device),
        num_phonemes=batch.num_phonemes,
        speaker_ids=batch.speaker_ids.to(device),
        durations=batch.durations.to(device),
        mels=batch.mels.to(device),
    )
    return batch_on_device


def validate(
    model: NonAttentiveTacotron,
    criterion: NonAttentiveTacotronLoss,
    val_loader: DataLoader,
    mels_weight: float,
    duration_weight: float,
    global_step: int,
    writer: SummaryWriter,
) -> None:

    model.eval()
    with torch.no_grad():

        val_loss = 0.0
        val_loss_prenet = 0.0
        val_loss_postnet = 0.0
        val_loss_durations = 0.0
        for i, batch in enumerate(val_loader):
            batch = batch_to_device(batch, model.device)
            durations, mel_outputs_postnet, mel_outputs = model(batch)
            loss_prenet, loss_postnet, loss_durations = criterion(
                mel_outputs, mel_outputs_postnet, durations, batch.durations, batch.mels
            )
            loss = (
                mels_weight * (loss_prenet + loss_postnet)
                + duration_weight * loss_durations
            )
            val_loss += loss.item()
            val_loss_prenet += loss_prenet.item()
            val_loss_postnet += loss_postnet.item()
            val_loss_durations += loss_durations.item()

        val_loss = val_loss / (i + 1)
        val_loss_prenet = val_loss_prenet / (i + 1)
        val_loss_postnet = val_loss_postnet / (i + 1)
        val_loss_durations = val_loss_durations / (i + 1)
        writer.add_scalar(
            "Loss/valid/total", scalar_value=val_loss, global_step=global_step
        )
        writer.add_scalar(
            "Loss/valid/prenet", scalar_value=val_loss_prenet, global_step=global_step
        )
        writer.add_scalar(
            "Loss/valid/postnet", scalar_value=val_loss_postnet, global_step=global_step
        )
        writer.add_scalar(
            "Loss/valid/durations",
            scalar_value=val_loss_durations,
            global_step=global_step,
        )

    model.train()


def generate_samples(
    model: NonAttentiveTacotron,
    sample_rate: int,
    global_step: int,
    generator,
    val_data: VctkDataset,
    device: torch.device,
    writer: SummaryWriter,
):
    idx = np.random.choice(np.arange(len(val_data)), SAMPLE_SIZE, replace=False)
    for i in idx:
        sample = val_data[i]
        batch = (
            torch.LongTensor([sample.phonemes]).to(device),
            torch.LongTensor([sample.num_phonemes]),
            torch.LongTensor([sample.speaker_id]).to(device),
        )
        output = model.inference(batch)
        output = output.permute(0, 2, 1).squeeze(0)
        output = output * MELS_STD.to(device) + MELS_MEAN.to(device)
        audio = inference(generator, output, device)
        writer.add_audio(
            "Audio/Val", audio, sample_rate=sample_rate * 2, global_step=global_step
        )


def train(config: TrainParams):

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    checkpoint_path = CHECKPOINT_DIR / config.checkpoint_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_dir = LOG_DIR / config.checkpoint_name
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device)

    train_loader, val_loader, phonemes_count, speaker_count = prepare_dataloaders(
        checkpoint_path, config
    )
    model = load_model(config, phonemes_count, speaker_count)
    generator = load_hifi(config.hifi, device)

    optimizer_config = config.optimizer
    optimizer = Adam(
        model.parameters(),
        lr=optimizer_config.learning_rate,
        weight_decay=optimizer_config.reg_weight,
        betas=(optimizer_config.adam_beta1, optimizer_config.adam_beta2),
        eps=optimizer_config.adam_epsilon,
    )

    scheduler = StepLR(
        optimizer=optimizer,
        step_size=config.scheduler.decay_steps,
        gamma=config.scheduler.decay_rate,
    )

    criterion = NonAttentiveTacotronLoss(
        sample_rate=config.sample_rate, hop_size=config.hop_size
    )

    if os.path.isfile(checkpoint_path / MODEL_NAME):
        model, optimizer, scheduler = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    model.train()
    writer = SummaryWriter(log_dir=log_dir)

    mels_weight = config.loss.mels_weight
    duration_weight = config.loss.duration_weight

    for epoch in range(config.epochs):
        for i, batch in enumerate(train_loader, start=1):
            global_step = epoch * len(train_loader) + i
            batch = batch_to_device(batch, device)
            optimizer.zero_grad()
            durations, mel_outputs_postnet, mel_outputs = model(batch)

            loss_prenet, loss_postnet, loss_durations = criterion(
                mel_outputs,
                mel_outputs_postnet,
                durations,
                batch.durations,
                batch.mels,
            )

            loss = (
                mels_weight * (loss_prenet + loss_postnet)
                + duration_weight * loss_durations
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_thresh)

            optimizer.step()
            if config.scheduler.start_decay <= i + 1 <= config.scheduler.last_epoch:
                scheduler.step()

            if global_step % config.log_steps == 0:
                writer.add_scalar("Loss/train/total", loss, global_step=global_step)
                writer.add_scalar(
                    "Loss/train/prenet", loss_prenet, global_step=global_step
                )
                writer.add_scalar(
                    "Loss/train/postnet", loss_postnet, global_step=global_step
                )
                writer.add_scalar(
                    "Loss/train/durations", loss_durations, global_step=global_step
                )

            if global_step % config.iters_per_checkpoint == 0:
                validate(
                    model=model,
                    criterion=criterion,
                    val_loader=val_loader,
                    mels_weight=mels_weight,
                    duration_weight=duration_weight,
                    global_step=global_step,
                    writer=writer,
                )
                generate_samples(
                    model=model,
                    generator=generator,
                    sample_rate=config.sample_rate,
                    val_data=val_loader.dataset,
                    global_step=global_step,
                    device=device,
                    writer=writer,
                )
                save_checkpoint(
                    filepath=checkpoint_path / MODEL_NAME,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=True, help='configuration file path'
    )
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == '__main__':
    main()
