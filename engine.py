import os
from typing import Union

import torch
from einops import rearrange
from omegaconf import DictConfig
from torch import nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from utils import CheckpointManager, ProgressBar


class Trainer:
    def __init__(
        self, model: nn.Module, log_dir: str, checkpoint_callback_params: Union[dict, DictConfig], device: str = 'cpu'
    ):
        self.model = model.to(device)

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.ckpt_manager = CheckpointManager(
            self.model, os.path.join(self.log_dir, 'checkpoints'), **checkpoint_callback_params
        )
        self.device = device

    def _prepare_for_training(self, optimizer_cfg, num_steps_per_epoch, epochs):
        # Create the loss function
        self.criterion = nn.CrossEntropyLoss()

        # Prepare the optimizer
        assert optimizer_cfg['name'] in ['adamw', 'adam'], 'Optimizer must be either AdamW or Adam'
        if optimizer_cfg['name'] == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=optimizer_cfg['lr'],
                weight_decay=optimizer_cfg['weight_decay'],
                betas=optimizer_cfg['betas'],
            )
        elif optimizer_cfg['name'] == 'adam':
            self.optimizer = Adam(
                self.model.parameters(),
                lr=optimizer_cfg['lr'],
                weight_decay=optimizer_cfg['weight_decay'],
                betas=optimizer_cfg['betas'],
            )

        # Setup scheduler
        self.scheduler = None
        if optimizer_cfg['scheduler']['name'] is not None:
            assert optimizer_cfg['scheduler']['name'] in ['CosineAnnealing'], 'Scheduler must be CosineAnnealing'

            total_steps = num_steps_per_epoch * epochs
            cfg_warmup_steps = optimizer_cfg['scheduler']['warmup_steps']
            cfg_warmup_ratio = optimizer_cfg['scheduler']['warmup_ratio']

            if cfg_warmup_steps is not None:
                assert cfg_warmup_steps < total_steps
                warmup_steps = cfg_warmup_steps
            elif cfg_warmup_ratio is not None:
                assert cfg_warmup_ratio < 1
                warmup_steps = int(total_steps * cfg_warmup_ratio)
            else:
                raise ValueError('Either warmup_steps or warmup_ratio must be specified')

            if optimizer_cfg['scheduler']['name'] == 'CosineAnnealing':
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps,
                )

    def train(self, loader):
        self.model.train()
        pbar = ProgressBar(target=len(loader), width=8)
        for batch_idx, (video, transcript) in enumerate(loader):
            video = video.to(self.device)
            transcript = transcript.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(video)

            # Compute the loss
            outputs = rearrange(outputs, 'b t d -> b d t')
            loss = self.criterion(outputs, transcript)

            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # Update progress bar
            pbar.update(batch_idx, values=[('Loss', round(loss.item(), 4))])

        pbar.add(
            1,
            values=[
                ('Loss', round(loss.item(), 4)),  # type: ignore[reportUnboundVariable]
            ],
        )

    def evaluate(self, loader, data_type='dev'):
        self.model.eval()
        eval_loss = 0
        with torch.no_grad():
            for video, transcript in loader:
                video = video.to(self.device)
                transcript = transcript.to(self.device)

                outputs = self.model(video)

                outputs = rearrange(outputs, 'b t d -> b d t')
                loss = self.criterion(outputs, transcript)
                eval_loss += loss.item()

        # Compute the average loss
        eval_loss /= len(loader)

        print(f'{"Validation" if data_type == "dev" else "Test"} set: ' f'Average loss: {eval_loss:.4f}\n')

        return eval_loss

    def fit(
        self, train_loader: DataLoader, dev_loader: DataLoader, optimizer_cfg: Union[dict, DictConfig], epochs: int
    ):
        self._prepare_for_training(optimizer_cfg, len(train_loader), epochs)

        for epoch in range(1, epochs + 1):
            print(f'\nEpoch {epoch}:')
            self.train(train_loader)
            eval_loss = self.evaluate(dev_loader)

            # Log the progress
            self.ckpt_manager.log(epoch, eval_loss)
