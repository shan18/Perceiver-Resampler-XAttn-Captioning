import os

import torch
from torch import nn


class CheckpointManager:
    """Checkpoint Manager to keep track of model training and save the best weights."""

    def __init__(self, model: nn.Module, checkpoint_dir: str, monitor: str, mode: str, save_top_k: int):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.recent_checkpoints = []

        self.best_score = float('inf') if self.mode == 'min' else float('-inf')

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def log(self, epoch: int, score: float):
        if self.mode == 'min':
            if score < self.best_score:
                print(f'{self.monitor} improved from {self.best_score:.4f} to {score:.4f}. Saving model...')
                self.best_score = score
                self.save_checkpoint(epoch, score)
        else:
            if score > self.best_score:
                print(f'{self.monitor} improved from {self.best_score:.4f} to {score:.4f}. Saving model...')
                self.best_score = score
                self.save_checkpoint(epoch, score)

    def save_checkpoint(self, epoch: int, score: float):
        self.recent_checkpoints.append(
            os.path.join(self.checkpoint_dir, f'ckpt-ep_{epoch}-{self.monitor}_{score:.4f}.pt')
        )
        if len(self.recent_checkpoints) > self.save_top_k:
            oldest_ckpt_file = self.recent_checkpoints.pop(0)
            os.remove(oldest_ckpt_file)
        torch.save(self.model.state_dict(), self.recent_checkpoints[-1])
