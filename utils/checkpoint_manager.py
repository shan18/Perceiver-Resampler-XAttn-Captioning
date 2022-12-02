import os

import torch
from torch import nn


class CheckpointManager:
    """Checkpoint Manager to keep track of model training and save the best weights.

    Args:
        model: model to be tracked
        checkpoint_dir: directory to save the checkpoints
        monitor: metric to monitor
        mode: whether to minimize or maximize the metric
        save_top_k: number of checkpoints to save
    """

    def __init__(self, model: nn.Module, checkpoint_dir: str, monitor: str, mode: str, save_top_k: int):
        assert monitor in [
            'train_loss',
            'val_loss',
            'val_bleu',
        ], 'Monitor must be either train_loss, val_loss or val_bleu'

        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.recent_checkpoints = []

        self.metrics = {'train_loss': [], 'val_loss': [], 'val_bleu': []}
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _log_metrics(self, train_loss: float, eval_loss: float, eval_bleu: float):
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(eval_loss)
        self.metrics['val_bleu'].append(eval_bleu)

    def log(self, epoch: int, train_loss: float, eval_loss: float, eval_bleu: float):
        """Logs the score and saves the checkpoint if the score is the best so far.

        Args:
            epoch: current epoch
            score: score to be logged
        """

        self._log_metrics(train_loss, eval_loss, eval_bleu)
        score = self.metrics[self.monitor][-1]
        if (self.mode == 'min' and score < self.best_score) or (self.mode == 'max' and score > self.best_score):
            print(f'{self.monitor} improved from {self.best_score:.4f} to {score:.4f}')
            self.best_score = score
            self.save_checkpoint(epoch, score)
        else:
            print(f'{self.monitor} did not improve. Best {self.monitor} is {self.best_score:.4f}')

    def save_checkpoint(self, epoch: int, score: float):
        self.recent_checkpoints.append(
            os.path.join(self.checkpoint_dir, f'ckpt-epoch_{epoch}-{self.monitor}_{score:.4f}.pt')
        )
        if len(self.recent_checkpoints) > self.save_top_k:
            oldest_ckpt_file = self.recent_checkpoints.pop(0)
            os.remove(oldest_ckpt_file)
        self.model.save_weights(self.recent_checkpoints[-1])
