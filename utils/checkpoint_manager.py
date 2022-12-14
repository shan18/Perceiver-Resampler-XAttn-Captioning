import os
import shutil

import torch
from torch import nn


class CheckpointManager:
    """Checkpoint Manager to keep track of model training and save the best weights.

    Args:
        model: model to be tracked
        checkpoint_dir: directory to save the checkpoints
        exp_name: name of the experiment
        monitor: metric to monitor
        mode: whether to minimize or maximize the metric
        save_top_k: number of checkpoints to save
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_dir: str,
        exp_name: str,
        monitor: str,
        save_top_k: int,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    ):
        assert monitor in [
            'train_loss',
            'val_loss',
        ], 'Monitor must be either train_loss, val_loss'

        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.exp_name = exp_name
        self.monitor = monitor
        self.mode = 'min' if monitor.endswith('loss') else 'max'
        self.save_top_k = save_top_k
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.recent_checkpoints = []

        self.metrics = {'train_loss': [], 'val_loss': []}
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _log_metrics(self, train_loss: float, eval_loss: float):
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(eval_loss)

    def log(self, epoch: int, train_loss: float, eval_loss: float):
        """Logs the score and saves the checkpoint if the score is the best so far.

        Args:
            epoch: current epoch
            score: score to be logged
        """

        self._log_metrics(train_loss, eval_loss)
        score = self.metrics[self.monitor][-1]
        if (self.mode == 'min' and score < self.best_score) or (self.mode == 'max' and score > self.best_score):
            print(f'{self.monitor} improved from {self.best_score:.4f} to {score:.4f}')
            self.best_score = score
            self.save_checkpoint(epoch, score)
        else:
            print(f'{self.monitor} did not improve. Best {self.monitor} is {self.best_score:.4f}')

    def save_checkpoint(self, epoch: int, score: float):
        self.recent_checkpoints.append(
            os.path.join(self.checkpoint_dir, f'{self.exp_name}-epoch_{epoch}-{self.monitor}_{score:.4f}.pt')
        )
        if len(self.recent_checkpoints) > self.save_top_k:
            oldest_ckpt_file = self.recent_checkpoints.pop(0)
            os.remove(oldest_ckpt_file)

        self.save_state(epoch, self.recent_checkpoints[-1])

    def save_state(self, epoch, path):
        state_dict = {
            'epoch': epoch,
            'model_cfg': self.model.cfg,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        if self.scheduler is not None:
            state_dict['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(state_dict, path)

    def save_current_state(self, epoch):
        path = os.path.join(self.checkpoint_dir, f'{self.exp_name}-last.pt')
        self.save_state(epoch, path)

    def save_best_state(self):
        assert len(self.recent_checkpoints) > 0, 'No checkpoints found'
        shutil.copy(self.recent_checkpoints[-1], os.path.join(self.checkpoint_dir, f'{self.exp_name}.pt'))
        print('Best checkpoint saved at', os.path.join(self.checkpoint_dir, f'{self.exp_name}.pt'))
