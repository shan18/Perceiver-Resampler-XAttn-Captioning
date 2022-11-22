import os

import torch
from einops import rearrange
from torch import nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from utils import ProgressBar


class Trainer:

    def __init__(self, model: nn.Module, checkpoint_dir: str, device: str = 'cpu'):
        self.model = model.to(device)
        self.checkpoint_dir = checkpoint_dir
        self.recent_checkpoints = []
        self.device = device

    def _prepare_for_training(self, num_steps_per_epoch, epochs):
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = AdamW(self.model.parameters(), lr=0.001)

        total_steps = num_steps_per_epoch * epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=total_steps // 10,  # 10% of the total number of steps
            num_training_steps=total_steps,
        )

    def train(self, loader):
        self.model.train()
        pbar = ProgressBar(target=len(loader), width=8)
        for batch_idx, (video, transcript, text_attn_mask) in enumerate(loader):
            video = video.to(self.device)
            transcript = transcript.to(self.device)
            text_attn_mask = text_attn_mask.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(video, text_attn_mask)

            # Compute the loss
            outputs = rearrange(outputs, 'b t d -> b d t')
            loss = self.criterion(outputs, transcript)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Update progress bar
            pbar.update(batch_idx, values=[('Loss', round(loss.item(), 4))])

        pbar.add(1, values=[
            ('Loss', round(loss.item(), 4)),
        ])

    def evaluate(self, loader, data_type='dev'):
        self.model.eval()
        eval_loss = 0
        with torch.no_grad():
            for video, transcript, text_attn_mask in loader:
                video = video.to(self.device)
                transcript = transcript.to(self.device)
                text_attn_mask = text_attn_mask.to(self.device)

                outputs = self.model(video, text_attn_mask)

                outputs = rearrange(outputs, 'b t d -> b d t')
                loss = self.criterion(outputs, transcript)
                eval_loss += loss.item()

        # Compute the average loss
        eval_loss /= len(loader)

        print(
            f'{"Validation" if data_type == "dev" else "Test"} set: '
            f'Average loss: {eval_loss:.4f}\n'
        )

        return eval_loss

    def fit(self, train_loader, dev_loader, epochs):
        self._prepare_for_training(len(train_loader), epochs)
        best_eval_loss = float('inf')
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}:')
            self.train(train_loader)

            eval_loss = self.evaluate(dev_loader)
            if eval_loss < best_eval_loss:
                print(f'Validation loss improved from {best_eval_loss:.4f} to {eval_loss:.4f}. Saving model...')
                best_eval_loss = eval_loss
                self.save_checkpoint(epoch, eval_loss)

    def save_checkpoint(self, epoch, loss):
        self.recent_checkpoints.append(os.path.join(self.checkpoint_dir, f'ckpt-ep_{epoch}-loss_{loss:.4f}.pt'))
        if len(self.recent_checkpoints) > 5:  # TODO: Assign the top k with a parameter
            oldest_ckpt_file = self.recent_checkpoints.pop(0)
            os.remove(oldest_ckpt_file)
        torch.save(self.model.state_dict(), self.recent_checkpoints[-1])
