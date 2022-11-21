import os

import torch
from einops import rearrange

from utils import ProgressBar


class Trainer:

    def __init__(self, model, optimizer, criterion, checkpoint_dir, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.checkpoint_dir = checkpoint_dir
        self.device = device

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
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f'ckpt_ep{epoch}_loss{loss}.pt'))
