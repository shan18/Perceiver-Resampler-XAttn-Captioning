import os

import torch

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
        for batch_idx, (video, transcript) in enumerate(loader, 0):
            video, transcript = video.to(self.device), transcript.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(video, transcript)

            # Compute the loss
            loss = self.criterion(outputs, transcript)

            loss.backward()
            self.optimizer.step()

            # Update progress bar
            pbar.update(batch_idx, values=[('Loss', round(loss.item(), 4))])

        pbar.add(1, values=[
            ('Loss', round(loss.item(), 4)),
        ])

    def fit(self, train_loader, dev_loader, epochs):
        best_eval_loss = float('inf')
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}:')
            self.train(train_loader)

            eval_loss = self.evaluate(dev_loader)
            if eval_loss < best_eval_loss:
                print(f'Validation loss improved from {best_eval_loss:.4f} to {eval_loss:.4f}. Saving model...')
                best_eval_loss = eval_loss
                self.save_checkpoint(epoch, eval_loss)

    def evaluate(self, loader, data_type='dev'):
        self.model.eval()
        eval_loss = 0
        with torch.no_grad():
            for video, transcript in loader:
                video, transcript = video.to(self.device), transcript.to(self.device)
                outputs = self.model(video, transcript)
                loss = self.criterion(outputs, transcript)
                eval_loss += loss.item()

        # Compute the average loss
        eval_loss /= len(loader)

        print(
            f'{"Validation" if data_type == "dev" else "Test"} set: '
            f'Average loss: {eval_loss:.4f}\n'
        )

        return eval_loss

    def save_checkpoint(self, epoch, loss):
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f'ckpt_ep{epoch}_loss{loss}.pt'))
