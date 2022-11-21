import os
from tqdm import tqdm


class Trainer:

    def __init__(self, model, optimizer, criterion, checkpoint_dir, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.checkpoint_dir = checkpoint_dir
        self.device = device

    def train(self, loader):
        self.model.train()
        for video, transcript in tqdm(loader):
            video, transcript = video.to(self.device), transcript.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(video, transcript)

            # Compute the loss
            loss = self.criterion(outputs, transcript)

            loss.backward()
            self.optimizer.step()

    def fit(self, train_loader, dev_loader, epochs):
        best_eval_loss = float('inf')
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}:')
            self.train(train_loader)

            eval_loss = self.evaluate(dev_loader)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                self.save_checkpoint(epoch, eval_loss)

    def evaluate(self, loader):
        self.model.eval()
        eval_loss = 0
        with torch.no_grad():
            for video, transcript in tqdm(loader):
                video, transcript = video.to(self.device), transcript.to(self.device)
                outputs = self.model(video, transcript)
                loss = self.criterion(outputs, transcript)
                eval_loss += loss.item()

        return eval_loss / len(loader)

    def save_checkpoint(self, epoch, loss):
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f'ckpt_ep{epoch}_loss{loss}.pt'))
