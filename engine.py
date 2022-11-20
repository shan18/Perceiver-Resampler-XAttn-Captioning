from tqdm import tqdm


class Trainer:

    def __init__(self, model, optimizer, criterion, checkpoint_dir, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.checkpoint_dir = checkpoint_dir
        self.device = device

    def train(self, loader):
        self.model.train()
        pass

    def fit(self, train_loader, dev_loader, epochs):
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}:')
            self.train(train_loader)

            self.evaluate(dev_loader)

            self.save_checkpoint()

    def evaluate(self, loader):
        self.model.eval()
        pass

    def save_checkpoint(self):
        pass
