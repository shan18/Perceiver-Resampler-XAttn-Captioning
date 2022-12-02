import torch
from omegaconf import DictConfig
from torch import nn
from torchinfo import summary


class BaseModel(nn.Module):
    def __init__(self, trainable: bool, cfg: DictConfig = None):
        super().__init__()
        self.trainable = trainable
        self.cfg = cfg

    def _update_trainable_state(self):
        """Freeze or unfreeze the model"""
        for param in self.parameters():
            param.requires_grad = self.trainable

    def get_input_shape(self):
        """Get the input shape of the model"""
        raise NotImplementedError

    def get_output_shape(self):
        """Get the output shape of the model"""
        raise NotImplementedError

    def freeze(self):
        self.trainable = False
        self._update_trainable_state()

    def unfreeze(self):
        self.trainable = True
        self._update_trainable_state()

    def summary(self):
        return summary(self)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
