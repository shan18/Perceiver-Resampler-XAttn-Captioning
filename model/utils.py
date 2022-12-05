import torch
from torch import nn


class SquaredReLU(nn.Module):
    """Squared ReLU activation function"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.pow(torch.relu(x), 2)


def feed_forward_layer(dim: int, mult: int = 4, activation: str = 'gelu'):
    """Feed forward layer with given activation function"""

    activations = dict(gelu=nn.GELU, sqrelu=SquaredReLU, relu=nn.ReLU)
    assert activation in activations, f'activation can only be one of {activations.keys()}'

    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        activations[activation](),
        nn.Linear(inner_dim, dim, bias=False),
    )
