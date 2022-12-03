from typing import Optional

import torch
from omegaconf import DictConfig

from .encoder_decoder import VideoTextModel


def build_model(model_cfg: Optional[DictConfig] = None, pretrained_name: Optional[str] = None, device: str = 'cpu'):
    assert model_cfg is not None or pretrained_name is not None, 'Either model_cfg or pretrained_name must be provided'

    if pretrained_name is not None:
        checkpoint = torch.load(pretrained_name, map_location=device)
        model_cfg = checkpoint['model_cfg']

    model = VideoTextModel(model_cfg.vision, model_cfg.resampler, model_cfg.text, cfg=model_cfg).to(device)  # type: ignore[reportOptionalMemberAccess]

    if pretrained_name is not None:
        model.load_state_dict(checkpoint['model_state_dict'])  # type: ignore[reportUnboundVariable]

    return model
