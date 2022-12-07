# type: ignore[reportOptionalMemberAccess]

from typing import Optional

import torch
from omegaconf import DictConfig, open_dict

from .encoder_decoder import VideoTextModel


def build_model(model_cfg: Optional[DictConfig] = None, pretrained_name: Optional[str] = None, device: str = 'cpu'):
    assert model_cfg is not None or pretrained_name is not None, 'Either model_cfg or pretrained_name must be provided'

    if pretrained_name is not None:
        checkpoint = torch.load(pretrained_name, map_location=device)
        model_cfg = checkpoint['model_cfg']

    # Add visual dim size to text config if resampler is enabled
    if model_cfg.enable_resampler_xattn:
        with open_dict(model_cfg.text.xattn):
            model_cfg.text.xattn.dim_visual = -1  # NOTE: This is a placeholder value. It will be updated in the model

    # Remove xattn parameters if resampler is disabled
    if not model_cfg.enable_resampler_xattn:
        with open_dict(model_cfg.text):
            del model_cfg.text.xattn

    model = VideoTextModel(
        vision_encoder_cfg=model_cfg.vision,
        text_generator_cfg=model_cfg.text,
        resampler_cfg=model_cfg.resampler if model_cfg.enable_resampler_xattn else None,
        cfg=model_cfg,
    ).to(device)

    if pretrained_name is not None:
        model.load_state_dict(checkpoint['model_state_dict'])  # type: ignore[reportUnboundVariable]

    return model
