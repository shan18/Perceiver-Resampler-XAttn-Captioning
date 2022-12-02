import torch

from .encoder_decoder import VideoTextModel


def build_model(model_cfg, pretrained_name, device):
    model = VideoTextModel(model_cfg.vision, model_cfg.resampler, model_cfg.text, cfg=model_cfg).to(device)
    if pretrained_name is not None:
        model.load_state_dict(torch.load(pretrained_name, map_location=device)['model_state_dict'])
    return model
