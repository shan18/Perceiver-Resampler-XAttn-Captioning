from typing import List

from einops import rearrange
from omegaconf import DictConfig
from torch import nn
from transformers import CLIPVisionModel, GPT2LMHeadModel, logging

from .gated_cross_attention import ModifiedLMBlock
from .resampler import PerceiverResampler


class VisionEncoder(nn.Module):
    """Model to encode the video frames.

    Args:
        pretrained_name: name of the pretrained model from huggingface
        trainable: whether the model is trainable
    """

    def __init__(self, pretrained_name: str, trainable: bool):
        super().__init__()

        logging.set_verbosity_error()
        self._model = CLIPVisionModel.from_pretrained(pretrained_name)
        logging.set_verbosity_warning()

        self._update_trainable_state(trainable)

    def _update_trainable_state(self, trainable: bool = False):
        for param in self.parameters():
            param.requires_grad = trainable

    def get_input_shape(self, batch_size: int = 16, timesteps: int = 75):
        return (batch_size, timesteps, 3, 224, 224)

    def get_output_shape(self, batch_size: int = 16):
        return (batch_size, 50, 768)

    def forward(self, video):
        """
        Args:
            video: Batch of video frames with shape (batch_size, timesteps, 3, 224, 224)

        Returns:
            Video embeddings with shape (batch_size, timesteps, hidden_dim, 768)
        """
        batch_size = video.shape[0]

        # Get embeddings for each frame in the video
        video = rearrange(video, 'b t ... -> (b t) ...')
        embeddings = self._model(pixel_values=video).last_hidden_state
        embeddings = rearrange(embeddings, '(b t) ... -> b t ...', b=batch_size)

        return embeddings


class TextGenerator(nn.Module):
    """Model to generate the text.

    Args:
        pretrained_name: name of the pretrained model from huggingface
        trainable: whether the model is trainable
    """

    def __init__(
        self,
        pretrained_name: str,
        trainable: bool,
        dim: int,
        dim_visual: int,
        dim_head: int,
        heads: int,
        num_latents: int,
        ff_mult: int,
        activation: str,
        freq: int,
    ):
        super().__init__()
        self.dim = dim
        self.dim_visual = dim_visual
        self.dim_head = dim_head
        self.heads = heads
        self.num_latents = num_latents
        self.ff_mult = ff_mult
        self.activation = activation
        self.freq = freq

        self._model = GPT2LMHeadModel.from_pretrained(pretrained_name)
        self._update_trainable_state(trainable)

        # Add the gated cross attention layers
        self._add_gated_cross_attention()

    def _update_trainable_state(self, trainable: bool = False):
        for param in self.parameters():
            param.requires_grad = trainable

    def _init_layers(self, lm_layers: nn.ModuleList):
        """Adding cross attention layer between LM layers"""
        self.modified_layers: List[ModifiedLMBlock] = []

        for i, lm_layer in enumerate(lm_layers):
            if i % self.freq != 0:
                continue

            modified_layer = ModifiedLMBlock(
                lm_layer,
                dim=self.dim,
                dim_visual=self.dim_visual,
                dim_head=self.dim_head,
                heads=self.heads,
                n_visual=self.num_latents,
                ff_mult=self.ff_mult,
                act=self.activation,
            )
            self.modified_layers.append(modified_layer)
            lm_layers[i] = modified_layer

    def _add_gated_cross_attention(self):
        self._init_layers(self._model.transformer.h)
        for xattn in self.modified_layers:
            for param in xattn.xattn_block.parameters():
                param.requires_grad = True

    def get_input_shape(self, batch_size: int = 16, timesteps: int = 75, num_latents: int = 64):
        return (batch_size, timesteps, 50, self._model.transformer.embed_dim), (
            batch_size,
            num_latents,
            self._model.transformer.embed_dim,
        )

    def get_output_shape(self, batch_size: int = 16, timesteps: int = 75):
        return (batch_size, timesteps, self._model.lm_head.out_features)

    def forward(self, video_embeddings, resampled_embeddings):
        """
        Args:
            video_embeddings: video embeddings with shape (batch_size, timesteps, hidden_dim, 768)
            resampled_embeddings: video embeddings with shape (batch_size, num_latents, 768)

        Returns:
            Logits of the output transcript
        """
        visual_features = resampled_embeddings.unsqueeze(1)
        for xattn in self.modified_layers:
            xattn.condition(visual_features, None)

        # FIXME: Insert a learnable parameter between GPT and CLIP embeddings
        return self._model(inputs_embeds=video_embeddings.mean(dim=2)).logits


class VideoTextModel(nn.Module):
    """Model to encode the video frames and generate the text.

    Args:
        vision_encoder_cfg: config for the vision encoder
        resampler_cfg: config for the resampler
        text_generator_cfg: config for the text generator
        cfg: full model config. Redundant, but useful when saving and restoring the checkpoint
    """

    def __init__(
        self,
        vision_encoder_cfg: DictConfig,
        resampler_cfg: DictConfig,
        text_generator_cfg: DictConfig,
        cfg: DictConfig = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = VisionEncoder(**vision_encoder_cfg)
        self.resampler = PerceiverResampler(self.vision_encoder.get_output_shape()[-1], **resampler_cfg)
        self.text_generator = TextGenerator(
            text_generator_cfg.pretrained_name, text_generator_cfg.trainable, **text_generator_cfg['xattn']
        )

    def forward(self, video, video_length):
        # Encode video
        video_embeddings = self.vision_encoder(video)

        # Resample video embeddings
        resampled_embeddings = self.resampler(video_embeddings, video_length)

        # Generate text
        text_output = self.text_generator(video_embeddings, resampled_embeddings)

        return text_output
