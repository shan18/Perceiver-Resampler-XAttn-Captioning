from typing import List

from einops import rearrange
from omegaconf import DictConfig
from torch import nn
from transformers import CLIPVisionModel, GPT2LMHeadModel, logging

from .base_model import BaseModel
from .gated_cross_attention import ModifiedLMBlock
from .resampler import PerceiverResampler


class VisionEncoder(BaseModel):
    """Model to encode the video frames.

    Args:
        pretrained_name: name of the pretrained model from huggingface
        trainable: whether the model is trainable
    """

    def __init__(self, pretrained_name: str, trainable: bool):
        super().__init__(trainable)

        logging.set_verbosity_error()
        self._model = CLIPVisionModel.from_pretrained(pretrained_name)
        logging.set_verbosity_warning()

        self._update_trainable_state()

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


class TextGenerator(BaseModel):
    """Model to generate the text.

    Args:
        pretrained_name: name of the pretrained model from huggingface
        trainable: whether the model is trainable
    """

    def __init__(
        self,
        pretrained_name: str,
        trainable: bool,
        xattn: DictConfig,
    ):
        super().__init__(trainable)
        self.dim = xattn.dim
        self.dim_visual = xattn.dim_visual
        self.dim_head = xattn.dim_head
        self.heads = xattn.heads
        self.num_latents = xattn.num_latents
        self.ff_mult = xattn.ff_mult
        self.activation = xattn.activation
        self.freq = xattn.freq

        self._model = GPT2LMHeadModel.from_pretrained(pretrained_name)
        self._update_trainable_state()
        self.modified_layers: List[ModifiedLMBlock] = []
        self._init_layers(self._model.transformer.h)
        for xattn in self.modified_layers:
            for param in xattn.xattn_block.parameters():
                param.requires_grad = True

    def get_input_shape(self, batch_size: int = 16, timesteps: int = 75):
        return (batch_size, timesteps, self._model.transformer.embed_dim), (batch_size, timesteps)

    def get_output_shape(self, batch_size: int = 16, timesteps: int = 75):
        return (batch_size, timesteps, self._model.lm_head.out_features)

    def _init_layers(self, lm_layers: nn.ModuleList):
        """
        Adding cross attention layer between LM layers
        """
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

    def forward(self, video_embeddings, resampled_embeddings):
        """
        Args:
            input_embeddings: video embeddings with shape (batch_size, seq_length, 768)

        Returns:
            Logits of the output transcript
        """
        visual_features = resampled_embeddings.unsqueeze(1)
        for xattn in self.modified_layers:
            xattn.condition(visual_features, None)

        return self._model(inputs_embeds=video_embeddings.mean(dim=2)).logits


class VideoTextModel(BaseModel):
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
        super().__init__(trainable=True, cfg=cfg)
        self.vision_encoder = VisionEncoder(**vision_encoder_cfg)
        self.resampler = PerceiverResampler(self.vision_encoder.get_output_shape()[-1], **resampler_cfg)
        self.text_generator = TextGenerator(**text_generator_cfg)

    def forward(self, video, video_length):
        # Encode video
        video_embeddings = self.vision_encoder(video)

        # Resample video embeddings
        resampled_embeddings = self.resampler(video_embeddings, video_length)

        # Generate text
        text_output = self.text_generator(video_embeddings, resampled_embeddings)

        return text_output
