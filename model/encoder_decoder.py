from typing import List

import torch
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
        self.vision = CLIPVisionModel.from_pretrained(pretrained_name)
        logging.set_verbosity_warning()

        self.dim = self.vision.config.hidden_size

        self._update_trainable_state(trainable)

    def _update_trainable_state(self, trainable: bool = False):
        for param in self.parameters():
            param.requires_grad = trainable

    def forward(self, video, mask: torch.BoolTensor = None):
        """
        Args:
            video: Batch of video frames with shape (batch_size, n_frames, 3, 224, 224)
            mask: Mask tensor for the video frames with shape (batch_size, n_frames)

        Returns:
            Video embeddings with shape (batch_size, n_frames, hidden_dim, 768)
        """
        batch_size = video.shape[0]

        # Get embeddings for each frame in the video
        video = rearrange(video, 'b t ... -> (b t) ...')
        embeddings = self.vision(pixel_values=video).last_hidden_state
        embeddings = rearrange(embeddings, '(b t) ... -> b t ...', b=batch_size)

        # Mask the embeddings
        if mask is not None:
            embeddings[~mask] = 0

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
        dim_visual: int,
        dim_head: int,
        heads: int,
        num_latents: int,
        ff_mult: int,
        activation: str,
        freq: int,
    ):
        super().__init__()
        self.dim_visual = dim_visual
        self.dim_head = dim_head
        self.heads = heads
        self.num_latents = num_latents
        self.ff_mult = ff_mult
        self.activation = activation
        self.freq = freq

        self.lm = GPT2LMHeadModel.from_pretrained(pretrained_name)
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
                dim=self.lm.config.hidden_size,
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
        self._init_layers(self.lm.transformer.h)
        for xattn in self.modified_layers:
            for param in xattn.xattn_block.parameters():
                param.requires_grad = True

    def forward(self, video_embeddings, resampler_embeddings, tokens=None, mask: torch.FloatTensor = None):
        """
        Args:
            video_embeddings: video embeddings with shape (batch_size, n_frames, 768)
            resampler_embeddings: video embeddings with shape (batch_size, num_latents, 768)
            tokens: text tokens with shape (batch_size, n_tokens)
            mask: Attention masks with shape (batch_size, n_frames + n_tokens)

        Returns:
            Logits of the output transcript
        """
        resampler_embeddings = resampler_embeddings.unsqueeze(1)
        for xattn in self.modified_layers:
            xattn.condition(resampler_embeddings, None)

        text_embeddings = self.lm.transformer.wte(tokens)  # (batch_size, n_tokens, 768)
        embeddings = torch.cat((video_embeddings, text_embeddings), dim=1)  # (batch_size, n_frames + n_tokens, 768)
        logits = self.lm(inputs_embeds=embeddings, attention_mask=mask).logits  # (batch_size, n_frames + n_tokens, vocab_size)

        # Consider logits only for the text tokens
        logits = logits[:, video_embeddings.shape[1] - 1:-1]  # (batch_size, n_tokens, vocab_size)

        return logits


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
        self.resampler = PerceiverResampler(self.vision_encoder.dim, **resampler_cfg)
        self.text_generator = TextGenerator(
            text_generator_cfg.pretrained_name, text_generator_cfg.trainable, self.resampler.dim, **text_generator_cfg['xattn']
        )

    def _create_video_mask(self, video_length):
        """Create a mask for the video embeddings.

        Args:
            video_length: length of the video

        Returns:
            Mask for the video embeddings (batch_size, max_length)
        """
        batch_size = video_length.shape[0]
        max_length = video_length.max().item()
        mask = torch.arange(max_length, device=video_length.device).expand(batch_size, max_length) < video_length.unsqueeze(1)
        return mask

    def forward(self, video, video_length, tokens=None, tokens_mask=None):
        # Encode video
        video_mask = self._create_video_mask(video_length)
        video_embeddings = self.vision_encoder(video, mask=video_mask)

        # Resample video embeddings
        resampled_embeddings = self.resampler(video_embeddings, mask=video_mask)

        # Generate text
        text_mask = torch.cat((video_mask.float(), tokens_mask), dim=1)
        # FIXME: Insert a learnable parameter between GPT and CLIP embeddings
        text_output = self.text_generator(video_embeddings.mean(dim=2), resampled_embeddings, tokens=tokens, mask=text_mask)

        return text_output
