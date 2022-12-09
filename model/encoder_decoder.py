from typing import List, Optional

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
        self.num_features = self.vision.vision_model.embeddings.position_embedding.weight.shape[0]

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
        trainable_lm_head: bool,
        dim_visual: Optional[int] = None,
        dim_head: Optional[int] = None,
        heads: Optional[int] = None,
        num_latents: Optional[int] = None,
        ff_mult: Optional[int] = None,
        activation: Optional[str] = None,
        freq: Optional[int] = None,
    ):
        super().__init__()
        self.enable_gated_xattn = dim_visual is not None
        if self.enable_gated_xattn:
            self.dim_visual = dim_visual
            self.dim_head = dim_head
            self.heads = heads
            self.num_latents = num_latents
            self.ff_mult = ff_mult
            self.activation = activation
            self.freq = freq

        self.lm = GPT2LMHeadModel.from_pretrained(pretrained_name)
        self._update_trainable_state(trainable, trainable_lm_head)

        # Add the gated cross attention layers
        if self.enable_gated_xattn:
            self._add_gated_cross_attention()

    def _update_trainable_state(self, trainable: bool = False, trainable_lm_head: bool = False):
        # Update the trainable state of the LM
        for param in self.parameters():
            param.requires_grad = trainable

        # Update the trainable state of the LM head
        for param in self.lm.lm_head.parameters():
            param.requires_grad = trainable_lm_head

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

    def forward(self, video_embeddings, resampler_embeddings=None, tokens=None, mask: torch.FloatTensor = None):
        """
        Args:
            video_embeddings: video embeddings with shape (batch_size, n_frames, 768)
            resampler_embeddings: video embeddings with shape (batch_size, num_latents, 768)
            tokens: text tokens with shape (batch_size, n_tokens)
            mask: Attention masks with shape (batch_size, n_frames + n_tokens)

        Returns:
            Logits of the output transcript
        """
        if self.enable_gated_xattn:
            assert resampler_embeddings is not None, "Resampler embeddings are required for gated cross attention"
            resampler_embeddings = resampler_embeddings.unsqueeze(1)  # type: ignore[reportOptionalMemberAccess]
            for xattn in self.modified_layers:
                xattn.condition(resampler_embeddings, None)

        text_embeddings = self.lm.transformer.wte(tokens)  # (batch_size, n_tokens, 768)
        embeddings = torch.cat((video_embeddings, text_embeddings), dim=1)  # (batch_size, n_frames + n_tokens, 768)
        logits = self.lm(
            inputs_embeds=embeddings, attention_mask=mask
        ).logits  # (batch_size, n_frames + n_tokens, vocab_size)

        # Consider logits only for the text tokens
        logits = logits[:, video_embeddings.shape[1] - 1 : -1]  # (batch_size, n_tokens, vocab_size)

        return logits


class TransformerMapper(nn.Module):
    """Map video embeddings to text generator embeddings.

    Args:
        embedding_dim: dimensions of the visual embeddings
        depth: number of encoder layers
        heads: number of heads in the multi-head attention
        feature_avg_mode: mode for averaging the num_features in the visual embeddings.
            Can be either 'mean' or 'mlp'
        num_features: number of features in the visual embeddings
        trainable: whether the mapper layers are trainable
    """

    def __init__(
        self,
        embedding_dim: int,
        depth: int,
        heads: int,
        dim_feedforward: int,
        feature_avg_mode: Optional[str] = 'mean',
        num_features: Optional[int] = None,
        trainable: Optional[bool] = True,
    ):
        super().__init__()
        assert feature_avg_mode in ['mean', 'mlp'], 'feature_avg_mode must be either mean or mlp'
        self.feature_avg_mode = feature_avg_mode

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, heads, dim_feedforward=dim_feedforward, batch_first=True),
            num_layers=depth,
        )

        if feature_avg_mode == 'mlp':
            assert num_features is not None, 'num_features is required for MLP feature averaging'
            self.linear = nn.Linear(num_features * embedding_dim, embedding_dim)

        self._update_trainable_state(trainable)

    def _update_trainable_state(self, trainable: bool = True):
        for param in self.parameters():
            param.requires_grad = trainable

    def forward(self, x, mask: torch.BoolTensor):
        """
        Args:
            video_embeddings: video embeddings with shape (batch_size, n_frames, 50, 768)
            mask: attention mask with shape (batch_size, n_frames)

        Returns:
            Mapped embeddings
        """
        if self.feature_avg_mode == 'mean':
            x = x.mean(dim=2)  # (batch_size, n_frames, 768)
        elif self.feature_avg_mode == 'linear':
            x = rearrange(x, 'b t n e -> b t (n e)')  # (batch_size, n_frames, 50 * 768)
            x = self.linear(x)  # (batch_size, n_frames, 768)

        out = self.transformer(x, src_key_padding_mask=~mask)
        return out


class VideoTextModel(nn.Module):
    """Model to encode the video frames and generate the text.

    Args:
        vision_encoder_cfg: config for the vision encoder
        resampler_cfg: config for the resampler
        text_generator_cfg: config for the text generator
        mapper_cfg: config for the mapper in between clip and GPT
        cfg: full model config. Redundant, but useful when saving and restoring the checkpoint
    """

    def __init__(
        self,
        vision_encoder_cfg: DictConfig,
        text_generator_cfg: DictConfig,
        mapper_cfg: Optional[DictConfig] = None,
        resampler_cfg: Optional[DictConfig] = None,
        cfg: Optional[DictConfig] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.enable_resampler_xattn = resampler_cfg is not None
        self.enable_mapper = mapper_cfg is not None

        # Build the vision encoder
        self.vision_encoder = VisionEncoder(**vision_encoder_cfg)

        # Build the mapper
        if self.enable_mapper:
            self.mapper = TransformerMapper(
                self.vision_encoder.dim, **mapper_cfg, num_features=self.vision_encoder.num_features
            )

        # Build the resampler
        if self.enable_resampler_xattn:
            text_generator_cfg['xattn']['dim_visual'] = self.vision_encoder.dim
            self.resampler = PerceiverResampler(self.vision_encoder.dim, **resampler_cfg)

        # Build the text generator
        xattn_kwargs = {} if not self.enable_resampler_xattn else text_generator_cfg['xattn']
        self.text_generator = TextGenerator(
            text_generator_cfg.pretrained_name,
            text_generator_cfg.trainable,
            text_generator_cfg.trainable_lm_head,
            **xattn_kwargs
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
        mask = torch.arange(max_length, device=video_length.device).expand(
            batch_size, max_length
        ) < video_length.unsqueeze(1)
        return mask

    def forward(self, video, video_length, tokens=None, tokens_mask=None):
        # Encode video
        video_mask = self._create_video_mask(video_length)
        video_embeddings = self.vision_encoder(video, mask=video_mask)

        # Pass the video embeddings through the mapper
        video_embeddings_mapped = (
            video_embeddings.mean(dim=2) if not self.enable_mapper else self.mapper(video_embeddings, video_mask)
        )

        # Resample video embeddings
        resampled_embeddings = None
        if self.enable_resampler_xattn:
            resampled_embeddings = self.resampler(video_embeddings, mask=video_mask)

        # Generate text
        text_mask = torch.cat((video_mask.float(), tokens_mask), dim=1)
        text_output = self.text_generator(
            video_embeddings_mapped, resampler_embeddings=resampled_embeddings, tokens=tokens, mask=text_mask
        )

        return text_output
