import torch
from einops import rearrange
from torch import nn
from transformers import CLIPVisionModel
from resampler import PerceiverResampler


class MLSLTModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.video_encoder = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')
        self.resampler = PerceiverResampler(
            dim=768,
            depth=6,
            dim_head=64,
            heads=8,
            num_latents=64,
            num_time_embeds=500, #FIXME: Need to give dynamic value based on number of frames.
            ff_mult=4,
            act='gelu',
        )

    def _encode_video_frames(self, video):
        """Generate embeddings for each frame in the video

        Args:
            video: Batch of video frames with shape (batch_size, timesteps, 3, 224, 224)

        Returns:
            video embeddings with shape (batch_size, timesteps, 512)
        """
        batch_size = video.shape[0]
        video = rearrange(video, 'b t ... -> (b t) ...')
        embeddings = self.video_encoder(pixel_values=video).last_hidden_state
        return rearrange(embeddings, '(b t) ... -> b t ...', b=batch_size)

    def forward(self, video: torch.Tensor, transcript: torch.Tensor):
        """
        Args:
            video: Batch of video frames with shape (batch_size, timesteps, 3, 224, 224)
        """
        # Get embeddings for each frame in the video
        video_embeddings = self._encode_video_frames(video)
        visual_features = self.resampler(video_embeddings)

        return visual_features

        # TODO: Add positional encoding to video embeddings

        # TODO: Get unified video embeddings

        # TODO: Generate output text

        # return video_embeddings, transcript
