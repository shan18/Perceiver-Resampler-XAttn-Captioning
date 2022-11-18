import torch
from einops import rearrange
from torch import nn
from transformers import CLIPVisionModel


class MLSLTModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.video_encoder = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')

    def _encode_video_frames(self, video):
        """Generate embeddings for each frame in the video

        Args:
            video: Batch of video frames with shape (batch_size, timesteps, 3, 224, 224)

        Returns:
            video embeddings with shape (batch_size, timesteps, 512)
        """
        batch_size = video.shape[0]
        video = rearrange(video, 'b t ... -> (b t) ...')
        embeddings = self.video_encoder(pixel_values=video).last_hidden_state.mean(dim=1)
        return rearrange(embeddings, '(b t) ... -> b t ...', b=batch_size)

    def forward(self, video: torch.Tensor, transcript: torch.Tensor):
        """
        Args:
            video: Batch of video frames with shape (batch_size, timesteps, 3, 224, 224)
        """
        # Get embeddings for each frame in the video
        video_embeddings = self._encode_video_frames(video)

        # TODO: Get unified video embeddings

        # TODO: Generate output text

        return video_embeddings
