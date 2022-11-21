from einops import rearrange
from torch import nn
from transformers import CLIPVisionModel, GPT2LMHeadModel

from .resampler import PerceiverResampler


class VisionEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.video_encoder = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')
        self.resampler = PerceiverResampler(
            dim=768,
            depth=6,
            dim_head=64,
            heads=8,
            num_latents=64,
            num_time_embeds=500,  # TODO: Need to give dynamic value based on number of frames.
            ff_mult=4,
            activation='gelu',
        )

    def _freeze_params(self):
        for param in self.video_encoder.parameters():
            param.requires_grad = False

    def forward(self, video):
        """
        Args:
            video: Batch of video frames with shape (batch_size, timesteps, 3, 224, 224)

        Returns:
            Video embeddings with shape (batch_size, dim_head, 768)
        """
        batch_size = video.shape[0]

        # Get embeddings for each frame in the video
        video = rearrange(video, 'b t ... -> (b t) ...')
        embeddings = self.video_encoder(pixel_values=video).last_hidden_state
        embeddings = rearrange(embeddings, '(b t) ... -> b t ...', b=batch_size)

        # Pass the video embeddings through the perceiver resampler
        return self.resampler(embeddings)


class VideoTextModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.vision_encoder = VisionEncoder()
        self.text_generator = GPT2LMHeadModel.from_pretrained('gpt2')
        self._freeze_params()

    def _freeze_params(self):
        self.vision_encoder._freeze_params()

        # Freeze text generator
        for param in self.text_generator.parameters():
            param.requires_grad = False

        # Unfreeze the lm head
        for param in self.text_generator.lm_head.parameters():
            param.requires_grad = True

    def forward(self, video):
        video_embeddings = self.vision_encoder(video)
        text_output = self.text_generator(inputs_embeds=video_embeddings).logits
        return text_output
