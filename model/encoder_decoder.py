from einops import rearrange
from omegaconf import DictConfig
from transformers import CLIPVisionModel, GPT2LMHeadModel, logging

from .base_model import BaseModel
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

    def __init__(self, pretrained_name: str, trainable: bool):
        super().__init__(trainable)
        self._model = GPT2LMHeadModel.from_pretrained(pretrained_name)
        self._update_trainable_state()

    def get_input_shape(self, batch_size: int = 16, timesteps: int = 75):
        return (batch_size, timesteps, self._model.transformer.embed_dim), (batch_size, timesteps)

    def get_output_shape(self, batch_size: int = 16, timesteps: int = 75):
        return (batch_size, timesteps, self._model.lm_head.out_features)

    def forward(self, input_embeddings):
        """
        Args:
            input_embeddings: video embeddings with shape (batch_size, seq_length, 768)

        Returns:
            Logits of the output transcript
        """
        return self._model(inputs_embeds=input_embeddings).logits


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
        text_output = self.text_generator(resampled_embeddings)

        return text_output
