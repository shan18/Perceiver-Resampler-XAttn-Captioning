"""
This script contains the Dataset class and the fuctions to create its dataloaders.
"""

import json
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
from transformers import CLIPProcessor, GPT2Tokenizer


class MLSLTDataset(Dataset):
    def __init__(self, video_dir, json_path, max_length):
        super().__init__()

        assert os.path.exists(video_dir), 'The videos directory does not exist.'

        self.max_length = max_length
        self.video_dir = video_dir
        self._get_labels(json_path)

        self.image_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = '<|pad|>'

    def _get_labels(self, json_path):
        """Reads the json file and creates a label dictionary"""
        with open(json_path) as f:
            metadata = json.load(f)

        self.json_data = [
            {'id': video['id'], 'transcript': {label['lang']: label['text'] for label in video['sign_list']}}
            for video in metadata
        ]

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        sample_id = self.json_data[index]['id']
        transcript = self.json_data[index]['transcript']['en']
        video_path = os.path.join(self.video_dir, str(sample_id), 'en.mp4')

        # Process video
        video, _, _ = read_video(video_path, output_format='TCHW', pts_unit='sec')
        video = self.image_processor(images=[x for x in video], return_tensors='pt')['pixel_values']
        video_length = torch.tensor(video.shape[0])

        # Process text
        transcript = self.tokenizer(
            f'{self.tokenizer.bos_token} {transcript} {self.tokenizer.eos_token}',
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
        )

        return video, video_length, transcript['input_ids'].squeeze(0)

    def _collate_pad(self, batch_samples):
        pad_video, pad_transcript, pad_video_length = [], [], []
        max_video_len = len(max(batch_samples, key=lambda x: len(x[0]))[0])
        for video, video_length, transcript in batch_samples:
            # Pad video frames
            if video_length < max_video_len:
                video = torch.cat(
                    [video, torch.zeros(max_video_len - len(video), *video.shape[1:], dtype=video.dtype)], dim=0
                )
            pad_video.append(video)
            pad_video_length.append(video_length)

            pad_transcript.append(transcript)

        return torch.stack(pad_video), torch.stack(pad_video_length), torch.stack(pad_transcript)

    def get_dataloader(self, batch_size, num_workers=1, shuffle=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_pad,
        )
