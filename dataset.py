"""
This script contains the Dataset class and the fuctions to create its dataloaders.
"""

import json
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
from transformers import CLIPProcessor


class MLSLTDataset(Dataset):

    def __init__(self, video_root, json_path):
        super().__init__()

        assert os.path.exists(video_root), 'The videos directory does not exist.'

        self.video_root = video_root
        self._get_labels(json_path)

        self.image_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

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
        video_path = os.path.join(self.video_root, str(sample_id), 'en.mp4')

        video, _, _ = read_video(video_path, output_format='TCHW', pts_unit='sec')
        video = self.image_processor(images=[x for x in video], return_tensors='pt')['pixel_values']

        # TODO: tokenize text and convert to tensor

        return video, transcript

    def _collate_pad(self, batch_samples):
        pad_video, pad_transcript = [], []
        max_video_len = len(max(batch_samples, key=lambda x: len(x[0]))[0])
        for video, transcript in batch_samples:
            # Pad video frames
            if len(video) < max_video_len:
                video = torch.cat([video, torch.zeros(max_video_len - len(video), *video.shape[1:])], dim=0)
            pad_video.append(video)

            # Pad transcripts
            pad_transcript.append(transcript)

        return torch.stack(pad_video), pad_transcript

    def get_dataloader(self, batch_size, shuffle=True, num_workers=1):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_pad,
        )
