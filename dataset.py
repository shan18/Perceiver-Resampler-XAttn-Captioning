"""
This script contains the Dataset class and the fuctions to create its dataloaders.
"""

import json
import os

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

        return video, transcript
