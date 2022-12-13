"""
This script contains the Dataset class and the fuctions to create its dataloaders.
"""

import json
import os
import random

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
from transformers import CLIPProcessor, GPT2Tokenizer

SUPPORTED_LANGUAGES = ['zh', 'uk', 'ru', 'bg', 'is', 'de', 'it', 'sv', 'lt', 'en']


class MLSLTDataset(Dataset):
    def __init__(self, video_dir, json_path, sign_languages=['en'], tokenizer='gpt2'):
        super().__init__()
        assert os.path.exists(video_dir), 'The videos directory does not exist.'
        for sign_lang in sign_languages:
            assert sign_lang in SUPPORTED_LANGUAGES, f'{sign_lang} sign language not supported.'

        self.sign_languages = sign_languages
        self.video_dir = video_dir
        self._get_samples(json_path)

        self.image_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self._prepare_tokenizer(tokenizer)

    def _prepare_tokenizer(self, tokenizer):
        if os.path.exists(tokenizer) and os.path.isfile(tokenizer):
            self.tokenizer = CustomTokenizer(tokenizer)
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer)
            self.tokenizer.pad_token = '<|pad|>'

    def _get_samples(self, json_path):
        """Reads the json file and create dataset samples."""
        with open(json_path) as f:
            metadata = json.load(f)

        self.videos = []
        self.transcripts = {}
        for video_data in metadata:
            # Get video samples
            self.videos.extend(
                [
                    {
                        'id': video_data['id'],
                        'lang': sign_lang,
                    }
                    for sign_lang in self.sign_languages
                ]
            )

            # Get transcripts
            self.transcripts[video_data['id']] = {
                text_data['lang']: text_data['text'] for text_data in video_data['sign_list']
            }

        random.shuffle(self.videos)

        # Get max length of transcripts
        self.max_length = max([len(x['en']) for x in self.transcripts.values()])

    def __len__(self):
        return len(self.videos)

    def _process_video(self, video_path):
        video, _, _ = read_video(video_path, output_format='TCHW', pts_unit='sec')
        video = self.image_processor(images=[x for x in video], return_tensors='pt')['pixel_values']
        video_length = torch.tensor(video.shape[0])

        return video, video_length

    def _process_text(self, text):
        # Convert the text to tokens
        tokenized_text = self.tokenizer(
            f'{self.tokenizer.bos_token} ' + text + f' {self.tokenizer.eos_token}',
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        tokens = tokenized_text['input_ids'].squeeze(0)
        attention_mask = tokenized_text['attention_mask'].squeeze(0)

        # Set padding to zero
        tokens[attention_mask == 0] = 0

        return tokens, attention_mask

    def __getitem__(self, index):
        sample_id = self.videos[index]['id']
        video_path = os.path.join(self.video_dir, str(sample_id), f'{self.videos[index]["lang"]}.mp4')
        transcript = self.transcripts[sample_id]['en']

        # Process video
        video, video_length = self._process_video(video_path)

        # Process text
        tokens, tokens_mask = self._process_text(transcript)

        # NOTE: We return video path because it's required to store results during evaluation
        return video, video_length, tokens, tokens_mask, video_path

    def _collate_pad(self, batch_samples):
        pad_video, pad_video_length, pad_tokens, pad_tokens_mask, pad_video_path = [], [], [], [], []
        max_video_len = len(max(batch_samples, key=lambda x: len(x[0]))[0])
        for video, video_length, tokens, tokens_mask, video_path in batch_samples:
            # Pad video frames
            if video_length < max_video_len:
                video = torch.cat(
                    (video, torch.zeros(max_video_len - len(video), *video.shape[1:], dtype=video.dtype)), dim=0
                )
            pad_video.append(video)
            pad_video_length.append(video_length)
            pad_video_path.append(video_path)

            # Pad tokens
            pad_tokens.append(tokens)
            pad_tokens_mask.append(tokens_mask)

        return (
            torch.stack(pad_video),
            torch.stack(pad_video_length),
            torch.stack(pad_tokens),
            torch.stack(pad_tokens_mask),
            pad_video_path,
        )

    def get_dataloader(self, batch_size, num_workers=1, shuffle=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_pad,
        )


class CustomTokenizer:
    """Wrapper class for the custom tokenizer."""

    def __init__(self, file_path):
        self._tokenizer = Tokenizer.from_file(file_path)
        self.eos_token = "<|endoftext|>"
        self.bos_token = "<|startoftext|>"
        self.pad_token = "<|pad|>"
        self.eos_token_id = self._tokenizer.token_to_id(self.eos_token)
        self.bos_token_id = self._tokenizer.token_to_id(self.bos_token)
        self.pad_token_id = self._tokenizer.token_to_id(self.pad_token)
        self.custom_tokenizer = True

    def __len__(self):
        return self._tokenizer.get_vocab_size()

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]

    def decode(self, sequence, skip_special_tokens=True):
        return self._tokenizer.decode(list(sequence), skip_special_tokens=skip_special_tokens)

    def __call__(self, sequence, max_length=None, truncation=False, padding=None, return_tensors=None):
        tokenized_seq = self._tokenizer.encode(sequence)
        tokenized_seq_ids = torch.tensor(tokenized_seq.ids)
        tokenized_seq_attn = torch.tensor(tokenized_seq.attention_mask)

        # Truncate the long sequences
        if truncation and max_length is not None:
            tokenized_seq_ids = tokenized_seq_ids[:max_length]
            tokenized_seq_attn = tokenized_seq_attn[:max_length]

        # Pad the short sequences
        if padding == 'max_length' and max_length is not None:
            tokenized_seq_ids = torch.cat(
                (tokenized_seq_ids, torch.zeros(max_length - len(tokenized_seq_ids), dtype=tokenized_seq_ids.dtype)),
                dim=0,
            )
            tokenized_seq_attn = torch.cat(
                (
                    tokenized_seq_attn,
                    torch.zeros(max_length - len(tokenized_seq_attn), dtype=tokenized_seq_attn.dtype),
                ),
                dim=0,
            )

        return {
            'input_ids': tokenized_seq_ids,
            'attention_mask': tokenized_seq_attn,
        }
