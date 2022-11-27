import os

import hydra
import torch
from omegaconf import OmegaConf
from torchinfo import summary

from dataset import MLSLTDataset
from engine import Trainer
from model import VideoTextModel


def create_dataset(video_dir: str, json_path: str, batch_size: int, num_workers: int, shuffle: bool):
    """Creates the dataset and its dataloaders

    Args:
        video_dir: The directory containing the videos
        json_path: Path to the json file with the transcripts
        batch_size: The batch size of the dataloader
        num_workers: The number of workers for the dataloader
        shuffle: Whether to shuffle the dataset

    Returns:
        The MLSLTDataset dataset object and the data loader
    """
    print(f'Loading dataset from {video_dir} and {json_path}')
    dataset = MLSLTDataset(video_dir, json_path)
    loader = dataset.get_dataloader(batch_size, num_workers=num_workers, shuffle=shuffle)
    return dataset, loader


@hydra.main(version_base=None, config_path=os.path.dirname(os.path.abspath(__file__)), config_name='config.yaml')
def main(cfg):
    print(f'Hydra config:\n{OmegaConf.to_yaml(cfg)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create dataloaders
    print('Creating dataloaders...')
    _, train_loader = create_dataset(**cfg.dataset.train_ds)
    _, dev_loader = create_dataset(**cfg.dataset.validation_ds)

    # Create model
    print('Creating model...')
    model = VideoTextModel(cfg.model.vision, cfg.model.resampler, cfg.model.text).to(device)
    model.summary()

    # Create trainer
    print('Creating trainer...')
    trainer = Trainer(model, cfg.exp_manager.exp_dir, device=device)

    # Train
    print('Training...')
    trainer.fit(train_loader, dev_loader, cfg.trainer.epochs)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
