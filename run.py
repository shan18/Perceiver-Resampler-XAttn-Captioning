import argparse
import os

import torch
from torch import nn
from torchinfo import summary

from dataset import MLSLTDataset
from engine import Trainer
from model import VideoTextModel
from omegaconf import OmegaConf



def create_dataset(data_root: str, data_type: str, batch_size: int, num_workers: int):
    """Creates the dataset and its dataloaders

    Args:
        data_root: The root directory of the dataset containing
            the video directory and the json file.
        data_type: The type of the dataset. Can be either `train`, `dev`, or `test`.
        batch_size: The batch size of the dataloader.
        num_workers: The number of workers for the dataloader.

    Returns:
        The MLSLTDataset dataloader and the dataset object
    """
    print(f'Loading dataset from {data_root}/{data_type} and {data_root}/{data_type}.json')
    dataset = MLSLTDataset(
        os.path.join(data_root, data_type), os.path.join(data_root, f'{data_type}.json')
    )
    loader = dataset.get_dataloader(batch_size, num_workers=num_workers)
    return dataset, loader


def main(args):
    config = OmegaConf.load(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Create dataloaders
    print('Creating dataloaders...')
    _, train_loader = create_dataset(args.data_root, 'train', args.batch_size, args.num_workers)
    _, dev_loader = create_dataset(args.data_root, 'dev', args.batch_size, args.num_workers)

    # Create model
    print('Creating model...')
    model = VideoTextModel(config, device).to(device)
    summary(model)

    # Create trainer
    print('Creating trainer...')
    trainer = Trainer(model, args.checkpoint_dir, device=device)

    # Train
    print('Training...')
    trainer.fit(train_loader, dev_loader, args.epochs)


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        default=os.path.join(BASE_DIR, 'data'),
        help='Path containing the train, dev and test dataset',
    )
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--checkpoint_dir', default=os.path.join(BASE_DIR, 'checkpoints'), help='Checkpoint directory')
    parser.add_argument('--config', default=os.path.join(BASE_DIR, 'config.yaml'), help='Config path')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    main(args)
