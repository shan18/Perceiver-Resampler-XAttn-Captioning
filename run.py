"""
This is the main script that is used to train the model on the
Multilingual Sign Language Translation dataset.

Usage:
======

Training
--------

HYDRA_FULL_ERROR=1 python run.py \
    --config-path=<Directory containing the config file> \
    --config-name=config.yaml \
    name=<name of the experiment> \
    mode=train \
    dataset.train_ds.visual_dir=<Directory containing the videos or images for train set> \
    dataset.train_ds.json_path=<Path to the json file with the transcripts for train set> \
    dataset.validation_ds.visual_dir=<Directory containing the videos or images for validation set> \
    dataset.validation_ds.json_path=<Path to the json file with the transcripts for validation set> \
    trainer.exp_dir=<Directory to save the checkpoints and logs>

Testing
-------

HYDRA_FULL_ERROR=1 python run.py \
    --config-path=<Directory containing the config file> \
    --config-name=config.yaml \
    name=<name of the experiment> \
    mode=test \
    pretrained_name=<path to the checkpoint> \
    dataset.test_ds.visual_dir=<Directory containing the videos or images for test set> \
    dataset.test_ds.json_path=<Path to the json file with the transcripts for test set> \
    trainer.exp_dir=<Directory to save the logs>
"""


import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torchinfo import summary

from dataset import COCODataset, MLSLTDataset
from engine import Trainer
from model import build_model


def check_mandatory_args(cfg: DictConfig):
    """Checks if the mandatory arguments are present in the config."""
    for k, v in cfg.items():
        if isinstance(v, DictConfig):
            check_mandatory_args(v)
        elif v is None:
            raise ValueError(f'Argument {k} is not specified in the config')


def setup_log_dir(cfg: DictConfig):
    """Sets up the log directory for the experiment if not specified in the config."""
    if cfg.trainer.exp_dir is None:
        with open_dict(cfg.trainer):
            cfg.trainer.exp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_logs')


def restore_cfg(cfg: DictConfig, checkpoint_path: str):
    """Restores parts of config from the checkpoint."""
    # Restore the model config
    with open_dict(cfg.model):
        model_cfg = torch.load(checkpoint_path, map_location='cpu')['model_cfg']
        cfg.model = model_cfg

    # Restore the tokenizer and languages info within model config
    if cfg.mode == 'test' or (cfg.mode == 'train' and cfg.resume_pretrained_state):
        with open_dict(cfg.dataset):
            cfg.dataset.tokenizer = model_cfg.tokenizer
            cfg.dataset.sign_languages = model_cfg.sign_languages


def create_dataset(
    name: str,
    visual_dir: str,
    json_path: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    sign_languages: list = ['en'],
    tokenizer: str = 'gpt2',
):
    """Creates the dataset and its dataloaders

    Args:
        name: Dataset name. Can be 'mlslt' or 'coco'
        visual_dir: The directory containing the videos
        json_path: Path to the json file with the transcripts
        batch_size: The batch size of the dataloader
        num_workers: The number of workers for the dataloader
        shuffle: Whether to shuffle the dataset
        max_length: The maximum length of each sample in the dataset
        sign_languages: The list of sign languages to be used
        tokenizer: Tokenizer for encoding and decoding the transcripts

    Returns:
        The MLSLTDataset dataset object and the data loader
    """
    print(f'Loading dataset from {visual_dir} and {json_path}')
    if name == 'mlslt':
        dataset = MLSLTDataset(visual_dir, json_path, sign_languages=sign_languages, tokenizer=tokenizer)
    else:
        dataset = COCODataset(visual_dir, json_path, tokenizer=tokenizer)
    loader = dataset.get_dataloader(batch_size, num_workers=num_workers, shuffle=shuffle)
    return dataset, loader


@hydra.main(version_base=None, config_path=os.path.dirname(os.path.abspath(__file__)), config_name='config.yaml')
def main(cfg):
    assert cfg.mode == 'train' or cfg.pretrained_name is not None, 'Need to specify the checkpoint path in test mode'

    if cfg.mode == 'train':
        check_mandatory_args(cfg.dataset.train_ds)
        check_mandatory_args(cfg.dataset.validation_ds)
    else:
        check_mandatory_args(cfg.dataset.test_ds)
    setup_log_dir(cfg)

    if cfg.pretrained_name is not None:
        print('Restoring config from checkpoint:', cfg.pretrained_name)
        restore_cfg(cfg, cfg.pretrained_name)
    else:
        # Save the tokenizer and languages info within model config
        with open_dict(cfg.model):
            cfg.model.tokenizer = cfg.dataset.tokenizer
            cfg.model.sign_languages = cfg.dataset.sign_languages

    print(f'Hydra config:\n{OmegaConf.to_yaml(cfg)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create dataloaders
    print('Creating dataloaders...')
    if cfg.mode == 'train':
        train_dataset, train_loader = create_dataset(
            cfg.dataset.name, **cfg.dataset.train_ds, sign_languages=cfg.dataset.sign_languages, tokenizer=cfg.dataset.tokenizer
        )
        _, dev_loader = create_dataset(
            cfg.dataset.name, **cfg.dataset.validation_ds, sign_languages=cfg.dataset.sign_languages, tokenizer=cfg.dataset.tokenizer
        )
        tokenizer = train_dataset.tokenizer
        text_max_length = train_dataset.max_length
    else:
        test_dataset, test_loader = create_dataset(
            cfg.dataset.name,
            **cfg.dataset.test_ds,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            sign_languages=cfg.dataset.sign_languages,
            tokenizer=cfg.dataset.tokenizer,
        )
        tokenizer = test_dataset.tokenizer
        text_max_length = test_dataset.max_length

    # Build model
    print('Creating model...')
    model = build_model(
        model_cfg=cfg.model,
        pretrained_name=cfg.pretrained_name,
        vocab_size=len(tokenizer) if hasattr(tokenizer, 'custom_tokenizer') else None,
        device=device,
    )
    summary(model)

    # Create trainer
    print('Creating trainer...')
    trainer = Trainer(
        model,
        tokenizer,
        text_max_length,
        cfg.trainer.exp_dir,
        cfg.trainer.exp_name,
        cfg.trainer.checkpoint_callback_params,
        device=device,
    )

    # Train and evaluate
    if cfg.mode == 'train':
        print('Training...')
        trainer.fit(
            train_loader,  # type: ignore[reportUnboundVariable]
            dev_loader,  # type: ignore[reportUnboundVariable]
            cfg.model.optimizer,
            cfg.trainer.epochs,
            check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
            restore_ckpt=cfg.pretrained_name if (cfg.pretrained_name is not None and cfg.resume_pretrained_state) else None
        )
    else:
        print(f'Testing with {cfg.trainer.test.decoding_strategy} decoding...')
        trainer.inference(test_loader, **cfg.trainer.test)  # type: ignore[reportUnboundVariable]


if __name__ == '__main__':
    main()
