import json
import os
from typing import Optional, Union

import evaluate as hf_evaluate
import torch
from einops import rearrange
from omegaconf import DictConfig
from torch import nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from utils import CheckpointManager, ProgressBar
from utils.decoding import decode_output


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        text_max_length: int,
        log_dir: str,
        exp_name: str,
        checkpoint_callback_params: Optional[Union[dict, DictConfig]] = None,
        device: Optional[str] = 'cpu',
    ):
        self.device = device
        self.model = model.to(device)

        self.text_max_length = text_max_length
        self.exp_name = exp_name
        self.checkpoint_callback_params = checkpoint_callback_params
        self.log_dir = os.path.join(log_dir, exp_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.tokenizer = tokenizer
        self.bleu_fn = hf_evaluate.load('bleu', experiment_id=exp_name)
        self.rouge_fn = hf_evaluate.load('rouge', experiment_id=exp_name)

    def _setup_loss(self):
        """Create the loss function"""
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def _prepare_for_training(self, optimizer_cfg, num_steps_per_epoch, epochs, restore_ckpt=None):
        self._setup_loss()

        # Prepare the optimizer
        assert optimizer_cfg['name'] in ['adamw', 'adam'], 'Optimizer must be either AdamW or Adam'
        if optimizer_cfg['name'] == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=optimizer_cfg['lr'],
                weight_decay=optimizer_cfg['weight_decay'],
                betas=optimizer_cfg['betas'],
            )
        elif optimizer_cfg['name'] == 'adam':
            self.optimizer = Adam(
                self.model.parameters(),
                lr=optimizer_cfg['lr'],
                weight_decay=optimizer_cfg['weight_decay'],
                betas=optimizer_cfg['betas'],
            )

        # Setup scheduler
        self.scheduler = None
        if optimizer_cfg['scheduler']['name'] is not None:
            assert optimizer_cfg['scheduler']['name'] in [
                'CosineAnnealing',
                'Linear',
            ], 'Scheduler must be CosineAnnealing or Linear'

            total_steps = num_steps_per_epoch * epochs
            cfg_warmup_steps = optimizer_cfg['scheduler']['warmup_steps']
            cfg_warmup_ratio = optimizer_cfg['scheduler']['warmup_ratio']

            if cfg_warmup_steps is not None:
                assert cfg_warmup_steps < total_steps
                warmup_steps = cfg_warmup_steps
            elif cfg_warmup_ratio is not None:
                assert cfg_warmup_ratio < 1
                warmup_steps = int(total_steps * cfg_warmup_ratio)
            else:
                raise ValueError('Either warmup_steps or warmup_ratio must be specified')

            if optimizer_cfg['scheduler']['name'] == 'CosineAnnealing':
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps,
                )
            elif optimizer_cfg['scheduler']['name'] == 'Linear':
                self.scheduler = get_constant_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                )

        # Restore states from checkpoint
        start_epoch = 1
        if restore_ckpt is not None:
            start_epoch = self._restore_state(restore_ckpt)

        # Create checkpoint manager
        assert self.checkpoint_callback_params is not None, 'Checkpoint callback params must be specified'
        self.ckpt_manager = CheckpointManager(
            self.model,
            os.path.join(self.log_dir, 'checkpoints'),
            self.exp_name,
            **self.checkpoint_callback_params,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        return start_epoch

    def _restore_state(self, restore_ckpt: str):
        checkpoint = torch.load(restore_ckpt, map_location=self.device)

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['epoch'] + 1

    def train(self, loader):
        self.model.train()
        pbar = ProgressBar(target=len(loader), width=8)
        for batch_idx, (video, video_length, tokens, tokens_mask, _) in enumerate(loader):
            video = video.to(self.device)
            video_length = video_length.to(self.device)
            tokens = tokens.to(self.device)
            tokens_mask = tokens_mask.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(video, video_length, tokens=tokens, tokens_mask=tokens_mask)

            # Compute the loss
            outputs = rearrange(outputs, 'b t d -> b d t')
            loss = self.criterion(outputs, tokens)

            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # Update progress bar
            pbar.update(batch_idx, values=[('Loss', round(loss.item(), 4))])

        pbar.add(
            1,
            values=[
                ('Loss', round(loss.item(), 4)),  # type: ignore[reportUnboundVariable]
            ],
        )

        return loss.item()  # type: ignore[reportUnboundVariable]

    def _compute_metrics(self, predictions, targets):
        """Compute BLEU1, BLEU4 and ROUGE metrics score"""
        # Remove the empty predictions and their corresponding targets
        valid_predictions, valid_targets = [], []
        for pred, target in zip(predictions, targets):
            pred = pred.strip()
            if pred != '':
                valid_predictions.append(pred)
                valid_targets.append([target])

        bleu1, bleu4, rouge = 0, 0, 0
        if len(valid_predictions) > 0:
            bleu1 = self.bleu_fn.compute(predictions=valid_predictions, references=valid_targets, max_order=1)['bleu']  # type: ignore[reportOptionalSubscript]
            bleu4 = self.bleu_fn.compute(predictions=valid_predictions, references=valid_targets, max_order=4)['bleu']  # type: ignore[reportOptionalSubscript]
            rouge = self.rouge_fn.compute(predictions=valid_predictions, references=valid_targets)['rougeL']  # type: ignore[reportOptionalSubscript]

        # Get the weighted average of the bleu score with the empty predictions
        bleu1 = (bleu1 * len(valid_predictions)) / len(predictions)
        bleu4 = (bleu4 * len(valid_predictions)) / len(predictions)
        rouge = (rouge * len(valid_predictions)) / len(predictions)

        return bleu1, bleu4, rouge

    def evaluate(self, loader, data_type='dev'):
        self.model.eval()

        # NOTE: This runs when we call the evaluate function directly without
        # running the _preprar_for_training function
        if not hasattr(self, 'criterion'):
            self._setup_loss()

        eval_loss = 0
        with torch.no_grad():
            for video, video_lengths, tokens, tokens_mask, _ in loader:
                video = video.to(self.device)
                video_lengths = video_lengths.to(self.device)
                tokens = tokens.to(self.device)
                tokens_mask = tokens_mask.to(self.device)

                # Get predictions
                outputs = self.model(video, video_lengths, tokens=tokens, tokens_mask=tokens_mask)

                # Compute the loss
                outputs = rearrange(outputs, 'b t d -> b d t')
                loss = self.criterion(outputs, tokens)
                eval_loss += loss.item()

        eval_loss /= len(loader)
        print(f'{"Validation" if data_type == "dev" else "Test"} set: ' f'Average loss: {eval_loss:.4f}')
        return eval_loss

    def inference(
        self,
        loader: DataLoader,
        decoding_strategy: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: Optional[float] = 0.8,
        top_k: Optional[int] = None,
        beam_width: Optional[int] = 5,
    ):
        """Test the model.

        Args:
            loader: data loader
            max_length: maximum length of each sample
            top_p: top-p sampling
            top_k: top-k sampling
            temperature: temperature for sampling
        """
        self.model.eval()

        test_loss = 0
        predictions, targets, video_paths = [], [], []
        pbar = ProgressBar(target=len(loader), width=8)

        with torch.no_grad():
            for batch_idx, (video, video_lengths, tokens, tokens_mask, video_path) in enumerate(loader):
                video = video.to(self.device)
                video_lengths = video_lengths.to(self.device)
                tokens = tokens.to(self.device)
                tokens_mask = tokens_mask.to(self.device)

                # Encode video
                video_embeddings, resampled_embeddings, _ = self.model.encode_video(video, video_lengths)

                # Get predictions
                decoded_prediction = decode_output(
                    self.model,
                    self.tokenizer,
                    video_embeddings,
                    resampled_embeddings=resampled_embeddings,
                    decoding_strategy=decoding_strategy,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    beam_width=beam_width,
                )

                predictions.extend(decoded_prediction)
                tokens[
                    tokens_mask == 0
                ] = self.tokenizer.pad_token_id  # This is done to make tokenizer ignore the pad tokens
                targets.extend([x.strip() for x in self.tokenizer.batch_decode(tokens, skip_special_tokens=True)])
                video_paths.extend(video_path)

                # Update progress bar
                pbar.update(batch_idx)

        pbar.add(1)

        # Compute the bleu score
        bleu1, bleu4, rouge = self._compute_metrics(predictions, targets)

        print(
            f'Test set: Average loss: {test_loss:.4f} '
            f'- Bleu1: {bleu1:.4f} '
            f'- Bleu4: {bleu4:.4f} '
            f'- Rouge: {rouge:.4f}\n'
        )

        results = [
            {
                'video_path': video_paths[i],
                'prediction': predictions[i],
                'target': targets[i],
            }
            for i in range(len(video_paths))
        ]

        with open(os.path.join(self.log_dir, f'inference.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Inference results saved to {os.path.join(self.log_dir, "inference.json")}')

    def fit(
        self,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        optimizer_cfg: DictConfig,
        epochs: int,
        restore_ckpt: Optional[str] = None,
    ):
        start_epoch = self._prepare_for_training(optimizer_cfg, len(train_loader), epochs, restore_ckpt=restore_ckpt)

        for epoch in range(start_epoch, epochs + 1):
            print(f'\nEpoch {epoch}:')
            train_loss = self.train(train_loader)
            eval_loss = self.evaluate(dev_loader)

            # Log the progress
            self.ckpt_manager.log(epoch, train_loss, eval_loss)

        # Store the last checkpoint weigths
        self.ckpt_manager.save_current_state()

        # Store the best checkpoint with the the experiment name
        self.ckpt_manager.save_best_state()
