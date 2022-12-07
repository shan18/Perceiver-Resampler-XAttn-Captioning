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
from transformers import PreTrainedTokenizer, get_cosine_schedule_with_warmup

from utils import CheckpointManager, ProgressBar


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        text_max_length: int,
        log_dir: str,
        exp_name: str,
        checkpoint_callback_params: Union[dict, DictConfig],
        save_test_results: bool = True,
        device: Optional[str] = 'cpu',
    ):
        self.device = device
        self.model = model.to(device)

        self.text_max_length = text_max_length
        self.exp_name = exp_name
        self.checkpoint_callback_params = checkpoint_callback_params
        self.save_test_results = save_test_results
        self.log_dir = os.path.join(log_dir, exp_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.tokenizer = tokenizer
        self.bleu_fn = hf_evaluate.load('bleu', experiment_id=exp_name)

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
            assert optimizer_cfg['scheduler']['name'] in ['CosineAnnealing'], 'Scheduler must be CosineAnnealing'

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

        # Restore states from checkpoint
        start_epoch = 1
        if restore_ckpt is not None:
            start_epoch = self._restore_state(restore_ckpt)

        # Create checkpoint manager
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

    def _compute_bleu(self, predictions, targets):
        """Compute BLEU4 score"""
        # Remove the empty predictions and their corresponding targets
        valid_predictions, valid_targets = [], []
        for pred, target in zip(predictions, targets):
            pred = pred.strip()
            if pred != '':
                valid_predictions.append(pred)
                valid_targets.append([target.strip()])

        bleu_score = 0 if len(valid_predictions) == 0 else self.bleu_fn.compute(predictions=valid_predictions, references=valid_targets)['bleu']  # type: ignore[reportOptionalSubscript]

        # Get the weighted average of the bleu score with the empty predictions
        bleu_score = (bleu_score * len(valid_predictions)) / len(predictions)

        return bleu_score

    def evaluate(self, loader, data_type='dev'):
        self.model.eval()

        # NOTE: This runs when we call the evaluate function directly without
        # running the _preprar_for_training function
        if not hasattr(self, 'criterion'):
            self._setup_loss()

        eval_loss = 0
        prediction_ids, target_ids, video_paths = [], [], []

        with torch.no_grad():
            for video, video_lengths, tokens, tokens_mask, video_path in loader:
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

                # Get sentence predictions for computing BLEU score
                outputs_ids = torch.argmax(outputs, dim=1)  # (batch_size, n_tokens)
                prediction_ids.extend(outputs_ids.tolist())

                # Remove the padding tokens from the targets
                target_ids.extend([t[:l] for t, l in zip(tokens.tolist(), tokens_mask.sum(dim=1).tolist())])

                video_paths.extend(video_path)

        # Remove tokens from predictions after EOS
        prediction_ids = [
            pred_id[:pred_id.index(self.tokenizer.eos_token_id)] if self.tokenizer.eos_token_id in pred_id else pred_id[:self.text_max_length]
            for pred_id in prediction_ids
        ]

        # Get the prediction and target strings
        predictions = self.tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
        targets = self.tokenizer.batch_decode(target_ids, skip_special_tokens=True)

        # Compute the average loss and bleu score
        eval_loss /= len(loader)
        bleu_score = self._compute_bleu(predictions, targets)

        # Store the results in a json format
        results = []
        if data_type == 'test' and self.save_test_results:
            results.extend(
                [
                    {
                        'video_path': video_paths[i],
                        'prediction': predictions[i],
                        'target': targets[i],
                    }
                    for i in range(len(video_paths))
                ]
            )

        print(
            f'{"Validation" if data_type == "dev" else "Test"} set: '
            f'Average loss: {eval_loss:.4f} - '
            f'Bleu Score: {bleu_score:.4f}\n'
        )

        # Write the results to a json file
        if data_type == 'test' and self.save_test_results:
            with open(os.path.join(self.log_dir, f'inference.json'), 'w') as f:
                json.dump(results, f, indent=2)
            print(f'Inference results saved to {os.path.join(self.log_dir, "inference.json")}')

        return eval_loss, bleu_score

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
            eval_loss, eval_bleu = self.evaluate(dev_loader)

            # Log the progress
            self.ckpt_manager.log(epoch, train_loss, eval_loss, eval_bleu)

        # Store the best checkpoint with the the experiment name
        self.ckpt_manager.save_best_state()
