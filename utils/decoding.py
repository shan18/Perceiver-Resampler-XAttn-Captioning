from typing import Optional

import numpy as np
import torch
import torch.nn.functional as nnf


def generate_nucleus_sampling(
    model,
    tokenizer,
    video_embeddings,
    resampled_embeddings,
    number_to_generate: int = 1,
    max_length: int = 100,
    top_p: float = 0.8,
    top_k=None,
    temperature: float = 1.0,
):
    """Generate text using nucleus sampling.

    Args:
        model: model to use for generation
        tokenizer: tokenizer for decoding
        video_embeddings: input video embeddings
        resampled_embeddings: resampled video embeddings
        number_to_generate: number of samples to generate
        max_length: maximum length of each sample
        top_p: top-p sampling
        top_k: top-k sampling
        temperature: temperature for sampling

    Returns:
        list of generated texts
    """
    device = video_embeddings.device
    batch_size = video_embeddings.shape[0]
    generations = []

    with torch.no_grad():
        for _ in range(number_to_generate):
            tokens = torch.tensor([tokenizer.bos_token_id]).repeat(batch_size, 1).to(device)
            for _ in range(max_length):
                logits = model.text_generator(
                    video_embeddings,
                    resampler_embeddings=resampled_embeddings,
                    tokens=tokens,
                    inference_mode=True,
                )  # (batch_size, vocab_size)
                logits /= temperature if temperature > 0 else 1.0

                if top_k is None:
                    top_k = logits.shape[-1]
                if top_p is None:
                    top_p = 1.0

                p, largest_p_idx = nnf.softmax(logits, dim=-1).topk(top_k, dim=-1)
                cumulative_p = p.cumsum(dim=-1)
                threshold_repeated = top_p + torch.zeros((len(p), 1)).to(device)
                idx = torch.searchsorted(cumulative_p, threshold_repeated).clip(max=top_k - 1).squeeze()
                cutoffs = cumulative_p[torch.arange(len(cumulative_p)), idx]
                censored_p = (cumulative_p <= cutoffs[:, None]) * p
                renormalized_p = censored_p / censored_p.sum(dim=-1, keepdims=True)

                final_p = torch.zeros_like(logits)
                row_idx = torch.arange(len(p)).unsqueeze(1).repeat(1, top_k).to(device)
                final_p[row_idx, largest_p_idx] = renormalized_p.to(final_p.dtype)

                next_token = torch.multinomial(final_p, num_samples=1)
                tokens = torch.cat((tokens, next_token), dim=1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list, skip_special_tokens=True).strip()

            generations.append(output_text)

    return generations


def generate_beam(
    model,
    tokenizer,
    video_embeddings,
    resampled_embeddings,
    number_to_generate: int = 1,
    beam_size: int = 3,
    max_length: int = 100,
    temperature: float = 1.0,
):
    """Generate text using nucleus sampling.

    Args:
        model: model to use for generation
        tokenizer: tokenizer for decoding
        video_embeddings: input video embeddings
        resampled_embeddings: resampled video embeddings
        number_to_generate: number of samples to generate
        max_length: maximum length of each sample
        temperature: temperature for sampling

    Returns:
        list of generated texts
    """
    device = video_embeddings.device
    batch_size = video_embeddings.shape[0]
    generations = []

    with torch.no_grad():
        for _ in range(number_to_generate):
            tokens = torch.tensor([tokenizer.bos_token_id]).repeat(batch_size, 1).to(device)
            for _ in range(max_length):
                logits = model.text_generator(
                    video_embeddings,
                    resampler_embeddings=resampled_embeddings,
                    tokens=tokens,
                    inference_mode=True,
                )  # (batch_size, vocab_size)
                logits /= temperature if temperature > 0 else 1.0
                logits = logits.softmax(-1).log()

                scores, next_tokens = logits.topk(beam_size, -1)
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                scores, _ = scores.view(batch_size, -1).topk(beam_size, sorted=True)
                order = scores.argmax(dim=-1)

                next_token = next_tokens[order.item(), :].unsqueeze(0)
                tokens = torch.cat((tokens, next_token), dim=1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list, skip_special_tokens=True).strip()

            generations.append(output_text)

    return generations
