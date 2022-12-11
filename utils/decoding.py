from typing import Optional

import torch
import torch.nn.functional as nnf


def greedy_search(
    model,
    tokenizer,
    video_embeddings,
    resampled_embeddings,
    number_to_generate: int = 1,
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
    assert video_embeddings.shape[0] == 1, "Greedy search only supports batch size of 1."

    device = video_embeddings.device
    generations = []

    with torch.no_grad():
        for _ in range(number_to_generate):
            tokens = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0).to(device)
            for _ in range(max_length):
                logits = model.text_generator(
                    video_embeddings,
                    resampler_embeddings=resampled_embeddings,
                    tokens=tokens,
                    inference_mode=True,
                )  # (batch_size, vocab_size)
                logits /= temperature if temperature > 0 else 1.0

                _, next_token = nnf.softmax(logits, dim=-1).topk(1, dim=-1)
                tokens = torch.cat((tokens, next_token), dim=1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list, skip_special_tokens=True).strip()

            generations.append(output_text)

    return generations


def nucleus_sampling(
    model,
    tokenizer,
    video_embeddings,
    resampled_embeddings,
    number_to_generate: int = 1,
    max_length: int = 100,
    temperature: float = 1.0,
    top_p: Optional[float] = 0.8,
    top_k: Optional[float] = None,
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
        top_p: top-p sampling
        top_k: top-k sampling

    Returns:
        list of generated texts
    """
    assert video_embeddings.shape[0] == 1, "Nucleus sampling only supports batch size of 1."

    device = video_embeddings.device
    generations = []

    with torch.no_grad():
        for _ in range(number_to_generate):
            tokens = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0).to(device)
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
                idx = torch.searchsorted(cumulative_p, threshold_repeated).clip(max=top_k - 1).squeeze()  # type: ignore[reportOptionalOperand]
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


def beam_search(
    model,
    tokenizer,
    video_embeddings,
    resampled_embeddings,
    number_to_generate: int = 1,
    max_length: int = 100,
    temperature: float = 1.0,
    beam_width: int = 5,
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
        beam_width: beam width

    Returns:
        list of generated texts
    """
    assert video_embeddings.shape[0] == 1, "Beam search only supports batch size of 1"

    device = video_embeddings.device
    tokens = None
    scores = None

    seq_lengths = torch.ones(beam_width, device=device)
    has_stopped = torch.zeros(beam_width, dtype=torch.bool, device=device)
    generations = []

    with torch.no_grad():
        tokens = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0).to(device)
        for _ in range(number_to_generate):
            for _ in range(max_length):
                logits = model.text_generator(
                    video_embeddings,
                    resampler_embeddings=resampled_embeddings,
                    tokens=tokens,
                    inference_mode=True,
                )  # (batch_size, vocab_size)
                logits /= temperature if temperature > 0 else 1.0
                logits = nnf.log_softmax(logits, dim=-1)

                if scores is None:
                    scores, next_tokens = logits.topk(beam_width, -1)
                    video_embeddings = video_embeddings.expand(beam_width, *video_embeddings.shape[1:])
                    resampled_embeddings = resampled_embeddings.expand(beam_width, *resampled_embeddings.shape[1:])
                    next_tokens = next_tokens.permute(1, 0)
                    scores = scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_width, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[has_stopped] = float('-inf')
                    logits[has_stopped, 0] = 0

                    scores_sum = scores[:, None] + logits
                    seq_lengths[~has_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_width, -1)

                    next_tokens_source = torch.div(next_tokens, scores_sum.shape[1], rounding_mode='trunc')
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)

                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)

                    video_embeddings = video_embeddings[next_tokens_source]
                    resampled_embeddings = resampled_embeddings[next_tokens_source]
                    scores = scores_sum_average * seq_lengths

                    has_stopped = has_stopped[next_tokens_source]

                has_stopped = has_stopped + next_tokens.eq(tokenizer.eos_token_id).squeeze()
                if has_stopped.all():
                    break

            scores = scores / seq_lengths
            output_list = tokens.cpu().numpy()
            output_texts = [tokenizer.decode(output[:int(length)], skip_special_tokens=True) for output, length in zip(output_list, seq_lengths)]

            order = scores.argsort(descending=True)
            output_texts = [output_texts[i] for i in order][0]

            generations.append(output_texts)

    return generations


def decode_output(
    model,
    tokenizer,
    video_embeddings,
    resampled_embeddings,
    decoding_strategy,
    number_to_generate: int = 1,
    max_length: int = 100,
    temperature: float = 1.0,
    top_p: Optional[float] = 0.8,
    top_k: Optional[int] = None,
    beam_width: Optional[int] = 5,
):
    """Decode model output.

    Args:
        decoding_strategy: strategy for decoding output
        model: model to use for generation
        tokenizer: tokenizer for decoding
        video_embeddings: input video embeddings
        resampled_embeddings: resampled video embeddings
        number_to_generate: number of samples to generate
        max_length: maximum length of each sample
        temperature: temperature for sampling
        top_p: top-p sampling
        top_k: top-k sampling
        beam_width: beam width

    Returns:
        list of generated texts
    """
    assert decoding_strategy in ['greedy', 'nucleus', 'beam']

    if decoding_strategy == 'greedy':
        return greedy_search(
            model,
            tokenizer,
            video_embeddings,
            resampled_embeddings,
            number_to_generate=number_to_generate,
            max_length=max_length,
            temperature=temperature,
        )
    elif decoding_strategy == 'nucleus':
        return nucleus_sampling(
            model,
            tokenizer,
            video_embeddings,
            resampled_embeddings,
            number_to_generate=number_to_generate,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    elif decoding_strategy == 'beam':
        return beam_search(
            model,
            tokenizer,
            video_embeddings,
            resampled_embeddings,
            number_to_generate=number_to_generate,
            max_length=max_length,
            temperature=temperature,
            beam_width=beam_width,
        )
