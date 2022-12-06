# This code is referenced from https://github.com/dhansmair/flamingo-mini/blob/main/flamingo_mini/gated_cross_attention.py

"""
Gated cross-attention layer adapted from flamingo-pytorch.
"""
from typing import Optional, Tuple

import torch
from einops import rearrange
from einops_exts import rearrange_many
from torch import einsum, nn, tanh

from .utils import feed_forward_layer


class MaskedCrossAttention(nn.Module):
    """Cross attention layer with masking.

    Args:
        dim: d_token, d_visual dimensionality of language and visual tokens
        dim_head: dimensionality of the q, k, v vectors inside one attention head
        heads: number of attention heads
    """

    def __init__(self, dim: int, dim_visual: int, dim_head: int = 64, heads: int = 8, n_visual: int = 64):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.n_visual = n_visual
        inner_dim = dim_head * heads
        self.layer_norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        y: torch.FloatTensor,
        visual_features: Optional[torch.FloatTensor] = None,
        previous_kv: tuple = None,
        output_kv: bool = False,
    ):
        """
        Args:
            y: language features (batch_size, n_token, d_token)
            visual_features: visual features (batch_size, n_images, n_queries, d_visual)
            previous_kv: tuple of previous keys and values. Passed when caching is used during text generation
            output_kv: whether to return the keys and values

        Returns:
            Tensor (batch_size, n_token, d_token)
        """
        y = self.layer_norm(y)

        # Compute the queries from the text tokens
        q = self.to_q(y)
        q = q * self.scale

        # Compute the keys and values from the visual tokens
        if previous_kv is None:
            visual_features = rearrange(visual_features, 'b t n d -> b (t n) d')
            k, v = self.to_kv(visual_features).chunk(2, dim=-1)
            q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=self.heads)
        else:
            k, v = previous_kv
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        # Compute the attention scores from the queries and keys
        sim = einsum('... i d, ... j d -> ... i j', q, k)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        alphas = sim.softmax(dim=-1)

        out = einsum('... i j, ... j d -> ... i d', alphas, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        conditioned_tokens = self.to_out(out)

        if output_kv:
            return conditioned_tokens, (k, v)

        return conditioned_tokens, None


class GatedCrossAttentionBlock(nn.Module):
    """
    Args:
        dim: d_token, d_visual
        dim_head: dimensionality of q, k, v inside the attention head
        heads: number of attention heads
        ff_mult: factor for the number of inner neurons in the ffw layer
    """

    def __init__(
        self,
        dim: int,
        dim_visual: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        act: str = 'gelu',
        n_visual: int = 64,
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim, dim_visual=dim_visual, dim_head=dim_head, heads=heads, n_visual=n_visual
        )
        self.alpha_attn = nn.Parameter(torch.tensor([0.0]))  # type: ignore[reportPrivateUsage]

        self.ffw = feed_forward_layer(dim, mult=ff_mult, activation=act)
        self.alpha_ffw = nn.Parameter(torch.tensor([0.0]))  # type: ignore[reportPrivateUsage]

    def forward(self, y: torch.LongTensor, visual_features: torch.FloatTensor, previous_kv=None, output_kv=False):
        """
        Args:
            y: language features from previous LM layer (batch_size, n_tokens, d_token)
            media: visual features, encoded by perceiver resample (batch_size, n_media, n_queries, dim)
        """
        if previous_kv is None:
            assert visual_features.ndim == 4
        shape_before = y.shape

        # kv will be None if output_kv=False
        attn_out, kv = self.attn(y, visual_features, previous_kv=previous_kv, output_kv=output_kv)
        y = y + tanh(self.alpha_attn) * attn_out
        assert y.shape == shape_before
        y = y + tanh(self.alpha_ffw) * self.ffw(y)
        assert y.shape == shape_before
        return y, kv


class ModifiedLMBlock(nn.Module):
    """
    A block that wraps a gated cross-attention layer, followed by a LM layer.
    We replace the original layers in the LM with these at a certain frequency
    to introduce the xattn layer. This layer mimics the functionality and behavior
    of the underlying LM block. This way, the LM can be used in the same way as before,
    and we can do the conditioning without altering the LM implementation.

    One drawback of this approach is that we cannot pass the visual features to forward()
    directly, but instead we need to pass them before the actual forward pass, via a
    side-channel, which is the condition() method. In addition, when use_cache is used,
    the cached keys and values for the xattn layers need to be retrieved separately from
    the kv_output property.

    (!) This implementation works with GPT-2 layers, but hasn't been tested with other LMs yet.
    """

    def __init__(self, lm_block, **kwargs):
        super().__init__()
        self.xattn_block = GatedCrossAttentionBlock(**kwargs)
        self.lm_block = lm_block
        self.visual_features = None
        self.xattn_layer_past = None
        self.kv_output = None

    def condition(self, visual_features: torch.FloatTensor, xattn_layer_past=None):
        """
        Conditioning. Called from outside of the LM before passing the text input to the LM.
        This way, the gated cross-attention layers get informed about the visual input
        without the need to pipe the visual input through the LM forward() function.

        xattn_layer_past can contain the cached cross-attention keys and values (computed
        from the visual input). Passing them is useful to speed up the autoregressive text
        generation where the keys and values will be the same for every word, since the
        visual input doesn't change.
        If both visual_features and xattn_layer past are passed, visual_features will be
        ignored in the xattn layers.
        """
        self.visual_features = visual_features
        self.xattn_layer_past = xattn_layer_past

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]], use_cache: Optional[bool] = False, **kwargs):
        """
        This forward function mimics forward() of GPT2Block, so it has the same input and output.
        """

        # Pass through xattn
        hidden_states, kv = self.xattn_block(
            y=hidden_states,
            visual_features=self.visual_features,
            previous_kv=self.xattn_layer_past,
            output_kv=use_cache,
        )
        self.kv_output = kv

        # Pass through original LM layer
        return self.lm_block(hidden_states, use_cache=use_cache, **kwargs)
