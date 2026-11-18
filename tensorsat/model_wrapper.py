from typing import Optional, Tuple
import torch
import torch.nn as nn
from tensorsat.attention import SingletAttention, KVCache

class TensorSATBlock(nn.Module):
    """
    Production-ready Transformer Decoder Block utilizing SingletAttention.
    Can be used as a drop-in replacement for standard Llama/Mistral attention blocks.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.0,
        is_causal: bool = True,
        use_triton: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.intermediate_dim = intermediate_dim or (embed_dim * 4)

        # Pre-LN architecture (matching modern LLMs)
        self.input_layernorm = nn.LayerNorm(embed_dim)
        self.self_attn = SingletAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            is_causal=is_causal,
            use_triton=use_triton,
        )

        self.post_attention_layernorm = nn.LayerNorm(embed_dim)
        # SwiGLU / MLP feed-forward block
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, self.intermediate_dim),
            nn.SiLU(),
            nn.Linear(self.intermediate_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        # Self-Attention with residual connection
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn_out, kv_cache = self.self_attn(normed, kv_cache=kv_cache, use_cache=use_cache)
        hidden_states = residual + attn_out

        # MLP with residual connection
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states, kv_cache
