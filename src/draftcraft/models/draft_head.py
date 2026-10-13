"""Authentic PyTorch implementation of the EAGLE speculative draft head."""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from draftcraft.core.config import EagleDraftHeadConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed


class EagleDecoderLayer(nn.Module):
    """Single transformer decoder layer with Multi-Head Attention and SwiGLU MLP."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        intermediate_dim: int,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # SwiGLU MLP
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

        # Norms
        self.input_layernorm = RMSNorm(hidden_dim, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_dim, eps=rms_norm_eps)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LN Self-Attention
        normed_x = self.input_layernorm(x)
        b_sz, seq_len, _ = normed_x.shape

        q = self.q_proj(normed_x).view(b_sz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed_x).view(b_sz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed_x).view(b_sz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                # Invert boolean mask: True means keep, False means mask with -inf
                scores = scores.masked_fill(~attention_mask, float("-inf"))
            else:
                scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(b_sz, seq_len, self.hidden_dim)
        attn_out = self.o_proj(context)

        # First residual connection
        x = x + self.dropout(attn_out)

        # Pre-LN SwiGLU MLP
        normed_mlp_in = self.post_attention_layernorm(x)
        gate = F.silu(self.gate_proj(normed_mlp_in))
        up = self.up_proj(normed_mlp_in)
        mlp_out = self.down_proj(gate * up)

        # Second residual connection
        x = x + self.dropout(mlp_out)
        return x


class EagleDraftHead(nn.Module):
    """EAGLE speculative draft head architecture."""

    def __init__(self, config: Optional[EagleDraftHeadConfig] = None):
        super().__init__()
        self.config = config or EagleDraftHeadConfig()

        # Token embedding for proposed sequence
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_dim)

        # Input feature projection from target LLM hidden dimension
        self.fc_in = nn.Linear(self.config.target_hidden_dim, self.config.hidden_dim)

        # Fusion layer merging projected teacher state and token embedding
        self.fusion_layer = nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim)

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            EagleDecoderLayer(
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.num_heads,
                intermediate_dim=self.config.intermediate_dim,
                rms_norm_eps=self.config.rms_norm_eps,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.num_layers)
        ])

        # Final normalization
        self.final_norm = RMSNorm(self.config.hidden_dim, eps=self.config.rms_norm_eps)

        # Output projection back to target hidden dimension
        self.fc_out = nn.Linear(self.config.hidden_dim, self.config.target_hidden_dim)

        # Language modeling head mapping target dimension to vocabulary
        self.lm_head = nn.Linear(self.config.target_hidden_dim, self.config.vocab_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting the next hidden state and next token logits.

        Args:
            hidden_states: [Batch, Seq_Len, Target_Hidden_Dim]
            input_ids: [Batch, Seq_Len]
            attention_mask: Optional 2D or 4D attention mask

        Returns:
            Tuple of:
            - predicted_target_hidden_states: [Batch, Seq_Len, Target_Hidden_Dim]
            - predicted_logits: [Batch, Seq_Len, Vocab_Size]
        """
        # Embed tokens
        tok_embeds = self.embed_tokens(input_ids)

        # Project target hidden states into draft hidden dim
        proj_h = self.fc_in(hidden_states)

        # Fuse hidden features and token embeddings
        fused = self.fusion_layer(torch.cat([proj_h, tok_embeds], dim=-1))

        # Pass through decoder layers
        x = fused
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        # Norm and project back to target hidden space
        x_normed = self.final_norm(x)
        predicted_target_h = self.fc_out(x_normed)

        # Compute next-token logits
        logits = self.lm_head(predicted_target_h)

        return predicted_target_h, logits

    @torch.no_grad()
    def generate_draft_candidates(
        self,
        initial_hidden_state: torch.Tensor,
        initial_token_id: int,
        lookahead_k: int = 4,
        temperature: float = 1.0,
    ) -> Tuple[List[int], torch.Tensor]:
        """Autoregressively propose K candidate tokens and their predicted hidden states.

        Args:
            initial_hidden_state: [1, 1, Target_Hidden_Dim]
            initial_token_id: Starting token ID integer.
            lookahead_k: Number of speculative candidate tokens to predict.
            temperature: Sampling temperature (uses argmax if <= 0).

        Returns:
            Tuple of (candidate_token_ids, predicted_hidden_states_tensor).
        """
        device = initial_hidden_state.device
        curr_h = initial_hidden_state
        curr_tok = torch.tensor([[initial_token_id]], dtype=torch.long, device=device)

        candidate_tokens: List[int] = []
        collected_hidden_states: List[torch.Tensor] = []

        for _ in range(lookahead_k):
            pred_h, logits = self.forward(curr_h, curr_tok)
            next_logits = logits[:, -1, :]

            if temperature > 0.0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token_id = int(torch.multinomial(probs, num_samples=1).item())
            else:
                next_token_id = int(torch.argmax(next_logits, dim=-1).item())

            candidate_tokens.append(next_token_id)
            collected_hidden_states.append(pred_h)

            # Update for next step in draft chain
            curr_h = pred_h
            curr_tok = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

        all_hidden = torch.cat(collected_hidden_states, dim=1) if collected_hidden_states else curr_h
        return candidate_tokens, all_hidden
