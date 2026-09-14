"""One-layer ReLU transformer, faithful to Nanda et al. 2023 Appendix A.

Input: tokens "a b =" (a, b one-hot over P values, '=' is a special token with
index P).  Learned token + positional embeddings.  One attention layer with
`n_heads` heads (query/key/value/output, no biases).  An MLP with ReLU and
biases.  An unembedding matrix W_U (P x d, no bias).  Loss is the
cross-entropy of the logits read off the final ('=') token.

Notation follows the paper (Elhage et al. 2021 conventions):
    x_i(0) = W_E t_i + p_i
    A_j    = softmax over keys of  x(0)^T W_Kj^T W_Qj x_2(0)
    x(1)   = x(0) + sum_j W_Oj W_Vj (A_j @ x(0))
    MLP    = ReLU(W_in x_2(1) + b_in)
    x(2)   = x_2(1) + W_out MLP + b_out
    Logits = W_U x(2)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_head: int):
        super().__init__()
        self.d_head = d_head
        self.W_Q = nn.Linear(d_model, d_head, bias=False)
        self.W_K = nn.Linear(d_model, d_head, bias=False)
        self.W_V = nn.Linear(d_model, d_head, bias=False)
        self.W_O = nn.Linear(d_head, d_model, bias=False)
        self.scale = 1.0 / math.sqrt(d_head)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B, T, d_model]. Returns (head output added to stream, attention [B,T,T])."""
        q = self.W_Q(x)                      # [B, T, d_head]
        k = self.W_K(x)
        v = self.W_V(x)
        attn = torch.softmax((q @ k.transpose(-1, -2)) * self.scale, dim=-1)  # [B,T,T]
        out = self.W_O(attn @ v)             # [B, T, d_model]
        return out, attn


class OneLayerTransformer(nn.Module):
    """The mainline model.  For n_layers > 1 we simply stack attention + MLP
    blocks (used only for the loss-curve experiments of Appendix C.2.2); the
    detailed reverse engineering assumes a single block (the paper's mainline)."""

    EQ_INDEX: int = -1   # placeholder, set in __init__

    def __init__(self, p: int, d_model: int = 128, n_heads: int = 4,
                 d_head: int = 32, n_mlp: int = 512, n_layers: int = 1,
                 init_std: float = 0.02, seed: Optional[int] = None):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_mlp = n_mlp
        self.n_layers = n_layers
        self.eq_index = p            # token id of the special '=' token
        self.n_positions = 3         # a, b, '='

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # token embeddings: indices 0..p-1 are the values, index p is '='
        self.token_embed = nn.Embedding(p + 1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(self.n_positions, d_model))

        self.heads = nn.ModuleList([AttentionHead(d_model, d_head) for _ in range(n_heads)])
        self.mlp_in = nn.Linear(d_model, n_mlp)
        self.mlp_out = nn.Linear(n_mlp, d_model)
        self.unembed = nn.Linear(d_model, p, bias=False)

        self._init_weights(init_std)

    def _init_weights(self, init_std: float) -> None:
        # embeddings: small gaussian (matching the paper's small initial embeddings)
        nn.init.normal_(self.token_embed.weight, std=init_std)
        nn.init.normal_(self.pos_embed, std=init_std)
        # linear layers keep PyTorch's default (kaiming-uniform) init

    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_tokens(a: torch.Tensor, b: torch.Tensor, eq_index: int) -> torch.Tensor:
        """Build [B, 3] token id tensors for inputs a, b (ints) plus '='."""
        B = a.shape[0]
        eq = torch.full((B, 1), eq_index, dtype=a.dtype, device=a.device)
        return torch.cat([a.unsqueeze(-1), b.unsqueeze(-1), eq], dim=-1)

    def embed_inputs(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """x(0): [B, 3, d_model] = W_E t_i + p_i."""
        tok = self._make_tokens(a, b, self.eq_index)
        x = self.token_embed(tok)                     # [B, 3, d]
        x = x + self.pos_embed[: self.n_positions].unsqueeze(0)
        return x

    def forward_full(self, a: torch.Tensor, b: torch.Tensor):
        """Run the network and return (logits[B, p], diagnostics dict)."""
        x0 = self.embed_inputs(a, b)                  # [B, 3, d]
        x = x0
        attn_patterns: List[torch.Tensor] = []
        for _ in range(self.n_layers):
            attn_out = torch.zeros_like(x)
            for head in self.heads:
                h_out, attn = head(x)
                attn_out = attn_out + h_out
                attn_patterns.append(attn)            # [B, T, T]
            x = x + attn_out                          # residual around attention

        # read off the final token
        resid2 = x[:, -1]                             # [B, d]
        pre_act = self.mlp_in(resid2)
        h = torch.relu(pre_act)                       # MLP activations [B, n_mlp]
        out = self.mlp_out(h)
        x2 = resid2 + out                             # residual around MLP (with bias in out)
        logits = self.unembed(x2)                     # [B, p]

        diag = {
            "x0": x0,
            "attn_patterns": attn_patterns,           # one [B,T,T] per head (per block)
            "mlp_acts": h,                            # ReLU activations on the final token
            "resid2": resid2,
            "x2": x2,
        }
        return logits, diag

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_full(a, b)
        return logits

    # ------------------------------------------------------------------ #
    # helpers used by the analysis (operating on cpu numpy-free tensors)
    def all_pairs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """All P*P (a, b) input pairs on the model's device."""
        p = self.p
        a = torch.arange(p, device=next(self.parameters()).device)
        aa, bb = torch.meshgrid(a, a, indexing="ij")
        return aa.reshape(-1), bb.reshape(-1)

    @torch.no_grad()
    def logits_grid(self, chunk: int = 512, device: Optional[str] = None) -> torch.Tensor:
        """Logits for every (a, b): [P, P, P] tensor over the full input grid."""
        a, b = self.all_pairs()
        L = torch.empty(self.p, self.p, self.p)
        for i in range(0, a.numel(), chunk):
            aa = a[i:i + chunk]
            bb = b[i:i + chunk]
            lg, _ = self.forward_full(aa, bb)
            L[aa, bb] = lg
        return L

    # ------------------------------------------------------------------ #
    # weight accessors (naming follows the paper)
    @property
    def W_E(self) -> torch.Tensor:
        """Token embedding for the P values (excludes the '=' row): [d, p] in the
        paper's convention we return [p, d] here (row = value)."""
        return self.token_embed.weight[: self.p].detach()   # [p, d]

    @property
    def W_U(self) -> torch.Tensor:
        return self.unembed.weight.detach()                  # [p, d]

    @property
    def W_L(self) -> torch.Tensor:
        """W_L = W_U @ W_out : [p, n_mlp] mapping MLP activations -> logits."""
        return self.W_U @ self.mlp_out.weight.detach()       # [p, n_mlp]

    def parameter_summary(self) -> int:
        return sum(p.numel() for p in self.parameters())
