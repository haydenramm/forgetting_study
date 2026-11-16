import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from forgetting_study.config import ModelConfig


class Block(nn.Module):
    """Single-head causal self-attention (no output projection) followed by an MLP."""

    def __init__(self, cfg: ModelConfig, tap):
        super().__init__()
        d, h = cfg.dim, cfg.dim * cfg.mlp_ratio
        self.cfg = cfg
        self.tap = tap
        self.norm_attn = nn.LayerNorm(d) if cfg.norm else nn.Identity()
        self.q = nn.Linear(d, d, bias=cfg.bias)
        self.k = nn.Linear(d, d, bias=cfg.bias)
        self.v = nn.Linear(d, d, bias=cfg.bias)
        self.norm_mlp = nn.LayerNorm(d) if cfg.norm else nn.Identity()
        self.up = nn.Linear(d, h, bias=cfg.bias)
        self.down = nn.Linear(h, d, bias=cfg.bias)

    def forward(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        T = x.shape[1]
        h = self.norm_attn(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        # Scores are tapped before masking so the stored tensor stays finite;
        # gradients of the masked (future) entries are zero.
        scores = self.tap(f"{prefix}.attn.scores", q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]))
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
        probs = self.tap(f"{prefix}.attn.probs", scores.masked_fill(causal, float("-inf")).softmax(-1))
        a = self.tap(f"{prefix}.attn.out", probs @ v)
        x = x + a if self.cfg.residual else a

        h = self.norm_mlp(x)
        u = self.tap(f"{prefix}.mlp.up", self.up(h))
        r = self.tap(f"{prefix}.mlp.act", F.relu(u))
        m = self.tap(f"{prefix}.mlp.down", self.down(r))
        if self.cfg.residual:
            return self.tap(f"{prefix}.out", x + m)
        return m


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int, max_len: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(max_len, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg, self._tap) for _ in range(cfg.n_blocks)])
        self.norm_out = nn.LayerNorm(cfg.dim) if cfg.norm else nn.Identity()
        self.unembed = nn.Linear(cfg.dim, vocab_size, bias=cfg.bias)
        self._taps: dict[str, torch.Tensor] | None = None

    def _tap(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if self._taps is not None:
            if x.requires_grad:
                x.retain_grad()
            self._taps[name] = x
        return x

    @contextmanager
    def recording(self):
        """Within this context, forward passes store intermediate tensors (with
        retained gradients) in the yielded dict, keyed by name."""
        self._taps = {}
        try:
            yield self._taps
        finally:
            self._taps = None

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self._tap("embed", self.tok_emb(idx) + self.pos_emb(pos))
        for i, block in enumerate(self.blocks):
            x = block(x, prefix=f"block{i}")
        return self._tap("logits", self.unembed(self.norm_out(x)))
