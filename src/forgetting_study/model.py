import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from forgetting_study.config import ModelConfig


class Block(nn.Module):
    """Single-head causal self-attention (no output projection) followed by an MLP."""

    def __init__(self, cfg: ModelConfig, tap: callable) -> None:
        """Initialization function for single cross-attention + MLP block.

        @param cfg: Instance of the ModelConfig object used to specify hyperparameters of the model.
        @param tap: Tensor preprocessing function used to keep track of activation map names.
        """
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
        """Transformer block forwards pass.

        @param x:  Size (N, dim)
        @return m: Size (N, dim)
        """
        T = x.shape[1]
        h = self.norm_attn(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        # Scores are tapped before masking so the stored tensor stays finite;
        # gradients of the masked (future) entries are zero.
        scores = self.tap(
            f"{prefix}.attn.scores", q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        )
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
        probs = self.tap(
            f"{prefix}.attn.probs",
            scores.masked_fill(causal, float("-inf")).softmax(-1),
        )
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
    """Transformer object with variable number of heads, MLP size, vocabulary size, and context length.

    @attribute cfg:        Instance of the ModelConfig object used to specify hyperparameters of the model.
    @attribute vocab_size: Number of distinct tokens in the dictionary.
    @attribute tok_emb:    Embedding matrix for tokens, producing embeddings in cfg["dim"]-dimensional space.
    @attribute pos_emb:    Embedding matrix for token position, producing positional embeddings in cfg["dim"]-dimensional space.
    @attribute blocks:     List of block (single-headed attention + MLP) submodules composing the network.
    @attribute norm_out:   LayerNorm operation performed before final unembedding operation (which is non-standard).
    @attribute unembed:    Learned unembedding matrix mapping to a (N x vocab_size) length tensor of logits. Note that
                           this is not the transpose of the embedding matrix.
    @attribute _taps:      Dictionary of activation maps, indexed by name.
    """

    def __init__(self, cfg: ModelConfig, vocab_size: int, max_len: int) -> None:
        """Initialization function for the transformer object.

        @param cfg:        Instance of the ModelConfig object used to specify hyperparameters of the model.
        @param vocab_size: Number of distinct tokens in the dictionary.
        @param max_len:    Context length for inputs to the model.
        """
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(max_len, cfg.dim)
        self.blocks = nn.ModuleList(
            [Block(cfg, self._tap) for _ in range(cfg.n_blocks)]
        )
        self.norm_out = nn.LayerNorm(cfg.dim) if cfg.norm else nn.Identity()
        self.unembed = nn.Linear(cfg.dim, vocab_size, bias=cfg.bias)
        self._taps: dict[str, torch.Tensor] | None = None

    def _tap(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Tensor pre-processing operation that assigns each activation map a name and stores the tensor
        of the activation map inside the self._taps object.

        @param name: Name, passed as string, by which the map will be keyed in self._taps
        @param x:    Tensor input to layer.

        @return x:   Tensor input to layer, with grad_retained if necessary.
        """
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
        """Network forwards pass.

        @param idx: Input tensor of one-hot encoded token vectors, size (N, vocab_size)
        @return:    Output tensor of logits, size (N, vocab_size)
        """
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self._tap("embed", self.tok_emb(idx) + self.pos_emb(pos))
        for i, block in enumerate(self.blocks):
            x = block(x, prefix=f"block{i}")
        return self._tap("logits", self.unembed(self.norm_out(x)))
