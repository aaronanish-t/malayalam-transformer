"""A decoder-only transformer, written out by hand.

Nothing here uses nn.MultiheadAttention or nn.Transformer. The pieces:

    token embedding (+ positional encoding: learned | sinusoidal | none)
    N x [ LayerNorm -> causal multi-head self-attention -> residual
          LayerNorm -> MLP (4x, GELU)                  -> residual ]
    LayerNorm -> linear to vocab (weights tied to the token embedding)

"none" positional encoding is a real ablation, not a bug: a causal decoder can
still infer position weakly from the attention mask, so it trains, just worse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 128     # context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    pos_enc: str = "learned"  # learned | sinusoidal | none

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must divide evenly across heads"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, nh, T, T)
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v                                                    # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


def sinusoidal_table(block_size: int, n_embd: int) -> torch.Tensor:
    pos = torch.arange(block_size, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, n_embd, 2, dtype=torch.float32) * (-math.log(10000.0) / n_embd))
    table = torch.zeros(block_size, n_embd)
    table[:, 0::2] = torch.sin(pos * div)
    table[:, 1::2] = torch.cos(pos * div)
    return table


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        if cfg.pos_enc == "learned":
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        elif cfg.pos_enc == "sinusoidal":
            self.register_buffer("pos_table", sinusoidal_table(cfg.block_size, cfg.n_embd), persistent=False)
        elif cfg.pos_enc != "none":
            raise ValueError(f"unknown pos_enc {cfg.pos_enc!r}")
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)
        # scale the residual-path output projections down with depth (GPT-2 trick)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.cfg.pos_enc == "learned":
            n -= self.pos_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        x = self.tok_emb(idx)
        if self.cfg.pos_enc == "learned":
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        elif self.cfg.pos_enc == "sinusoidal":
            x = x + self.pos_table[:T]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
