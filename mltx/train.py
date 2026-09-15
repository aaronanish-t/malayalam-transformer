"""Training loop. One run = one directory under runs/ with:

    config.json    model + training config, parameter count
    log.jsonl      one line per eval: step, train_loss, val_loss, lr, elapsed
    ckpt_*.pt      checkpoints at 10%, 50% and 100% of training
    samples.md     fixed-seed generations from each checkpoint

    python -m mltx.train --name baseline
    python -m mltx.train --name no_pe --pos-enc none
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from .model import ModelConfig, Transformer
from .tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"

# Every checkpoint is sampled with the same prompts and seed so the three
# snapshots in the README are directly comparable.
SAMPLE_PROMPTS = ["കേരളം ", "ഭാരതത്തിലെ ", "അദ്ദേഹം "]
CHECKPOINT_FRACS = (0.1, 0.5, 1.0)


@dataclass
class TrainConfig:
    name: str = "baseline"
    # model
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    pos_enc: str = "learned"
    # optimisation. tokens_per_step is held fixed across context lengths so
    # that ablations over block_size see the same amount of data.
    steps: int = 3000
    tokens_per_step: int = 8192
    lr: float = 3e-3
    min_lr_ratio: float = 0.1
    warmup: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # eval
    eval_every: int = 100
    eval_batches: int = 20
    sample_tokens: int = 300
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def batch_size(self) -> int:
        return max(1, self.tokens_per_step // self.block_size)


class Batches:
    def __init__(self, split: str, cfg: TrainConfig, gen: torch.Generator):
        self.data = torch.from_numpy(np.fromfile(DATA_DIR / f"{split}.bin", dtype=np.uint16).astype(np.int64))
        self.cfg = cfg
        self.gen = gen

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        T, B = self.cfg.block_size, self.cfg.batch_size
        ix = torch.randint(len(self.data) - T - 1, (B,), generator=self.gen)
        x = torch.stack([self.data[i:i + T] for i in ix])
        y = torch.stack([self.data[i + 1:i + 1 + T] for i in ix])
        return x.to(self.cfg.device), y.to(self.cfg.device)


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine)


@torch.no_grad()
def estimate_loss(model: Transformer, batches: dict[str, Batches], n: int) -> dict[str, float]:
    model.eval()
    out = {}
    for split, get in batches.items():
        losses = torch.zeros(n)
        for i in range(n):
            x, y = get()
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def sample(model: Transformer, tok: CharTokenizer, cfg: TrainConfig) -> str:
    model.eval()
    torch.manual_seed(cfg.seed)  # multinomial uses the global generator
    parts = []
    for prompt in SAMPLE_PROMPTS:
        idx = torch.tensor([tok.encode(prompt)], device=cfg.device)
        out = model.generate(idx, cfg.sample_tokens, temperature=0.8, top_k=40)
        parts.append(tok.decode(out[0].tolist()))
    model.train()
    return "\n\n---\n\n".join(parts)


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    run_dir = RUNS_DIR / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    tok = CharTokenizer.load(DATA_DIR / "tokenizer.json")
    mcfg = ModelConfig(vocab_size=tok.vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
                       n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=cfg.dropout, pos_enc=cfg.pos_enc)
    model = Transformer(mcfg).to(cfg.device)
    n_params = model.num_params()
    print(f"[{cfg.name}] {n_params / 1e6:.2f}M params | batch {cfg.batch_size} x {cfg.block_size} | {cfg.device}")

    gen = torch.Generator().manual_seed(cfg.seed)
    batches = {"train": Batches("train", cfg, gen), "val": Batches("val", cfg, gen)}

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg.lr, betas=(0.9, 0.95))
    use_amp = cfg.device == "cuda" and torch.cuda.is_bf16_supported()
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else torch.autocast("cpu", enabled=False)

    (run_dir / "config.json").write_text(json.dumps({**asdict(cfg), "batch_size": cfg.batch_size,
                                                     "model": mcfg.to_dict(), "n_params": n_params}, indent=2))
    log = (run_dir / "log.jsonl").open("w", encoding="utf-8")
    samples = (run_dir / "samples.md").open("w", encoding="utf-8")
    ckpt_steps = {max(1, round(f * cfg.steps)) for f in CHECKPOINT_FRACS}

    t0 = time.time()
    best_val = float("inf")
    for step in range(1, cfg.steps + 1):
        lr = lr_at(step - 1, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = batches["train"]()
        with autocast:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.eval_every == 0 or step == cfg.steps:
            losses = estimate_loss(model, batches, cfg.eval_batches)
            best_val = min(best_val, losses["val"])
            rec = {"step": step, "train_loss": losses["train"], "val_loss": losses["val"],
                   "lr": lr, "elapsed": time.time() - t0}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[{cfg.name}] step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| bpc {losses['val'] / math.log(2):.3f} | lr {lr:.2e} | {rec['elapsed']:.0f}s")

        if step in ckpt_steps:
            torch.save({"model": model.state_dict(), "model_config": mcfg.to_dict(), "step": step},
                       run_dir / f"ckpt_{step}.pt")
            samples.write(f"## step {step} ({100 * step / cfg.steps:.0f}%)\n\n")
            samples.write(sample(model, tok, cfg) + "\n\n")
            samples.flush()

    log.close()
    samples.close()
    final = {"name": cfg.name, "n_params": n_params, "best_val": best_val, "elapsed": time.time() - t0}
    (run_dir / "result.json").write_text(json.dumps({**final, "final": rec}, indent=2))
    return final


def parse_args(argv=None) -> TrainConfig:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f, default in asdict(TrainConfig()).items():
        flag = "--" + f.replace("_", "-")
        ap.add_argument(flag, type=type(default), default=default)
    return TrainConfig(**vars(ap.parse_args(argv)))


if __name__ == "__main__":
    train(parse_args())
