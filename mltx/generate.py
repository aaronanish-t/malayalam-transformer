"""Sample from a checkpoint.

    python -m mltx.generate runs/baseline/ckpt_3000.pt --prompt "കേരളം " --tokens 400
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import ModelConfig, Transformer
from .tokenizer import CharTokenizer
from .train import DATA_DIR


def load(ckpt: Path, device: str) -> tuple[Transformer, CharTokenizer]:
    state = torch.load(ckpt, map_location=device)
    model = Transformer(ModelConfig(**state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, CharTokenizer.load(DATA_DIR / "tokenizer.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--prompt", default="കേരളം ")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.seed is not None:
        torch.manual_seed(args.seed)
    model, tok = load(args.ckpt, device)
    idx = torch.tensor([tok.encode(args.prompt)], device=device)
    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
