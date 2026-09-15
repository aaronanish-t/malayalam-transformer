"""Run the ablation grid. Each run differs from the baseline in exactly one thing.

    python -m mltx.ablate                 # full grid
    python -m mltx.ablate --only no_pe    # a subset
    python -m mltx.ablate --steps 200     # quick smoke test

Tokens per optimiser step are held constant (8192), so ctx_64 uses batch 128,
ctx_128 batch 64 and ctx_256 batch 32. Every run sees the same data budget.
"""

from __future__ import annotations

import argparse
import json

from .train import TrainConfig, train, RUNS_DIR

GRID: dict[str, dict] = {
    "baseline":   {},
    "no_pe":      {"pos_enc": "none"},
    "sinusoidal": {"pos_enc": "sinusoidal"},
    "heads_2":    {"n_head": 2},
    "ctx_64":     {"block_size": 64},
    "ctx_256":    {"block_size": 256},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, help="run names to include")
    ap.add_argument("--steps", type=int, default=TrainConfig.steps)
    ap.add_argument("--eval-every", type=int, default=TrainConfig.eval_every)
    ap.add_argument("--skip-done", action="store_true", help="skip runs that already have result.json")
    args = ap.parse_args()

    names = args.only or list(GRID)
    results = []
    for name in names:
        if args.skip_done and (RUNS_DIR / name / "result.json").exists():
            print(f"[{name}] already done, skipping")
            continue
        cfg = TrainConfig(name=name, steps=args.steps, eval_every=args.eval_every, **GRID[name])
        results.append(train(cfg))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
