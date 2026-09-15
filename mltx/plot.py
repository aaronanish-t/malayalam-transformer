"""Loss curves from runs/*/log.jsonl -> results/loss_curves.png

Left panel: baseline train vs val. Right panel: val loss of every run.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .train import RUNS_DIR, ROOT  # noqa: E402

RESULTS_DIR = ROOT / "results"


def load_log(run: Path) -> list[dict]:
    path = run / "log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    runs = {p.name: load_log(p) for p in sorted(RUNS_DIR.iterdir()) if p.is_dir()}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        raise SystemExit("no runs found under runs/")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    base = runs.get("baseline") or next(iter(runs.values()))
    steps = [r["step"] for r in base]
    ax1.plot(steps, [r["train_loss"] for r in base], label="train")
    ax1.plot(steps, [r["val_loss"] for r in base], label="val")
    ax1.set_title("baseline: train vs val")
    ax1.set_xlabel("step")
    ax1.set_ylabel("cross-entropy (nats / char)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    for name, log in runs.items():
        ax2.plot([r["step"] for r in log], [r["val_loss"] for r in log], label=name,
                 lw=2.2 if name == "baseline" else 1.4)
    ax2.set_title("validation loss, all runs")
    ax2.set_xlabel("step")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = RESULTS_DIR / "loss_curves.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
