"""Fill the results section of README.md from runs/.

Everything between <!-- RESULTS:START --> and <!-- RESULTS:END --> is replaced:
the ablation table, the loss-curve figure and the baseline samples at the
three checkpoints. Run after `python -m mltx.plot`.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .train import RUNS_DIR, ROOT

README = ROOT / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def load_run(run: Path) -> dict | None:
    cfg, res = run / "config.json", run / "result.json"
    if not (cfg.exists() and res.exists()):
        return None
    return {"name": run.name, **json.loads(cfg.read_text()), **json.loads(res.read_text())}


def ablation_table(runs: list[dict]) -> str:
    base = next((r for r in runs if r["name"] == "baseline"), None)
    rows = ["| run | positional enc. | heads | context | batch | params | val loss | val bpc | Δ vs baseline | time |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        val = r["best_val"]
        delta = "" if base is None else ("—" if r is base else f"{val - base['best_val']:+.3f}")
        rows.append(f"| {r['name']} | {r['pos_enc']} | {r['n_head']} | {r['block_size']} | {r['batch_size']} "
                    f"| {r['n_params'] / 1e6:.2f}M | {val:.3f} | {val / math.log(2):.3f} | {delta} "
                    f"| {r['elapsed'] / 60:.1f} min |")
    return "\n".join(rows)


def samples_block(run: Path) -> str:
    path = run / "samples.md"
    if not path.exists():
        return "_no samples yet_"
    text = path.read_text(encoding="utf-8")
    # demote headings and wrap generations in code fences so wiki-ish text
    # does not get interpreted as markdown
    out = []
    for part in re.split(r"^## ", text, flags=re.M):
        if not part.strip():
            continue
        head, _, body = part.partition("\n")
        out.append(f"#### {head.strip()}\n\n```text\n{body.strip()}\n```")
    return "\n\n".join(out)


def build(runs: list[dict]) -> str:
    steps = runs[0]["steps"] if runs else "?"
    tps = runs[0]["tokens_per_step"] if runs else "?"
    return f"""{START}
### Ablations

Every run trains for {steps} steps at {tps} tokens per step (the same data
budget), with a single change from the baseline. **val bpc** is bits per
character (val loss / ln 2). Lower is better.

{ablation_table(runs)}

### Loss curves

![loss curves](results/loss_curves.png)

### Samples from the baseline at three checkpoints

Prompts are fixed and the sampling seed is the same at every checkpoint
(temperature 0.8, top-k 40), so differences are the model, not the dice.

{samples_block(RUNS_DIR / "baseline")}
{END}"""


def main() -> None:
    runs = [r for r in (load_run(p) for p in sorted(RUNS_DIR.iterdir()) if p.is_dir()) if r]
    if not runs:
        raise SystemExit("no finished runs under runs/")
    # baseline first, then the grid order
    runs.sort(key=lambda r: (r["name"] != "baseline", r["name"]))
    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit("README.md is missing the RESULTS markers")
    pre, _, rest = text.partition(START)
    _, _, post = rest.partition(END)
    README.write_text(pre + build(runs) + post, encoding="utf-8")
    print(f"updated {README} with {len(runs)} runs")


if __name__ == "__main__":
    main()
