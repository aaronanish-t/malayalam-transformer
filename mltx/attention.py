"""Plot what every attention head looks at over one Malayalam sentence.

    python -m mltx.attention runs/baseline/ckpt_3000.pt
    python -m mltx.attention runs/baseline/ckpt_3000.pt --text "..." --out results/attention.png

One panel per (layer, head). Row i shows where position i attends, scaled so
the row's strongest link is brightest. On a
character-level model the interesting heads are usually easy to spot: one
looks at the previous character, one jumps back to the last space (word
start), and one locks onto the consonant a vowel sign belongs to.

matplotlib's default font has no Malayalam glyphs, so this looks for an
installed Malayalam font (Nirmala UI on Windows, Noto Sans Malayalam or the
SMC fonts on Linux; on Colab `apt install fonts-noto-core` provides one).
Without one, the axes are labelled with code points instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from .generate import load  # noqa: E402
from .train import ROOT  # noqa: E402

DEFAULT_TEXT = "കേരളം ഇന്ത്യയിലെ ഒരു സംസ്ഥാനമാണ്."
MALAYALAM_FONTS = ["Nirmala UI", "Noto Sans Malayalam", "Manjari", "Rachana", "Meera", "Kartika"]


def malayalam_font() -> str | None:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((name for name in MALAYALAM_FONTS if name in installed), None)


@torch.no_grad()
def attention_maps(model, idx: torch.Tensor) -> list[torch.Tensor]:
    """One (n_head, T, T) tensor per layer, for a single sequence."""
    for block in model.blocks:
        block.attn.record = True
    try:
        model(idx)
        return [block.attn.last_att[0].cpu() for block in model.blocks]
    finally:
        for block in model.blocks:
            block.attn.record = False
            block.attn.last_att = None


def plot(maps: list[torch.Tensor], chars: list[str], out: Path) -> Path:
    font = malayalam_font()
    labels = chars if font else [f"{ord(c):04X}" for c in chars]
    labels = [("_" if c == " " else c) for c in labels]  # space is otherwise invisible
    n_layer, n_head = len(maps), maps[0].shape[0]
    size = max(2.4, 0.16 * len(chars))
    fig, axes = plt.subplots(n_layer, n_head, figsize=(size * n_head, size * n_layer), squeeze=False)
    for l, layer in enumerate(maps):
        for h in range(n_head):
            ax = axes[l][h]
            # scale each row to its own max: later rows spread attention over
            # more characters, and on a shared scale their pattern disappears
            att = layer[h] / layer[h].amax(dim=-1, keepdim=True)
            ax.imshow(att.numpy(), cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"layer {l + 1}, head {h + 1}", fontsize=9)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=6, fontname=font, rotation=90 if not font else 0)
            ax.set_yticklabels(labels, fontsize=6, fontname=font)
            ax.tick_params(length=0)
    fig.supxlabel("attended-to character")
    fig.supylabel("query character")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "attention.png")
    args = ap.parse_args()

    model, tok = load(args.ckpt, "cpu")
    text = args.text[: model.cfg.block_size]
    idx = torch.tensor([tok.encode(text)])
    out = plot(attention_maps(model, idx), list(text), args.out)
    if not malayalam_font():
        print("no Malayalam font found; axes show code points")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
