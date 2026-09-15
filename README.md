# A character-level Malayalam transformer, from scratch

A small GPT-style decoder trained on Malayalam Wikipedia, one Unicode code
point at a time. No `nn.Transformer`, no `nn.MultiheadAttention`, no
tokenizer library: the attention, the positional encodings, the tokenizer and
the training loop are all in this repo, in about 500 lines.

The point is not the model. It is the **ablation table** below: the same
network trained six times with one knob changed each time, on the same data
budget, so the effect of each knob can be read off directly.

## Why Malayalam, and why characters?

Malayalam is an abugida. What looks like one letter on screen, say **ക്ഷ**,
is three code points: a consonant, a virama (്) that suppresses the inherent
vowel, and a second consonant. Vowel signs (ാ ി ു ...) are separate code
points that attach to the consonant before them. A character-level model gets
none of that for free: it has to learn that a virama only follows a
consonant, that two vowel signs never stack, that a chillu (ൻ ർ ൽ ...) ends a
syllable. Ill-formed clusters are visible in the samples early in training and
disappear as the loss falls, which makes the checkpoints fun to compare.

The vocabulary is about 130 code points (anything rarer than 1 in 50,000
characters, mostly stray CJK and Cyrillic from Wikipedia, maps to a single
unknown token), so the embedding table is tiny and almost all of the
parameters are in the attention and MLP blocks.

## Model

| | |
|---|---|
| architecture | pre-norm decoder, GPT-2 style |
| layers / heads / width | 4 / 4 / 128 |
| context | 128 characters |
| parameters | ~0.8M |
| positional encoding | learned (ablated against sinusoidal and none) |
| optimiser | AdamW, lr 3e-3, cosine to 3e-4, 100 warmup steps, wd 0.1 |
| batch | 8192 characters per step, 3000 steps (~25M characters seen) |
| precision | bf16 autocast on GPU |

## Run it

```bash
pip install -r requirements.txt
python -m mltx.data                # ~30M chars of Malayalam Wikipedia -> data/*.bin
python -m mltx.train --name baseline
python -m mltx.ablate              # the whole grid, ~6 runs
python -m mltx.plot                # results/loss_curves.png
python -m mltx.report              # fills the section below into this README
python -m mltx.generate runs/baseline/ckpt_3000.pt --prompt "കേരളം "
```

On a free Colab T4 the baseline takes a few minutes and the full grid well
under an hour. `notebooks/colab.ipynb` does all of the above in order.

Sanity tests (tokenizer round trip, causal mask, overfitting a toy sequence):

```bash
pytest
```

## Results

<!-- RESULTS:START -->
_Run `python -m mltx.plot && python -m mltx.report` after training to fill this in._
<!-- RESULTS:END -->

## Reading the ablations

Things to look for once the table is filled in (edit this section to match
what you actually observe):

- **No positional encoding** should be clearly worse but not catastrophic. A
  causal decoder leaks position through the mask (the first token attends
  only to itself, and so on), so the model has a weak position signal even
  with nothing added to the embeddings.
- **Sinusoidal vs learned** should be close at this context length. Learned
  embeddings have 128 positions to learn and plenty of data per position.
- **2 vs 4 heads** at the same width means fewer, wider heads. Character-level
  modelling wants at least one head that tracks "previous consonant" and one
  that tracks word boundaries, so 2 heads has less room to specialise.
- **Context 64 / 128 / 256** with the batch scaled to keep tokens per step
  constant. Longer context helps until the model is too small to use it;
  watch whether 256 actually beats 128 or just costs more time per step.

## Layout

```
mltx/
  tokenizer.py   code-point tokenizer, NFC normalisation
  data.py        download + clean Malayalam Wikipedia, write train/val .bin
  model.py       the transformer (attention, MLP, blocks, positional encodings)
  train.py       training loop, eval, checkpoints, fixed-seed samples
  ablate.py      the ablation grid
  plot.py        loss curves
  report.py      writes the results section of this README
  generate.py    sample from a checkpoint
tests/           shape, causality, tokenizer and overfit tests
notebooks/       Colab notebook that runs everything
```
