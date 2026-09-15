import torch

from mltx.model import ModelConfig, Transformer
from mltx.tokenizer import CharTokenizer, normalize


def small(**kw) -> Transformer:
    cfg = ModelConfig(vocab_size=40, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0, **kw)
    return Transformer(cfg)


def test_tokenizer_roundtrip():
    text = normalize("കേരളം ഭാരതത്തിലെ ഒരു സംസ്ഥാനമാണ്.\n")
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text
    # unseen chars map to the replacement character rather than crashing
    assert tok.decode(tok.encode("xyz")) == CharTokenizer.UNK * 3


def test_output_shape_and_loss():
    model = small()
    x = torch.randint(0, 40, (3, 16))
    logits, loss = model(x, x)
    assert logits.shape == (3, 16, 40)
    assert loss.item() > 0


def test_causal_mask_blocks_the_future():
    """Changing token t must not change logits at positions < t."""
    torch.manual_seed(0)
    model = small().eval()
    x = torch.randint(0, 40, (1, 16))
    y = x.clone()
    y[0, 10] = (y[0, 10] + 1) % 40
    a, _ = model(x)
    b, _ = model(y)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-6)
    assert not torch.allclose(a[0, 10:], b[0, 10:])


def test_all_positional_encodings_run():
    for pe in ("learned", "sinusoidal", "none"):
        logits, _ = small(pos_enc=pe)(torch.zeros(1, 8, dtype=torch.long))
        assert logits.shape == (1, 8, 40)


def test_can_overfit_a_tiny_sequence():
    torch.manual_seed(0)
    model = small()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    seq = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]])
    x, y = seq[:, :-1], seq[:, 1:]
    _, first = model(x, y)
    for _ in range(150):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first.item() * 0.2


def test_generate_extends_sequence():
    model = small().eval()
    out = model.generate(torch.zeros(1, 4, dtype=torch.long), max_new_tokens=20, top_k=5)
    assert out.shape == (1, 24)
