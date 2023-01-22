"""tests for char tokenizer."""

import os
import tempfile

from src.data import CharTokenizer


def test_char_tokenizer_roundtrip():
    text = "hello world\n the quick brown fox"
    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert tok.vocab_size == len(set(text))


def test_char_tokenizer_save_load():
    text = "abracadabra"
    tok = CharTokenizer.from_text(text)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tok.pkl")
        tok.save(p)
        tok2 = CharTokenizer.load(p)
    assert tok2.vocab == tok.vocab
    assert tok2.encode(text) == tok.encode(text)
