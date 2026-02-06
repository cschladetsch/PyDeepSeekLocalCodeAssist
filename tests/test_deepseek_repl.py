import os
from pathlib import Path

import deepseek_repl


class _DummyFile:
    def __init__(self, name: str):
        self.name = name


def test_get_max_length_uses_tokenizer_cap():
    class _Tok:
        model_max_length = 4096

    assert deepseek_repl.get_max_length(_Tok()) == 2048


def test_get_max_length_default_when_missing_attr():
    class _Tok:
        pass

    assert deepseek_repl.get_max_length(_Tok()) == 1024


def test_process_files_reads_text_and_binary(tmp_path: Path):
    text_path = tmp_path / "note.txt"
    text_path.write_text("hello world", encoding="utf-8")

    bin_path = tmp_path / "image.png"
    bin_path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff\x00\xff")

    result = deepseek_repl.process_files(
        [_DummyFile(str(text_path)), _DummyFile(str(bin_path))]
    )

    assert "File: note.txt" in result
    assert "Content:\nhello world" in result
    assert "Binary File: image.png" in result
    assert "File type: png" in result
