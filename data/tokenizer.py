from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import sentencepiece as spm


@dataclass(frozen=True)
class TokenizerMeta:
    unk_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 3


class TokenizerLike(Protocol):
    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...

    def decode(self, token_ids: Iterable[int]) -> str: ...


class SentencePieceTokenizer:
    """Minimal wrapper around the SentencePiece tokenizer."""

    def __init__(self, model_path: str | Path, meta: TokenizerMeta | None = None) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"SentencePiece tokenizer file not found: {path}")
        self.model_path = str(path)
        self.sp_model = spm.SentencePieceProcessor(model_file=str(path))
        self.meta = meta or TokenizerMeta()

    @property
    def vocab_size(self) -> int:
        return int(self.sp_model.get_piece_size())

    @property
    def unk_token_id(self) -> int:
        return self.meta.unk_token_id

    @property
    def bos_token_id(self) -> int:
        return self.meta.bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self.meta.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self.meta.pad_token_id

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        token_ids = list(self.sp_model.encode(text, out_type=int))
        if add_bos:
            token_ids.insert(0, self.bos_token_id)
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def encode_batch(
        self,
        texts: list[str],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[list[int]]:
        encoded = self.sp_model.encode(texts, out_type=int, add_bos=add_bos, add_eos=add_eos)
        return [list(token_ids) for token_ids in encoded]

    def decode(self, token_ids: Iterable[int]) -> str:
        return self.sp_model.decode(list(token_ids))
