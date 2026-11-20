"""Engine interface. An engine turns one chunk of 16 kHz audio into words."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

from ..types import Word


@dataclass
class ChunkContext:
    """Text context an engine may use to bias recognition.

    ``vocabulary``: course terms supplied by the user.
    ``previous_text``: the end of the already-transcribed preceding chunk
    (only used by engines whose ``uses_previous_text`` option is on).
    """

    vocabulary: list[str] = field(default_factory=list)
    previous_text: str = ""
    language: str = "en"


class Engine(abc.ABC):
    name: str = "base"
    #: maximum padded chunk length in seconds the engine is fed
    max_chunk_s: float = 30.0
    #: context padding on each side of a chunk core
    pad_s: float = 2.0
    #: chunks handed to transcribe_batch at once (1 = no batching)
    batch_size: int = 1
    #: make every padded chunk window exactly max_chunk_s long (constant shapes)
    fixed_window: bool = False

    @abc.abstractmethod
    def load(self) -> None: ...

    @abc.abstractmethod
    def transcribe_chunk(self, audio: np.ndarray, ctx: ChunkContext) -> list[Word]:
        """Return words with timestamps relative to the start of ``audio``."""

    def transcribe_batch(self, audios: list[np.ndarray], ctxs: list[ChunkContext]) -> list[list[Word]]:
        """Transcribe several independent chunks. Engines override this when they can batch."""
        return [self.transcribe_chunk(a, c) for a, c in zip(audios, ctxs)]

    def unload(self) -> None:
        import gc

        import mlx.core as mx

        for attr in list(vars(self)):
            if attr.startswith("_m"):
                setattr(self, attr, None)
        gc.collect()
        mx.clear_cache()

    @abc.abstractmethod
    def describe(self) -> dict: ...


def words_from_tokens(tokens: list[tuple[str, float, float, float | None]]) -> list[Word]:
    """Group subword tokens (SentencePiece style: leading space starts a word)."""
    out: list[Word] = []
    confs: list[list[float]] = []
    pending_space = False
    for text, start, end, conf in tokens:
        if not text:
            continue
        piece = text.strip()
        if not piece:
            # A bare-space token (Parakeet emits ' ' before digits and some
            # words) still marks a word boundary.
            pending_space = True
            continue
        starts_word = text[0].isspace() or pending_space or not out
        pending_space = False
        if starts_word:
            out.append(Word(piece, start, end))
            confs.append([] if conf is None else [conf])
        else:
            out[-1].text += piece
            out[-1].end = max(out[-1].end, end)
            if conf is not None:
                confs[-1].append(conf)
    for w, c in zip(out, confs):
        # A word is only as reliable as its least confident piece.
        w.confidence = min(c) if c else None
    return out
