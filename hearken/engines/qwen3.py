"""Alibaba Qwen3-ASR via mlx-qwen3-asr (Apache-2.0 code and weights).

An audio-encoder + LLM-decoder model. It accepts free-text *context*, which we
fill with the course vocabulary (and optionally the preceding transcript),
and it is paired with Qwen3-ForcedAligner for word timestamps. It produces no
per-word confidence.
"""

from __future__ import annotations

import re

import numpy as np

from .. import models
from ..errors import EngineError
from ..types import Word
from .base import ChunkContext, Engine

_LANG = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
         "pt": "Portuguese", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ru": "Russian"}
_ALNUM = re.compile(r"[\W_]+", re.UNICODE)


def _key(s: str) -> str:
    return _ALNUM.sub("", s).casefold()


def attach_timings(text: str, aligned: list[tuple[str, float, float]]) -> list[Word]:
    """Map aligner items (punctuation-free tokens) back onto the punctuated text.

    Text words are the source of truth for *what* was said; the aligner only
    contributes times. Words the aligner skipped get times interpolated from
    their neighbours and are flagged ``interpolated_time``.
    """
    words = [Word(t, -1.0, -1.0) for t in text.split()]
    j = 0
    for w in words:
        target = _key(w.text)
        if not target:
            continue
        acc, k, t0, t1 = "", j, None, None
        while k < len(aligned) and len(acc) < len(target):
            piece = _key(aligned[k][0])
            if not target.startswith(acc + piece):
                break
            acc += piece
            t0 = aligned[k][1] if t0 is None else t0
            t1 = aligned[k][2]
            k += 1
        if acc == target and t0 is not None:
            w.start, w.end = float(t0), float(t1)
            j = k
    _interpolate(words)
    return words


def _interpolate(words: list[Word]) -> None:
    """Spread each run of untimed words evenly between its timed neighbours."""
    if not any(w.start >= 0 for w in words):
        return
    i = 0
    while i < len(words):
        if words[i].start >= 0:
            i += 1
            continue
        j = i
        while j < len(words) and words[j].start < 0:
            j += 1
        t0 = words[i - 1].end if i > 0 else words[j].start
        t1 = words[j].start if j < len(words) else words[i - 1].end
        step = (t1 - t0) / (j - i)
        for n, w in enumerate(words[i:j]):
            w.start, w.end = t0 + n * step, t0 + (n + 1) * step
            w.flags.append("interpolated_time")
        i = j


class Qwen3Engine(Engine):
    name = "qwen3"

    def __init__(self, model_key: str = "qwen3-asr-1.7b", aligner_key: str = "qwen3-forced-aligner-0.6b",
                 *, max_chunk_s: float = 30.0, pad_s: float = 1.5, use_previous_text: bool = False):
        self.model_key = model_key
        self.aligner_key = aligner_key
        self.max_chunk_s = max_chunk_s
        self.pad_s = pad_s
        self.use_previous_text = use_previous_text
        self._model = None
        self._maligner = None

    def load(self) -> None:
        from mlx_qwen3_asr import ForcedAligner, load_model

        path = models.resolve_local(self.model_key)
        apath = models.resolve_local(self.aligner_key)
        try:
            self._model, _ = load_model(str(path))
            self._maligner = ForcedAligner(str(apath))
        except Exception as e:
            raise EngineError(f"failed to load {self.model_key}: {e}") from e

    @staticmethod
    def build_context(ctx: ChunkContext, use_previous: bool) -> str:
        parts = []
        if ctx.vocabulary:
            parts.append("Technical terms that may occur: " + ", ".join(ctx.vocabulary) + ".")
        if use_previous and ctx.previous_text:
            parts.append(ctx.previous_text)
        return "\n".join(parts)

    def transcribe_chunk(self, audio: np.ndarray, ctx: ChunkContext) -> list[Word]:
        from mlx_qwen3_asr import transcribe

        if self._model is None:
            raise EngineError("engine not loaded")
        x = np.asarray(audio, dtype=np.float32)
        if len(x) < 0.3 * 16000:
            return []
        lang = _LANG.get(ctx.language, ctx.language)
        try:
            res = transcribe(x, model=self._model, context=self.build_context(ctx, self.use_previous_text),
                             language=lang, return_timestamps=False)
        except Exception as e:
            raise EngineError(f"qwen3 decoding failed: {e}") from e
        text = res.text.strip()
        if not text:
            return []
        try:
            aligned = self._maligner.align(x, text, lang)
        except Exception as e:
            raise EngineError(f"qwen3 forced alignment failed: {e}") from e
        words = attach_timings(text, [(a.text, a.start_time, a.end_time) for a in aligned])
        if res.truncated:
            for w in words[-3:]:
                w.flags.append("decoder_truncated")
        return words

    def describe(self) -> dict:
        s, a = models.spec(self.model_key), models.spec(self.aligner_key)
        return {"engine": self.name, "model": s.key, "repo": s.repo, "revision": s.revision,
                "license": s.license, "aligner": a.repo, "aligner_revision": a.revision,
                "decoding": "greedy (temperature 0)", "max_chunk_s": self.max_chunk_s,
                "pad_s": self.pad_s, "use_previous_text": self.use_previous_text}
