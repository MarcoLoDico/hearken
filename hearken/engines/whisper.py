"""OpenAI Whisper via mlx-whisper (MIT code and weights).

Chunks are VAD-cut and at most 30 s, i.e. a single Whisper window each, which
removes Whisper's own seek/sliding logic (a known source of repetition loops
and skipped audio on long files). ``condition_on_previous_text`` is off for
the same reason; the vocabulary goes into the initial prompt instead.
"""

from __future__ import annotations

import numpy as np

from .. import models
from ..errors import EngineError
from ..types import Word
from .base import ChunkContext, Engine


class WhisperEngine(Engine):
    name = "whisper"

    def __init__(self, model_key: str = "whisper-large-v3", *, max_chunk_s: float = 30.0,
                 pad_s: float = 1.5, beam_size: int | None = None, use_previous_text: bool = False):
        self.model_key = model_key
        self.max_chunk_s = max_chunk_s
        self.pad_s = pad_s
        self.beam_size = beam_size  # mlx-whisper does not implement beam search yet
        self.use_previous_text = use_previous_text
        self._path: str | None = None

    def load(self) -> None:
        self._path = str(models.resolve_local(self.model_key))

    def transcribe_chunk(self, audio: np.ndarray, ctx: ChunkContext) -> list[Word]:
        import mlx_whisper

        if self._path is None:
            raise EngineError("engine not loaded")
        prompt_parts = []
        if ctx.vocabulary:
            prompt_parts.append("Glossary: " + ", ".join(ctx.vocabulary) + ".")
        if self.use_previous_text and ctx.previous_text:
            prompt_parts.append(ctx.previous_text)
        try:
            res = mlx_whisper.transcribe(
                np.asarray(audio, dtype=np.float32), path_or_hf_repo=self._path, language=ctx.language,
                word_timestamps=True, condition_on_previous_text=False,
                initial_prompt=" ".join(prompt_parts) or None,
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=2.4,
                logprob_threshold=-1.0, no_speech_threshold=0.6,
                hallucination_silence_threshold=2.0, verbose=None)
        except Exception as e:
            raise EngineError(f"whisper decoding failed: {e}") from e
        out = []
        for seg in res.get("segments", []):
            for w in seg.get("words", []):
                t = w["word"].strip()
                if t:
                    out.append(Word(t, float(w["start"]), float(w["end"]), float(w.get("probability", 0.0))))
        return out

    def describe(self) -> dict:
        s = models.spec(self.model_key)
        return {"engine": self.name, "model": s.key, "repo": s.repo, "revision": s.revision,
                "license": s.license, "decoding": "greedy + temperature fallback",
                "condition_on_previous_text": False, "max_chunk_s": self.max_chunk_s, "pad_s": self.pad_s}
