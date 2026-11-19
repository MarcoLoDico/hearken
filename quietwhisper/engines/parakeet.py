"""NVIDIA Parakeet TDT via parakeet-mlx (Apache-2.0 code, CC-BY-4.0 weights).

Token-and-Duration Transducer: frame-synchronous decoding, so it cannot
"run on" and invent sentences during silence the way attention decoders can,
and every token carries a native timestamp and confidence.

Vocabulary biasing uses parakeet-mlx's trie booster, which requires beam search.

Front-end note: parakeet-mlx (<=0.7.0) computes the STFT magnitude as
|re| + |im| rather than sqrt(re^2 + im^2) as NeMo does. ``frontend="nemo"``
(default) uses a NeMo-faithful log-mel instead; ``frontend="library"`` keeps
the library's. The choice is benchmarked in docs/RESULTS.md.
"""

from __future__ import annotations

import numpy as np

from .. import models
from ..errors import EngineError
from ..types import Word
from .base import ChunkContext, Engine, words_from_tokens


def nemo_logmel(x, args):
    """NeMo-style log-mel (AudioToMelSpectrogramPreprocessor, inference mode).

    Adapted from parakeet_mlx.audio.get_logmel (Apache-2.0) with the true STFT
    magnitude, NeMo's log zero-guard (2**-24) and no dither. float32 throughout.
    """
    import mlx.core as mx
    from parakeet_mlx.audio import hanning, stft

    x = x.astype(mx.float32)
    if args.preemph is not None:
        x = mx.concat([x[:1], x[1:] - args.preemph * x[:-1]], axis=0)
    window = hanning(args.win_length).astype(mx.float32)
    spec = stft(x, args.n_fft, args.hop_length, args.win_length, window)
    mag = mx.abs(spec)  # complex modulus
    if args.mag_power != 1.0:
        mag = mx.power(mag, args.mag_power)
    mel = mx.matmul(args._filterbanks.astype(mx.float32), mag.T)
    mel = mx.log(mel + 2.0 ** -24)
    if args.normalize == "per_feature":
        mean = mx.mean(mel, axis=1, keepdims=True)
        std = mx.sqrt(mx.sum((mel - mean) ** 2, axis=1, keepdims=True) / max(mel.shape[1] - 1, 1))
    else:
        mean, std = mx.mean(mel), mx.std(mel)
    mel = (mel - mean) / (std + 1e-5)
    return mx.expand_dims(mel.T, axis=0)


class ParakeetEngine(Engine):
    name = "parakeet"

    def __init__(self, model_key: str = "parakeet-tdt-0.6b-v2", *, beam_size: int = 4,
                 boost_weight: float = 2.0, max_chunk_s: float = 120.0, pad_s: float = 5.0,
                 frontend: str = "nemo"):
        self.model_key = model_key
        self.beam_size = beam_size
        self.boost_weight = boost_weight
        self.max_chunk_s = max_chunk_s
        self.pad_s = pad_s
        if frontend not in ("nemo", "library"):
            raise ValueError(f"unknown frontend {frontend!r}")
        self.frontend = frontend
        self._model = None
        self._booster_vocab: tuple[str, ...] | None = None
        self._decoding = None

    def load(self) -> None:
        from parakeet_mlx import from_pretrained

        path = models.resolve_local(self.model_key)
        try:
            self._model = from_pretrained(str(path))
        except Exception as e:
            raise EngineError(f"failed to load {self.model_key}: {e}") from e

    def _decoding_config(self, vocab: list[str]):
        from parakeet_mlx.boost import lift, words
        from parakeet_mlx.parakeet import Beam, DecodingConfig, Greedy

        key = tuple(sorted(vocab))
        if self._decoding is not None and key == self._booster_vocab:
            return self._decoding
        if self.beam_size <= 1:
            dec = Greedy()
        elif vocab:
            # The trie booster matches at word boundaries itself.
            weights = {term: self.boost_weight for term in vocab if term.strip()}
            booster = lift(words(weights), self._model.vocabulary)
            dec = Beam(beam_size=self.beam_size, booster=booster)
        else:
            dec = Beam(beam_size=self.beam_size)
        self._decoding = DecodingConfig(decoding=dec)
        self._booster_vocab = key
        return self._decoding

    def transcribe_chunk(self, audio: np.ndarray, ctx: ChunkContext) -> list[Word]:
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        if self._model is None:
            raise EngineError("engine not loaded")
        cfg = self._model.preprocessor_config
        if len(audio) < cfg.hop_length * 4:
            return []
        try:
            x = mx.array(np.asarray(audio, dtype=np.float32))
            mel = nemo_logmel(x, cfg) if self.frontend == "nemo" else get_logmel(x, cfg)
            result = self._model.generate(mel.astype(mx.bfloat16),
                                          decoding_config=self._decoding_config(ctx.vocabulary))[0]
        except Exception as e:
            raise EngineError(f"parakeet decoding failed: {e}") from e
        toks = [(t.text, t.start, t.end, t.confidence) for t in result.tokens]
        return words_from_tokens(toks)

    def describe(self) -> dict:
        s = models.spec(self.model_key)
        return {"engine": self.name, "model": s.key, "repo": s.repo, "revision": s.revision,
                "license": s.license, "decoding": "beam" if self.beam_size > 1 else "greedy",
                "beam_size": self.beam_size, "boost_weight": self.boost_weight, "frontend": self.frontend,
                "max_chunk_s": self.max_chunk_s, "pad_s": self.pad_s}
