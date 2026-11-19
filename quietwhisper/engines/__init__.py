"""ASR engine adapters. Imports of the heavy ML libraries happen lazily inside each engine."""

from .base import ChunkContext, Engine


def create(name: str, **kwargs) -> Engine:
    if name == "parakeet":
        from .parakeet import ParakeetEngine
        return ParakeetEngine(**kwargs)
    if name == "qwen3":
        from .qwen3 import Qwen3Engine
        return Qwen3Engine(**kwargs)
    if name == "whisper":
        from .whisper import WhisperEngine
        return WhisperEngine(**kwargs)
    raise ValueError(f"unknown engine {name!r}")


__all__ = ["ChunkContext", "Engine", "create"]
