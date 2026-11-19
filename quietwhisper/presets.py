"""Named, reproducible pipeline configurations.

A preset fixes the engines, their decoding parameters and their order. The
first engine is the *primary*: its words form the transcript. Further engines
are *verifiers*: they never change the primary text, they produce evidence
for the uncertain-passage list.

Engine choice and defaults are justified by the benchmark in docs/RESULTS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import ConfigError


@dataclass(frozen=True)
class EngineSpec:
    engine: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    engines: tuple[EngineSpec, ...]


PRESETS: dict[str, Preset] = {
    p.name: p
    for p in [
        Preset("max", "Most accurate: Qwen3-ASR 1.7B transcript, cross-checked by Parakeet TDT v2.",
               (EngineSpec("qwen3", {}), EngineSpec("parakeet", {"beam_size": 4}))),
        Preset("max-parakeet", "Parakeet TDT v2 transcript (beam 4), cross-checked by Qwen3-ASR 1.7B.",
               (EngineSpec("parakeet", {"beam_size": 4}), EngineSpec("qwen3", {}))),
        Preset("triple", "Qwen3-ASR primary, verified by both Parakeet and Whisper large-v3.",
               (EngineSpec("qwen3", {}), EngineSpec("parakeet", {"beam_size": 4}), EngineSpec("whisper", {}))),
        Preset("qwen3", "Single engine: Qwen3-ASR 1.7B (no cross-check).", (EngineSpec("qwen3", {}),)),
        Preset("qwen3-8bit", "Single engine: Qwen3-ASR 1.7B, 8-bit weights (lighter).",
               (EngineSpec("qwen3", {"model_key": "qwen3-asr-1.7b-8bit"}),)),
        Preset("parakeet", "Single engine: Parakeet TDT 0.6B v2, beam 4 (no cross-check).",
               (EngineSpec("parakeet", {"beam_size": 4}),)),
        Preset("parakeet-greedy", "Single engine: Parakeet TDT 0.6B v2, greedy (fastest).",
               (EngineSpec("parakeet", {"beam_size": 1}),)),
        Preset("whisper", "Single engine: Whisper large-v3 (baseline).", (EngineSpec("whisper", {}),)),
        Preset("parakeet-libfe", "Benchmark only: Parakeet beam 4 with parakeet-mlx's own log-mel front-end.",
               (EngineSpec("parakeet", {"beam_size": 4, "frontend": "library"}),)),
    ]
}

DEFAULT_PRESET = "max"


def get(name: str) -> Preset:
    try:
        return PRESETS[name]
    except KeyError:
        raise ConfigError(f"unknown preset {name!r}; choose from: {', '.join(PRESETS)}") from None


def model_keys(preset: Preset) -> list[str]:
    keys = []
    for e in preset.engines:
        if e.engine == "parakeet":
            keys.append(e.params.get("model_key", "parakeet-tdt-0.6b-v2"))
        elif e.engine == "qwen3":
            keys += [e.params.get("model_key", "qwen3-asr-1.7b"), "qwen3-forced-aligner-0.6b"]
        elif e.engine == "whisper":
            keys.append(e.params.get("model_key", "whisper-large-v3"))
    return list(dict.fromkeys(keys))
