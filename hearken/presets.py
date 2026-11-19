"""Named, reproducible pipeline configurations.

A preset fixes the engines, their decoding parameters and their order. The
first engine is the *primary*: its words form the transcript. Further engines
are *verifiers*: they never change the primary text, they produce evidence
for the uncertain-passage list.

Engine choice and defaults are justified by the benchmark in docs/RESULTS.md
(docs/MODEL_CHOICE.md for the short version).
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
        Preset("accurate", "Default. Qwen3-ASR 1.7B (8-bit) transcript, cross-checked by Parakeet TDT v2. "
               "~4-5 min per audio hour, ~6 GB.",
               (EngineSpec("qwen3", {"model_key": "qwen3-asr-1.7b-8bit"}), EngineSpec("parakeet", {"beam_size": 1}))),
        Preset("accurate-full", "As 'accurate' with full-precision Qwen3 weights. Measured no more accurate; ~8 GB.",
               (EngineSpec("qwen3", {}), EngineSpec("parakeet", {"beam_size": 1}))),
        Preset("fast", "Parakeet TDT v2 only. ~1 min per audio hour, ~2.4 GB. Drops some repetitions/fillers; "
               "no cross-check.", (EngineSpec("parakeet", {"beam_size": 1}),)),
        Preset("qwen3-8bit", "Qwen3-ASR 1.7B 8-bit only (no cross-check).",
               (EngineSpec("qwen3", {"model_key": "qwen3-asr-1.7b-8bit"}),)),
        Preset("qwen3", "Qwen3-ASR 1.7B full precision only (no cross-check).", (EngineSpec("qwen3", {}),)),
        Preset("triple", "Qwen3-ASR 8-bit verified by Parakeet and Whisper large-v3 (needs the [whisper] extra).",
               (EngineSpec("qwen3", {"model_key": "qwen3-asr-1.7b-8bit"}), EngineSpec("parakeet", {"beam_size": 1}),
                EngineSpec("whisper", {}))),
        Preset("whisper", "Whisper large-v3 only; baseline (needs the [whisper] extra).", (EngineSpec("whisper", {}),)),
        Preset("parakeet-beam", "Benchmark: Parakeet beam 4 (measured no better than greedy, 3x slower).",
               (EngineSpec("parakeet", {"beam_size": 4}),)),
        Preset("parakeet-libfe", "Benchmark: Parakeet beam 4 with parakeet-mlx's own log-mel front-end.",
               (EngineSpec("parakeet", {"beam_size": 4, "frontend": "library"}),)),
    ]
}

DEFAULT_PRESET = "accurate"


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
