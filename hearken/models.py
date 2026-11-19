"""Model registry with pinned revisions, and offline-only resolution.

Revisions are pinned to exact Hugging Face commit hashes so a given version of
hearken always runs byte-identical weights. `hearken setup` is the only code
path that touches the network; transcription resolves models with
``local_files_only=True`` and sets HF_HUB_OFFLINE=1.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .errors import ModelNotAvailableError


@dataclass(frozen=True)
class ModelSpec:
    key: str
    engine: str
    repo: str
    revision: str | None  # None = resolve whatever is cached (unpinned extra)
    license: str
    languages: str
    notes: str = ""


MODELS: dict[str, ModelSpec] = {
    m.key: m
    for m in [
        ModelSpec("parakeet-tdt-0.6b-v2", "parakeet", "mlx-community/parakeet-tdt-0.6b-v2",
                  "8ae155301e23d820d82aa60d24817c900e69e487", "CC-BY-4.0", "en",
                  "FastConformer-TDT, English. Low hallucination; native word timings."),
        ModelSpec("parakeet-tdt-0.6b-v3", "parakeet", "mlx-community/parakeet-tdt-0.6b-v3",
                  None, "CC-BY-4.0", "25 European languages"),
        ModelSpec("qwen3-asr-1.7b", "qwen3", "Qwen/Qwen3-ASR-1.7B",
                  "7278e1e70fe206f11671096ffdd38061171dd6e5", "Apache-2.0", "52 languages/dialects",
                  "Audio-LLM; accepts free-text context for terminology."),
        ModelSpec("qwen3-asr-1.7b-8bit", "qwen3", "mlx-community/Qwen3-ASR-1.7B-8bit",
                  "a8379a2e2f9e313c9292cdf1af4055ab56d50d55", "Apache-2.0", "52 languages/dialects",
                  "8-bit MLX quantisation of Qwen3-ASR-1.7B (~half the memory)."),
        ModelSpec("qwen3-forced-aligner-0.6b", "aligner", "Qwen/Qwen3-ForcedAligner-0.6B",
                  "c7cbfc2048c462b0d63a45797104fc9db3ad62b7", "Apache-2.0", "11 languages", "Word timestamps for Qwen3-ASR."),
        ModelSpec("whisper-large-v3", "whisper", "mlx-community/whisper-large-v3-mlx",
                  "49e6aa286ad60c14352c404340ded53710378a11", "MIT", "99 languages"),
    ]
}

VAD_URL = "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx"
PINNED_VAD_SHA256 = "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"


def home() -> Path:
    p = Path(os.environ.get("HEARKEN_HOME", Path.home() / ".cache" / "hearken"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def vad_path() -> Path:
    return home() / "silero_vad_v5.onnx"


def spec(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        raise ModelNotAvailableError(f"unknown model {key!r}; known: {', '.join(MODELS)}") from None


def resolve_local(key: str) -> Path:
    """Return the local snapshot directory for a model, never touching the network."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    s = spec(key)
    try:
        return Path(snapshot_download(s.repo, revision=s.revision, local_files_only=True))
    except (LocalEntryNotFoundError, FileNotFoundError, OSError) as e:
        raise ModelNotAvailableError(
            f"model {key} ({s.repo}@{s.revision or 'latest'}) is not downloaded; "
            f"run `hearken setup --models {key}` once while online"
        ) from e


def download(key: str) -> Path:
    from huggingface_hub import snapshot_download

    s = spec(key)
    return Path(snapshot_download(s.repo, revision=s.revision))


def download_vad() -> Path:
    dst = vad_path()
    if not dst.is_file():
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(VAD_URL, tmp)
        tmp.rename(dst)
    digest = hashlib.sha256(dst.read_bytes()).hexdigest()
    if PINNED_VAD_SHA256 and digest != PINNED_VAD_SHA256:
        dst.unlink()
        raise ModelNotAvailableError(f"VAD model checksum mismatch ({digest}); refusing to use it")
    return dst


def is_cached(key: str) -> bool:
    try:
        resolve_local(key)
        return True
    except ModelNotAvailableError:
        return False


def go_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

