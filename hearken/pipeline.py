"""End-to-end orchestration: import -> analyse -> VAD -> transcribe -> raw -> verify -> outputs."""

from __future__ import annotations

import base64
import copy
import datetime as dt
import hashlib
import logging
import os
import platform
import resource
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__, audio as audio_mod, engines, models, presets, vad as vad_mod
from .chunking import plan_chunks
from .engines.base import ChunkContext, Engine
from .errors import ConfigError, EngineError
from .timing import SNAP_THRESHOLD, snap_onsets
from .output import FormatOptions, to_html, to_json, to_srt, to_txt, to_vtt, write_text, sentences
from .types import EngineResult, UncertainSpan, Word
from .verify import VerifyConfig, verify

log = logging.getLogger(__name__)

FORMATS = ("txt", "srt", "vtt", "json", "html")
RAW_SCHEMA = "hearken.raw/1"
DOC_SCHEMA = "hearken.transcript/1"

Progress = Callable[[str, float, str], None]  # (stage, fraction 0..1, detail)


@dataclass
class RunOptions:
    preset: str = presets.DEFAULT_PRESET
    language: str = "en"
    vocabulary: list[str] = field(default_factory=list)
    formats: tuple[str, ...] = ("txt", "srt", "vtt", "json")
    fmt: FormatOptions = field(default_factory=FormatOptions)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    preprocess: bool = True
    engine_overrides: dict[str, dict] = field(default_factory=dict)


@dataclass
class RunResult:
    outputs: dict[str, Path]
    raw_path: Path
    n_words: int
    n_uncertain: int
    duration_s: float
    metrics: dict


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _peak_metrics() -> dict:
    try:
        import mlx.core as mx
        gpu = mx.get_peak_memory() / 1e9
    except Exception:
        gpu = None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = rss / 1e9 if platform.system() == "Darwin" else rss / 1e6  # bytes on macOS, KiB on Linux
    return {"peak_rss_gb": round(rss_gb, 3), "peak_mlx_gb": round(gpu, 3) if gpu is not None else None}


def run_engine(engine: Engine, samples: np.ndarray, duration: float, speech: vad_mod.SpeechProbs,
               vocabulary: list[str], language: str, progress: Progress | None = None,
               label: str = "") -> EngineResult:
    sr = audio_mod.SAMPLE_RATE
    chunks = plan_chunks(duration, speech, max_len=engine.max_chunk_s, pad=engine.pad_s)
    words: list[Word] = []
    meta = []
    prev = ""
    t_engine = time.perf_counter()
    for c in chunks:
        last = c.index == len(chunks) - 1
        seg = np.asarray(samples[int(c.pad_start * sr):int(c.pad_end * sr)], dtype=np.float32)
        t0 = time.perf_counter()
        error = None
        raw: list[Word] = []
        for attempt in (1, 2):
            try:
                raw = engine.transcribe_chunk(seg, ChunkContext(vocabulary, prev, language))
                error = None
                break
            except EngineError as e:
                error = str(e)
                log.warning("%s chunk %d attempt %d failed: %s", engine.name, c.index, attempt, e)
        kept = []
        for w in raw:
            w.start = min(max(w.start + c.pad_start, c.pad_start), c.pad_end)
            w.end = min(max(w.end + c.pad_start, w.start), c.pad_end)
            if c.owns(w.mid, last):
                kept.append(w)
        # Enforce monotonic timestamps across chunk joins.
        if words and kept and kept[0].start < words[-1].end:
            for w in kept:
                if w.start < words[-1].end:
                    w.start = words[-1].end
                    w.end = max(w.end, w.start)
        words += kept
        prev = " ".join(w.text for w in words[-40:])
        meta.append({"index": c.index, "core": [round(c.core_start, 3), round(c.core_end, 3)],
                     "padded": [round(c.pad_start, 3), round(c.pad_end, 3)], "words_decoded": len(raw),
                     "words_kept": len(kept), "seconds": round(time.perf_counter() - t0, 3),
                     **({"error": error} if error else {})})
        if progress:
            progress(label or engine.name, c.core_end / duration, f"chunk {c.index + 1}/{len(chunks)}")
    return EngineResult(engine.name, engine.describe(), words, meta, time.perf_counter() - t_engine)


def _encode_probs(p: np.ndarray) -> str:
    return base64.b64encode(np.clip(np.round(p * 255), 0, 255).astype(np.uint8).tobytes()).decode()


def _decode_probs(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.uint8).astype(np.float32) / 255.0


def _unique(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return path.with_name(path.name.replace(".raw.json", f".raw.{stamp}.json"))


def transcribe_file(src: str | Path, out_dir: str | Path, opts: RunOptions,
                    progress: Progress | None = None, stem: str | None = None) -> RunResult:
    src = Path(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in opts.formats:
        if f not in FORMATS:
            raise ConfigError(f"unknown output format {f!r}; choose from {', '.join(FORMATS)}")
    preset = presets.get(opts.preset)
    models.go_offline()
    for key in presets.model_keys(preset):
        models.resolve_local(key)  # fail fast, before minutes of decoding

    timings: dict[str, float] = {}
    t_all = time.perf_counter()
    if progress:
        progress("decode", 0.0, src.name)
    t = time.perf_counter()
    with audio_mod.decode(src, workdir=out_dir) as decoded:
        timings["decode_s"] = time.perf_counter() - t
        analysis = audio_mod.analyze(decoded)
        if opts.preprocess:
            audio_mod.preprocess(decoded, analysis)
        for w in analysis.warnings:
            log.warning("audio: %s", w)

        t = time.perf_counter()
        if progress:
            progress("vad", 0.0, "")
        speech = vad_mod.detect(decoded.samples, models.vad_path())
        timings["vad_s"] = time.perf_counter() - t

        results: list[EngineResult] = []
        for i, es in enumerate(preset.engines):
            params = {**es.params, **opts.engine_overrides.get(es.engine, {})}
            eng = engines.create(es.engine, **params)
            role = "primary" if i == 0 else "verifier"
            log.info("loading %s (%s)", es.engine, role)
            eng.load()
            try:
                results.append(run_engine(eng, decoded.samples, decoded.duration, speech, opts.vocabulary,
                                          opts.language, progress, f"{es.engine} ({role})"))
            finally:
                eng.unload()
            timings[f"{es.engine}_s"] = results[-1].seconds
            failed = [c for c in results[-1].chunks if "error" in c]
            if failed and len(failed) == len(results[-1].chunks):
                raise EngineError(f"{es.engine} failed on every chunk: {failed[0]['error']}")
            if failed:
                log.warning("%s failed on %d/%d chunks; those spans are listed as uncertain",
                            es.engine, len(failed), len(results[-1].chunks))
        duration = decoded.duration

    stem = stem or src.stem
    raw = {
        "schema": RAW_SCHEMA,
        "tool": {"name": "hearken", "version": __version__},
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": {"name": src.name, "sha256": _sha256(src), "bytes": src.stat().st_size},
        "preset": preset.name,
        "language": opts.language,
        "vocabulary": opts.vocabulary,
        "analysis": analysis.to_dict(),
        "vad": {"method": speech.method, "frame_s": vad_mod.FRAME_S, "probs_u8_b64": _encode_probs(speech.probs)},
        "engines": [{"engine": r.engine, "model": r.model, "seconds": round(r.seconds, 2),
                     "chunks": r.chunks, "words": [w.to_dict() for w in r.words]} for r in results],
    }
    raw_path = _unique(out_dir / f"{stem}.raw.json")
    write_text(raw_path, to_json(raw))
    os.chmod(raw_path, 0o444)  # immutable record of what the models produced

    timings["total_s"] = time.perf_counter() - t_all
    metrics = {"timings": {k: round(v, 2) for k, v in timings.items()},
               "realtime_factor": round(timings["total_s"] / max(duration, 1e-9), 4), **_peak_metrics()}
    outputs = render_from_raw(raw, raw_path, out_dir, stem, opts, src.name, metrics)
    n_words = len(results[0].words)
    log.info("done: %d words, %.1fs audio, %.1fs wall", n_words, duration, timings["total_s"])
    return RunResult(outputs, raw_path, n_words, outputs.pop("_n_uncertain"), duration, metrics)  # type: ignore[arg-type]


def load_raw(path: str | Path) -> dict:
    import json
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != RAW_SCHEMA:
        raise ConfigError(f"{path} is not a hearken raw transcript")
    return doc


def render_from_raw(raw: dict, raw_path: Path, out_dir: Path, stem: str, opts: RunOptions,
                    media_name: str, metrics: dict | None = None) -> dict:
    """Derive every output from the raw record. Deterministic; never edits ``raw``."""
    raw = copy.deepcopy(raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = [EngineResult(e["engine"], e["model"], [Word.from_dict(w) for w in e["words"]], e["chunks"], e["seconds"])
           for e in raw["engines"]]
    speech = vad_mod.SpeechProbs(_decode_probs(raw["vad"]["probs_u8_b64"]), raw["vad"]["method"])
    vcfg = copy.copy(opts.verify)
    vcfg.language = raw["language"]
    for e in res:
        snap_onsets(e.words, speech)
    spans: list[UncertainSpan] = verify(res[0], res[1:], speech, vcfg)
    for e in res:
        for c in e.chunks:
            if "error" in c:
                spans.append(UncertainSpan(c["core"][0], c["core"][1], [f"{e.engine}_failed"], "", {}))
    spans.sort(key=lambda s: s.start)
    words = res[0].words

    doc = {
        "schema": DOC_SCHEMA,
        "tool": raw["tool"],
        "created": raw["created"],
        "source": raw["source"],
        "raw_transcript": {"file": raw_path.name, "sha256": _sha256(raw_path)},
        "preset": raw["preset"],
        "language": raw["language"],
        "engines": [{"role": "primary" if i == 0 else "verifier", **e.model, "seconds": round(e.seconds, 2)}
                    for i, e in enumerate(res)],
        "audio_analysis": raw["analysis"],
        "vad_method": raw["vad"]["method"],
        "timing_refinement": {"vad_onset_snap_threshold": SNAP_THRESHOLD},
        "format_options": asdict(opts.fmt),
        "verify_options": asdict(vcfg),
        "metrics": metrics or {},
        "text": " ".join(w.text for w in words),
        "segments": [{"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text,
                      "uncertain": s.uncertain} for s in sentences(words)],
        "words": [w.to_dict() for w in words],
        "uncertain": [s.to_dict() for s in spans],
    }
    outputs: dict = {}
    writers = {
        "txt": lambda: to_txt(words, opts.fmt),
        "srt": lambda: to_srt(words, opts.fmt),
        "vtt": lambda: to_vtt(words, opts.fmt),
        "json": lambda: to_json(doc),
        "html": lambda: to_html(words, spans, media_name, stem, opts.fmt),
    }
    for f in opts.formats:
        p = out_dir / (f"{stem}.review.html" if f == "html" else f"{stem}.{f}")
        write_text(p, writers[f]())
        outputs[f] = p
    outputs["_n_uncertain"] = len(spans)
    return outputs
