"""Audio import, analysis and (conservative) preprocessing.

Decoding uses PyAV (bundled FFmpeg libraries), so no system ffmpeg is needed.
Every container/codec FFmpeg understands works: MP4, MOV, MKV, MP3, M4A/AAC,
WAV, FLAC, OGG/Opus, WebM, ...

All engines consume 16 kHz mono float32, which is what every model evaluated
here was trained on. We decode straight from the source to that format in a
single high-quality resampling step (libswresample), and stream the samples to
a private temporary file that is memory-mapped, so a 3-hour lecture costs
~0.7 GB of address space rather than several in-memory copies.

Preprocessing policy (see docs/ARCHITECTURE.md for the reasoning):
  * DC offset removal and linear gain only. Both are invertible and do not
    alter the spectral shape the acoustic models see.
  * NO noise reduction, spectral gating, compression or EQ by default.
    Published results consistently show that speech-enhancement front-ends
    trained for human listening tend to *raise* WER of modern ASR models,
    which are trained on noisy data and are hurt by enhancement artifacts.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .errors import AudioError

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass
class AudioAnalysis:
    duration_s: float
    source_codec: str | None
    source_sample_rate: int | None
    source_channels: int | None
    peak_dbfs: float
    rms_dbfs: float
    noise_floor_dbfs: float  # 10th percentile of 100 ms frame levels
    speech_level_dbfs: float  # 95th percentile of 100 ms frame levels
    est_snr_db: float
    clipped_fraction: float
    dc_offset: float
    applied: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


def _db(x: float | np.ndarray) -> float | np.ndarray:
    return 20.0 * np.log10(np.maximum(x, 1e-10))


class DecodedAudio:
    """16 kHz mono float32 samples backed by a private memory-mapped temp file."""

    def __init__(self, samples: np.ndarray, tmp_path: Path | None, info: dict):
        self.samples = samples
        self._tmp_path = tmp_path
        self.info = info

    @property
    def duration(self) -> float:
        return len(self.samples) / SAMPLE_RATE

    def close(self) -> None:
        self.samples = np.zeros(0, dtype=np.float32)
        if self._tmp_path is not None:
            try:
                self._tmp_path.unlink()
            except FileNotFoundError:
                pass
            self._tmp_path = None

    def __enter__(self) -> "DecodedAudio":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def decode(path: str | Path, workdir: str | Path | None = None) -> DecodedAudio:
    """Decode any audio/video file to 16 kHz mono float32."""
    import av  # imported lazily: heavy, and not needed for e.g. `hearken review`

    path = Path(path)
    if not path.is_file():
        raise AudioError(f"input file not found: {path}")

    try:
        container = av.open(str(path))
    except (av.FFmpegError, OSError) as e:
        raise AudioError(f"cannot open {path.name}: {e}") from e

    with container:
        streams = [s for s in container.streams if s.type == "audio"]
        if not streams:
            raise AudioError(f"{path.name} contains no audio stream")
        # Prefer the default-disposition / most-channels stream.
        stream = max(streams, key=lambda s: (s.codec_context.channels or 0))
        info = {
            "codec": stream.codec_context.name,
            "sample_rate": stream.codec_context.sample_rate,
            "channels": stream.codec_context.channels,
        }
        # Resample per channel (planar float) and average the channels ourselves:
        # FFmpeg's default stereo->mono matrix is (L+R)*0.707, which adds 3 dB
        # to centred speech and can push loud recordings past full scale.
        resampler = av.AudioResampler(format="fltp", rate=SAMPLE_RATE)

        fd, tmp = tempfile.mkstemp(prefix="hearken-audio-", suffix=".f32", dir=workdir)
        os.chmod(tmp, 0o600)
        n = 0
        try:
            with os.fdopen(fd, "wb") as fh:
                def _write(frames) -> None:
                    nonlocal n
                    for fr in frames:
                        arr = fr.to_ndarray()  # (channels, samples)
                        arr = (arr.mean(axis=0) if arr.shape[0] > 1 else arr[0]).astype(np.float32, copy=False)
                        fh.write(arr.tobytes())
                        n += arr.size

                try:
                    for packet in container.demux(stream):
                        for frame in packet.decode():
                            _write(resampler.resample(frame))
                except av.FFmpegError as e:
                    # Truncated / partially corrupt files: keep what decoded and
                    # warn loudly rather than silently dropping the whole file.
                    log.warning("decoder error after %.1fs of audio: %s", n / SAMPLE_RATE, e)
                    info["decode_error"] = str(e)
                _write(resampler.resample(None))
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    if n == 0:
        Path(tmp).unlink(missing_ok=True)
        raise AudioError(f"{path.name}: audio stream decoded to zero samples")
    samples = np.memmap(tmp, dtype=np.float32, mode="r+", shape=(n,))
    log.info("decoded %.1fs of audio (%s, %s Hz, %s ch)", n / SAMPLE_RATE,
             info["codec"], info["sample_rate"], info["channels"])
    return DecodedAudio(samples, Path(tmp), info)


def from_array(samples: np.ndarray) -> DecodedAudio:
    """Wrap an in-memory 16 kHz mono array (tests, benchmarks)."""
    arr = np.ascontiguousarray(samples, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise AudioError("expected a non-empty 1-D array")
    return DecodedAudio(arr.copy(), None, {"codec": None, "sample_rate": SAMPLE_RATE, "channels": 1})


def analyze(audio: DecodedAudio) -> AudioAnalysis:
    x = audio.samples
    frame = SAMPLE_RATE // 10
    n_frames = max(1, len(x) // frame)
    # Streaming per-frame RMS, chunked to bound memory on long inputs.
    levels = np.empty(n_frames, dtype=np.float64)
    peak, sum_sq, total, clipped, dc_sum = 0.0, 0.0, 0, 0, 0.0
    step = frame * 6000  # 10 minutes per block
    for b in range(0, n_frames * frame, step):
        blk = np.asarray(x[b:min(b + step, n_frames * frame)], dtype=np.float64)
        f = blk.reshape(-1, frame)
        levels[b // frame:b // frame + len(f)] = np.sqrt(np.mean(f * f, axis=1))
        peak = max(peak, float(np.max(np.abs(blk))))
        sum_sq += float(np.sum(blk * blk))
        total += blk.size
        clipped += int(np.count_nonzero(np.abs(blk) >= 0.999))
        dc_sum += float(np.sum(blk))
    lv_db = _db(levels)
    nonzero = lv_db[lv_db > -99]
    floor = float(np.percentile(nonzero, 10)) if nonzero.size else -100.0
    speech = float(np.percentile(nonzero, 95)) if nonzero.size else -100.0
    a = AudioAnalysis(
        duration_s=audio.duration,
        source_codec=audio.info.get("codec"),
        source_sample_rate=audio.info.get("sample_rate"),
        source_channels=audio.info.get("channels"),
        peak_dbfs=float(_db(peak)),
        rms_dbfs=float(_db(np.sqrt(sum_sq / max(total, 1)))),
        noise_floor_dbfs=floor,
        speech_level_dbfs=speech,
        est_snr_db=speech - floor,
        clipped_fraction=clipped / max(total, 1),
        dc_offset=dc_sum / max(total, 1),
    )
    if a.clipped_fraction > 1e-4:
        a.warnings.append(f"clipping detected ({a.clipped_fraction:.2%} of samples); expect errors in loud passages")
    if a.est_snr_db < 15:
        a.warnings.append(f"low estimated SNR ({a.est_snr_db:.0f} dB); noisy recording")
    if a.speech_level_dbfs < -45:
        a.warnings.append("very quiet recording")
    if "decode_error" in audio.info:
        a.warnings.append(f"decoder error, transcript may be incomplete: {audio.info['decode_error']}")
    return a


def preprocess(audio: DecodedAudio, analysis: AudioAnalysis, *, target_peak_dbfs: float = -1.0,
               normalize_below_dbfs: float = -12.0) -> None:
    """Apply only lossless-in-spirit corrections, in place. Records what was done."""
    x = audio.samples
    if abs(analysis.dc_offset) > 1e-3:
        x -= np.float32(analysis.dc_offset)
        analysis.applied.append(f"dc_offset_removed({analysis.dc_offset:+.4f})")
    if -100 < analysis.peak_dbfs < normalize_below_dbfs:
        gain_db = target_peak_dbfs - analysis.peak_dbfs
        g = np.float32(10 ** (gain_db / 20))
        step = SAMPLE_RATE * 600
        for b in range(0, len(x), step):
            x[b:b + step] *= g
        analysis.applied.append(f"linear_gain(+{gain_db:.1f} dB)")
    if isinstance(x, np.memmap):
        x.flush()
