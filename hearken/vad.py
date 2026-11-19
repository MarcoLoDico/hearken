"""Voice activity detection.

VAD is used ONLY for (a) choosing where to cut chunks and (b) providing
evidence for review flags (words in apparent silence, speech with no words).
It is never used to delete audio before recognition: every sample of the
recording is transcribed, so quiet speech the VAD misses is still heard by
the ASR models.

Primary detector: Silero VAD v5 (MIT) through onnxruntime (MIT). Fallback,
if the model file is missing: an adaptive energy detector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE

log = logging.getLogger(__name__)

FRAME = 512  # samples per Silero frame at 16 kHz (32 ms)
FRAME_S = FRAME / SAMPLE_RATE
_CONTEXT = 64


@dataclass
class SpeechProbs:
    probs: np.ndarray  # one probability per 32 ms frame
    method: str

    def at(self, t0: float, t1: float) -> np.ndarray:
        i0 = max(0, int(t0 / FRAME_S))
        i1 = max(i0 + 1, int(np.ceil(t1 / FRAME_S)))
        return self.probs[i0:i1]

    def regions(self, threshold: float = 0.5, min_dur: float = 0.25,
                merge_gap: float = 0.3) -> list[tuple[float, float]]:
        """Contiguous speech regions above ``threshold``."""
        on = self.probs >= threshold
        out: list[list[float]] = []
        idx = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(np.int8), [0]])))
        for s, e in zip(idx[::2], idx[1::2]):
            t0, t1 = s * FRAME_S, e * FRAME_S
            if out and t0 - out[-1][1] <= merge_gap:
                out[-1][1] = t1
            else:
                out.append([t0, t1])
        return [(a, b) for a, b in out if b - a >= min_dur]


def silero_probs(samples: np.ndarray, model_path: Path, streams: int = 32) -> SpeechProbs:
    """Run Silero v5 over the whole signal.

    The signal is split into ``streams`` contiguous parts processed as one
    batch (each with its own recurrent state), which makes multi-hour inputs
    tractable. Each part is warmed up on the 2 s preceding it so that the
    batch split does not introduce state-reset artefacts.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(model_path), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    n_frames = len(samples) // FRAME + (1 if len(samples) % FRAME else 0)
    streams = max(1, min(streams, n_frames // 200 or 1))
    per = int(np.ceil(n_frames / streams))
    warm = int(2.0 / FRAME_S)
    total = per + warm
    batch = np.zeros((streams, total * FRAME), dtype=np.float32)
    for b in range(streams):
        f0 = b * per - (warm if b else 0)
        s0 = max(0, f0 * FRAME)
        seg = np.asarray(samples[s0:(b + 1) * per * FRAME], dtype=np.float32)
        off = (total - (per + (warm if b else 0))) * FRAME
        batch[b, off:off + len(seg)] = seg
    state = np.zeros((2, streams, 128), dtype=np.float32)
    ctx = np.zeros((streams, _CONTEXT), dtype=np.float32)
    sr = np.array(SAMPLE_RATE, dtype=np.int64)
    out = np.zeros((streams, total), dtype=np.float32)
    for i in range(total):
        x = batch[:, i * FRAME:(i + 1) * FRAME]
        inp = np.concatenate([ctx, x], axis=1)
        p, state = sess.run(None, {"input": inp, "state": state, "sr": sr})
        out[:, i] = p[:, 0]
        ctx = inp[:, -_CONTEXT:]
    probs = out[:, warm:].reshape(-1)[:n_frames]
    return SpeechProbs(probs=probs.astype(np.float32), method="silero_v5")


def energy_probs(samples: np.ndarray) -> SpeechProbs:
    """Adaptive energy detector: a soft score relative to the noise floor."""
    n = len(samples) // FRAME
    x = np.asarray(samples[: n * FRAME], dtype=np.float64).reshape(n, FRAME)
    db = 10 * np.log10(np.mean(x * x, axis=1) + 1e-12)
    floor = np.percentile(db, 10)
    probs = 1 / (1 + np.exp(-(db - floor - 12) / 3))
    return SpeechProbs(probs=probs.astype(np.float32), method="energy")


def detect(samples: np.ndarray, model_path: Path | None) -> SpeechProbs:
    if model_path is not None and model_path.is_file():
        try:
            return silero_probs(samples, model_path)
        except Exception as e:  # onnxruntime failures should not abort a 3 h job
            log.warning("Silero VAD failed (%s); falling back to energy VAD", e)
    else:
        log.warning("Silero VAD model not found; using energy VAD (run `hearken setup`)")
    return energy_probs(samples)
