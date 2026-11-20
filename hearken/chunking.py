"""Split a long recording into decodable chunks without losing or duplicating words.

Each chunk has a *core* interval and a *padded* interval:

    padded:   |--pad--|=========== core ===========|--pad--|
    words kept:        [ words whose midpoint is in core )

Cores tile the recording exactly (no gaps, no overlap), so every word is
assigned to exactly one chunk. Padding gives the model acoustic context on
both sides, so a word straddling a cut is decoded completely in at least one
chunk. Cuts are placed at the least speech-like point (lowest smoothed VAD
probability) inside an allowed window, i.e. normally in a pause.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vad import FRAME_S, SpeechProbs


@dataclass(frozen=True)
class Chunk:
    index: int
    core_start: float
    core_end: float
    pad_start: float
    pad_end: float

    def owns(self, t: float, is_last: bool) -> bool:
        return self.core_start <= t < self.core_end or (is_last and t >= self.core_end)


def plan_chunks(duration: float, vad: SpeechProbs, *, max_len: float, pad: float,
                min_len: float | None = None, fixed_window: bool = False) -> list[Chunk]:
    """Plan chunks whose padded length never exceeds ``max_len`` seconds.

    With ``fixed_window``, every padded window is exactly ``max_len`` seconds
    (or the whole file, if shorter): a short core gets extra real-audio context
    instead of a shorter window. Ownership still follows the cores, so the
    words kept are unaffected. This keeps tensor shapes constant: MLX caches
    buffers per shape, and varying shapes cost ~3x encoder time and >10 GB of
    cache on a 25-minute file.
    """
    core_max = max_len - 2 * pad
    if core_max <= 1.0:
        raise ValueError("max_len too small for the requested padding")
    min_len = min_len if min_len is not None else 0.5 * core_max

    # Smooth over ~0.3 s so a single low frame inside a word is not a "pause".
    k = max(1, int(round(0.3 / FRAME_S)))
    p = vad.probs.astype(np.float64)
    smooth = np.convolve(p, np.ones(k) / k, mode="same") if len(p) >= k else p

    cuts = [0.0]
    pos = 0.0
    while duration - pos > core_max:
        lo, hi = pos + min_len, pos + core_max
        i0, i1 = int(lo / FRAME_S), int(hi / FRAME_S)
        window = smooth[i0:i1] if i1 <= len(smooth) else None
        if window is None or window.size == 0:
            cut = hi
        else:
            # Prefer later cuts among near-equal minima: longer chunks give
            # the model more context. 0.02 tolerance on probability.
            best = window.min()
            cand = np.flatnonzero(window <= best + 0.02)
            cut = (i0 + cand[-1]) * FRAME_S
        cuts.append(cut)
        pos = cut
    cuts.append(duration)

    chunks = []
    for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        p0, p1 = max(0.0, a - pad), min(duration, b + pad)
        if fixed_window:
            if duration <= max_len:
                p0, p1 = 0.0, duration
            else:
                extra = max_len - (b - a)
                p0 = min(max(0.0, a - extra / 2), duration - max_len)
                p1 = p0 + max_len
        chunks.append(Chunk(i, a, b, p0, p1))
    return chunks
