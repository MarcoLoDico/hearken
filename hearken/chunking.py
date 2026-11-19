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
                min_len: float | None = None) -> list[Chunk]:
    """Plan chunks whose padded length never exceeds ``max_len`` seconds."""
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
        chunks.append(Chunk(i, a, b, max(0.0, a - pad), min(duration, b + pad)))
    return chunks
