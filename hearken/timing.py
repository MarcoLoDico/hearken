"""Timestamp refinement. Changes times only, never words.

Transducer (Parakeet) and attention (Whisper) timestamps put the start of a
word that follows a pause too early: the leading silence is absorbed into the
word. On the terms benchmark the median onset error was -186 ms (Parakeet) and
-540 ms (Whisper). Symmetrically, the last word before a pause often has its
end stretched up to ~1 s into the silence. If a word starts (ends) in frames
the VAD considers non-speech but contains speech, its start (end) is moved to
the speech edge. Words with no speech frame at all are left untouched (they
are flagged ``no_speech_detected`` by verify instead).
"""

from __future__ import annotations

import numpy as np

from .types import Word
from .vad import FRAME_S, SpeechProbs

SNAP_THRESHOLD = 0.3
MIN_WORD_S = 0.05


def snap_onsets(words: list[Word], speech: SpeechProbs, threshold: float = SNAP_THRESHOLD) -> int:
    """Move word starts out of leading silence and ends out of trailing silence, in place.

    Returns the number of words whose start or end moved.
    """
    p = speech.probs
    moved = 0
    for w in words:
        i0 = int(w.start / FRAME_S)
        i1 = max(i0 + 1, int(np.ceil(w.end / FRAME_S)))
        seg = p[i0:i1]
        on = np.flatnonzero(seg >= threshold) if seg.size else seg
        if not on.size:
            continue
        changed = False
        if seg[0] < threshold:
            new = min((i0 + on[0]) * FRAME_S, w.end - MIN_WORD_S)
            if new > w.start:
                w.start, changed = new, True
        if seg[-1] < threshold:
            new = max((i0 + on[-1] + 1) * FRAME_S, w.start + MIN_WORD_S)
            if new < w.end:
                w.end, changed = new, True
        moved += changed
    return moved
