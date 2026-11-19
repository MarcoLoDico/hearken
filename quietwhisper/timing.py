"""Timestamp refinement. Changes times only, never words.

Transducer (Parakeet) and attention (Whisper) timestamps put the start of a
word that follows a pause too early: the leading silence is absorbed into the
word. On the terms benchmark the median onset error was -186 ms (Parakeet) and
-540 ms (Whisper). If a word starts in a frame the VAD considers non-speech
but speech begins later inside the word, its start is moved to that speech
onset. Words with no speech frame at all are left untouched (they are flagged
``no_speech_detected`` by verify instead).
"""

from __future__ import annotations

import numpy as np

from .types import Word
from .vad import FRAME_S, SpeechProbs

SNAP_THRESHOLD = 0.3
MIN_WORD_S = 0.05


def snap_onsets(words: list[Word], speech: SpeechProbs, threshold: float = SNAP_THRESHOLD) -> int:
    """Move word starts out of leading silence, in place. Returns number of words moved."""
    p = speech.probs
    moved = 0
    for w in words:
        i0 = int(w.start / FRAME_S)
        i1 = max(i0 + 1, int(np.ceil(w.end / FRAME_S)))
        seg = p[i0:i1]
        if not seg.size or seg[0] >= threshold:
            continue
        on = np.flatnonzero(seg >= threshold)
        if not on.size:
            continue
        new = min((i0 + on[0]) * FRAME_S, w.end - MIN_WORD_S)
        if new > w.start:
            w.start = new
            moved += 1
    return moved
