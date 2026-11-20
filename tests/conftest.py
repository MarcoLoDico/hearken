import numpy as np
import pytest

from hearken.engines.base import ChunkContext, Engine
from hearken.types import Word

SR = 16000


def burst_signal(n_bursts: int, gap: float = 1.3, dur: float = 0.25, lead: float = 1.0,
                 long_pause_every: int = 0, long_pause: float = 0.0):
    """Tone bursts ('words') whose frequency encodes their index; returns (x, onsets)."""
    t, parts, onsets = lead, [np.zeros(int(lead * SR), np.float32)], []
    for i in range(n_bursts):
        f = 300 + 25 * i
        n = int(dur * SR)
        tt = np.arange(n) / SR
        parts.append((0.3 * np.sin(2 * np.pi * f * tt) * np.hanning(n)).astype(np.float32))
        onsets.append(t)
        g = gap + (long_pause if long_pause_every and (i + 1) % long_pause_every == 0 else 0.0)
        parts.append(np.zeros(int(g * SR), np.float32))
        t += dur + g
    return np.concatenate(parts), onsets


class BurstEngine(Engine):
    """Fake ASR: emits one word per tone burst, named by the burst's frequency.

    Bursts cut by the chunk edge are emitted with clipped times, mimicking a
    real model hearing half a word at a chunk boundary.
    """

    name = "burst"

    def __init__(self, max_chunk_s=20.0, pad_s=2.0, fixed_window=False, batch_size=1, **_):
        self.max_chunk_s, self.pad_s = max_chunk_s, pad_s
        self.fixed_window, self.batch_size = fixed_window, batch_size
        self.calls = 0

    def load(self):
        pass

    def unload(self):
        pass

    def describe(self):
        return {"engine": "burst", "model": "fake", "repo": None, "revision": None}

    def transcribe_chunk(self, audio, ctx: ChunkContext):
        self.calls += 1
        frame = 160
        n = len(audio) // frame
        e = (np.asarray(audio[: n * frame]).reshape(n, frame) ** 2).mean(1)
        on = e > 1e-5
        words, i = [], 0
        while i < n:
            if on[i]:
                j = i
                while j < n and on[j]:
                    j += 1
                seg = audio[i * frame:j * frame]
                spec = np.abs(np.fft.rfft(seg, n=SR))
                f = int(np.argmax(spec))
                idx = round((f - 300) / 25)
                words.append(Word(f"w{idx}", i * frame / SR, j * frame / SR, 0.9))
                i = j
            else:
                i += 1
        return words


@pytest.fixture
def burst():
    return burst_signal
