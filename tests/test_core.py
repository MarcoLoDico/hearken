import numpy as np
import pytest

from hearken.chunking import plan_chunks
from hearken.engines.base import words_from_tokens
from hearken.engines.qwen3 import attach_timings
from hearken.textnorm import diff_words, normalize
from hearken.types import EngineResult, Word
from hearken.vad import FRAME_S, SpeechProbs, energy_probs
from hearken.verify import find_repetitions, verify


def probs_from_regions(duration, regions):
    p = np.zeros(int(duration / FRAME_S) + 1, np.float32)
    for a, b in regions:
        p[int(a / FRAME_S):int(b / FRAME_S)] = 0.95
    return SpeechProbs(p, "test")


# ---------------------------------------------------------------- chunking
@pytest.mark.parametrize("duration", [5.0, 29.0, 61.3, 3 * 3600.0])
def test_chunk_cores_tile_exactly(duration):
    speech = probs_from_regions(duration, [(t, t + 2.5) for t in np.arange(0, duration, 3.0)])
    chunks = plan_chunks(duration, speech, max_len=30.0, pad=1.5)
    assert chunks[0].core_start == 0.0 and chunks[-1].core_end == duration
    for a, b in zip(chunks, chunks[1:]):
        assert a.core_end == b.core_start
    for c in chunks:
        assert c.pad_end - c.pad_start <= 30.0 + 1e-6
        assert c.core_end > c.core_start


def test_chunk_cuts_land_in_pauses():
    duration = 300.0
    regions = [(t, t + 2.5) for t in np.arange(0, duration, 3.0)]  # 0.5 s pauses
    chunks = plan_chunks(duration, probs_from_regions(duration, regions), max_len=30.0, pad=1.5)
    for c in chunks[1:]:
        assert any(b <= c.core_start <= b + 0.5 + FRAME_S for _, b in regions), c.core_start


def test_chunk_without_any_pause_still_progresses():
    speech = SpeechProbs(np.ones(int(100 / FRAME_S), np.float32), "test")
    chunks = plan_chunks(100.0, speech, max_len=30.0, pad=2.0)
    assert len(chunks) >= 4 and chunks[-1].core_end == 100.0


# ---------------------------------------------------------------- engines helpers
def test_words_from_tokens_bare_space_token_is_boundary():
    toks = [(" it", 0, .1, .9), ("'", .1, .15, .9), ("s", .15, .2, .9), (" ", .2, .2, .9),
            ("es", .2, .3, .8), ("pe", .3, .4, .5), ("cially", .4, .5, .9), (" ", .5, .5, .9),
            ("4", .5, .6, .9), ("0", .6, .7, .9)]
    words = words_from_tokens(toks)
    assert [w.text for w in words] == ["it's", "especially", "40"]
    assert words[1].confidence == 0.5  # min over pieces
    assert words[1].start == .2 and words[1].end == .5


def test_attach_timings_keeps_punctuation_and_interpolates():
    text = "Hello, world! It's 3.5% faster."
    aligned = [("Hello", 0.0, 0.4), ("world", 0.5, 0.9), ("It's", 1.0, 1.2), ("faster", 2.0, 2.4)]
    words = attach_timings(text, aligned)
    assert [w.text for w in words] == ["Hello,", "world!", "It's", "3.5%", "faster."]
    assert (words[0].start, words[1].end) == (0.0, 0.9)
    assert words[3].flags == ["interpolated_time"] and 1.2 <= words[3].start < words[3].end <= 2.0


# ---------------------------------------------------------------- normalisation / diff
def test_normalize_equivalences():
    assert normalize("Ten thousand") == normalize("10,000")
    assert normalize("colour") == normalize("color")


def test_diff_ignores_formatting_but_finds_real_errors():
    a = "we measured 10,000 samples in the colour test".split()
    b = "we measured ten thousand samples in the color test".split()
    assert diff_words(a, b) == []
    c = "we measured ten thousand examples in the color test".split()
    r = diff_words(a, c)
    # A real error next to a number-format difference is reported as one region
    # covering both; it must contain the error and nothing beyond the number.
    assert len(r) == 1 and "samples" in a[r[0].a0:r[0].a1] and "examples" in c[r[0].b0:r[0].b1]
    assert r[0].a1 - r[0].a0 <= 2


# ---------------------------------------------------------------- verify
def _words(texts, start=0.0, step=0.4):
    return [Word(t, start + i * step, start + i * step + 0.3) for i, t in enumerate(texts)]


def test_repetition_detection():
    ws = _words("so the the result is is is is clear thank you thank you thank you".split())
    reps = find_repetitions(ws)
    texts = [" ".join(w.text for w in ws[a:b]) for a, b in reps]
    assert "is is is is" in texts and "thank you thank you thank you" in texts
    assert all("the the" != t for t in texts)  # two repeats is normal speech


def test_verify_flags_without_rewriting():
    prim = _words("the quick brown fox jumps".split())
    sec = _words("the quick brown box jumps".split())
    late = Word("hallucinated", 20.0, 20.5)
    prim.append(late)
    speech = probs_from_regions(40.0, [(0, 2.0), (30.0, 33.0)])
    spans = verify(EngineResult("a", {}, prim, [], 0), [EngineResult("b", {}, sec, [], 0)], speech)
    reasons = {r for s in spans for r in s.reasons}
    assert {"model_disagreement", "no_speech_detected", "speech_without_words"} <= reasons
    dis = next(s for s in spans if "model_disagreement" in s.reasons)
    assert dis.text == "fox" and dis.alternatives["b"] == "box"
    assert [w.text for w in prim] == "the quick brown fox jumps hallucinated".split()
    assert "no_speech_detected" in late.flags


def test_energy_vad_detects_bursts(burst):
    x, onsets = burst(10)
    sp = energy_probs(x)
    regs = sp.regions(threshold=0.5, min_dur=0.05)
    assert len(regs) == 10
    assert all(abs(r[0] - o) < 0.1 for r, o in zip(regs, onsets))


def test_snap_onsets_moves_only_starts_in_silence():
    from hearken.timing import snap_onsets
    speech = probs_from_regions(10.0, [(1.0, 2.0), (3.0, 4.0)])
    early = Word("late", 0.6, 1.5)        # starts 0.4 s before speech
    ok = Word("fine", 3.1, 3.6)           # already inside speech
    ghost = Word("ghost", 6.0, 6.5)       # no speech at all: leave for verify to flag
    n = snap_onsets([early, ok, ghost], speech)
    assert n == 1
    assert abs(early.start - 1.0) <= FRAME_S and early.end == 1.5 and early.text == "late"
    tail = Word("tail", 3.5, 4.8)          # ends 0.8 s into the pause
    assert snap_onsets([tail], speech) == 1 and abs(tail.end - 4.0) <= FRAME_S and tail.start == 3.5
    assert (ok.start, ghost.start) == (3.1, 6.0)
