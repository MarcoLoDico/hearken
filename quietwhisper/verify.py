"""Accuracy checks on a finished transcript. Flags, never rewrites.

Evidence sources:
  * cross-model disagreement   (secondary engine heard something different)
  * low engine confidence      (Parakeet token confidence)
  * words in non-speech audio  (possible hallucination; VAD says no speech)
  * speech with no words       (possible omission; VAD says speech)
  * repetition loops           (classic decoder failure)
  * engine-specific warnings   (truncated decode, interpolated timestamps)

Each finding becomes an UncertainSpan that the reviewer can jump to. The
primary transcript's words are kept verbatim; flags are attached to them.
"""

from __future__ import annotations

from dataclasses import dataclass

from .textnorm import diff_words
from .types import EngineResult, UncertainSpan, Word
from .vad import SpeechProbs


@dataclass
class VerifyConfig:
    low_confidence: float = 0.35
    silence_prob: float = 0.15  # max VAD prob under a word below this => "no speech"
    missed_min_dur: float = 1.5
    missed_prob: float = 0.8
    merge_gap: float = 0.75
    language: str = "en"


def _text(words: list[Word]) -> str:
    return " ".join(w.text for w in words)


def _in_range(words: list[Word], t0: float, t1: float) -> list[Word]:
    return [w for w in words if t0 <= w.mid < t1]


def find_repetitions(words: list[Word]) -> list[tuple[int, int]]:
    """Index ranges where an n-gram (n<=4) repeats back-to-back suspiciously often."""
    keys = [w.text.casefold().strip(".,!?;:") for w in words]
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(keys):
        found = None
        for n in (1, 2, 3, 4):
            need = 4 if n == 1 else 3
            reps = 1
            while i + (reps + 1) * n <= len(keys) and keys[i + reps * n:i + (reps + 1) * n] == keys[i:i + n]:
                reps += 1
            if reps >= need:
                found = (i, i + reps * n)
                break
        if found:
            out.append(found)
            i = found[1]
        else:
            i += 1
    return out


def verify(primary: EngineResult, secondaries: list[EngineResult], vad: SpeechProbs,
           cfg: VerifyConfig = VerifyConfig()) -> list[UncertainSpan]:
    words = primary.words
    spans: list[UncertainSpan] = []

    def add(t0: float, t1: float, reason: str, alts: dict[str, str] | None = None,
            flag_words: list[Word] | None = None) -> None:
        for w in flag_words or []:
            if reason not in w.flags:
                w.flags.append(reason)
        spans.append(UncertainSpan(t0, t1, [reason], "", dict(alts or {})))

    for w in words:
        if w.confidence is not None and w.confidence < cfg.low_confidence:
            add(w.start, w.end, "low_confidence", flag_words=[w])
        p = vad.at(w.start, w.end)
        if p.size and float(p.max()) < cfg.silence_prob:
            add(w.start, w.end, "no_speech_detected", flag_words=[w])
        for f in ("decoder_truncated", "interpolated_time"):
            if f in w.flags:
                add(w.start, w.end, f)

    for a, b in find_repetitions(words):
        add(words[a].start, words[b - 1].end, "repetition", flag_words=words[a:b])

    for sec in secondaries:
        sw = sec.words
        for r in diff_words([w.text for w in words], [w.text for w in sw], cfg.language):
            pa, sb = words[r.a0:r.a1], sw[r.b0:r.b1]
            times = [x for x in (pa or sb)]
            if not times:
                continue
            t0 = min(x.start for x in times)
            t1 = max(x.end for x in times)
            if not pa:  # secondary has words the primary lacks: anchor between neighbours
                t0 = min(t0, words[r.a0 - 1].end if r.a0 > 0 else t0)
                t1 = max(t1, words[r.a0].start if r.a0 < len(words) else t1)
            add(t0, t1, "model_disagreement", {sec.engine: _text(sb)}, flag_words=pa)

    for t0, t1 in vad.regions(threshold=cfg.missed_prob, min_dur=cfg.missed_min_dur):
        # Shrink by 0.3 s to tolerate timestamp jitter at region edges.
        if not _in_range(words, t0 - 0.3, t1 + 0.3):
            alts = {s.engine: _text(_in_range(s.words, t0, t1)) for s in secondaries}
            add(t0, t1, "speech_without_words", {k: v for k, v in alts.items() if v})

    merged = _merge(sorted(spans, key=lambda s: (s.start, s.end)), cfg.merge_gap)
    for s in merged:
        s.text = _text(_in_range(words, s.start, s.end + 1e-6))
        # Show each secondary engine's reading of the whole merged span.
        for sec in secondaries:
            if sec.engine in s.alternatives or "speech_without_words" in s.reasons:
                s.alternatives[sec.engine] = _text(_in_range(sec.words, s.start, s.end + 1e-6))
    return merged


def _merge(spans: list[UncertainSpan], gap: float) -> list[UncertainSpan]:
    out: list[UncertainSpan] = []
    for s in spans:
        if out and s.start - out[-1].end <= gap:
            m = out[-1]
            m.end = max(m.end, s.end)
            for r in s.reasons:
                if r not in m.reasons:
                    m.reasons.append(r)
            m.alternatives.update(s.alternatives)
        else:
            out.append(UncertainSpan(s.start, s.end, list(s.reasons), "", dict(s.alternatives)))
    return out
