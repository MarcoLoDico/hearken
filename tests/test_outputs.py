import re

from quietwhisper.bench import hallucination, onset_errors, score, term_recall
from quietwhisper.output import FormatOptions, to_srt, to_txt, to_vtt, ts
from quietwhisper.types import Word


def _ws():
    texts = "Hello everyone. Um, today we study the Lagrangian. It is, uh, important.".split()
    out = []
    for i, t in enumerate(texts):
        out.append(Word(t, i * 0.5 + (3.0 if i >= 6 else 0.0), i * 0.5 + 0.4 + (3.0 if i >= 6 else 0.0)))
    out[6].flags.append("low_confidence")
    return out


def test_timestamp_format():
    assert ts(3725.0456) == "01:02:05,046"
    assert ts(0.5, ".") == "00:00:00.500"


def _words_of(text):
    return re.sub(r"\[\d\d:\d\d:\d\d\]|\[\?|\?\]", " ", text).split()


def test_readable_and_verbatim_have_identical_words():
    ws = _ws()
    a = _words_of(to_txt(ws, FormatOptions(mode="readable")))
    b = _words_of(to_txt(ws, FormatOptions(mode="verbatim")))
    assert a == b == [w.text for w in ws]


def test_markers_and_fillers():
    ws = _ws()
    txt = to_txt(ws, FormatOptions())
    assert "[? the ?]" in txt
    assert "[?" not in to_txt(ws, FormatOptions(markers=False))
    dropped = _words_of(to_txt(ws, FormatOptions(drop_fillers=True, markers=False)))
    assert "Um," not in dropped and "uh," not in dropped
    assert len(dropped) == len(ws) - 2


def test_srt_and_vtt_structure():
    ws = _ws()
    srt = to_srt(ws, FormatOptions())
    blocks = [b for b in srt.strip().split("\n\n")]
    assert blocks[0].splitlines()[0] == "1"
    pat = re.compile(r"^\d\d:\d\d:\d\d,\d{3} --> \d\d:\d\d:\d\d,\d{3}$")
    prev_end = -1
    for b in blocks:
        line = b.splitlines()[1]
        assert pat.match(line)
        a, e = line.split(" --> ")
        assert a < e and a >= (prev_end if isinstance(prev_end, str) else "")
        prev_end = e
    vtt = to_vtt(ws, FormatOptions())
    assert vtt.startswith("WEBVTT\n\n") and "-->" in vtt and "," not in vtt.split("\n")[2]
    joined = " ".join(l for b in blocks for l in b.splitlines()[2:])
    assert joined.split() == [w.text for w in ws]


def test_score_counts_missing_and_duplicates():
    # (not number words: the Whisper normaliser fuses "one two three" into digits)
    ref = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo mike lima"
    hyp = "alpha bravo charlie delta delta echo foxtrot golf lima"
    s = score(ref, hyp)
    assert s.missing_words == 5  # hotel..mike deleted as one run of >= 5
    assert s.duplicated_words == 1
    assert s.ins == 1 and s.dele == 5


def test_term_recall_and_hallucination_and_onsets():
    r = term_recall("the Lagrangian and the Hamiltonian and the Lagrangian", "the lagrangian and the hamilton",
                    ["Lagrangian", "Hamiltonian", "Jacobian"])
    assert r["occurrences"] == 3 and r["hits"] == 1
    h = hallucination([{"text": "x", "start": 10.0, "end": 10.2, "flags": ["no_speech_detected"]},
                       {"text": "y", "start": 30.0, "end": 30.2, "flags": []}], [[5.0, 15.0]])
    assert h["words_in_nonspeech"] == 1 and h["flagged"] == 1
    o = onset_errors([{"text": "Today", "start": 1.05, "end": 1.4}], [[1.0, "Today we"]])
    assert o["n"] == 1 and o["median_ms"] == 50
