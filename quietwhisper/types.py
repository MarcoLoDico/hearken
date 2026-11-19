"""Core data structures shared by engines, post-processing and writers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Word:
    """One recognised word with absolute timestamps (seconds).

    ``text`` is exactly what the engine emitted for this word, including any
    attached punctuation. It is never rewritten after creation; formatting
    happens at output time.
    """

    text: str
    start: float
    end: float
    confidence: float | None = None
    flags: list[str] = field(default_factory=list)

    @property
    def mid(self) -> float:
        return 0.5 * (self.start + self.end)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["start"] = round(self.start, 3)
        d["end"] = round(self.end, 3)
        if self.confidence is not None:
            d["confidence"] = round(self.confidence, 4)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Word":
        return cls(
            text=d["text"],
            start=float(d["start"]),
            end=float(d["end"]),
            confidence=d.get("confidence"),
            flags=list(d.get("flags", [])),
        )


@dataclass
class Segment:
    """A run of consecutive words (sentence / subtitle cue / paragraph unit)."""

    words: list[Word]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def uncertain(self) -> bool:
        return any(w.flags for w in self.words)


@dataclass
class UncertainSpan:
    """A time range a human reviewer should check, with the evidence why."""

    start: float
    end: float
    reasons: list[str]
    text: str  # what the primary transcript says here ("" if nothing)
    alternatives: dict[str, str] = field(default_factory=dict)  # engine -> text

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "reasons": self.reasons,
            "text": self.text,
            "alternatives": self.alternatives,
        }


@dataclass
class EngineResult:
    """Raw output of one engine over the whole recording."""

    engine: str
    model: dict[str, Any]
    words: list[Word]
    chunks: list[dict[str, Any]]
    seconds: float  # wall-clock time spent in this engine
