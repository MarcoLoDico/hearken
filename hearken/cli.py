"""Command-line interface: `hearken <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from . import __version__, models, presets
from .errors import QuietWhisperError

log = logging.getLogger("hearken")


class ProgressLine:
    """Single, self-overwriting progress line on stderr (plain lines when not a TTY)."""

    def __init__(self) -> None:
        self.t0 = time.time()
        self.stage = None
        self.stage_t0 = self.t0
        self.tty = sys.stderr.isatty()
        self.last_print = 0.0

    def __call__(self, stage: str, frac: float, detail: str) -> None:
        now = time.time()
        if stage != self.stage:
            if self.stage is not None and self.tty:
                sys.stderr.write("\n")
            self.stage, self.stage_t0 = stage, now
        if not self.tty and now - self.last_print < 15 and frac < 1:
            return
        self.last_print = now
        el = now - self.stage_t0
        eta = f" ETA {_hms(el / frac - el)}" if 0.02 < frac < 1 else ""
        msg = f"[{_hms(now - self.t0)}] {stage:<22} {frac * 100:5.1f}%  {detail}{eta}"
        sys.stderr.write(("\r" + msg.ljust(90)) if self.tty else msg + "\n")
        sys.stderr.flush()

    def done(self) -> None:
        if self.tty and self.stage is not None:
            sys.stderr.write("\n")


def _hms(s: float) -> str:
    s = int(max(s, 0))
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def cmd_setup(a: argparse.Namespace) -> int:
    keys = a.models or (list(models.MODELS) if a.all else presets.model_keys(presets.get(a.preset)))
    for k in keys:
        print(f"downloading {k} ({models.spec(k).repo}) ...", flush=True)
        print("  ->", models.download(k))
    print("downloading Silero VAD ...", flush=True)
    print("  ->", models.download_vad())
    print("Setup complete. Transcription now runs fully offline.")
    return 0


def cmd_models(a: argparse.Namespace) -> int:
    for k, s in models.MODELS.items():
        state = "cached" if models.is_cached(k) else "missing"
        print(f"{k:28} {state:8} {s.license:11} {s.repo}@{(s.revision or 'unpinned')[:10]}  [{s.languages}]")
    print(f"{'silero-vad-v5':28} {'cached' if models.vad_path().is_file() else 'missing':8} MIT")
    return 0


def cmd_presets(a: argparse.Namespace) -> int:
    for p in presets.PRESETS.values():
        mark = " (default)" if p.name == presets.DEFAULT_PRESET else ""
        print(f"{p.name:16}{mark}\n    {p.description}")
    return 0


def _run_options(a: argparse.Namespace):
    from . import vocab
    from .output import FormatOptions
    from .pipeline import RunOptions

    fmts = tuple(f.strip() for f in a.formats.split(",") if f.strip())
    fmt = FormatOptions(mode=a.mode, markers=not a.no_markers, subtitle_markers=a.subtitle_markers,
                        drop_fillers=a.drop_fillers)
    return RunOptions(preset=getattr(a, "preset", presets.DEFAULT_PRESET), language=getattr(a, "language", "en"),
                      vocabulary=vocab.load(getattr(a, "vocab", None)), formats=fmts, fmt=fmt,
                      preprocess=not getattr(a, "no_preprocess", False))


def cmd_transcribe(a: argparse.Namespace) -> int:
    from .pipeline import transcribe_file

    opts = _run_options(a)
    rc = 0
    for src in a.inputs:
        src = Path(src)
        out = Path(a.output) if a.output else src.parent / f"{src.stem}.transcript"
        prog = ProgressLine()
        try:
            res = transcribe_file(src, out, opts, progress=prog)
        except QuietWhisperError as e:
            prog.done()
            print(f"error: {src.name}: {e}", file=sys.stderr)
            rc = e.exit_code
            continue
        prog.done()
        print(f"{src.name}: {res.n_words} words, {res.n_uncertain} passages to review, "
              f"{res.duration_s / 60:.1f} min audio in {res.metrics['timings']['total_s'] / 60:.1f} min")
        for k, p in res.outputs.items():
            print(f"  {k:5} {p}")
        print(f"  raw   {res.raw_path}  (read-only)")
        if a.metrics_json:
            Path(a.metrics_json).write_text(json.dumps({**res.metrics, "duration_s": res.duration_s,
                                                        "n_words": res.n_words}, indent=1))
    return rc


def cmd_review(a: argparse.Namespace) -> int:
    from .output import clock

    doc = json.loads(Path(a.transcript).read_text(encoding="utf-8"))
    spans = doc.get("uncertain", [])
    if a.reason:
        spans = [s for s in spans if any(a.reason in r for r in s["reasons"])]
    words = doc.get("words", [])
    print(f"{len(spans)} uncertain passages in {doc['source']['name']}"
          + (f" (reason contains {a.reason!r})" if a.reason else ""))
    for i, s in enumerate(spans, 1):
        before = [w["text"] for w in words if s["start"] - a.context <= w["start"] < s["start"]]
        after = [w["text"] for w in words if s["end"] < w["start"] <= s["end"] + a.context]
        print(f"\n#{i}  {clock(s['start'])}–{clock(s['end'])}  [{', '.join(s['reasons'])}]")
        print(f"    transcript: …{' '.join(before[-8:])} >>{s['text'] or '(no words)'}<< {' '.join(after[:8])}…")
        for eng, alt in s.get("alternatives", {}).items():
            print(f"    {eng:>10}: {alt or '(nothing)'}")
    return 0


def cmd_reformat(a: argparse.Namespace) -> int:
    from .pipeline import load_raw, render_from_raw

    raw_path = Path(a.raw)
    raw = load_raw(raw_path)
    out = Path(a.output) if a.output else raw_path.parent
    stem = raw_path.name.split(".raw.")[0]
    opts = _run_options(a)
    outputs = render_from_raw(raw, raw_path, out, stem, opts, raw["source"]["name"])
    outputs.pop("_n_uncertain")
    for k, p in outputs.items():
        print(f"  {k:5} {p}")
    return 0


def _add_format_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--formats", default="txt,srt,vtt,json,html",
                   help="comma-separated: txt,srt,vtt,json,html (default: all)")
    p.add_argument("--mode", choices=["readable", "verbatim"], default="readable",
                   help="txt layout: readable paragraphs or one timestamped line per sentence")
    p.add_argument("--no-markers", action="store_true", help="do not wrap uncertain words in [? ... ?] in .txt")
    p.add_argument("--subtitle-markers", action="store_true", help="also mark uncertain words in .srt/.vtt")
    p.add_argument("--drop-fillers", action="store_true", help="omit stand-alone um/uh/erm (off by default)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hearken", description="Accuracy-first local transcription for long lectures.")
    ap.add_argument("--version", action="version", version=f"hearken {__version__}")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging (never logs transcript text)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("setup", help="download models once (the only command that uses the network)")
    p.add_argument("--preset", default=presets.DEFAULT_PRESET)
    p.add_argument("--models", nargs="*", help="explicit model keys (see `hearken models`)")
    p.add_argument("--all", action="store_true", help="download every registered model")
    p.set_defaults(fn=cmd_setup)

    sub.add_parser("models", help="list models and cache status").set_defaults(fn=cmd_models)
    sub.add_parser("presets", help="list presets").set_defaults(fn=cmd_presets)

    p = sub.add_parser("transcribe", help="transcribe audio/video files")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--output", help="output directory (default: <input>.transcript/ next to the input)")
    p.add_argument("-p", "--preset", default=presets.DEFAULT_PRESET, choices=list(presets.PRESETS))
    p.add_argument("--vocab", help="course terminology file, one term per line")
    p.add_argument("--language", default="en", help="spoken language code (default en)")
    p.add_argument("--no-preprocess", action="store_true", help="skip DC/gain correction")
    p.add_argument("--metrics-json", help="write timing/memory metrics to this file")
    _add_format_args(p)
    p.set_defaults(fn=cmd_transcribe)

    p = sub.add_parser("review", help="list uncertain passages of a transcript .json")
    p.add_argument("transcript")
    p.add_argument("--reason", help="filter by reason substring, e.g. disagreement, no_speech")
    p.add_argument("--context", type=float, default=4.0, help="seconds of context to show")
    p.set_defaults(fn=cmd_review)

    p = sub.add_parser("reformat", help="re-render outputs from a .raw.json without re-transcribing")
    p.add_argument("raw")
    p.add_argument("-o", "--output")
    _add_format_args(p)
    p.set_defaults(fn=cmd_reformat)

    from .bench import add_eval_parsers
    add_eval_parsers(sub)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("huggingface_hub", "urllib3", "numba", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        return a.fn(a)
    except QuietWhisperError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
