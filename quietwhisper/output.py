"""Segmentation, formatting and writers (.txt .srt .vtt .json .html).

Formatting modes
  verbatim  one line per sentence, each with its start time; words exactly
            as recognised.
  readable  paragraphs (split at long pauses), a timestamp per paragraph.
            Pure layout: the word sequence is identical to verbatim.
Optional ``drop_fillers`` removes only stand-alone hesitation tokens
(um, uh, erm, er, hmm, mm); it is off by default and recorded in metadata.
Uncertain words are wrapped in ``[? ... ?]`` in the text outputs unless
markers are disabled.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .types import Segment, UncertainSpan, Word

FILLERS = {"um", "uh", "erm", "er", "hmm", "mm", "uhm", "umm"}
_SENT_END = re.compile(r"[.?!…][\"'”’)\]]*$")
# Flags that do not indicate doubt about the words themselves.
_SILENT_FLAGS = {"interpolated_time"}


@dataclass
class FormatOptions:
    mode: str = "readable"  # "verbatim" | "readable"
    markers: bool = True
    subtitle_markers: bool = False
    drop_fillers: bool = False
    paragraph_pause: float = 2.0
    paragraph_max_s: float = 90.0


def is_filler(w: Word) -> bool:
    return w.text.casefold().strip(".,!?;:-…") in FILLERS


def doubtful(w: Word) -> bool:
    return any(f not in _SILENT_FLAGS for f in w.flags)


def _split(words: list[Word], brk) -> list[Segment]:
    segs, cur = [], []
    for i, w in enumerate(words):
        if cur and brk(cur, w):
            segs.append(Segment(cur))
            cur = []
        cur.append(w)
    if cur:
        segs.append(Segment(cur))
    return segs


def sentences(words: list[Word], pause: float = 1.5, max_s: float = 30.0) -> list[Segment]:
    return _split(words, lambda cur, w: bool(_SENT_END.search(cur[-1].text))
                  or w.start - cur[-1].end >= pause or w.end - cur[0].start > max_s)


def cues(words: list[Word], max_chars: int = 84, max_s: float = 6.0, pause: float = 0.8) -> list[Segment]:
    def brk(cur, w):
        n = sum(len(x.text) + 1 for x in cur) + len(w.text)
        return (bool(_SENT_END.search(cur[-1].text)) and cur[-1].end - cur[0].start > 1.0) \
            or w.start - cur[-1].end >= pause or n > max_chars or w.end - cur[0].start > max_s
    return _split(words, brk)


def paragraphs(words: list[Word], opts: FormatOptions) -> list[Segment]:
    return _split(words, lambda cur, w: (w.start - cur[-1].end >= opts.paragraph_pause
                                         and bool(_SENT_END.search(cur[-1].text)))
                  or (w.end - cur[0].start > opts.paragraph_max_s and bool(_SENT_END.search(cur[-1].text))))


def render_words(words: Iterable[Word], markers: bool) -> str:
    parts, open_ = [], False
    for w in words:
        d = markers and doubtful(w)
        if d and not open_:
            parts.append("[? " + w.text)
            open_ = True
        elif not d and open_:
            parts[-1] += " ?]"
            parts.append(w.text)
            open_ = False
        else:
            parts.append(w.text)
    if open_:
        parts[-1] += " ?]"
    return " ".join(parts)


def ts(t: float, sep: str = ",", hours: bool = True) -> str:
    ms = int(round(max(t, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}" if hours else f"{m:02d}:{s:02d}{sep}{ms:03d}"


def clock(t: float) -> str:
    return ts(t)[:8]


def select_words(words: list[Word], opts: FormatOptions) -> list[Word]:
    return [w for w in words if not (opts.drop_fillers and is_filler(w))]


def to_txt(words: list[Word], opts: FormatOptions) -> str:
    words = select_words(words, opts)
    if opts.mode == "verbatim":
        lines = [f"[{clock(s.start)}] {render_words(s.words, opts.markers)}" for s in sentences(words)]
        return "\n".join(lines) + "\n"
    out = []
    for p in paragraphs(words, opts):
        out.append(f"[{clock(p.start)}]\n{render_words(p.words, opts.markers)}\n")
    return "\n".join(out)


def _wrap2(text: str, width: int = 42) -> str:
    if len(text) <= width:
        return text
    mid = len(text) // 2
    left, right = text.rfind(" ", 0, mid + 1), text.find(" ", mid)
    cut = left if right < 0 or (left >= 0 and mid - left <= right - mid) else right
    return text if cut <= 0 else text[:cut] + "\n" + text[cut + 1:]


def _subtitle_cues(words: list[Word], opts: FormatOptions) -> list[tuple[float, float, str]]:
    out = []
    segs = cues(select_words(words, opts))
    for i, c in enumerate(segs):
        end = c.end
        nxt = segs[i + 1].start if i + 1 < len(segs) else None
        # Keep very short cues readable without overlapping the next one.
        end = max(end, min(c.start + 0.8, nxt if nxt is not None else c.start + 0.8))
        out.append((c.start, end, _wrap2(render_words(c.words, opts.subtitle_markers))))
    return out


def to_srt(words: list[Word], opts: FormatOptions) -> str:
    blocks = [f"{i}\n{ts(a)} --> {ts(b)}\n{t}\n" for i, (a, b, t) in enumerate(_subtitle_cues(words, opts), 1)]
    return "\n".join(blocks)


def to_vtt(words: list[Word], opts: FormatOptions) -> str:
    blocks = [f"{ts(a, '.')} --> {ts(b, '.')}\n{t}\n" for a, b, t in _subtitle_cues(words, opts)]
    return "WEBVTT\n\n" + "\n".join(blocks)


def to_json(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=False, indent=1)


def to_html(words: list[Word], spans: list[UncertainSpan], media_name: str, title: str,
            opts: FormatOptions) -> str:
    """Self-contained offline review page: click any sentence to play from there."""
    words = select_words(words, opts)
    esc = html.escape
    paras = []
    for p in paragraphs(words, opts):
        ws = []
        for w in p.words:
            cls = ' class="u"' if doubtful(w) else ""
            tip = f' title="{esc(", ".join(w.flags))}"' if w.flags else ""
            ws.append(f'<span data-t="{w.start:.2f}"{cls}{tip}>{esc(w.text)}</span>')
        paras.append(f'<p><a class="t" data-t="{p.start:.2f}">{clock(p.start)}</a> {" ".join(ws)}</p>')
    items = []
    for s in spans:
        alts = "".join(f"<div class='alt'><b>{esc(k)}:</b> {esc(v) or '<i>(nothing)</i>'}</div>"
                       for k, v in s.alternatives.items())
        items.append(f"<li data-t='{max(0, s.start - 1):.2f}'><a class='t'>{clock(s.start)}</a> "
                     f"<span class='r'>{esc(', '.join(s.reasons))}</span><div>{esc(s.text) or '<i>(no words)</i>'}</div>{alts}</li>")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{esc(title)}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#fff;--fg:#1a1a1a;--mut:#666;--u:#fff1b8;--line:#ddd}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--fg:#e8e8e8;--mut:#999;--u:#5a4a00;--line:#333}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 -apple-system,system-ui,sans-serif}}
header{{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 16px}}
audio,video{{width:100%;max-height:30vh}} main{{display:flex;gap:24px;padding:0 16px}}
#tx{{flex:3;max-width:80ch}} aside{{flex:2;max-height:calc(100vh - 120px);overflow:auto;position:sticky;top:110px}}
.t{{color:var(--mut);font-variant-numeric:tabular-nums;cursor:pointer;font-size:.85em}}
span[data-t]{{cursor:pointer}} .u{{background:var(--u);border-radius:3px}}
li{{margin-bottom:10px;cursor:pointer}} .r{{color:var(--mut);font-size:.8em}} .alt{{font-size:.9em;color:var(--mut)}}
@media (max-width:800px){{main{{flex-direction:column}} aside{{position:static;max-height:none}}}}
</style></head><body>
<header><b>{esc(title)}</b> — {len(spans)} passages to review. Click text to play. Keep the media file next to this page.
<video id="m" controls preload="metadata" src="{esc(media_name)}"></video></header>
<main><div id="tx">{''.join(paras)}</div><aside><h3>Uncertain passages</h3><ol>{''.join(items)}</ol></aside></main>
<script>
const m=document.getElementById('m');
document.addEventListener('click',e=>{{const el=e.target.closest('[data-t]');if(!el)return;m.currentTime=parseFloat(el.dataset.t);m.play();}});
</script></body></html>"""


def write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
