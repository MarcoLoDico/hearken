"""Repeatable evaluation against manually verified reference transcripts.

Manifest: JSON Lines, one recording per line::

    {"id": "talk1", "audio": "talk1.wav", "reference": "talk1.txt",
     "vocab": "terms.txt",                      # optional
     "nonspeech": [[t0, t1], ...],              # optional: known no-speech regions
     "onsets": [[t, "first words"], ...]}       # optional: known utterance onsets

Paths are relative to the manifest. Each (item, preset) is transcribed in a
fresh subprocess so that timing and peak-memory figures are not polluted by
earlier runs.

Metrics (all on Whisper-normalised text):
  WER / CER, substitution / deletion / insertion counts
  missing speech     reference words inside deletion runs >= 5 words
  duplicated speech  inserted words that repeat the immediately preceding words
  hallucination      words whose midpoint lies in a known no-speech region
  timestamps         |start of first word - true onset| per known onset
  term recall        occurrences of vocabulary terms recognised (bag-of-ngrams)
  RTF, peak RSS, peak MLX (unified GPU) memory
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .textnorm import normalize


@dataclass
class Score:
    ref_words: int
    hyp_words: int
    wer: float
    cer: float
    sub: int
    dele: int
    ins: int
    missing_words: int
    duplicated_words: int

    def to_dict(self) -> dict:
        d = dict(vars(self))
        d["wer"], d["cer"] = round(self.wer, 5), round(self.cer, 5)
        return d


def score(reference: str, hypothesis: str, language: str = "en", missing_run: int = 5) -> Score:
    import jiwer

    ref = normalize(reference, language)
    hyp = normalize(hypothesis, language)
    if not ref.strip():
        raise ValueError("reference is empty after normalisation")
    o = jiwer.process_words(ref, hyp)
    c = jiwer.cer(ref, hyp)
    hw = o.hypotheses[0]
    missing = dup = 0
    for ch in o.alignments[0]:
        n_ref = ch.ref_end_idx - ch.ref_start_idx
        if ch.type == "delete" and n_ref >= missing_run:
            missing += n_ref
        if ch.type == "insert":
            ins = hw[ch.hyp_start_idx:ch.hyp_end_idx]
            k = len(ins)
            if ch.hyp_start_idx >= k and hw[ch.hyp_start_idx - k:ch.hyp_start_idx] == ins:
                dup += k
    return Score(len(ref.split()), len(hyp.split()), o.wer, c, o.substitutions, o.deletions, o.insertions,
                 missing, dup)


def term_recall(reference: str, hypothesis: str, terms: list[str], language: str = "en") -> dict:
    """Per-term min(hyp_count, ref_count) / ref_count over normalised n-grams."""
    ref = f" {normalize(reference, language)} "
    hyp = f" {normalize(hypothesis, language)} "
    per, hit, tot = {}, 0, 0
    for t in terms:
        k = normalize(t, language)
        if not k:
            continue
        r = ref.count(f" {k} ")
        if r == 0:
            continue
        h = min(hyp.count(f" {k} "), r)
        per[t] = [h, r]
        hit += h
        tot += r
    return {"recall": round(hit / tot, 4) if tot else None, "hits": hit, "occurrences": tot, "per_term": per}


def hallucination(words: list[dict], regions: list[list[float]]) -> dict:
    inside = [w for w in words if any(a <= 0.5 * (w["start"] + w["end"]) < b for a, b in regions)]
    flagged = [w for w in inside if w.get("flags")]
    minutes = sum(b - a for a, b in regions) / 60
    return {"nonspeech_minutes": round(minutes, 2), "words_in_nonspeech": len(inside),
            "per_minute": round(len(inside) / minutes, 3) if minutes else None,
            "flagged": len(flagged)}


def onset_errors(words: list[dict], onsets: list[list], tol: float = 2.0) -> dict:
    errs, missed = [], 0
    for t, first in onsets:
        key = normalize(str(first).split()[0])
        cands = [w for w in words if abs(w["start"] - t) <= tol and normalize(w["text"]) == key]
        if not cands:
            missed += 1
            continue
        errs.append(min(abs(w["start"] - t) for w in cands))
    if not errs:
        return {"n": 0, "unmatched": missed}
    errs.sort()
    p90 = errs[min(len(errs) - 1, int(0.9 * len(errs)))]
    return {"n": len(errs), "unmatched": missed, "median_ms": round(1000 * statistics.median(errs)),
            "p90_ms": round(1000 * p90), "max_ms": round(1000 * errs[-1]),
            "within_100ms": round(sum(e <= 0.1 for e in errs) / len(errs), 3),
            "within_250ms": round(sum(e <= 0.25 for e in errs) / len(errs), 3)}


def _load_manifest(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            it = json.loads(line)
            for k in ("audio", "reference", "vocab"):
                if it.get(k):
                    it[k] = str((path.parent / it[k]).resolve())
            items.append(it)
    return items


def evaluate_output(item: dict, transcript_json: Path, metrics: dict, with_vocab: bool) -> dict:
    doc = json.loads(transcript_json.read_text(encoding="utf-8"))
    ref = Path(item["reference"]).read_text(encoding="utf-8")
    s = score(ref, doc["text"], doc.get("language", "en"))
    row = {"id": item["id"], "preset": doc["preset"], "vocab": with_vocab, **s.to_dict(),
           "audio_s": metrics.get("duration_s"), "rtf": metrics.get("realtime_factor"),
           "wall_s": metrics.get("timings", {}).get("total_s"),
           "peak_rss_gb": metrics.get("peak_rss_gb"), "peak_mlx_gb": metrics.get("peak_mlx_gb"),
           "uncertain_spans": len(doc["uncertain"]),
           "flagged_words": sum(1 for w in doc["words"] if w.get("flags"))}
    if item.get("vocab"):
        from .vocab import load
        row["terms"] = term_recall(ref, doc["text"], load(item["vocab"]))
    if item.get("nonspeech"):
        row["hallucination"] = hallucination(doc["words"], item["nonspeech"])
    if item.get("onsets"):
        row["timestamps"] = onset_errors(doc["words"], item["onsets"])
    return row


def cmd_run(a: argparse.Namespace) -> int:
    manifest = Path(a.manifest)
    items = _load_manifest(manifest)
    if a.only:
        items = [i for i in items if i["id"] in a.only]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    done = set()
    if results_path.exists() and not a.force:
        for line in results_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["id"], r["preset"], r["vocab"]))
    vocab_modes = {"off": [False], "on": [True], "both": [False, True]}[a.vocab]
    for preset in a.preset:
        for it in items:
            for use_vocab in vocab_modes:
                if use_vocab and not it.get("vocab"):
                    continue
                key = (it["id"], preset, use_vocab)
                if key in done:
                    continue
                run_dir = out / preset / (it["id"] + ("+vocab" if use_vocab else ""))
                mfile = run_dir / "metrics.json"
                run_dir.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, "-m", "quietwhisper.cli", "transcribe", it["audio"], "-o", str(run_dir),
                       "-p", preset, "--formats", "json,txt", "--metrics-json", str(mfile)]
                if use_vocab:
                    cmd += ["--vocab", it["vocab"]]
                print(f"== {preset} / {it['id']}{' +vocab' if use_vocab else ''}", flush=True)
                proc = subprocess.run(cmd)
                if proc.returncode != 0:
                    print(f"   FAILED (exit {proc.returncode})", flush=True)
                    continue
                stem = Path(it["audio"]).stem
                row = evaluate_output(it, run_dir / f"{stem}.json", json.loads(mfile.read_text()), use_vocab)
                with results_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"   WER {row['wer']:.2%}  CER {row['cer']:.2%}  RTF {row['rtf']}", flush=True)
    write_report(results_path, out / "report.md")
    print(f"report: {out / 'report.md'}")
    return 0


def write_report(results_path: Path, md_path: Path) -> None:
    rows = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    lines = ["# Benchmark report", "",
             "Pooled WER = total errors / total reference words across items (not a mean of per-file WERs).", "",
             "## Per preset (pooled)", "",
             "| preset | vocab | items | ref words | WER | CER | sub | del | ins | missing | duplicated | mean RTF | max peak RSS GB | max peak MLX GB |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["preset"], r["vocab"]), []).append(r)
    for (p, v), rs in sorted(groups.items()):
        n = sum(r["ref_words"] for r in rs)
        err = sum(r["sub"] + r["dele"] + r["ins"] for r in rs)
        cer_w = sum(r["cer"] * r["ref_words"] for r in rs) / n
        lines.append(f"| {p} | {'on' if v else 'off'} | {len(rs)} | {n} | {err / n:.2%} | {cer_w:.2%} | "
                     f"{sum(r['sub'] for r in rs)} | {sum(r['dele'] for r in rs)} | {sum(r['ins'] for r in rs)} | "
                     f"{sum(r['missing_words'] for r in rs)} | {sum(r['duplicated_words'] for r in rs)} | "
                     f"{statistics.mean(r['rtf'] or 0 for r in rs):.3f} | {max(r['peak_rss_gb'] or 0 for r in rs):.1f} | "
                     f"{max(r['peak_mlx_gb'] or 0 for r in rs):.1f} |")
    lines += ["", "## Per item", "", "| item | preset | vocab | WER | CER | missing | dup | uncertain spans | extra |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["id"], r["preset"], r["vocab"])):
        extra = []
        if "terms" in r and r["terms"]["recall"] is not None:
            extra.append(f"term recall {r['terms']['hits']}/{r['terms']['occurrences']}")
        if "hallucination" in r:
            h = r["hallucination"]
            extra.append(f"{h['words_in_nonspeech']} words in {h['nonspeech_minutes']} min non-speech ({h['flagged']} flagged)")
        if "timestamps" in r and r["timestamps"].get("n"):
            t = r["timestamps"]
            extra.append(f"onset err median {t['median_ms']} ms / p90 {t['p90_ms']} ms (n={t['n']})")
        lines.append(f"| {r['id']} | {r['preset']} | {'on' if r['vocab'] else 'off'} | {r['wer']:.2%} | {r['cer']:.2%} | "
                     f"{r['missing_words']} | {r['duplicated_words']} | {r['uncertain_spans']} | {'; '.join(extra)} |")
    md_path.write_text("\n".join(lines) + "\n")


def cmd_rescore(a: argparse.Namespace) -> int:
    """Re-derive every run's outputs from its raw record with the current code and re-score.

    Lets post-processing changes (verification, timing refinement) be evaluated
    without re-running the models. Timing/memory metrics are kept from the run.
    """
    from .pipeline import RunOptions, load_raw, render_from_raw

    out = Path(a.out)
    items = {i["id"]: i for i in _load_manifest(Path(a.manifest))}
    old = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines() if l.strip()]
    rows = []
    for r in old:
        it = items.get(r["id"])
        run_dir = out / r["preset"] / (r["id"] + ("+vocab" if r["vocab"] else ""))
        raws = sorted(run_dir.glob("*.raw*.json"))
        if it is None or not raws:
            rows.append(r)
            continue
        stem = Path(it["audio"]).stem
        raw_path = raws[-1]
        render_from_raw(load_raw(raw_path), raw_path, run_dir, stem, RunOptions(formats=("json", "txt")),
                        Path(it["audio"]).name, json.loads((run_dir / "metrics.json").read_text()))
        rows.append(evaluate_output(it, run_dir / f"{stem}.json", json.loads((run_dir / "metrics.json").read_text()),
                                    r["vocab"]))
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_report(out / "results.jsonl", out / "report.md")
    print(out / "report.md")
    return 0


def cmd_score(a: argparse.Namespace) -> int:
    doc = json.loads(Path(a.transcript).read_text(encoding="utf-8"))
    ref = Path(a.reference).read_text(encoding="utf-8")
    print(json.dumps(score(ref, doc["text"], doc.get("language", "en")).to_dict(), indent=1))
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    out = Path(a.out)
    write_report(out / "results.jsonl", out / "report.md")
    print(out / "report.md")
    return 0


def add_eval_parsers(sub) -> None:
    from . import presets

    p = sub.add_parser("eval", help="benchmark presets against reference transcripts")
    es = p.add_subparsers(dest="eval_cmd", required=True)
    r = es.add_parser("run", help="run presets over a manifest")
    r.add_argument("manifest")
    r.add_argument("--preset", action="append", required=True, choices=list(presets.PRESETS))
    r.add_argument("--out", default="bench_results")
    r.add_argument("--vocab", choices=["off", "on", "both"], default="off")
    r.add_argument("--only", nargs="*", help="item ids to include")
    r.add_argument("--force", action="store_true", help="re-run items already in results.jsonl")
    r.set_defaults(fn=cmd_run)
    s = es.add_parser("score", help="score one transcript .json against a reference .txt")
    s.add_argument("transcript")
    s.add_argument("reference")
    s.set_defaults(fn=cmd_score)
    rs = es.add_parser("rescore", help="re-render outputs from raw records with current code and re-score")
    rs.add_argument("manifest")
    rs.add_argument("--out", default="bench_results")
    rs.set_defaults(fn=cmd_rescore)
    rp = es.add_parser("report", help="regenerate report.md from results.jsonl")
    rp.add_argument("--out", default="bench_results")
    rp.set_defaults(fn=cmd_report)
    from .bench_data import add_data_parsers
    add_data_parsers(es)
