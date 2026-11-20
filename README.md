# hearken

Accuracy-first, fully local transcription for long lectures on Apple Silicon.

- **Local and private.** The network is used once, by `hearken setup`, to download pinned open-source models. Transcription is offline. Logs never contain transcript text.
- **Faithful.** The transcript is exactly what the primary model heard. Nothing is summarised or rewritten. Passages where models disagree, confidence is low, words appear in silence, or speech produced no words are marked for review instead of being "fixed".
- **Complete.** No VAD gating: every second is transcribed. Chunks are cut in pauses, with padded context and exact de-duplication.
- **Measured.** A reproducible benchmark (`hearken eval`) reports WER/CER, hallucination in silence, missing and duplicated speech, timestamp error, term recall, speed and peak memory. See [docs/RESULTS.md](docs/RESULTS.md).

Why these models, in one page: [docs/MODEL_CHOICE.md](docs/MODEL_CHOICE.md). Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Measured accuracy, limitations and failure modes: [docs/RESULTS.md](docs/RESULTS.md).

## Requirements

- macOS on Apple Silicon (M1 or later), Python 3.10–3.13
- ~2.5 GB free disk for the default model (~7 GB more for `accurate`)
- 8 GB unified memory is enough for the default preset (peak ~2.4 GB); `accurate` needs ~6 GB

No system `ffmpeg` needed: decoding uses PyAV's bundled FFmpeg libraries.

## Install

```bash
git clone https://github.com/MarcoLoDico/hearken.git && cd hearken
python3.12 -m venv .venv
.venv/bin/pip install -e .
```

Optional extras: `.[whisper]` adds the Whisper large-v3 engine (pulls in PyTorch); `.[eval]` adds benchmarking tools; `.[dev]` adds tests.

Download the default model once, while online (~2.5 GB; `hearken setup --preset accurate` adds the Qwen3 models):

```bash
.venv/bin/hearken setup
```

From then on, everything works with the network off.

## Use

```bash
hearken transcribe lecture.mp4
hearken transcribe lecture.m4a --vocab course_terms.txt -o out/
hearken transcribe *.mp3 --preset fast
```

Outputs, written to `<input>.transcript/` by default:

| file | contents |
|---|---|
| `lecture.txt` | readable paragraphs with timestamps; uncertain words wrapped in `[? … ?]` |
| `lecture.srt`, `lecture.vtt` | subtitles (≤ 2 lines × 42 chars, ≤ 6 s per cue) |
| `lecture.json` | words with timestamps and confidences, sentences, uncertain passages, engines and exact model revisions, audio analysis, timing and memory metrics |
| `lecture.review.html` | offline review page: click any word to play from there; side panel lists every uncertain passage with each engine's reading. Keep the media file next to it. |
| `lecture.raw.json` | **read-only** record of exactly what each model produced, before any post-processing |

Review uncertain passages in the terminal:

```bash
hearken review out/lecture.json
hearken review out/lecture.json --reason disagreement
```

Re-render with different formatting without re-transcribing:

```bash
hearken reformat out/lecture.raw.json --mode verbatim --no-markers
```

### Presets

| preset | what | TED WER | time per audio hour | peak memory |
|---|---|---|---|---|
| `fast` (default) | Parakeet TDT v2 only, no cross-check | 3.1% | ~20 s | 2.3 GB |
| `accurate` | Qwen3-ASR 1.7B 8-bit transcript, Parakeet cross-check | 2.2% | ~4.3 min | 6.2 GB |
| `accurate-full` | same, full-precision Qwen3 (no more accurate) | 2.2% | ~5.5 min | 8.2 GB |
| `qwen3-8bit`, `qwen3` | Qwen3 only, no cross-check | 2.2% | ~4 / ~6 min | 6.2 / 8.2 GB |
| `triple`, `whisper` | add / use Whisper large-v3 (needs `.[whisper]`) | 3.5% (Whisper) | ~8 min | 4.0 GB |

Measured on an M5 Pro with 24 GB. `hearken presets` lists all presets.

`fast` is the default because it is ~14× faster and ~2.5× lighter: a 3-hour lecture takes about a minute. Its extra errors are mostly dropped repetitions and fillers ("I am I am" → "I am"). It has no second model, so the review list only contains VAD-based checks (words in silence, speech without words, repetition loops). Use `--preset accurate` when every word matters, or for a recording with heavy disfluency.

### Course vocabulary

A UTF-8 text file with one term per line. Lines starting with `#` are ignored. Spell terms exactly as they should appear:

```
# MATH 241
Lagrangian
Cauchy-Schwarz
eigendecomposition
Tikhonov
```

Use it to measure whether vocabulary helps on *your* lectures: transcribe a 10-minute excerpt, correct it by hand into `ref.txt`, then run `hearken eval score out/excerpt.json ref.txt` with and without `--vocab`.

### Other languages

`--language de` (etc.) works with the Qwen3-ASR and Whisper engines. Parakeet TDT v2 is English-only; `parakeet-tdt-0.6b-v3` covers 25 European languages. Only English has been benchmarked.

## Benchmark yourself

```bash
pip install -e ".[eval]"
hearken eval prepare-tedlium path/to/tedlium-long-form/test.parquet --out data/eval/tedlium
hearken eval terms --out data/eval/terms
hearken eval run data/eval/tedlium/manifest.jsonl --preset accurate --preset fast --out bench_results/ted
```

Your own recordings: write a manifest line `{"id": "...", "audio": "x.m4a", "reference": "x.txt"}` per file.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

The unit and pipeline tests use a synthetic engine and run in seconds without models.
