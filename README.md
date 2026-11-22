# hearken

**Fast, faithful, fully private speech-to-text for Apple Silicon.**

Lectures, meetings, interviews, podcasts, voice memos: three hours of audio becomes a timestamped transcript and subtitles in under a minute, on your Mac, with no cloud and no account.

```bash
hearken transcribe recording.mp4
```

## Why hearken

- **Fast.** ~16 s per hour of audio. A custom batched decoder and constant-size chunks make it ~3.4× faster than stock parakeet-mlx.
- **Accurate.** 3.1% word error rate on TED talks by default, 2.2% with `--preset accurate`.
- **Private.** Models are downloaded once. After that, nothing leaves your machine, and logs never contain transcript text.
- **Faithful.** It never summarises or "fixes" what was said. Doubtful passages are marked `[? … ?]` and listed for review.
- **Complete.** Every second is transcribed. Chunks are cut in pauses, and tests check that no word is lost or duplicated at a cut.
- **Light.** About 3 GB of RAM, any audio or video format, no system ffmpeg.
- **Measured.** Every claim above comes from a reproducible benchmark; see [docs/RESULTS.md](docs/RESULTS.md).

## Install

Requires macOS on Apple Silicon and Python 3.10–3.13.

```bash
git clone https://github.com/MarcoLoDico/hearken.git && cd hearken
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
hearken setup          # one-time model download (~2.5 GB); offline from here on
```

## Use

```bash
hearken transcribe recording.mp4                       # any format: mp4, mov, m4a, mp3, wav, …
hearken transcribe *.m4a -o notes/                     # batch, custom output folder
hearken transcribe recording.mp4 --preset accurate     # max fidelity, cross-checked by a 2nd model
hearken transcribe recording.mp4 --vocab terms.txt     # names and jargon, one per line
hearken review notes/recording.json                    # list passages worth checking
hearken reformat notes/recording.raw.json --mode verbatim   # re-render without re-transcribing
```

Each run writes `<name>.transcript/` next to the input:

| file | what it is |
|---|---|
| `.txt` | readable paragraphs with timestamps |
| `.srt` / `.vtt` | subtitles |
| `.review.html` | click any word to play from there; uncertain passages in a side panel |
| `.json` | every word with time and confidence, plus review items and metadata |
| `.raw.json` | read-only record of exactly what the model heard |

## Presets

| preset | WER (TED) | per audio hour | RAM | use it when |
|---|---|---|---|---|
| `fast` (default) | 3.1% | ~16 s | 3.2 GB | everyday notes and subtitles |
| `accurate` | 2.2% | ~3 min | 6.7 GB | every word matters: verbatim quotes, disfluent speakers |

`fast` runs NVIDIA Parakeet TDT v2. `accurate` runs Qwen3-ASR 1.7B and has Parakeet check its work: every disagreement is flagged, and reviewing those flags finds errors ~9× faster than proofreading at random. Run `hearken setup --preset accurate` once before using it. `hearken presets` lists the rest.

`fast` mostly loses repeated words and fillers ("I am I am" → "I am").

**Best with one speaker at a time.** All benchmarks are single-speaker talks. Several speakers taking turns are transcribed, but not labelled by speaker. Overlapping or cross-talking speech has not been tested and is likely to lose words.

Other languages work with Qwen3 (`--preset qwen3-8bit --language de`) but are not benchmarked.

## Learn more

- [MODEL_CHOICE.md](docs/MODEL_CHOICE.md): why these models, in one page
- [RESULTS.md](docs/RESULTS.md): accuracy, speed, memory, failure modes
- [ARCHITECTURE.md](docs/ARCHITECTURE.md): how it works
- Benchmark it yourself: `pip install -e ".[eval]"`, then `hearken eval --help`
- Tests: `pip install -e ".[dev]" && pytest` (seconds, no models needed)

MIT licensed. Model weights: Parakeet CC-BY-4.0, Qwen3-ASR Apache-2.0.
