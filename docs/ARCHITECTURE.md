# hearken — architecture and model selection

## Goals, in priority order

1. **Fidelity.** The transcript says what the speaker said: nothing summarised, reworded or invented. When the system is unsure, it says so.
2. **Completeness.** Every second of the recording reaches a recogniser. Nothing is dropped by a VAD or truncated at a chunk edge.
3. **Privacy.** The network is used only by `hearken setup`, to download pinned model weights. Transcription runs with `HF_HUB_OFFLINE=1` and never opens a socket.
4. **Cost.** Speed and memory matter, but only after 1–3. Where a cheaper configuration is *measured* to be as accurate, it is the default.

## Pipeline

```
source file ──► decode (PyAV, 16 kHz mono f32, memory-mapped temp file)
            ──► analyse (levels, SNR estimate, clipping, DC) ──► preprocess (DC removal, linear gain only)
            ──► VAD (Silero v5, per 32 ms frame; evidence only, never deletes audio)
            ──► for each engine in the preset:
                   plan chunks (cuts at the least speech-like point; padded context; cores tile the file)
                   decode each padded chunk; keep words whose midpoint is in the core
            ──► RAW record (.raw.json, written read-only, all engine outputs + VAD + analysis)
            ──► verify (cross-model disagreement, low confidence, words in silence,
                        speech without words, repetition loops, engine warnings)
            ──► outputs: .txt .srt .vtt .json .review.html (all derived from RAW, reproducible
                         with `hearken reformat`)
```

### Decoding

PyAV ships its own FFmpeg libraries, so any container FFmpeg reads works (MP4, MOV, MKV, MP3, M4A/AAC, WAV, FLAC, OGG/Opus, WebM) without a system `ffmpeg`. Audio is resampled once, straight from the source rate to 16 kHz, with libswresample. Channels are averaged in NumPy. FFmpeg's default stereo→mono matrix adds +3 dB to centred speech and can clip loud recordings, so it is not used. Samples are streamed to a private (`0600`) temporary file and memory-mapped, so a 3-hour recording uses ~0.7 GB of address space and is never held in several in-RAM copies.

### Preprocessing — deliberately minimal

Only invertible, spectrally neutral operations are applied: DC-offset removal, and a linear gain when the peak is below −12 dBFS. **No denoising, spectral gating, compression or EQ by default.** All candidate models were trained on large amounts of noisy, real-world audio. Speech-enhancement front-ends tuned for human listeners produce artefacts (musical noise, smeared consonants) that these models have never seen, and they commonly *raise* WER. The stress benchmark (additive noise at 5 dB SNR, −30 dB quiet copy) measures robustness without enhancement. See RESULTS.md. Every step that is applied is recorded in `audio_analysis.applied`.

### VAD is evidence, not a gate

Many pipelines (WhisperX, faster-whisper's `vad_filter`) delete non-speech before recognition. That is fast, but a VAD miss deletes quiet speech irrecoverably: a quiet question from the back of the room, a mumbled aside. hearken never removes audio. The VAD is used to:

- place chunk cuts in pauses,
- flag recognised words that lie in apparent silence (`no_speech_detected`, possible hallucination),
- flag confident speech regions where the primary engine produced no words (`speech_without_words`, possible omission).

Silero v5 runs as a 32-stream batch (each stream warmed up on the 2 s before it) and takes a few seconds per hour of audio. If the model file is missing, an adaptive energy detector is used instead and recorded in the output.

### Chunking without loss or duplication

Each chunk has a *core* and a *padded* interval. Cores tile the recording exactly. The engine decodes the padded audio, and only words whose midpoint falls inside the core are kept. Every word therefore belongs to exactly one chunk. A word straddling a cut is heard whole in at least one padded window. Cuts are placed at the minimum of the 0.3 s-smoothed VAD probability inside the allowed window, preferring later minima for longer context. `tests/test_pipeline.py` checks that 60 synthetic "words" survive chunking with no loss, duplication or timing error, with and without padding.

Chunk length per engine: Parakeet 120 s (5 s pad); Qwen3-ASR and Whisper 30 s (1.5 s pad), which is one native window and avoids their internal seek logic.

Are breaks a source of errors? Across the 10 TED talks, all 84 Parakeet cuts landed where the VAD speech probability was 0. The word error rate within 2 s of a cut was 4.9%, vs 4.1% elsewhere: a difference of ~7 words out of 822, ~1.2 standard errors. There is no measurable loss at breaks, so no extra boundary pass is run.

### Concurrency (Parakeet)

Profiling one 120 s chunk: spectrogram 1%, Conformer encoder ~41% (GPU-compute-bound), TDT decoding loop ~59%. parakeet-mlx's loop makes three host↔GPU syncs per step for a tiny LSTM/joint step, so it is latency-bound. Measured alternatives that did *not* help: multiple worker processes (1 → 4 workers: 13.0 → 15.9 s, the GPU is shared), decoding on CPU (2× slower, and bf16 CPU numerics change output), batching the encoder (0.3–0.8×: already compute-saturated), and more VAD streams.

What hearken does instead:

1. **Constant-shape windows.** Every Parakeet window is exactly 120 s of real audio. A shorter core simply gets more context on each side. MLX keeps a buffer set per tensor shape. With variable windows, encoding a 25-minute talk took 16 s and grew the cache to 16.8 GB; with constant windows it takes 6 s and the cache stays at 1.6 GB. The process's real peak footprint on that talk went from 18.3 GB to 3.3 GB. The MLX cache is also capped at 1 GB for every engine: for Qwen3 that alone cut the footprint from 8.5 to 6.7 GB, with an identical transcript and no speed cost.
2. **Lockstep batched TDT decoding** (`engines/tdt_batch.py`). Up to 16 chunks decode together. The joint network's encoder projection is computed once for all frames, each step is one batched joint evaluation with a single sync, and the LSTM runs only for chunks that emitted a token. It applies the same decoding rule as parakeet-mlx. It also avoids parakeet-mlx's LSTM wrapper, which mixes up time and batch axes and is only correct at batch size 1.
3. **bf16 encoder input.** parakeet-mlx always produces a float32 log-mel, which makes MLX run the bf16 encoder with float32 activations. hearken casts the log-mel to bf16 first. That saves 1.1 GB and makes the encoder ~35% faster, with no measurable accuracy change.

Batching itself adds no memory: weights are shared, encoding stays one window at a time, and the decoder holds only ~3 MB per window. RESULTS.md, "Compared with stock parakeet-mlx", breaks down every gigabyte and second against the stock library.

Result on a 25-minute talk: the Parakeet stage drops from 27.8 s to 3.7 s and the whole run from 33.5 s to 5.7 s. On all 10 TED talks, WER is 3.07% vs 3.04% (7 words out of 26,513; bf16 rounding on near-ties, differences in both directions). Beam search and vocabulary boosting still use the per-chunk library path.

## Model survey and selection

Candidates were limited to models that are open-weights, run natively on Apple Silicon (MLX), and fit comfortably in 24 GB of unified memory.

| model | params | licence | decoder | strengths | weaknesses |
|---|---|---|---|---|---|
| **NVIDIA Parakeet TDT 0.6B v2** (parakeet-mlx) | 0.6 B | CC-BY-4.0 | transducer (TDT) | top-tier English WER, very fast, frame-synchronous (structurally cannot "run on" in silence), native per-token timestamps and confidence, trie-based vocabulary boosting | English only (v3 adds 25 European languages) |
| **Qwen3-ASR 1.7B** (mlx-qwen3-asr) | 1.7 B + 0.6 B aligner | Apache-2.0 | audio-LLM | among the best open models on the 2026 Open ASR Leaderboard; free-text context for terminology; 52 languages | autoregressive (can hallucinate or loop); no confidences; timestamps need a separate forced aligner; slower |
| **Whisper large-v3** (mlx-whisper) | 1.55 B | MIT | attention enc-dec | 99 languages, mature, prompt support | well-documented hallucination in silence and repetition loops; behind the newer models on English WER; pulls in PyTorch |
| Nemotron 3.5 Transcribe (30B-A4B) | 30 B MoE | OpenMDW | speech-LLM | leaderboard leader (late 2026) | ~35 GB even at 8-bit: does not fit a 24 GB Mac |
| Cohere Transcribe, Granite Speech 4.x, Canary | 1–8 B | various | — | competitive WER | no maintained MLX port at time of writing, or NeMo/CUDA-centric |

**Decision (from the benchmark, not leaderboard claims).** The default preset `fast` uses Parakeet TDT v2 alone (greedy): 3.07% WER at ~16 s per audio hour and 3.2 GB peak RAM (with the batched decoder below). It was chosen for speed and footprint; its extra errors are mostly dropped disfluencies. For maximum fidelity, `accurate` uses Qwen3-ASR 1.7B with 8-bit weights as the primary: 2.21% WER on TED-LIUM long-form vs 3.04% for Parakeet and 3.50% for Whisper, zero words in inserted silence, and the best timestamps. 8-bit weights are as accurate as bf16 and 32% faster. Parakeet TDT v2 (greedy) is the verifier. It is architecturally different (transducer vs. LLM decoder, different training data), so its errors are largely independent of Qwen3's, and it costs ~10–15% extra time. Its disagreements cover 1.8% of words and contain 17% of the primary's errors. Whisper is optional. The short version is in [MODEL_CHOICE.md](MODEL_CHOICE.md); full numbers are in [RESULTS.md](RESULTS.md).

### Why the verifier does not vote

ROVER-style voting between two systems cannot break ties, and with three it can silently replace a correct minority reading. The guiding rule is "never silently reinterpret the speaker", so the primary transcript is kept verbatim. Disagreements become review items showing every engine's reading, timestamped, in `.json`, `hearken review` and the HTML review page.

### Parakeet front-end fix

parakeet-mlx ≤ 0.7.0 computes the STFT magnitude as |re| + |im| instead of √(re² + im²) as NeMo does. hearken ships a NeMo-faithful log-mel (`frontend="nemo"`, default) and benchmarked it against the library's (`parakeet-libfe` preset). The difference is not measurable (3.04% vs 3.06% WER).

### Vocabulary

A plain text file with one term per line. Each engine gets it in its own way:

- Parakeet: trie booster during beam search (+2.0 log-score per completed term; a vocabulary switches Parakeet from greedy to beam 4, because the booster only acts in beam search),
- Qwen3-ASR: "Technical terms that may occur: …" as system context,
- Whisper: "Glossary: …" initial prompt.

Biasing can also cause *false* insertions of a boosted term. The terms benchmark therefore reports both term recall and overall WER with the vocabulary off and on.

## Raw vs. derived outputs

`<name>.raw.json` is written once, made read-only (`0444`), and never overwritten. A re-run writes `<name>.raw.<timestamp>.json`. It holds every engine's words exactly as decoded, the VAD track, the audio analysis and the exact model revisions. All other files are a deterministic function of it. `hearken reformat` regenerates them with different formatting options, without re-transcribing. The main `.json` records the raw file's SHA-256.

## Formatting modes

- `verbatim`: one line per sentence, each with a timestamp.
- `readable` (default): paragraphs split at pauses, a timestamp per paragraph. This is pure layout; tests assert that the word sequence is identical to verbatim.
- `--drop-fillers` (off by default) removes only standalone *um/uh/erm/er/hmm/mm*.
- Uncertain words are wrapped in `[? … ?]` in `.txt` (`--no-markers` to disable; `--subtitle-markers` to add them to `.srt`/`.vtt`).

## Privacy and logging

Logs contain durations, chunk counts, timings and warnings, never transcript text. Temporary audio lives in the output directory with mode `0600` and is deleted on exit. Models are resolved with `local_files_only=True`. HF telemetry is disabled.

## Licensing

hearken: MIT. Runtime dependencies: MLX (MIT), parakeet-mlx (Apache-2.0), mlx-qwen3-asr (Apache-2.0), PyAV (BSD; bundles LGPL FFmpeg), onnxruntime (MIT), Silero VAD (MIT), rapidfuzz (MIT), whisper-normalizer (MIT). Optional: mlx-whisper (MIT, pulls PyTorch, BSD). Weights: Parakeet TDT v2 **CC-BY-4.0** (attribute NVIDIA if you redistribute the model; transcripts you produce carry no obligation), Qwen3-ASR / ForcedAligner Apache-2.0, Whisper MIT. Evaluation data: TED-LIUM 3 is CC BY-NC-ND 3.0 and is used for evaluation only, never redistributed.
