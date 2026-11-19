# Validation results

Measured 2026-11-18 on an Apple M5 Pro with 24 GB unified memory, macOS 26.6, Python 3.12, MLX 0.32.4. The models are pinned to the commits in `hearken/models.py`. Every number below comes from `hearken eval` and can be regenerated with `scripts/run_benchmarks.sh` and `scripts/run_benchmarks_2.sh`.

The benchmark directories use earlier preset labels: `parakeet-greedy` = `fast`, `parakeet` = `parakeet-beam`, `max` = `accurate-full`, `max-8bit` = `accurate`.

## Data

| set | what | size | reference |
|---|---|---|---|
| TED-LIUM 3 long-form test | 10 complete talks, 5.7–25 min each; the 1-utterance item was excluded | 26,513 words, ~2.5 h | manually verified; lower-case, no punctuation |
| Stress (from *DanielKahneman*, 18 min) | `gaps`: 8 inserted non-speech gaps of 20–60 s (digital silence alternating with −55 dBFS room noise), 5.4 min total; `snr5`: pink noise at 5 dB SNR; `quiet30`: −30 dB | 3 × 3,015 words | as above |
| Terms (synthetic) | 24 sentences of lecture prose with 28 technical terms, macOS `say` (Samantha), 0.7–2.5 s random gaps, exact sentence onsets known | 278 words, 2.6 min | exact script |

All WER/CER figures use Whisper's English normaliser (the Open ASR Leaderboard convention). Pooled WER is total errors divided by total reference words.

## Accuracy: TED-LIUM long-form

| configuration | WER | CER | sub | del | ins | words in deletion runs ≥ 5 | duplicated* | RTF | peak MLX | peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen3-ASR 1.7B 8-bit** | **2.21%** | 1.21% | 255 | 237 | 95 | 19 | 14 | 0.067 | 6.2 GB | 3.2 GB |
| Qwen3-ASR 1.7B bf16 | 2.20% | 1.19% | 253 | 232 | 98 | 19 | 15 | 0.099 | 8.2 GB | 4.4 GB |
| Parakeet TDT v2, greedy | 3.04% | 1.81% | 261 | 450 | 96 | 8 | 0 | 0.017 | 2.4 GB | 1.1 GB |
| Parakeet TDT v2, beam 4 | 3.04% | 1.85% | 255 | 477 | 75 | 14 | 0 | 0.056 | 2.3 GB | 1.0 GB |
| Parakeet, beam 4, library front-end | 3.06% | 1.84% | 262 | 463 | 86 | 14 | 0 | 0.046 | 2.4 GB | 1.2 GB |
| Whisper large-v3 | 3.50% | 2.16% | 263 | 485 | 179 | 14 | 45 | 0.135 | 4.0 GB | 4.1 GB |

RTF is wall time ÷ audio duration, including model loading. 0.067 is about 4 minutes per audio hour.

\* "Duplicated" counts inserted words that repeat the preceding words. For Qwen3 these are real stutters the reference omits ("of of", "that that that that"), not chunk-boundary duplicates; none cluster at chunk cuts. Whisper's 45 include genuine repetition-loop output.

Per talk, 8-bit Qwen3 beat Parakeet on 7 talks and was narrowly worse on 3. The largest gap is the most disfluent speaker (James Cameron: 2.75% vs 7.63%). Without that talk, pooled WER is 2.14% (Qwen3 8-bit), 2.46% (Parakeet) and 3.04% (Whisper). Deletions explain most of Parakeet's deficit: it drops repeated and filler words that are in the audio.

8-bit vs bf16 Qwen3: per-talk differences range from −0.10 to +0.12 points, in both directions. **No measurable accuracy loss.**

### Two-engine presets (actual runs, 2 talks, 5,613 words)

| preset | WER | RTF | peak MLX |
|---|---|---|---|
| `accurate` (Qwen3 8-bit + Parakeet) | 2.67% | 0.072 | 6.2 GB |
| `accurate-full` (Qwen3 bf16 + Parakeet) | 2.62% | 0.091 | 8.2 GB |

Engines run sequentially and are unloaded in between, so peak memory is that of the larger engine. In the bf16 run, Parakeet accounted for 13.3 s of 131 s (10%).

## Review flags: do they find the errors?

Flags were scored against the reference with `hearken eval flags`. A word is "wrong" if it is substituted or inserted, or borders a deletion.

| primary | verifier | words flagged | precision (flagged words that are wrong) | recall (errors inside flags) |
|---|---|---|---|---|
| Qwen3 8-bit | Parakeet | 1.8% | 29.7% | 16.8% |
| Qwen3 bf16 | Parakeet | 1.8% | 29.8% | 17.1% |
| Qwen3 bf16 | Whisper | 2.0% | 31.6% | 20.3% |
| Qwen3 bf16 | none | 0.2% | 39.1% | 2.2% |
| Parakeet | Qwen3 | 1.4% | 36.2% | 12.1% |

The base error rate per word is 3.2% (Qwen3), so a flagged word is about 9× more likely to be wrong than an unflagged one. Some apparent errors are disfluencies the reference omits, so precision is a lower bound. Most errors are still *not* flagged: both models agree on them. Typical cases are proper nouns and words the reference itself spells differently.

## Hallucination during non-speech

The `gaps` variant inserts 5.42 min of non-speech at 8 known positions.

| engine | words in inserted non-speech | of which flagged |
|---|---|---|
| Qwen3-ASR (8-bit and bf16) | **0** | — |
| Parakeet (beam 4) | 2 | 2 |
| Whisper | 3 | 3 |

Before word-end snapping, Parakeet had 6 such words. 5 were real sentence-final words whose end timestamp ran into the pause. Only "You." (confidence 0.0001) was invented, and it was flagged.

WER on the `gaps` variant vs the original talk: Qwen3 2.07% vs 1.81%/1.91% (bf16/8-bit), Parakeet 2.33% vs 2.23%, Whisper 3.42% vs 3.02%. (The stress runs used Parakeet beam 4.)

## Robustness: noise and level

| engine | original | +pink noise at 5 dB SNR | −30 dB |
|---|---|---|---|
| Qwen3 8-bit | 1.91% | 3.25% | 1.87% |
| Parakeet (beam 4) | 2.23% | **2.86%** | 2.30% |
| Whisper | 3.02% | 4.93% | 3.22% |

No enhancement was applied. Very quiet audio costs nothing, thanks to linear gain and the models' level-invariant front-ends. In heavy noise, Parakeet degrades least and overtakes Qwen3. This is one talk with synthetic noise, so treat it as indicative.

## Timestamps

Synthetic terms set: absolute error of the first word's start vs the true sentence onset (n ≈ 23).

| engine | before VAD snapping (median) | after (median / p90) |
|---|---|---|
| Qwen3 (forced aligner) | 6 ms early | 9 ms / 56 ms |
| Parakeet | 186 ms early | 35 ms / 213 ms |
| Whisper | 540 ms early | 27 ms / 739 ms |

Snapping moves only boundaries that fall in VAD-detected silence; text is never touched. On real TED speech there is no ground-truth word timing. There, the end-to-end check confirmed that timestamps are monotonic and span the whole recording (last word ending at 344.996 s of a 345.23 s file), and that chunk joins neither drop nor repeat words.

## Technical vocabulary

| engine | WER vocab off → on | terms recognised off → on |
|---|---|---|
| Qwen3 8-bit | 1.80% → 1.08% | 27/28 → 27/28 |
| Qwen3 bf16 | 1.80% → 1.80% | 27/28 → 27/28 |
| Parakeet (beam + booster) | 2.88% → 2.88% | 25/28 → 26/28 |
| Whisper | 6.47% → 5.76% | 25/28 → 28/28 |

The differences are 0–2 words out of 278, so **no measurable effect** at this size. The strongest models already know these terms, and a single TTS voice is easy. Vocabulary is expected to matter for course-specific names, acronyms and coined notation that no model has seen. Measure on your own lectures with `hearken eval run --vocab both`.

## Speed and memory summary

| | default `fast` | `accurate` |
|---|---|---|
| processing per audio hour | ~1 min (3 h lecture ≈ 3 min) | ~4.3 min (3 h ≈ 13 min) |
| peak unified (GPU) memory | 2.4 GB | 6.2 GB |
| model disk | 2.5 GB | 6.8 GB |

Audio is memory-mapped in both: 3 h of audio is 0.7 GB.

## Known failure modes

1. **Errors both models share are not flagged.** Recall is ~17%: proper nouns, homophones ("their/they're"), and words neither model knows. A clean-looking passage can still be wrong.
2. **Heavy noise:** Qwen3 degrades more than Parakeet (above). For very noisy rooms, consider `--preset fast`, or read the flags carefully.
3. **Parakeet confidences are poorly calibrated** in greedy mode and rarely trigger `low_confidence`. Disagreement is the useful signal.
4. **Qwen3 has no per-word confidence.** Its flags come only from disagreement, VAD and repetition checks.
5. **Overlapping speakers / crosstalk:** not evaluated; there is no diarisation.
6. **Timestamp outliers:** after snapping, Parakeet and Whisper have p90 onset errors of 0.2–0.7 s. The SRT and VTT come from the primary (Qwen3 + aligner), which is much tighter.
7. **Chunk-boundary words:** cuts land in pauses where possible. In unbroken speech a word can still be cut, and the padded context mitigates this. The no-loss/no-duplication guarantee is tested with a synthetic engine, not proven for every real model behaviour.
8. **Number formatting** follows the model ("2 or 3" vs "two or three"). The normaliser treats these as equal for scoring, but the transcript keeps the model's form.
9. **Languages other than English** are wired through but unvalidated.

## Improvements likely to raise accuracy

1. **A larger primary when memory allows.** Nemotron 3.5 Transcribe or the next Qwen3-ASR would likely help, but need ≥ 48 GB unified memory today.
2. **Second-pass decoding of flagged spans:** re-decode only disagreement regions with a wider context window and beam search, then show the result as a third reading. It is cheap because it covers under 2% of the audio.
3. **Lecture-domain evaluation:** record and hand-correct 2–3 real lectures (accents, board noise, student questions). TED is cleaner than a classroom.
4. **Slide or syllabus vocabulary extraction:** build `--vocab` automatically from course PDFs, then measure it on real lectures.
5. **Speaker diarisation** for Q&A segments, as a separate, optional pass.
