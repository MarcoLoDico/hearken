# Model choice

**Default preset `fast`: Parakeet TDT 0.6B v2 alone.** It is ~12× faster and needs half the memory of the most accurate configuration (about 16 s per audio hour, 3.2 GB peak RAM), for a modest accuracy cost (3.1% vs 2.2% WER on TED lectures). Most of that cost is dropped repetitions and fillers, not misheard content words. That trade was chosen deliberately, as the better fit for everyday lecture notes.

**Most accurate preset `accurate`:** Qwen3-ASR 1.7B with 8-bit weights writes the transcript. NVIDIA Parakeet TDT 0.6B v2 independently re-transcribes the audio, and every place it disagrees is flagged for review. Parakeet never changes the text.

## The evidence

TED-LIUM 3 long-form test set: 10 complete talks, 26.5k words, ~2.5 h. Measured on an M5 Pro with 24 GB. Full tables: [RESULTS.md](RESULTS.md). Peak RAM is the process footprint as macOS reports it. †Whisper: live MLX allocations only, not re-measured.

| | WER | speed (min per audio hour) | peak RAM | words in 5 min of inserted silence |
|---|---|---|---|---|
| **Qwen3-ASR 1.7B, 8-bit** | **2.21%** | **~3** | 6.7 GB | **0** |
| Qwen3-ASR 1.7B, full precision | 2.20% | ~4.5 | 8.3 GB | 0 |
| Parakeet TDT v2 | 3.07% | ~0.3 | 3.2 GB | 2 (both flagged) |
| Whisper large-v3 | 3.50% | ~8 | ≥ 4.0 GB† | 3 |

## Why Qwen3-ASR writes the transcript in `accurate`

- **Most accurate.** It made 28% fewer errors than Parakeet and 37% fewer than Whisper. It is better on 7 of 10 talks (narrowly worse on 3), and stays ahead with the most disfluent talk removed (2.14% vs 2.46%).
- **Most faithful.** Its lead comes mainly from keeping what was said: Parakeet silently drops repetitions and fillers ("I am I am I am" → "I am") and Qwen3 keeps them. It also recognised the most technical terms (27/28).
- **No hallucinations in silence.** It produced 0 words across 5.4 minutes of inserted silence and room noise.
- **Best timestamps.** Thanks to its forced aligner, word onsets are within ~10 ms median on a controlled test.

## Why 8-bit

It is exactly as accurate as full precision (2.21% vs 2.20%, with per-talk differences of ±0.1 points in both directions), faster, and needs 1.6 GB less memory (6.7 vs 8.3 GB). The full-precision version is kept as `accurate-full`.

## Why Parakeet as the cross-check in `accurate`

- **Different architecture, independent errors.** Parakeet is a transducer, not a language-model decoder, so its mistakes rarely coincide with Qwen3's. Where the two disagree is where errors cluster.
- **Flags are worth reading.** The flags cover 1.8% of words but contain 17% of Qwen3's errors. A flagged word is ~30% likely to be wrong, against 3% elsewhere, so checking flags is about 9× more efficient than proofreading at random.
- **Cheap.** It adds about 10–15% to run time.
- **Whisper as checker** catches slightly more (20%), but costs about 8× as much and needs PyTorch.

## What was rejected, and why

- **Whisper large-v3 (mlx-whisper, whisper.cpp):** worst accuracy, most repeated and invented words, and needs PyTorch. whisper.cpp runs the same weights, so it would be lighter but no more accurate. Kept only as an optional baseline.
- **Parakeet as primary (now the default, `fast`):** ~12× faster (with hearken's batched decoder) and half the memory, at ~39% more errors. The extra errors are mostly dropped disfluencies. No cross-check, so far fewer errors are flagged for review.
- **Parakeet beam search:** no gain over greedy (3.04% both) at 3× the cost.
- **Nemotron 3.5 Transcribe (30B):** tops the late-2026 leaderboard but needs ~35 GB even at 8-bit, so it doesn't fit a 24 GB Mac.
- **Cohere Transcribe, Granite Speech, Canary:** no maintained Apple Silicon (MLX) build.
- **Voting between models:** it can silently replace a correct reading, so disagreements are flagged instead.

## Caveats

- English only. TED talks are lecture-like but more polished than a classroom.
- Under heavy noise (5 dB SNR), Parakeet did slightly better than Qwen3 (2.86% vs 3.25%) on the one talk tested.
- The terminology test is synthetic (text-to-speech). Course vocabulary showed no measurable effect there, so test it on your own lectures.
