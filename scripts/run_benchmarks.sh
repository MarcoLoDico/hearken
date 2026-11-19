#!/usr/bin/env bash
# Full benchmark used for docs/RESULTS.md. Resumable: finished runs are skipped.
set -u
cd "$(dirname "$0")/.."
QW=.venv/bin/hearken
SINGLE="--preset parakeet-beam --preset parakeet-libfe --preset fast --preset whisper --preset qwen3"
$QW eval run data/eval/terms/manifest.jsonl $SINGLE --vocab both --out bench_results/terms
$QW eval run data/eval/tedlium/manifest.jsonl $SINGLE --out bench_results/tedlium
$QW eval run data/eval/tedlium/stress.jsonl --preset fast --preset whisper --preset qwen3 --out bench_results/stress
