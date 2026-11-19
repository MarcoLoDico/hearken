#!/usr/bin/env bash
# Second benchmark round: quantised Qwen3 and the two-engine presets.
set -u
cd "$(dirname "$0")/.."
QW=.venv/bin/hearken
$QW eval run data/eval/tedlium/manifest.jsonl --preset qwen3-8bit --out bench_results/tedlium
$QW eval run data/eval/terms/manifest.jsonl --preset qwen3-8bit --vocab both --out bench_results/terms
$QW eval run data/eval/tedlium/stress.jsonl --preset qwen3-8bit --out bench_results/stress
$QW eval run data/eval/tedlium/manifest.jsonl --preset accurate-full --preset accurate --only BillGates GaryFlake --out bench_results/combo
