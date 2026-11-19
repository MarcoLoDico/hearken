"""Builders for evaluation sets.

prepare-tedlium   TED-LIUM 3 long-form test set (11 complete talks, ~2.5 h,
                  manually verified references; CC BY-NC-ND 3.0, evaluation only).
stress            derive stress variants from an item: inserted non-speech
                  gaps (silence and room-tone noise) at known positions,
                  additive noise at a fixed SNR, and a very quiet copy.
terms             synthetic technical-vocabulary lecture built with macOS
                  `say`, with exact sentence onsets for timestamp scoring.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SR = 16000


def _write_wav(path: Path, x: np.ndarray) -> None:
    import soundfile as sf
    sf.write(str(path), np.clip(x, -1, 1), SR, subtype="PCM_16")


def _read_wav(path: Path) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr}")
    return x.mean(axis=1)


def cmd_prepare_tedlium(a: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = pq.read_table(a.parquet)
    lines = []
    for i in range(t.num_rows):
        r = t.slice(i, 1).to_pylist()[0]
        n_words = len(r["text"].split())
        if n_words < a.min_words:
            continue
        sid = r["speaker_id"]
        (out / f"{sid}.wav").write_bytes(r["audio"]["bytes"])
        (out / f"{sid}.txt").write_text(r["text"].strip() + "\n", encoding="utf-8")
        lines.append(json.dumps({"id": sid, "audio": f"{sid}.wav", "reference": f"{sid}.txt"}))
    (out / "manifest.jsonl").write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} items -> {out / 'manifest.jsonl'}")
    return 0


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    f = np.fft.rfftfreq(n, 1 / SR)
    spec = rng.normal(size=f.size) + 1j * rng.normal(size=f.size)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return (x / (np.std(x) + 1e-12)).astype(np.float32)


def cmd_stress(a: argparse.Namespace) -> int:
    from .vad import detect, FRAME_S
    from . import models

    manifest = Path(a.manifest)
    items = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    item = next(i for i in items if i["id"] == a.id)
    src = manifest.parent / item["audio"]
    x = _read_wav(src)
    rng = np.random.default_rng(a.seed)
    out_lines = []
    base = manifest.parent

    # 1) inserted non-speech gaps at detected pauses
    speech = detect(x, models.vad_path())
    p = speech.probs
    pauses = [i for i in range(10, len(p) - 10) if p[i - 5:i + 5].max() < 0.1]
    n_gaps = a.gaps
    step = len(x) / SR / (n_gaps + 1)
    chosen = []
    for g in range(1, n_gaps + 1):
        target = g * step / FRAME_S
        if pauses:
            chosen.append(min(pauses, key=lambda i: abs(i - target)) * FRAME_S)
    chosen = sorted(set(round(c, 2) for c in chosen))
    pieces, regions, shift, prev = [], [], 0.0, 0
    for k, t in enumerate(chosen):
        cut = int(t * SR)
        pieces.append(x[prev:cut])
        dur = float(rng.uniform(a.gap_min, a.gap_max))
        n = int(dur * SR)
        # alternate digital silence and quiet room tone (-55 dBFS pink noise)
        gap = np.zeros(n, np.float32) if k % 2 == 0 else _pink(n, rng) * 10 ** (-55 / 20)
        start = t + shift
        regions.append([round(start, 3), round(start + dur, 3)])
        pieces.append(gap)
        shift += dur
        prev = cut
    pieces.append(x[prev:])
    gid = f"{a.id}_gaps"
    _write_wav(base / f"{gid}.wav", np.concatenate(pieces))
    out_lines.append({"id": gid, "audio": f"{gid}.wav", "reference": item["reference"], "nonspeech": regions})

    # 2) additive pink noise at fixed SNR (relative to active-speech RMS)
    active = np.repeat(p >= 0.5, int(FRAME_S * SR))[:len(x)]
    s_rms = float(np.sqrt(np.mean(x[:len(active)][active] ** 2)))
    noise = _pink(len(x), rng) * s_rms * 10 ** (-a.snr / 20)
    nid = f"{a.id}_snr{a.snr:g}"
    _write_wav(base / f"{nid}.wav", x + noise)
    out_lines.append({"id": nid, "audio": f"{nid}.wav", "reference": item["reference"]})

    # 3) quiet speaker: -30 dB
    qid = f"{a.id}_quiet30"
    _write_wav(base / f"{qid}.wav", x * 10 ** (-30 / 20))
    out_lines.append({"id": qid, "audio": f"{qid}.wav", "reference": item["reference"]})

    out = Path(a.out_manifest)
    with out.open("a") as f:
        for line in out_lines:
            f.write(json.dumps(line) + "\n")
    print(f"wrote {[l['id'] for l in out_lines]} -> {out}")
    return 0


TERMS_SCRIPT = [
    "Today we derive the Kullback-Leibler divergence between two Gaussian distributions.",
    "Recall that the eigendecomposition of a symmetric matrix gives orthogonal eigenvectors.",
    "The Lagrangian combines the objective with the constraints through Lagrange multipliers.",
    "We minimize the cross-entropy loss using stochastic gradient descent with a learning rate of zero point zero one.",
    "By the Cauchy-Schwarz inequality, the inner product is bounded by the product of the norms.",
    "The Navier-Stokes equations describe the motion of viscous fluids.",
    "In quantum mechanics, the Hamiltonian operator determines the time evolution of the wave function.",
    "A Lyapunov function proves that the equilibrium is asymptotically stable.",
    "Tikhonov regularization adds a penalty proportional to the squared norm of the parameters.",
    "The Nyquist frequency is half the sampling rate, so at forty four point one kilohertz it is twenty two point zero five kilohertz.",
    "Chebyshev polynomials minimize the maximum interpolation error on the interval.",
    "The softmax function converts logits into a probability distribution.",
    "We apply the Fourier transform and then the inverse discrete cosine transform.",
    "Kolmogorov complexity measures the length of the shortest program that outputs a string.",
    "The Michaelis-Menten equation models enzyme kinetics in biochemistry.",
    "Under the null hypothesis, the p-value was zero point zero three, so we reject it at the five percent level.",
    "The Schrödinger equation is a linear partial differential equation.",
    "Backpropagation computes gradients with the chain rule through every layer.",
    "In nineteen ninety seven, the algorithm ran in O of n log n time.",
    "Heteroscedasticity violates one of the Gauss-Markov assumptions.",
    "The Jacobian matrix collects all first order partial derivatives.",
    "Bayesian inference updates the prior with the likelihood to obtain the posterior.",
    "A Hilbert space is a complete inner product space.",
    "The Krebs cycle produces NADH and FADH two for oxidative phosphorylation.",
]
TERMS_VOCAB = ["Kullback-Leibler", "eigendecomposition", "Lagrangian", "Lagrange multipliers", "cross-entropy",
               "Cauchy-Schwarz", "Navier-Stokes", "Hamiltonian", "Lyapunov", "Tikhonov", "Nyquist", "Chebyshev",
               "softmax", "logits", "Kolmogorov", "Michaelis-Menten", "p-value", "Schrödinger", "Backpropagation",
               "Heteroscedasticity", "Gauss-Markov", "Jacobian", "Bayesian", "Hilbert space", "Krebs cycle",
               "NADH", "FADH", "oxidative phosphorylation"]


def cmd_terms(a: argparse.Namespace) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    pieces, onsets, t = [np.zeros(int(1.0 * SR), np.float32)], [], 1.0
    with tempfile.TemporaryDirectory() as td:
        for i, sentence in enumerate(TERMS_SCRIPT):
            f = Path(td) / f"{i}.wav"
            subprocess.run(["say", "-v", a.voice, "-r", str(a.rate), "--file-format=WAVE",
                            "--data-format=LEI16@16000", "-o", str(f), sentence], check=True)
            x = _read_wav(f)
            nz = np.flatnonzero(np.abs(x) > 1e-3)
            x = x[nz[0]:nz[-1] + 1] if nz.size else x
            onsets.append([round(t, 3), sentence.split()[0]])
            pieces.append(x.astype(np.float32))
            gap = rng.uniform(0.7, 2.5)
            pieces.append(np.zeros(int(gap * SR), np.float32))
            t += len(x) / SR + gap
    _write_wav(out / "terms.wav", np.concatenate(pieces))
    (out / "terms.txt").write_text(" ".join(TERMS_SCRIPT) + "\n", encoding="utf-8")
    (out / "terms_vocab.txt").write_text("\n".join(TERMS_VOCAB) + "\n", encoding="utf-8")
    item = {"id": "terms", "audio": "terms.wav", "reference": "terms.txt", "vocab": "terms_vocab.txt",
            "onsets": onsets}
    (out / "manifest.jsonl").write_text(json.dumps(item) + "\n")
    print(f"synthetic lecture {t:.0f}s -> {out / 'manifest.jsonl'}")
    return 0


def add_data_parsers(es) -> None:
    p = es.add_parser("prepare-tedlium", help="unpack TED-LIUM long-form parquet into wav/txt + manifest")
    p.add_argument("parquet")
    p.add_argument("--out", default="data/eval/tedlium")
    p.add_argument("--min-words", type=int, default=200)
    p.set_defaults(fn=cmd_prepare_tedlium)
    p = es.add_parser("stress", help="make silence-gap / noise / quiet variants of one item")
    p.add_argument("manifest")
    p.add_argument("--id", required=True)
    p.add_argument("--out-manifest", required=True)
    p.add_argument("--gaps", type=int, default=8)
    p.add_argument("--gap-min", type=float, default=20.0)
    p.add_argument("--gap-max", type=float, default=60.0)
    p.add_argument("--snr", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_stress)
    p = es.add_parser("terms", help="synthesise a technical-vocabulary test with macOS `say`")
    p.add_argument("--out", default="data/eval/terms")
    p.add_argument("--voice", default="Samantha")
    p.add_argument("--rate", type=int, default=175)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_terms)
