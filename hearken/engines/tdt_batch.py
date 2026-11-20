"""Batched greedy TDT decoding for Parakeet: many chunks advance in lockstep.

parakeet-mlx decodes one chunk at a time and makes three GPU round-trips per
decoding step. The step is tiny (an LSTM cell plus a joint layer), so the run
is bound by per-step latency, not compute. Here, B chunks are decoded
together:

  * the joint network's encoder projection is computed once for all frames
    of all chunks (one large matmul instead of one small one per step),
  * each step evaluates the joint for every still-active chunk in one batched
    call with a single host sync,
  * the prediction network (LSTM) runs only for the chunks that emitted a
    token on that step, since blanks leave its state unchanged.

The decoding rule (argmax token, argmax duration, TDT frame advance, the
max-symbols anti-stall rule, the entropy confidence) is the same as
parakeet_mlx.parakeet.ParakeetTDT.decode_greedy. bf16 batched matmuls can
round differently from per-row ones, so rare near-tie tokens may differ. WER
equivalence is checked on the benchmark (docs/RESULTS.md).
"""

from __future__ import annotations

import math

import mlx.core as mx


def _lstm_step(rnn, x, h, c):
    """One time step of the stacked LSTM for a batch.

    parakeet_mlx's LSTM wrapper transposes to time-major before calling
    mlx.nn.LSTM, which is batch-first, so it is only correct for batch size 1.
    The MLX cells are called directly here. x: [N, D]; h, c: [layers, N, H].
    """
    hs, cs = [], []
    out = x[:, None, :]
    for i, layer in enumerate(rnn.lstm):
        ah, ac = layer(out, hidden=h[i], cell=c[i])
        out = ah
        hs.append(ah[:, -1])
        cs.append(ac[:, -1])
    return out[:, -1], mx.stack(hs), mx.stack(cs)


def decode_greedy_batch(model, features: list[mx.array]) -> list[list[tuple[int, float, float, float]]]:
    """Decode encoder outputs ``features`` (each [frames, dim]).

    Returns, per chunk, a list of (token_id, start_s, duration_s, confidence).
    """
    B = len(features)
    if B == 0:
        return []
    V = len(model.vocabulary)  # blank id == V
    durations = list(model.durations)
    max_symbols = model.max_symbols
    ratio = model.time_ratio
    dtype = features[0].dtype
    lengths = [int(f.shape[0]) for f in features]
    S = max(lengths)

    padded = mx.stack([mx.pad(f, [(0, S - f.shape[0]), (0, 0)]) if f.shape[0] < S else f for f in features])
    joint = model.joint
    enc_proj = joint.enc(padded)  # [B, S, H]
    act, *_rest, out_layer = joint.joint_net

    pred = model.decoder
    rnn = pred.prediction["dec_rnn"]
    embed = pred.prediction["embed"]
    n_layers, hid = len(rnn.lstm), rnn.hidden_size
    zeros = mx.zeros((n_layers, B, hid))
    # Same start state as the library: zero embedding, zero LSTM state.
    o, h, c = _lstm_step(rnn, mx.zeros((B, pred.pred_hidden)), zeros, zeros)
    h, c = h.astype(dtype), c.astype(dtype)
    dec_proj = joint.pred(o.astype(dtype))  # [B, H]
    mx.eval(enc_proj, dec_proj, h, c)

    log_vocab = math.log(V + 1)
    step = [0] * B
    stuck = [0] * B
    out: list[list[tuple[int, float, float, float]]] = [[] for _ in range(B)]
    active = [b for b in range(B) if lengths[b] > 0]

    while active:
        ia = mx.array(active)
        e = enc_proj[ia, mx.array([step[b] for b in active])]  # [A, H]
        logits = out_layer(act(e + dec_proj[ia]))  # [A, V+1+D]
        tok_logits = logits[:, : V + 1]
        tok = mx.argmax(tok_logits, axis=-1)
        dur = mx.argmax(logits[:, V + 1:], axis=-1)
        p = mx.softmax(tok_logits, axis=-1)
        conf = 1.0 - (-mx.sum(p * mx.log(p + 1e-10), axis=-1)) / log_vocab
        mx.eval(tok, dur, conf)
        tok_l, dur_l, conf_l = tok.tolist(), dur.tolist(), conf.tolist()

        emitted, emitted_tok = [], []
        still = []
        for k, b in enumerate(active):
            t, d = tok_l[k], durations[dur_l[k]]
            if t != V:
                out[b].append((t, step[b] * ratio, d * ratio, float(conf_l[k])))
                emitted.append(b)
                emitted_tok.append(t)
            step[b] += d
            if d != 0:
                stuck[b] = 0
            else:
                stuck[b] += 1
                if max_symbols is not None and stuck[b] >= max_symbols:
                    step[b] += 1
                    stuck[b] = 0
            if step[b] < lengths[b]:
                still.append(b)

        if emitted:
            ie = mx.array(emitted)
            y = embed(mx.array(emitted_tok))
            o, hn, cn = _lstm_step(rnn, y, h[:, ie], c[:, ie])
            h[:, ie] = hn.astype(dtype)
            c[:, ie] = cn.astype(dtype)
            dec_proj[ie] = joint.pred(o.astype(dtype))
        active = still
    return out
