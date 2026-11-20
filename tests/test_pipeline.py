"""End-to-end pipeline with a fake engine: no lost or duplicated words across chunks,
all formats written, raw transcript immutable, decoding of real containers."""

import json
import os
import stat

import numpy as np
import pytest

from hearken import audio, engines, models, pipeline, presets
from hearken.errors import AudioError
from hearken.presets import EngineSpec, Preset

from .conftest import SR, BurstEngine, burst_signal


def _write(path, x, sr=SR, channels=1, codec=None):
    import av
    fmt = {"wav": ("wav", "pcm_s16le"), "m4a": ("mp4", "aac"), "mp3": ("mp3", "libmp3lame"),
           "mp4": ("mp4", "aac")}[path.suffix[1:]]
    with av.open(str(path), "w", format=fmt[0]) as c:
        s = c.add_stream(codec or fmt[1], rate=sr)
        s.layout = "stereo" if channels == 2 else "mono"
        data = np.tile(x, (channels, 1)) if channels == 2 else x[None, :]
        pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16)
        step = 1024
        for i in range(0, pcm.shape[1], step):
            blk = pcm[:, i:i + step]
            fr = __import__("av").AudioFrame.from_ndarray(
                blk.T.reshape(1, -1) if channels == 2 else blk, format="s16", layout=s.layout.name)
            fr.sample_rate = sr
            for p in s.encode(fr):
                c.mux(p)
        for p in s.encode(None):
            c.mux(p)


@pytest.fixture
def fake_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARKEN_HOME", str(tmp_path / "home"))
    monkeypatch.setitem(presets.PRESETS, "fake", Preset("fake", "test", (EngineSpec("burst", {}),)))
    monkeypatch.setitem(presets.PRESETS, "fake2", Preset("fake2", "test", (EngineSpec("burst", {}),
                                                                          EngineSpec("burst", {"max_chunk_s": 13.0}))))
    monkeypatch.setattr(presets, "model_keys", lambda p: [])
    created = []

    def create(name, **kw):
        e = BurstEngine(**kw)
        created.append(e)
        return e
    monkeypatch.setattr(engines, "create", create)
    return created


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("pad", [0.0, 2.0])
def test_no_word_lost_or_duplicated_across_chunks(fake_env, tmp_path, pad, fixed, batch):
    x, onsets = burst_signal(60, gap=0.9, long_pause_every=7, long_pause=1.5)
    src = tmp_path / "lecture.wav"
    _write(src, x)
    opts = pipeline.RunOptions(preset="fake", formats=("json",))
    over = {"pad_s": 0.0001} if pad == 0.0 else {}
    if fixed:
        over["fixed_window"] = True
    over["batch_size"] = batch
    opts.engine_overrides = {"burst": over}
    res = pipeline.transcribe_file(src, tmp_path / "out", opts)
    doc = json.loads(res.outputs["json"].read_text())
    names = [w["text"] for w in doc["words"]]
    assert names == [f"w{i}" for i in range(60)]
    assert fake_env[0].calls > 2  # really chunked
    for w, o in zip(doc["words"], onsets):
        assert abs(w["start"] - o) < 0.03


def test_all_formats_and_raw_immutability(fake_env, tmp_path):
    x, _ = burst_signal(20)
    src = tmp_path / "talk.m4a"
    _write(src, x, sr=44100 if False else SR)
    out = tmp_path / "out"
    opts = pipeline.RunOptions(preset="fake2", formats=pipeline.FORMATS)
    r1 = pipeline.transcribe_file(src, out, opts)
    for f in pipeline.FORMATS:
        assert r1.outputs[f].is_file() and r1.outputs[f].stat().st_size > 0
    mode = r1.raw_path.stat().st_mode
    assert not (mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    raw_bytes = r1.raw_path.read_bytes()
    doc = json.loads(r1.outputs["json"].read_text())
    assert doc["raw_transcript"]["file"] == r1.raw_path.name
    assert [e["role"] for e in doc["engines"]] == ["primary", "verifier"]
    r2 = pipeline.transcribe_file(src, out, opts)
    assert r2.raw_path != r1.raw_path and r1.raw_path.read_bytes() == raw_bytes
    # reformat from raw reproduces the same words
    again = pipeline.render_from_raw(pipeline.load_raw(r1.raw_path), r1.raw_path, tmp_path / "re", "talk",
                                     pipeline.RunOptions(formats=("json",)), "talk.m4a")
    assert json.loads(again["json"].read_text())["words"] == doc["words"]
    assert not any(p.name.startswith("hearken-audio-") for p in out.iterdir())  # temp audio cleaned


@pytest.mark.parametrize("suffix,sr,ch", [("wav", 44100, 2), ("mp3", 44100, 2), ("m4a", 48000, 1), ("mp4", 22050, 1)])
def test_decode_containers(tmp_path, suffix, sr, ch):
    t = np.arange(int(3.0 * sr)) / sr
    x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    p = tmp_path / f"a.{suffix}"
    _write(p, x, sr=sr, channels=ch)
    with audio.decode(p, workdir=tmp_path) as d:
        assert abs(d.duration - 3.0) < 0.1
        assert 0.15 < float(np.max(np.abs(d.samples))) < 0.35
        a = audio.analyze(d)
        assert a.clipped_fraction == 0


def test_decode_errors(tmp_path):
    with pytest.raises(AudioError):
        audio.decode(tmp_path / "missing.wav")
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not audio at all" * 100)
    with pytest.raises(AudioError):
        audio.decode(bad)


def test_quiet_audio_gets_linear_gain_only():
    x = (0.01 * np.sin(np.linspace(0, 2000, SR * 2))).astype(np.float32)
    d = audio.from_array(x)
    a = audio.analyze(d)
    audio.preprocess(d, a)
    assert any(s.startswith("linear_gain") for s in a.applied)
    ratio = d.samples[100:200] / x[100:200]
    assert np.allclose(ratio, ratio[0], rtol=1e-4)  # pure scaling, shape unchanged


def test_missing_model_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty"))
    monkeypatch.setitem(models.MODELS, "nope", models.ModelSpec("nope", "x", "nobody/nothing", None, "-", "-"))
    from hearken.errors import ModelNotAvailableError
    with pytest.raises(ModelNotAvailableError):
        models.resolve_local("nope")


def test_review_page_links_media_from_default_output_dir(fake_env, tmp_path):
    x, _ = burst_signal(5)
    src = tmp_path / "My Lecture.wav"
    _write(src, x)
    out = tmp_path / "My Lecture.transcript"
    r = pipeline.transcribe_file(src, out, pipeline.RunOptions(preset="fake", formats=("html",)))
    html = r.outputs["html"].read_text()
    assert 'src="../My%20Lecture.wav"' in html
    assert (out / "../My Lecture.wav").resolve() == src.resolve()
