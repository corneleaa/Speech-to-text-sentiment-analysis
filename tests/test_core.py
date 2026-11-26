"""Tests that need no model downloads."""
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.audio_input import AudioValidationError, load_audio_bytes, load_audio_path
from modules.sentiment_analysis import SentimentAnalyzer, SentimentError, normalize_label
from modules.visualization import build_record, record_to_csv, record_to_json, save_results

SCORES = {"positive": 70.0, "neutral": 20.0, "negative": 10.0}


def make_wav(path, seconds=1.0, amplitude=0.3):
    t = np.linspace(0, seconds, int(22050 * seconds), endpoint=False)
    sf.write(path, amplitude * np.sin(2 * np.pi * 440 * t), 22050)


def test_normalize_label_names_and_ids():
    assert normalize_label("POSITIVE") == "positive"
    assert normalize_label("LABEL_0") == "negative"
    assert normalize_label("LABEL_2") == "positive"
    with pytest.raises(SentimentError):
        normalize_label("joy")


def test_empty_text_rejected_before_model_load():
    with pytest.raises(SentimentError):
        SentimentAnalyzer(device="cpu").analyze("   ")


def test_audio_is_resampled_to_16k(tmp_path):
    path = tmp_path / "tone.wav"
    make_wav(path, 1.0)
    audio = load_audio_path(path)
    assert audio.sample_rate == 16000
    assert audio.samples.dtype == np.float32
    assert abs(audio.duration - 1.0) < 0.05


def test_audio_bytes_roundtrip(tmp_path):
    path = tmp_path / "tone.wav"
    make_wav(path)
    assert load_audio_bytes(path.read_bytes(), "tone.wav").duration > 0.9


def test_silence_and_bad_format_rejected(tmp_path):
    silent = tmp_path / "silent.wav"
    make_wav(silent, amplitude=0.0)
    with pytest.raises(AudioValidationError):
        load_audio_path(silent)
    with pytest.raises(AudioValidationError):
        load_audio_bytes(b"abc", "notes.txt")
    with pytest.raises(AudioValidationError):
        load_audio_path(tmp_path / "missing.wav")


def make_record():
    return build_record(
        source="a.wav", duration=1.0, language="lt", transcript="Labas, kaip sekasi?",
        sentiment="positive", scores=SCORES, whisper_model="small",
        sentiment_model="m", device="cpu",
    )


def test_record_exports(tmp_path):
    record = make_record()
    assert json.loads(record_to_json(record))["transcript"] == "Labas, kaip sekasi?"
    rows = list(csv.DictReader(io.StringIO(record_to_csv(record).decode("utf-8-sig"))))
    assert rows[0]["positive_pct"] == "70.0"
    paths = save_results(record, tmp_path)
    assert all(p.exists() for p in paths.values())


def test_default_device_is_cpu():
    import config
    assert config.get_device() == "cpu"


def test_transcribe_joins_segments_without_model_download(tmp_path):
    from types import SimpleNamespace
    from modules.speech_to_text import SpeechToText

    class FakeModel:
        def transcribe(self, samples, **kwargs):
            segs = iter([SimpleNamespace(start=0, end=1, text=" Labas "),
                         SimpleNamespace(start=1, end=2, text="pasauli")])
            return segs, SimpleNamespace(language="lt")

    path = tmp_path / "tone.wav"
    make_wav(path)
    stt = SpeechToText("tiny", "cpu")
    stt._model = FakeModel()
    result = stt.transcribe(load_audio_path(path), "lt")
    assert result.text == "Labas pasauli" and result.language == "lt"
