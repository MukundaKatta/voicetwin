"""Tests for voice profile reporting."""

import json

import numpy as np
import pytest

from voicetwin.models import AudioSample, VoiceProfile
from voicetwin.report import VoiceProfileReport, _make_serializable


class TestVoiceProfileReport:
    def test_generate_report(self):
        waveform = np.random.randn(22050).astype(np.float32) * 0.3
        sample = AudioSample.from_array(waveform, sample_rate=22050)
        profile = VoiceProfile(
            speaker_id="test",
            embedding=np.random.randn(256).astype(np.float32),
            source_duration_seconds=1.0,
        )

        reporter = VoiceProfileReport(sample_rate=22050)
        report = reporter.generate(sample, profile)

        assert report["speaker_id"] == "test"
        assert "pitch" in report
        assert "prosody" in report
        assert "quality" in report

    def test_save_json(self, tmp_path):
        waveform = np.random.randn(22050).astype(np.float32) * 0.3
        sample = AudioSample.from_array(waveform, sample_rate=22050)
        profile = VoiceProfile(
            speaker_id="test",
            embedding=np.random.randn(256).astype(np.float32),
            source_duration_seconds=1.0,
        )

        reporter = VoiceProfileReport(sample_rate=22050)
        report = reporter.generate(sample, profile)

        out_path = tmp_path / "report.json"
        reporter.save_json(report, out_path)

        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded["speaker_id"] == "test"


class TestMakeSerializable:
    def test_numpy_float(self):
        assert isinstance(_make_serializable(np.float32(1.0)), float)

    def test_numpy_int(self):
        assert isinstance(_make_serializable(np.int64(42)), int)

    def test_numpy_array(self):
        result = _make_serializable(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_nested_dict(self):
        data = {"a": np.float32(1.0), "b": {"c": np.array([1, 2])}}
        result = _make_serializable(data)
        assert result == {"a": 1.0, "b": {"c": [1, 2]}}
