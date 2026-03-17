"""Tests for VOICETWIN data models."""

import numpy as np
import pytest

from voicetwin.models import AudioSample, ProsodyFeatures, SynthesisResult, VoiceProfile


class TestAudioSample:
    def test_from_array_mono(self):
        waveform = np.random.randn(22050).astype(np.float32)
        sample = AudioSample.from_array(waveform, sample_rate=22050)
        assert sample.duration_seconds == pytest.approx(1.0)
        assert sample.num_channels == 1
        assert sample.num_samples == 22050

    def test_from_array_stereo(self):
        waveform = np.random.randn(2, 22050).astype(np.float32)
        sample = AudioSample.from_array(waveform, sample_rate=22050)
        assert sample.num_channels == 2

    def test_from_array_with_path(self, tmp_path):
        waveform = np.zeros(1000, dtype=np.float32)
        path = tmp_path / "test.wav"
        sample = AudioSample.from_array(waveform, file_path=path)
        assert sample.file_path == path


class TestVoiceProfile:
    def test_similarity_identical(self):
        embedding = np.random.randn(256).astype(np.float32)
        p1 = VoiceProfile(embedding=embedding)
        p2 = VoiceProfile(embedding=embedding.copy())
        assert p1.similarity(p2) == pytest.approx(1.0, abs=1e-5)

    def test_similarity_orthogonal(self):
        e1 = np.zeros(256, dtype=np.float32)
        e1[0] = 1.0
        e2 = np.zeros(256, dtype=np.float32)
        e2[1] = 1.0
        p1 = VoiceProfile(embedding=e1)
        p2 = VoiceProfile(embedding=e2)
        assert p1.similarity(p2) == pytest.approx(0.0, abs=1e-5)

    def test_similarity_opposite(self):
        e1 = np.random.randn(256).astype(np.float32)
        e2 = -e1
        p1 = VoiceProfile(embedding=e1)
        p2 = VoiceProfile(embedding=e2)
        assert p1.similarity(p2) == pytest.approx(-1.0, abs=1e-5)


class TestSynthesisResult:
    def test_creation(self):
        mel = np.random.randn(80, 100).astype(np.float32)
        result = SynthesisResult(
            mel_spectrogram=mel,
            text="hello",
            duration_seconds=1.0,
        )
        assert result.text == "hello"
        assert result.mel_spectrogram.shape == (80, 100)
        assert result.sample_rate == 22050


class TestProsodyFeatures:
    def test_creation(self):
        features = ProsodyFeatures(
            pitch_contour=np.zeros(100),
            energy_contour=np.ones(100),
            speaking_rate=4.5,
            pitch_mean=150.0,
            pitch_std=30.0,
            emphasis_indices=[10, 50, 80],
        )
        assert features.speaking_rate == 4.5
        assert len(features.emphasis_indices) == 3
