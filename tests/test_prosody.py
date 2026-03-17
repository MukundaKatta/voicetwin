"""Tests for prosody analysis."""

import numpy as np
import pytest

from voicetwin.features.prosody import ProsodyAnalyzer
from voicetwin.models import AudioSample


def _make_tone(freq_hz: float, duration_s: float = 1.0, sr: int = 22050) -> np.ndarray:
    """Generate a pure sine tone."""
    t = np.arange(int(sr * duration_s)) / sr
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


class TestProsodyAnalyzer:
    def setup_method(self):
        self.analyzer = ProsodyAnalyzer(sample_rate=22050)

    def test_analyze_returns_features(self):
        waveform = _make_tone(200.0, duration_s=1.0)
        sample = AudioSample.from_array(waveform, sample_rate=22050)
        features = self.analyzer.analyze(sample)
        assert features.pitch_contour.ndim == 1
        assert features.energy_contour.ndim == 1
        assert features.speaking_rate >= 0
        assert features.pitch_mean >= 0

    def test_pitch_detection_sine(self):
        freq = 200.0
        waveform = _make_tone(freq, duration_s=1.0)
        sample = AudioSample.from_array(waveform, sample_rate=22050)
        features = self.analyzer.analyze(sample)
        # Voiced frames should detect pitch near the true frequency
        voiced = features.pitch_contour[features.pitch_contour > 0]
        if len(voiced) > 0:
            assert abs(np.median(voiced) - freq) < 50.0

    def test_silent_input(self):
        waveform = np.zeros(22050, dtype=np.float32)
        sample = AudioSample.from_array(waveform, sample_rate=22050)
        features = self.analyzer.analyze(sample)
        assert features.pitch_mean == 0.0

    def test_energy_contour_positive(self):
        waveform = np.random.randn(22050).astype(np.float32) * 0.1
        sample = AudioSample.from_array(waveform, sample_rate=22050)
        features = self.analyzer.analyze(sample)
        assert np.all(features.energy_contour >= 0)
