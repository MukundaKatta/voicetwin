"""Tests for voice quality analysis."""

import numpy as np
import pytest

from voicetwin.features.quality import VoiceQualityAnalyzer
from voicetwin.models import AudioSample, VoiceProfile


class TestVoiceQualityAnalyzer:
    def setup_method(self):
        self.analyzer = VoiceQualityAnalyzer(sample_rate=22050)

    def test_clarity_range(self):
        waveform = np.random.randn(22050).astype(np.float32) * 0.3
        sample = AudioSample.from_array(waveform, sample_rate=22050)
        clarity = self.analyzer.measure_clarity(sample)
        assert 0.0 <= clarity <= 1.0

    def test_naturalness_range(self):
        waveform = np.random.randn(22050).astype(np.float32) * 0.3
        sample = AudioSample.from_array(waveform, sample_rate=22050)
        naturalness = self.analyzer.measure_naturalness(sample)
        assert 0.0 <= naturalness <= 1.0

    def test_similarity_identical_profiles(self):
        embedding = np.random.randn(256).astype(np.float32)
        p1 = VoiceProfile(embedding=embedding)
        p2 = VoiceProfile(embedding=embedding.copy())
        sim = self.analyzer.measure_similarity(p1, p2)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_full_analysis_keys(self):
        waveform = np.random.randn(22050).astype(np.float32) * 0.3
        sample = AudioSample.from_array(waveform, sample_rate=22050)
        metrics = self.analyzer.full_analysis(sample)
        assert "clarity" in metrics
        assert "naturalness" in metrics
        assert "snr_db" in metrics

    def test_silent_input(self):
        waveform = np.zeros(22050, dtype=np.float32)
        sample = AudioSample.from_array(waveform, sample_rate=22050)
        clarity = self.analyzer.measure_clarity(sample)
        assert clarity == 0.0

    def test_short_input(self):
        waveform = np.random.randn(100).astype(np.float32)
        sample = AudioSample.from_array(waveform, sample_rate=22050)
        naturalness = self.analyzer.measure_naturalness(sample)
        assert 0.0 <= naturalness <= 1.0
