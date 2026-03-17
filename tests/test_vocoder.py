"""Tests for waveform generator (vocoder)."""

import numpy as np
import pytest

from voicetwin.cloner.vocoder import WaveformGenerator
from voicetwin.models import SynthesisResult


class TestWaveformGenerator:
    def setup_method(self):
        self.vocoder = WaveformGenerator(n_mels=80)

    def test_generate_output_shape(self):
        mel = np.random.randn(80, 50).astype(np.float32)
        audio = self.vocoder.generate(mel)
        assert audio.ndim == 1
        assert len(audio) > 0

    def test_generate_normalized(self):
        mel = np.random.randn(80, 50).astype(np.float32)
        audio = self.vocoder.generate(mel)
        assert np.max(np.abs(audio)) <= 1.0 + 1e-6

    def test_generate_from_result(self):
        mel = np.random.randn(80, 50).astype(np.float32)
        result = SynthesisResult(
            mel_spectrogram=mel,
            text="test",
            duration_seconds=1.0,
        )
        audio = self.vocoder.generate_from_result(result)
        assert audio.ndim == 1
        assert len(audio) > 0

    def test_save_load_weights(self, tmp_path):
        path = tmp_path / "vocoder.pt"
        self.vocoder.save_weights(path)

        vocoder2 = WaveformGenerator(n_mels=80)
        vocoder2.load_weights(path)

        mel = np.random.randn(80, 20).astype(np.float32)
        a1 = self.vocoder.generate(mel)
        a2 = vocoder2.generate(mel)
        np.testing.assert_array_almost_equal(a1, a2)
