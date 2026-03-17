"""Tests for mel-spectrogram extraction."""

import numpy as np
import pytest

from voicetwin.features.mel import MelSpectrogramExtractor


class TestMelSpectrogramExtractor:
    def setup_method(self):
        self.extractor = MelSpectrogramExtractor(
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
        )

    def test_output_shape(self):
        # 1 second of audio at 22050 Hz
        waveform = np.random.randn(22050).astype(np.float32)
        mel = self.extractor.extract(waveform)
        assert mel.shape[0] == 80  # n_mels
        assert mel.shape[1] > 0  # time frames

    def test_output_dtype(self):
        waveform = np.random.randn(22050).astype(np.float32)
        mel = self.extractor.extract(waveform)
        assert mel.dtype == np.float32

    def test_silent_input(self):
        waveform = np.zeros(22050, dtype=np.float32)
        mel = self.extractor.extract(waveform)
        assert mel.shape[0] == 80
        # Silent input should produce very low dB values
        assert np.all(mel <= 0)

    def test_stereo_to_mono(self):
        stereo = np.random.randn(2, 22050).astype(np.float32)
        mel = self.extractor.extract(stereo)
        assert mel.shape[0] == 80

    def test_configurable_params(self):
        extractor = MelSpectrogramExtractor(
            sample_rate=16000,
            n_fft=512,
            hop_length=128,
            n_mels=40,
        )
        waveform = np.random.randn(16000).astype(np.float32)
        mel = extractor.extract(waveform)
        assert mel.shape[0] == 40

    def test_num_frames(self):
        frames = self.extractor.num_frames(22050)
        assert frames > 0
        assert isinstance(frames, int)

    def test_normalized_output(self):
        extractor = MelSpectrogramExtractor(normalized=True)
        waveform = np.random.randn(22050).astype(np.float32) * 0.5
        mel = extractor.extract(waveform)
        # Normalized should have roughly zero mean
        assert abs(np.mean(mel)) < 0.5
