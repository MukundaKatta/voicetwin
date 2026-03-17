"""Tests for voice synthesizer."""

import numpy as np
import pytest

from voicetwin.cloner.synthesizer import VoiceSynthesizer


class TestVoiceSynthesizer:
    def setup_method(self):
        self.synth = VoiceSynthesizer(n_mels=80, embedding_dim=256)

    def test_synthesize_output_type(self):
        embedding = np.random.randn(256).astype(np.float32)
        result = self.synth.synthesize("Hello world", embedding)
        assert result.mel_spectrogram.shape[0] == 80
        assert result.text == "Hello world"
        assert result.duration_seconds > 0

    def test_synthesize_with_speaker_embedding(self):
        embedding = np.random.randn(256).astype(np.float32)
        result = self.synth.synthesize("Test", embedding)
        assert result.speaker_embedding is not None
        np.testing.assert_array_equal(result.speaker_embedding, embedding)

    def test_synthesize_empty_text(self):
        embedding = np.random.randn(256).astype(np.float32)
        result = self.synth.synthesize("", embedding, max_duration_seconds=1.0)
        assert result.mel_spectrogram.shape[0] == 80

    def test_max_duration_limit(self):
        embedding = np.random.randn(256).astype(np.float32)
        long_text = "word " * 1000
        result = self.synth.synthesize(long_text, embedding, max_duration_seconds=2.0)
        # Should not exceed max duration by much
        assert result.duration_seconds <= 3.0

    def test_text_to_ids(self):
        ids = self.synth._text_to_ids("abc")
        assert ids.shape == (1, 3)
        assert ids[0, 0].item() == ord("a")
