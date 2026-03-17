"""Tests for speaker encoder."""

import numpy as np
import pytest

from voicetwin.cloner.encoder import SpeakerEncoder, SpeakerEncoderNetwork


class TestSpeakerEncoderNetwork:
    def test_output_shape(self):
        import torch

        net = SpeakerEncoderNetwork(n_mels=80, embedding_dim=256)
        mel = torch.randn(1, 50, 80)  # (batch, time, n_mels)
        embedding = net(mel)
        assert embedding.shape == (1, 256)

    def test_normalized_output(self):
        import torch

        net = SpeakerEncoderNetwork(n_mels=80, embedding_dim=256)
        mel = torch.randn(1, 50, 80)
        embedding = net(mel)
        norm = torch.norm(embedding, dim=1)
        assert norm.item() == pytest.approx(1.0, abs=0.1)

    def test_batch_processing(self):
        import torch

        net = SpeakerEncoderNetwork(n_mels=80, embedding_dim=128)
        mel = torch.randn(4, 30, 80)
        embeddings = net(mel)
        assert embeddings.shape == (4, 128)


class TestSpeakerEncoder:
    def test_encode_waveform(self):
        encoder = SpeakerEncoder(embedding_dim=256)
        waveform = np.random.randn(22050).astype(np.float32) * 0.1
        embedding = encoder.encode_waveform(waveform)
        assert embedding.shape == (256,)
        assert embedding.dtype == np.float32

    def test_embedding_consistency(self):
        encoder = SpeakerEncoder(embedding_dim=256)
        waveform = np.random.randn(22050).astype(np.float32) * 0.1
        e1 = encoder.encode_waveform(waveform)
        e2 = encoder.encode_waveform(waveform)
        np.testing.assert_array_almost_equal(e1, e2)

    def test_save_load_weights(self, tmp_path):
        encoder = SpeakerEncoder(embedding_dim=128)
        path = tmp_path / "encoder.pt"
        encoder.save_weights(path)

        encoder2 = SpeakerEncoder(embedding_dim=128)
        encoder2.load_weights(path)

        waveform = np.random.randn(22050).astype(np.float32) * 0.1
        e1 = encoder.encode_waveform(waveform)
        e2 = encoder2.encode_waveform(waveform)
        np.testing.assert_array_almost_equal(e1, e2)
