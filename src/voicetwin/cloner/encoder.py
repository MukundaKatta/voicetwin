"""Speaker encoder that extracts voice embeddings from audio using mel-spectrograms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from voicetwin.features.mel import MelSpectrogramExtractor
from voicetwin.models import AudioSample, VoiceProfile


class SpeakerEncoderNetwork(nn.Module):
    """Neural network for speaker embedding extraction.

    Architecture: 3-layer LSTM followed by a linear projection to produce
    a fixed-size d-vector from variable-length mel-spectrogram input.
    """

    def __init__(
        self,
        n_mels: int = 80,
        hidden_size: int = 256,
        embedding_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_size, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            mel: Mel-spectrogram tensor of shape (batch, time, n_mels).

        Returns:
            Speaker embedding of shape (batch, embedding_dim).
        """
        # LSTM over time steps
        output, (hidden, _) = self.lstm(mel)
        # Take the last hidden state from the top layer
        embedding = self.projection(hidden[-1])
        embedding = self.relu(embedding)
        # L2 normalize
        embedding = embedding / (torch.norm(embedding, dim=1, keepdim=True) + 1e-8)
        return embedding


class SpeakerEncoder:
    """High-level speaker encoder: loads audio, extracts mel-spectrogram,
    and produces a speaker embedding (d-vector).

    Args:
        n_mels: Number of mel bands.
        embedding_dim: Dimensionality of the output embedding.
        sample_rate: Expected audio sample rate.
        device: Torch device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        n_mels: int = 80,
        embedding_dim: int = 256,
        sample_rate: int = 22050,
        device: str = "cpu",
    ):
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        self.sample_rate = sample_rate
        self.device = torch.device(device)

        self.mel_extractor = MelSpectrogramExtractor(
            sample_rate=sample_rate,
            n_mels=n_mels,
        )
        self.network = SpeakerEncoderNetwork(
            n_mels=n_mels,
            embedding_dim=embedding_dim,
        ).to(self.device)
        self.network.eval()

    def encode(self, audio_path: str | Path) -> np.ndarray:
        """Encode audio file into a speaker embedding.

        Args:
            audio_path: Path to a WAV audio file.

        Returns:
            Speaker embedding as a numpy array of shape (embedding_dim,).
        """
        import librosa

        waveform, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        return self.encode_waveform(waveform)

    def encode_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """Encode a raw waveform array into a speaker embedding.

        Args:
            waveform: 1-D numpy array of audio samples.

        Returns:
            Speaker embedding as a numpy array of shape (embedding_dim,).
        """
        mel = self.mel_extractor.extract(waveform)
        # mel shape: (n_mels, time) -> transpose to (time, n_mels), add batch dim
        mel_tensor = torch.from_numpy(mel.T).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.network(mel_tensor)

        return embedding.squeeze(0).cpu().numpy()

    def build_profile(
        self, audio_path: str | Path, speaker_id: str = "unknown"
    ) -> VoiceProfile:
        """Build a full voice profile from an audio file.

        Args:
            audio_path: Path to a WAV audio file.
            speaker_id: Identifier for the speaker.

        Returns:
            VoiceProfile with embedding and metadata.
        """
        import librosa

        waveform, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        embedding = self.encode_waveform(waveform)
        duration = len(waveform) / self.sample_rate

        return VoiceProfile(
            speaker_id=speaker_id,
            embedding=embedding,
            embedding_dim=self.embedding_dim,
            source_duration_seconds=duration,
        )

    def load_weights(self, path: str | Path) -> None:
        """Load pre-trained encoder weights."""
        state_dict = torch.load(str(path), map_location=self.device)
        self.network.load_state_dict(state_dict)
        self.network.eval()

    def save_weights(self, path: str | Path) -> None:
        """Save encoder weights."""
        torch.save(self.network.state_dict(), str(path))
