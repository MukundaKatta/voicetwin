"""Waveform generator (vocoder): converts mel-spectrograms to audio waveforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from voicetwin.models import SynthesisResult


class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions for waveform generation."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size * dilation - dilation) // 2
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=padding
        )
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.leaky_relu(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class VocoderNetwork(nn.Module):
    """Neural vocoder network using upsampling convolutions and residual blocks.

    Converts mel-spectrograms into raw audio waveforms.
    """

    def __init__(
        self,
        n_mels: int = 80,
        upsample_rates: tuple[int, ...] = (8, 8, 2, 2),
        upsample_channels: int = 256,
        residual_channels: int = 256,
        num_residual_blocks: int = 3,
    ):
        super().__init__()
        self.input_conv = nn.Conv1d(n_mels, upsample_channels, kernel_size=7, padding=3)

        # Upsampling layers
        self.upsample_layers = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()

        channels = upsample_channels
        for rate in upsample_rates:
            self.upsample_layers.append(
                nn.ConvTranspose1d(
                    channels,
                    channels // 2,
                    kernel_size=rate * 2,
                    stride=rate,
                    padding=rate // 2,
                )
            )
            channels //= 2

            # Add residual blocks after each upsample
            for i in range(num_residual_blocks):
                dilation = 3 ** i
                self.residual_blocks.append(
                    ResidualBlock(channels, kernel_size=3, dilation=dilation)
                )

        self.output_conv = nn.Conv1d(channels, 1, kernel_size=7, padding=3)
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate waveform from mel-spectrogram.

        Args:
            mel: (batch, n_mels, time) mel-spectrogram tensor.

        Returns:
            (batch, 1, samples) waveform tensor.
        """
        x = self.input_conv(mel)

        res_idx = 0
        num_res_per_upsample = len(self.residual_blocks) // len(self.upsample_layers)

        for upsample in self.upsample_layers:
            x = self.leaky_relu(x)
            x = upsample(x)
            for _ in range(num_res_per_upsample):
                x = self.residual_blocks[res_idx](x)
                res_idx += 1

        x = self.leaky_relu(x)
        x = self.output_conv(x)
        x = self.tanh(x)
        return x


class WaveformGenerator:
    """High-level vocoder that converts mel-spectrograms to audio waveforms.

    Wraps the VocoderNetwork and provides convenience methods for generating
    and saving audio.

    Args:
        n_mels: Number of mel channels (must match synthesizer).
        sample_rate: Output audio sample rate.
        device: Torch device.
    """

    def __init__(
        self,
        n_mels: int = 80,
        sample_rate: int = 22050,
        device: str = "cpu",
    ):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.device = torch.device(device)

        self.network = VocoderNetwork(n_mels=n_mels).to(self.device)
        self.network.eval()

    def generate(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Convert a mel-spectrogram to an audio waveform.

        Args:
            mel_spectrogram: Numpy array of shape (n_mels, time_frames).

        Returns:
            1-D numpy array of audio samples.
        """
        mel_tensor = (
            torch.from_numpy(mel_spectrogram)
            .float()
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            waveform = self.network(mel_tensor)

        audio = waveform.squeeze().cpu().numpy()
        # Normalize to [-1, 1]
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        return audio

    def generate_from_result(self, result: SynthesisResult) -> np.ndarray:
        """Generate waveform from a SynthesisResult."""
        return self.generate(result.mel_spectrogram)

    def save(self, path: str | Path, audio: np.ndarray, sample_rate: int | None = None) -> None:
        """Save audio waveform to a WAV file.

        Args:
            path: Output file path.
            audio: 1-D numpy array of audio samples.
            sample_rate: Sample rate (defaults to self.sample_rate).
        """
        import soundfile as sf

        sr = sample_rate or self.sample_rate
        sf.write(str(path), audio, sr)

    def load_weights(self, path: str | Path) -> None:
        """Load pre-trained vocoder weights."""
        state_dict = torch.load(str(path), map_location=self.device)
        self.network.load_state_dict(state_dict)
        self.network.eval()

    def save_weights(self, path: str | Path) -> None:
        """Save vocoder weights."""
        torch.save(self.network.state_dict(), str(path))
