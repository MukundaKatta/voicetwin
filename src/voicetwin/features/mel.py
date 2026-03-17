"""Mel-spectrogram extraction with configurable FFT, hop, and mel parameters."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class MelSpectrogramExtractor:
    """Extract mel-spectrograms from raw audio waveforms.

    Converts time-domain audio into mel-frequency spectrograms suitable for
    speaker encoding, synthesis, and vocoding.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int | None = None,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float | None = None,
        power: float = 1.0,
        normalized: bool = False,
        ref_db: float = 20.0,
        min_db: float = -100.0,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2.0
        self.power = power
        self.normalized = normalized
        self.ref_db = ref_db
        self.min_db = min_db

        self._mel_basis: torch.Tensor | None = None

    def _build_mel_filterbank(self) -> torch.Tensor:
        """Build a mel-scale filterbank matrix."""
        n_freqs = self.n_fft // 2 + 1

        # Convert Hz to mel scale
        def hz_to_mel(hz: float) -> float:
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel: float) -> float:
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_min = hz_to_mel(self.f_min)
        mel_max = hz_to_mel(self.f_max)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = np.array([mel_to_hz(m) for m in mel_points])
        bin_indices = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        filterbank = np.zeros((self.n_mels, n_freqs), dtype=np.float32)
        for i in range(self.n_mels):
            left = bin_indices[i]
            center = bin_indices[i + 1]
            right = bin_indices[i + 2]
            for j in range(left, center):
                if center != left:
                    filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right != center:
                    filterbank[i, j] = (right - j) / (right - center)

        return torch.from_numpy(filterbank)

    def _get_mel_basis(self) -> torch.Tensor:
        if self._mel_basis is None:
            self._mel_basis = self._build_mel_filterbank()
        return self._mel_basis

    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """Extract mel-spectrogram from a waveform.

        Args:
            waveform: 1-D numpy array of audio samples.

        Returns:
            Mel-spectrogram as numpy array of shape (n_mels, time_frames).
        """
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        audio = torch.from_numpy(waveform.astype(np.float32))

        # Pad signal for complete frames
        pad_amount = self.n_fft // 2
        audio = F.pad(audio, (pad_amount, pad_amount), mode="reflect")

        # STFT
        window = torch.hann_window(self.win_length)
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        magnitudes = stft.abs()
        if self.power != 1.0:
            magnitudes = magnitudes.pow(self.power)

        # Apply mel filterbank
        mel_basis = self._get_mel_basis()
        mel_spec = torch.matmul(mel_basis, magnitudes)

        # Convert to log scale (dB)
        mel_spec = self._amplitude_to_db(mel_spec)

        if self.normalized:
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

        return mel_spec.numpy()

    def _amplitude_to_db(self, magnitudes: torch.Tensor) -> torch.Tensor:
        """Convert amplitude spectrogram to dB scale."""
        log_spec = self.ref_db * torch.log10(torch.clamp(magnitudes, min=1e-10))
        log_spec = torch.clamp(log_spec, min=self.min_db)
        return log_spec

    def num_frames(self, num_samples: int) -> int:
        """Calculate number of time frames for a given number of samples."""
        return (num_samples + self.n_fft) // self.hop_length
