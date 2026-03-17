"""Prosody analysis: pitch, rhythm, and emphasis pattern extraction."""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from voicetwin.models import AudioSample, ProsodyFeatures


class ProsodyAnalyzer:
    """Extracts prosodic features (pitch, rhythm, emphasis) from audio.

    Uses autocorrelation-based pitch detection and energy analysis to
    characterize the speaking style of a voice sample.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        frame_length: int = 1024,
        hop_length: int = 256,
        pitch_min_hz: float = 50.0,
        pitch_max_hz: float = 600.0,
        emphasis_threshold: float = 1.5,
    ):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.pitch_min_hz = pitch_min_hz
        self.pitch_max_hz = pitch_max_hz
        self.emphasis_threshold = emphasis_threshold

    def analyze(self, sample: AudioSample) -> ProsodyFeatures:
        """Analyze prosodic features of an audio sample.

        Args:
            sample: An AudioSample instance.

        Returns:
            ProsodyFeatures with pitch contour, energy, rate, and emphasis.
        """
        waveform = sample.waveform
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        pitch_contour = self._extract_pitch(waveform)
        energy_contour = self._extract_energy(waveform)
        speaking_rate = self._estimate_speaking_rate(energy_contour)
        emphasis_indices = self._detect_emphasis(energy_contour, pitch_contour)

        voiced_pitch = pitch_contour[pitch_contour > 0]
        pitch_mean = float(np.mean(voiced_pitch)) if len(voiced_pitch) > 0 else 0.0
        pitch_std = float(np.std(voiced_pitch)) if len(voiced_pitch) > 0 else 0.0

        return ProsodyFeatures(
            pitch_contour=pitch_contour,
            energy_contour=energy_contour,
            speaking_rate=speaking_rate,
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            emphasis_indices=emphasis_indices,
        )

    def _extract_pitch(self, waveform: np.ndarray) -> np.ndarray:
        """Extract pitch (F0) contour using autocorrelation method."""
        min_lag = int(self.sample_rate / self.pitch_max_hz)
        max_lag = int(self.sample_rate / self.pitch_min_hz)
        num_frames = (len(waveform) - self.frame_length) // self.hop_length + 1
        pitch = np.zeros(num_frames, dtype=np.float32)

        for i in range(num_frames):
            start = i * self.hop_length
            frame = waveform[start : start + self.frame_length]

            if np.max(np.abs(frame)) < 1e-6:
                continue

            # Normalized autocorrelation
            frame = frame - np.mean(frame)
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2 :]
            if corr[0] > 0:
                corr = corr / corr[0]

            # Search for peak in valid lag range
            search_start = min(min_lag, len(corr) - 1)
            search_end = min(max_lag, len(corr))
            if search_start >= search_end:
                continue

            segment = corr[search_start:search_end]
            if len(segment) == 0:
                continue

            peak_idx = np.argmax(segment)
            peak_val = segment[peak_idx]

            if peak_val > 0.3:
                lag = search_start + peak_idx
                pitch[i] = self.sample_rate / lag if lag > 0 else 0.0

        return pitch

    def _extract_energy(self, waveform: np.ndarray) -> np.ndarray:
        """Extract RMS energy contour."""
        num_frames = (len(waveform) - self.frame_length) // self.hop_length + 1
        energy = np.zeros(num_frames, dtype=np.float32)

        for i in range(num_frames):
            start = i * self.hop_length
            frame = waveform[start : start + self.frame_length]
            energy[i] = np.sqrt(np.mean(frame ** 2))

        return energy

    def _estimate_speaking_rate(self, energy: np.ndarray) -> float:
        """Estimate speaking rate in syllables per second from energy envelope.

        Counts energy peaks as approximate syllable nuclei.
        """
        if len(energy) < 3:
            return 0.0

        # Smooth energy
        kernel_size = min(5, len(energy))
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(energy, kernel, mode="same")

        # Find peaks above mean
        threshold = np.mean(smoothed) * 0.5
        above = smoothed > threshold

        # Count rising edges as syllable onsets
        transitions = np.diff(above.astype(int))
        syllable_count = int(np.sum(transitions > 0))

        duration_seconds = len(energy) * self.hop_length / self.sample_rate
        if duration_seconds > 0:
            return syllable_count / duration_seconds
        return 0.0

    def _detect_emphasis(
        self, energy: np.ndarray, pitch: np.ndarray
    ) -> list[int]:
        """Detect frames with emphasis/stress based on energy and pitch spikes."""
        if len(energy) == 0:
            return []

        mean_energy = np.mean(energy)
        std_energy = np.std(energy)
        if std_energy < 1e-8:
            return []

        min_len = min(len(energy), len(pitch))
        emphasis = []
        for i in range(min_len):
            energy_z = (energy[i] - mean_energy) / (std_energy + 1e-8)
            if energy_z > self.emphasis_threshold:
                emphasis.append(i)
            elif pitch[i] > 0:
                voiced_pitch = pitch[pitch > 0]
                if len(voiced_pitch) > 0:
                    pitch_mean = np.mean(voiced_pitch)
                    pitch_std = np.std(voiced_pitch)
                    if pitch_std > 0:
                        pitch_z = (pitch[i] - pitch_mean) / pitch_std
                        if pitch_z > self.emphasis_threshold:
                            emphasis.append(i)

        return emphasis
