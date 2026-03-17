"""Voice quality analysis: clarity, naturalness, and similarity metrics."""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from voicetwin.models import AudioSample, VoiceProfile


class VoiceQualityAnalyzer:
    """Measures voice quality attributes: clarity, naturalness, and similarity.

    Provides objective metrics for evaluating both original recordings and
    synthesized voice output.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        frame_length: int = 1024,
        hop_length: int = 256,
    ):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length

    def measure_clarity(self, sample: AudioSample) -> float:
        """Measure voice clarity based on harmonics-to-noise ratio.

        Returns a score between 0.0 (poor) and 1.0 (excellent).
        """
        waveform = sample.waveform
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        if len(waveform) < self.frame_length:
            return 0.0

        hnr_values = []
        num_frames = (len(waveform) - self.frame_length) // self.hop_length + 1

        for i in range(num_frames):
            start = i * self.hop_length
            frame = waveform[start : start + self.frame_length].copy()

            if np.max(np.abs(frame)) < 1e-6:
                continue

            frame = frame - np.mean(frame)
            autocorr = np.correlate(frame, frame, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]

            if autocorr[0] <= 0:
                continue

            # Find first peak after zero crossing as harmonic component
            min_lag = int(self.sample_rate / 600)
            max_lag = int(self.sample_rate / 50)
            search = autocorr[min_lag : min(max_lag, len(autocorr))]

            if len(search) == 0:
                continue

            peak_val = np.max(search)
            harmonic_power = max(peak_val, 0)
            noise_power = max(autocorr[0] - harmonic_power, 1e-10)

            hnr = harmonic_power / noise_power
            hnr_values.append(hnr)

        if not hnr_values:
            return 0.0

        mean_hnr = np.mean(hnr_values)
        # Map HNR to 0-1 range (HNR of ~10 is good, ~20+ is excellent)
        clarity = float(np.clip(mean_hnr / 2.0, 0.0, 1.0))
        return clarity

    def measure_naturalness(self, sample: AudioSample) -> float:
        """Measure voice naturalness based on spectral variation and dynamics.

        Natural speech has moderate spectral variation and dynamic range.
        Returns a score between 0.0 and 1.0.
        """
        waveform = sample.waveform
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        if len(waveform) < self.frame_length * 2:
            return 0.0

        # Compute spectral flatness across frames
        flatness_values = []
        num_frames = (len(waveform) - self.frame_length) // self.hop_length + 1

        for i in range(num_frames):
            start = i * self.hop_length
            frame = waveform[start : start + self.frame_length]
            spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
            spectrum = spectrum + 1e-10

            geometric_mean = np.exp(np.mean(np.log(spectrum)))
            arithmetic_mean = np.mean(spectrum)
            flatness = geometric_mean / (arithmetic_mean + 1e-10)
            flatness_values.append(flatness)

        if not flatness_values:
            return 0.0

        flatness_arr = np.array(flatness_values)

        # Natural speech: moderate flatness variation, not too uniform or noisy
        mean_flatness = float(np.mean(flatness_arr))
        flatness_var = float(np.var(flatness_arr))

        # Score components
        # Ideal mean flatness for speech: ~0.1-0.4
        flatness_score = 1.0 - abs(mean_flatness - 0.25) * 4.0
        flatness_score = max(0.0, min(1.0, flatness_score))

        # Some variation is natural
        var_score = min(1.0, flatness_var * 100.0)

        # Dynamic range contribution
        rms_values = []
        for i in range(num_frames):
            start = i * self.hop_length
            frame = waveform[start : start + self.frame_length]
            rms_values.append(np.sqrt(np.mean(frame ** 2)))

        rms_arr = np.array(rms_values)
        if np.max(rms_arr) > 0:
            dynamic_range = np.max(rms_arr) / (np.min(rms_arr[rms_arr > 0]) + 1e-10)
            dr_score = float(np.clip(np.log10(dynamic_range + 1) / 2.0, 0.0, 1.0))
        else:
            dr_score = 0.0

        naturalness = 0.4 * flatness_score + 0.3 * var_score + 0.3 * dr_score
        return float(np.clip(naturalness, 0.0, 1.0))

    def measure_similarity(
        self, profile_a: VoiceProfile, profile_b: VoiceProfile
    ) -> float:
        """Measure similarity between two voice profiles.

        Uses cosine similarity on speaker embeddings.
        Returns a score between 0.0 (different) and 1.0 (identical).
        """
        return max(0.0, profile_a.similarity(profile_b))

    def full_analysis(self, sample: AudioSample) -> dict[str, float]:
        """Run full quality analysis on an audio sample.

        Returns dict with clarity, naturalness, and SNR metrics.
        """
        clarity = self.measure_clarity(sample)
        naturalness = self.measure_naturalness(sample)
        snr = self._estimate_snr(sample)

        return {
            "clarity": clarity,
            "naturalness": naturalness,
            "snr_db": snr,
        }

    def _estimate_snr(self, sample: AudioSample) -> float:
        """Estimate signal-to-noise ratio in dB."""
        waveform = sample.waveform
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        if len(waveform) == 0:
            return 0.0

        # Simple SNR estimate: ratio of signal power to noise floor
        sorted_power = np.sort(waveform ** 2)
        n = len(sorted_power)
        noise_floor = np.mean(sorted_power[: max(1, n // 10)]) + 1e-10
        signal_power = np.mean(sorted_power[n // 2 :]) + 1e-10

        snr_db = 10.0 * np.log10(signal_power / noise_floor)
        return float(snr_db)
