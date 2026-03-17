"""Pydantic data models for VOICETWIN."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class AudioSample(BaseModel):
    """Represents a loaded audio sample with metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    file_path: Optional[Path] = None
    waveform: np.ndarray = Field(..., description="Raw audio waveform as numpy array")
    sample_rate: int = Field(default=22050, description="Sample rate in Hz")
    duration_seconds: float = Field(..., description="Duration in seconds")
    num_channels: int = Field(default=1, description="Number of audio channels")

    @property
    def num_samples(self) -> int:
        return self.waveform.shape[-1]

    @classmethod
    def from_array(
        cls, waveform: np.ndarray, sample_rate: int = 22050, file_path: Path | None = None
    ) -> AudioSample:
        duration = waveform.shape[-1] / sample_rate
        channels = 1 if waveform.ndim == 1 else waveform.shape[0]
        return cls(
            file_path=file_path,
            waveform=waveform,
            sample_rate=sample_rate,
            duration_seconds=duration,
            num_channels=channels,
        )


class VoiceProfile(BaseModel):
    """A speaker's voice profile containing embedding and characteristics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    speaker_id: str = Field(default="unknown", description="Speaker identifier")
    embedding: np.ndarray = Field(
        ..., description="Speaker embedding vector (d-vector)"
    )
    embedding_dim: int = Field(default=256, description="Dimensionality of embedding")
    mean_pitch_hz: Optional[float] = Field(
        default=None, description="Average fundamental frequency in Hz"
    )
    pitch_range_hz: Optional[tuple[float, float]] = Field(
        default=None, description="(min, max) pitch range in Hz"
    )
    speaking_rate: Optional[float] = Field(
        default=None, description="Syllables per second estimate"
    )
    clarity_score: Optional[float] = Field(
        default=None, description="Voice clarity 0.0-1.0"
    )
    naturalness_score: Optional[float] = Field(
        default=None, description="Voice naturalness 0.0-1.0"
    )
    source_duration_seconds: Optional[float] = Field(
        default=None, description="Duration of source audio used to build profile"
    )

    def similarity(self, other: VoiceProfile) -> float:
        """Compute cosine similarity between two voice profiles."""
        a = self.embedding / (np.linalg.norm(self.embedding) + 1e-8)
        b = other.embedding / (np.linalg.norm(other.embedding) + 1e-8)
        return float(np.dot(a, b))


class ProsodyFeatures(BaseModel):
    """Extracted prosody features from an audio sample."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pitch_contour: np.ndarray = Field(
        ..., description="Frame-level pitch (F0) values in Hz"
    )
    energy_contour: np.ndarray = Field(
        ..., description="Frame-level energy/loudness values"
    )
    speaking_rate: float = Field(..., description="Estimated syllables per second")
    pitch_mean: float = Field(..., description="Mean pitch in Hz")
    pitch_std: float = Field(..., description="Pitch standard deviation")
    emphasis_indices: list[int] = Field(
        default_factory=list,
        description="Frame indices where emphasis/stress is detected",
    )


class SynthesisResult(BaseModel):
    """Result from voice synthesis."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mel_spectrogram: np.ndarray = Field(
        ..., description="Synthesized mel-spectrogram"
    )
    alignment: Optional[np.ndarray] = Field(
        default=None, description="Attention alignment matrix"
    )
    text: str = Field(..., description="Input text that was synthesized")
    duration_seconds: float = Field(..., description="Estimated output duration")
    sample_rate: int = Field(default=22050, description="Target sample rate")
    speaker_embedding: Optional[np.ndarray] = Field(
        default=None, description="Speaker embedding used for synthesis"
    )
