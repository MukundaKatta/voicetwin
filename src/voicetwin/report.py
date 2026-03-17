"""Voice profile reporting with rich console output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from voicetwin.features.prosody import ProsodyAnalyzer
from voicetwin.features.quality import VoiceQualityAnalyzer
from voicetwin.models import AudioSample, VoiceProfile


class VoiceProfileReport:
    """Generates comprehensive voice profile reports.

    Combines speaker embedding, prosody analysis, and quality metrics
    into a structured report with optional rich console rendering.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
    ):
        self.sample_rate = sample_rate
        self.prosody_analyzer = ProsodyAnalyzer(sample_rate=sample_rate)
        self.quality_analyzer = VoiceQualityAnalyzer(sample_rate=sample_rate)

    def generate(
        self,
        sample: AudioSample,
        profile: VoiceProfile,
    ) -> dict[str, Any]:
        """Generate a full voice profile report.

        Args:
            sample: The audio sample analyzed.
            profile: The speaker's voice profile.

        Returns:
            Dictionary containing all analysis results.
        """
        prosody = self.prosody_analyzer.analyze(sample)
        quality = self.quality_analyzer.full_analysis(sample)

        # Update profile with extracted features
        profile.mean_pitch_hz = prosody.pitch_mean
        voiced = prosody.pitch_contour[prosody.pitch_contour > 0]
        if len(voiced) > 0:
            profile.pitch_range_hz = (float(np.min(voiced)), float(np.max(voiced)))
        profile.speaking_rate = prosody.speaking_rate
        profile.clarity_score = quality["clarity"]
        profile.naturalness_score = quality["naturalness"]

        report = {
            "speaker_id": profile.speaker_id,
            "source_duration_seconds": profile.source_duration_seconds,
            "embedding_dim": profile.embedding_dim,
            "pitch": {
                "mean_hz": prosody.pitch_mean,
                "std_hz": prosody.pitch_std,
                "range_hz": profile.pitch_range_hz,
            },
            "prosody": {
                "speaking_rate_sps": prosody.speaking_rate,
                "num_emphasis_points": len(prosody.emphasis_indices),
            },
            "quality": quality,
        }
        return report

    def save_json(self, report: dict[str, Any], path: str | Path) -> None:
        """Save report to JSON file."""
        serializable = _make_serializable(report)
        with open(str(path), "w") as f:
            json.dump(serializable, f, indent=2)

    def print_report(self, report: dict[str, Any]) -> None:
        """Print a formatted report to the console using rich."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        console.print(
            Panel(
                f"[bold]Speaker:[/bold] {report['speaker_id']}\n"
                f"[bold]Duration:[/bold] {report.get('source_duration_seconds', 'N/A'):.1f}s\n"
                f"[bold]Embedding dim:[/bold] {report['embedding_dim']}",
                title="VOICETWIN Profile Report",
                border_style="blue",
            )
        )

        # Pitch table
        pitch = report.get("pitch", {})
        table = Table(title="Pitch Analysis")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Mean F0", f"{pitch.get('mean_hz', 0):.1f} Hz")
        table.add_row("Std F0", f"{pitch.get('std_hz', 0):.1f} Hz")
        pitch_range = pitch.get("range_hz")
        if pitch_range:
            table.add_row("Range", f"{pitch_range[0]:.1f} - {pitch_range[1]:.1f} Hz")
        console.print(table)

        # Prosody table
        prosody = report.get("prosody", {})
        table = Table(title="Prosody")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Speaking Rate", f"{prosody.get('speaking_rate_sps', 0):.2f} syl/s")
        table.add_row("Emphasis Points", str(prosody.get("num_emphasis_points", 0)))
        console.print(table)

        # Quality table
        quality = report.get("quality", {})
        table = Table(title="Voice Quality")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Clarity", f"{quality.get('clarity', 0):.3f}")
        table.add_row("Naturalness", f"{quality.get('naturalness', 0):.3f}")
        table.add_row("SNR", f"{quality.get('snr_db', 0):.1f} dB")
        console.print(table)


def _make_serializable(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_make_serializable(item) for item in obj]
        return tuple(converted) if isinstance(obj, tuple) else converted
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    return obj
