"""VOICETWIN command-line interface."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
from rich.console import Console

console = Console()


def _load_sample(path: str, sample_rate: int = 22050):
    """Load an audio file into an AudioSample."""
    import librosa
    from voicetwin.models import AudioSample

    waveform, sr = librosa.load(path, sr=sample_rate, mono=True)
    return AudioSample.from_array(waveform, sample_rate=sr, file_path=Path(path))


@click.group()
@click.version_option(package_name="voicetwin")
def cli():
    """VOICETWIN - AI Voice Cloning from 30-second samples."""
    pass


@cli.command()
@click.option("--sample", "-s", required=True, type=click.Path(exists=True), help="Path to voice sample WAV file.")
@click.option("--text", "-t", required=True, help="Text to synthesize in the cloned voice.")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output WAV file path.")
@click.option("--device", "-d", default="cpu", help="Torch device (cpu/cuda).")
def clone(sample: str, text: str, output: str, device: str):
    """Clone a voice and synthesize speech."""
    from voicetwin.cloner.encoder import SpeakerEncoder
    from voicetwin.cloner.synthesizer import VoiceSynthesizer
    from voicetwin.cloner.vocoder import WaveformGenerator

    console.print("[bold blue]VOICETWIN[/bold blue] - Cloning voice...\n")

    with console.status("Extracting speaker embedding..."):
        encoder = SpeakerEncoder(device=device)
        embedding = encoder.encode(sample)
    console.print("[green]Speaker embedding extracted.[/green]")

    with console.status("Synthesizing speech..."):
        synthesizer = VoiceSynthesizer(device=device)
        result = synthesizer.synthesize(text, embedding)
    console.print(f"[green]Synthesized {result.duration_seconds:.2f}s of speech.[/green]")

    with console.status("Generating waveform..."):
        vocoder = WaveformGenerator(device=device)
        audio = vocoder.generate_from_result(result)
        vocoder.save(output, audio)
    console.print(f"[green]Saved output to {output}[/green]")


@cli.command()
@click.option("--sample", "-s", required=True, type=click.Path(exists=True), help="Path to voice sample WAV file.")
def analyze(sample: str):
    """Analyze voice quality of a sample."""
    from voicetwin.features.quality import VoiceQualityAnalyzer

    console.print("[bold blue]VOICETWIN[/bold blue] - Analyzing voice quality...\n")

    audio_sample = _load_sample(sample)
    analyzer = VoiceQualityAnalyzer(sample_rate=audio_sample.sample_rate)
    metrics = analyzer.full_analysis(audio_sample)

    from rich.table import Table

    table = Table(title="Voice Quality Analysis")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Clarity", f"{metrics['clarity']:.3f}")
    table.add_row("Naturalness", f"{metrics['naturalness']:.3f}")
    table.add_row("SNR", f"{metrics['snr_db']:.1f} dB")
    console.print(table)


@cli.command()
@click.option("--sample", "-s", required=True, type=click.Path(exists=True), help="Path to voice sample WAV file.")
@click.option("--output", "-o", default=None, type=click.Path(), help="Optional JSON output path.")
@click.option("--speaker-id", default="unknown", help="Speaker identifier.")
@click.option("--device", "-d", default="cpu", help="Torch device (cpu/cuda).")
def report(sample: str, output: str | None, speaker_id: str, device: str):
    """Generate a full voice profile report."""
    from voicetwin.cloner.encoder import SpeakerEncoder
    from voicetwin.report import VoiceProfileReport

    console.print("[bold blue]VOICETWIN[/bold blue] - Generating voice profile report...\n")

    audio_sample = _load_sample(sample)

    with console.status("Building voice profile..."):
        encoder = SpeakerEncoder(device=device)
        profile = encoder.build_profile(sample, speaker_id=speaker_id)

    reporter = VoiceProfileReport(sample_rate=audio_sample.sample_rate)
    report_data = reporter.generate(audio_sample, profile)
    reporter.print_report(report_data)

    if output:
        reporter.save_json(report_data, output)
        console.print(f"\n[green]Report saved to {output}[/green]")


if __name__ == "__main__":
    cli()
