# VOICETWIN

AI Voice Cloning from 30-second samples.

VOICETWIN analyzes a short voice sample (as little as 30 seconds) and generates
speech in the cloned voice. It extracts speaker embeddings, prosody patterns,
and voice quality metrics to produce natural-sounding synthesized audio.

## Architecture

```
Audio Sample (30s)
    |
    v
MelSpectrogramExtractor --> mel-spectrogram
    |
    v
SpeakerEncoder --> voice embedding (d-vector)
    |
    +---> ProsodyAnalyzer --> pitch/rhythm/emphasis
    +---> VoiceQualityAnalyzer --> clarity/naturalness/similarity
    |
    v
VoiceSynthesizer (encoder + decoder) --> synthesized mel-spectrogram
    |
    v
WaveformGenerator (vocoder) --> output waveform (.wav)
```

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from voicetwin.cloner.encoder import SpeakerEncoder
from voicetwin.cloner.synthesizer import VoiceSynthesizer
from voicetwin.cloner.vocoder import WaveformGenerator

# Extract voice embedding from a 30-second sample
encoder = SpeakerEncoder()
embedding = encoder.encode("sample.wav")

# Synthesize speech in the cloned voice
synthesizer = VoiceSynthesizer(encoder)
result = synthesizer.synthesize("Hello, this is my cloned voice.", embedding)

# Convert to waveform
vocoder = WaveformGenerator()
audio = vocoder.generate(result.mel_spectrogram)
vocoder.save("output.wav", audio)
```

## CLI

```bash
# Clone a voice and synthesize speech
voicetwin clone --sample voice.wav --text "Hello world" --output output.wav

# Analyze voice quality
voicetwin analyze --sample voice.wav

# Generate a voice profile report
voicetwin report --sample voice.wav --output report.json
```

## Project Structure

```
src/voicetwin/
    cli.py              - Click CLI interface
    models.py           - Pydantic data models
    report.py           - Voice profile reporting
    cloner/
        encoder.py      - SpeakerEncoder (voice embeddings)
        synthesizer.py  - VoiceSynthesizer (text-to-mel)
        vocoder.py      - WaveformGenerator (mel-to-waveform)
    features/
        mel.py          - Mel-spectrogram extraction
        prosody.py      - Pitch/rhythm/emphasis analysis
        quality.py      - Voice quality metrics
```

## Requirements

- Python 3.10+
- PyTorch
- librosa
- numpy / scipy
- pydantic
- click / rich

## License

MIT
