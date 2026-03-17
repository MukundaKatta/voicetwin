"""Voice synthesizer: generates speech mel-spectrograms in a cloned voice
using an encoder-decoder architecture conditioned on speaker embeddings."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from voicetwin.models import SynthesisResult


class TextEncoder(nn.Module):
    """Encodes character/phoneme sequences into hidden representations."""

    def __init__(self, vocab_size: int = 256, embed_dim: int = 256, hidden_size: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers=2, batch_first=True, bidirectional=True
        )
        self.projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """Encode text token IDs.

        Args:
            text_ids: (batch, seq_len) integer tensor of character codes.

        Returns:
            Encoder output of shape (batch, seq_len, hidden_size).
        """
        embedded = self.embedding(text_ids)
        output, _ = self.lstm(embedded)
        return self.projection(output)


class AttentionDecoder(nn.Module):
    """Autoregressive decoder with location-sensitive attention.

    Produces mel-spectrogram frames conditioned on encoder output and
    speaker embedding.
    """

    def __init__(self, n_mels: int = 80, hidden_size: int = 256, embedding_dim: int = 256):
        super().__init__()
        self.n_mels = n_mels
        self.hidden_size = hidden_size

        # Pre-net: transforms previous mel frame
        self.prenet = nn.Sequential(
            nn.Linear(n_mels, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # Attention
        self.attention_query = nn.Linear(hidden_size, hidden_size)
        self.attention_key = nn.Linear(hidden_size, hidden_size)

        # Decoder LSTM
        self.decoder_lstm = nn.LSTMCell(hidden_size * 2 + embedding_dim, hidden_size)

        # Output projection to mel
        self.mel_projection = nn.Linear(hidden_size, n_mels)

        # Stop token prediction
        self.stop_projection = nn.Linear(hidden_size, 1)

    def forward(
        self,
        encoder_output: torch.Tensor,
        speaker_embedding: torch.Tensor,
        max_steps: int = 500,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode mel-spectrogram frames.

        Args:
            encoder_output: (batch, seq_len, hidden_size) from TextEncoder.
            speaker_embedding: (batch, embedding_dim) speaker d-vector.
            max_steps: Maximum number of mel frames to generate.

        Returns:
            Tuple of (mel_output, stop_logits).
            mel_output: (batch, max_steps, n_mels)
            stop_logits: (batch, max_steps)
        """
        batch_size = encoder_output.size(0)
        device = encoder_output.device

        # Initialize
        mel_frame = torch.zeros(batch_size, self.n_mels, device=device)
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)

        keys = self.attention_key(encoder_output)
        mel_outputs = []
        stop_outputs = []

        for _ in range(max_steps):
            # Pre-net on previous mel frame
            prenet_out = self.prenet(mel_frame)

            # Attention: query from decoder state
            query = self.attention_query(h).unsqueeze(1)
            attn_weights = torch.bmm(query, keys.transpose(1, 2))
            attn_weights = torch.softmax(attn_weights, dim=-1)
            context = torch.bmm(attn_weights, encoder_output).squeeze(1)

            # Concatenate context, prenet output, and speaker embedding
            decoder_input = torch.cat([prenet_out, context, speaker_embedding], dim=-1)

            # Decoder LSTM step
            h, c = self.decoder_lstm(decoder_input, (h, c))

            # Project to mel frame
            mel_frame = self.mel_projection(h)
            stop_logit = self.stop_projection(h).squeeze(-1)

            mel_outputs.append(mel_frame)
            stop_outputs.append(stop_logit)

        mel_output = torch.stack(mel_outputs, dim=1)
        stop_logits = torch.stack(stop_outputs, dim=1)
        return mel_output, stop_logits


class VoiceSynthesizer:
    """High-level synthesizer that generates speech in a cloned voice.

    Combines a TextEncoder and AttentionDecoder conditioned on speaker
    embeddings from SpeakerEncoder.

    Args:
        n_mels: Number of mel channels.
        embedding_dim: Speaker embedding dimensionality.
        hidden_size: Internal hidden size.
        sample_rate: Target sample rate.
        device: Torch device.
    """

    def __init__(
        self,
        n_mels: int = 80,
        embedding_dim: int = 256,
        hidden_size: int = 256,
        sample_rate: int = 22050,
        device: str = "cpu",
    ):
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        self.sample_rate = sample_rate
        self.device = torch.device(device)

        self.text_encoder = TextEncoder(
            hidden_size=hidden_size,
        ).to(self.device)

        self.decoder = AttentionDecoder(
            n_mels=n_mels,
            hidden_size=hidden_size,
            embedding_dim=embedding_dim,
        ).to(self.device)

        self.text_encoder.eval()
        self.decoder.eval()

    def _text_to_ids(self, text: str) -> torch.Tensor:
        """Convert text string to integer token IDs (simple character-level)."""
        ids = [min(ord(ch), 255) for ch in text]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def synthesize(
        self,
        text: str,
        speaker_embedding: np.ndarray,
        max_duration_seconds: float = 10.0,
    ) -> SynthesisResult:
        """Synthesize speech mel-spectrogram for the given text and voice.

        Args:
            text: Text to synthesize.
            speaker_embedding: Speaker d-vector from SpeakerEncoder.
            max_duration_seconds: Maximum output duration.

        Returns:
            SynthesisResult containing the mel-spectrogram and metadata.
        """
        # Estimate max steps from text length and duration
        chars_per_second = 15.0
        estimated_duration = min(len(text) / chars_per_second, max_duration_seconds)
        hop_length = 256
        max_steps = int(estimated_duration * self.sample_rate / hop_length)
        max_steps = max(max_steps, 10)

        text_ids = self._text_to_ids(text)
        emb_tensor = (
            torch.from_numpy(speaker_embedding).float().unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            encoder_output = self.text_encoder(text_ids)
            mel_output, stop_logits = self.decoder(
                encoder_output, emb_tensor, max_steps=max_steps
            )

        mel_np = mel_output.squeeze(0).cpu().numpy()  # (time, n_mels)
        mel_np = mel_np.T  # (n_mels, time)

        # Trim silence using stop token
        stop_probs = torch.sigmoid(stop_logits).squeeze(0).cpu().numpy()
        stop_idx = np.argmax(stop_probs > 0.5)
        if stop_idx > 0:
            mel_np = mel_np[:, :stop_idx]

        actual_duration = mel_np.shape[1] * hop_length / self.sample_rate

        return SynthesisResult(
            mel_spectrogram=mel_np,
            text=text,
            duration_seconds=actual_duration,
            sample_rate=self.sample_rate,
            speaker_embedding=speaker_embedding,
        )

    def load_weights(self, encoder_path: str, decoder_path: str) -> None:
        """Load pre-trained synthesizer weights."""
        self.text_encoder.load_state_dict(
            torch.load(encoder_path, map_location=self.device)
        )
        self.decoder.load_state_dict(
            torch.load(decoder_path, map_location=self.device)
        )
        self.text_encoder.eval()
        self.decoder.eval()
