from __future__ import annotations

import torch
from torch import nn
from transformers import WhisperModel


class GRUTurnClassifier(nn.Module):
    """Many-to-many GRU classifier over one Whisper embedding per audio window."""

    def __init__(self, input_size: int = 384, hidden_size: int = 192, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size // 2, 1))

    def forward(self, embeddings: torch.Tensor, lengths: torch.Tensor | None = None, hidden: torch.Tensor | None = None):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_output, hidden = self.gru(packed, hidden)
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=embeddings.shape[1])
        else:
            output, hidden = self.gru(embeddings, hidden)
        return self.head(output).squeeze(-1), hidden


class WhisperGRUTurnClassifier(nn.Module):
    """Fine-tunable Whisper encoder + GRU. Whisper remains stateless between windows."""

    def __init__(self, hidden_size: int = 192, num_layers: int = 1, unfreeze_last_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained("openai/whisper-tiny")
        for parameter in self.whisper.parameters():
            parameter.requires_grad = False
        encoder_layers = self.whisper.encoder.layers
        for layer in encoder_layers[-unfreeze_last_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        self.sequence_model = GRUTurnClassifier(self.whisper.config.d_model, hidden_size, num_layers, dropout)

    def forward(self, input_features: torch.Tensor, lengths: torch.Tensor):
        # (batch, steps, 80, 3000) -> independent Whisper pass for every window.
        batch, steps = input_features.shape[:2]
        flat = input_features.flatten(0, 1)
        encoded = self.whisper.encoder(flat).last_hidden_state.mean(dim=1)
        embeddings = encoded.unflatten(0, (batch, steps))
        return self.sequence_model(embeddings, lengths)
