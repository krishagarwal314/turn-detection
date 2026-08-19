from __future__ import annotations

from pathlib import Path

import torch
from transformers import WhisperModel, WhisperProcessor

from .constants import WHISPER_MODEL_ID
from .model import GRUTurnClassifier
from .policy import ConsecutiveThresholdPolicy


class StreamingTurnDetector:
    """Runs a fresh Whisper encoder pass per 1 s window, retaining only GRU state."""

    def __init__(self, checkpoint_path: str | Path, threshold: float = 0.9, consecutive_frames: int = 2, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if state.get("model_type") != "cached_gru":
            raise ValueError("The lightweight demo currently expects a frozen-baseline cached_gru checkpoint.")
        self.processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID)
        self.encoder = WhisperModel.from_pretrained(WHISPER_MODEL_ID).encoder.to(self.device).eval()
        self.model = GRUTurnClassifier(**state["model_config"]).to(self.device).eval()
        self.model.load_state_dict(state["model_state"])
        self.policy = ConsecutiveThresholdPolicy(threshold, consecutive_frames)
        self.hidden = None

    def reset(self):
        self.hidden = None; self.policy.reset()

    @torch.inference_mode()
    def step(self, window):
        features = self.processor(window, sampling_rate=16_000, return_tensors="pt").input_features.to(self.device)
        embedding = self.encoder(features).last_hidden_state.mean(dim=1).unsqueeze(1)
        logits, self.hidden = self.model(embedding, hidden=self.hidden)
        probability = float(torch.sigmoid(logits[0, 0]).item())
        return probability, self.policy.update(probability)
