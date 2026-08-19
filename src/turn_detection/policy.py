from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConsecutiveThresholdPolicy:
    """Decision policy kept separate from raw model probabilities."""

    threshold: float = 0.9
    consecutive_frames: int = 2
    _run: int = 0
    triggered: bool = False

    def reset(self) -> None:
        self._run = 0
        self.triggered = False

    def update(self, probability: float) -> bool:
        if self.triggered:
            return False
        self._run = self._run + 1 if probability >= self.threshold else 0
        if self._run >= self.consecutive_frames:
            self.triggered = True
            return True
        return False

    def first_trigger(self, probabilities: list[float]) -> int | None:
        self.reset()
        for index, probability in enumerate(probabilities):
            if self.update(float(probability)):
                return index
        return None
