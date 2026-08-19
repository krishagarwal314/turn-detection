from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .constants import HOP_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES


def load_mono_audio(path: str | Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a recording as mono float32 audio at `target_sr`."""
    import librosa
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != target_sr:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
    return np.asarray(audio, dtype=np.float32)


def load_dataset_audio(audio: dict[str, Any], target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode a Hugging Face ``Audio(decode=False)`` value without torchcodec.

    This keeps the data-preparation path portable to Colab images whose bundled
    PyTorch version may not match torchcodec. The dataset supplies either audio
    bytes or a local path; both are decoded by the pinned soundfile/librosa stack.
    """
    import librosa
    import soundfile as sf

    source: str | io.BytesIO
    if audio.get("bytes") is not None:
        source = io.BytesIO(audio["bytes"])
    elif audio.get("path"):
        source = str(audio["path"])
    else:
        raise ValueError("Dataset audio value has neither bytes nor path")
    samples, sample_rate = sf.read(source, always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    samples = np.asarray(samples, dtype=np.float32)
    if sample_rate != target_sr:
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=target_sr)
    return np.asarray(samples, dtype=np.float32)


def windows_for_audio(audio: np.ndarray) -> list[np.ndarray]:
    """Return 1 s windows at 200 ms hops; the last window ends at clip end.

    Windows shorter than one second (including very short clips) are zero padded.
    Every item is an independent raw-audio window; no encoder state is shared.
    """
    if len(audio) == 0:
        audio = np.zeros(1, dtype=np.float32)
    max_start = max(0, len(audio) - WINDOW_SAMPLES)
    starts = list(range(0, max_start + 1, HOP_SAMPLES))
    if not starts or starts[-1] != max_start:
        starts.append(max_start)
    output: list[np.ndarray] = []
    for start in starts:
        chunk = audio[start : start + WINDOW_SAMPLES]
        if len(chunk) < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
        output.append(chunk.astype(np.float32, copy=False))
    return output


def labels_for_clip(audio: np.ndarray, endpoint: bool) -> np.ndarray:
    labels = np.zeros(len(windows_for_audio(audio)), dtype=np.float32)
    if endpoint:
        labels[-1] = 1.0
    return labels


def iter_windows(audio: np.ndarray) -> Iterable[tuple[int, np.ndarray]]:
    for index, window in enumerate(windows_for_audio(audio)):
        yield index, window
