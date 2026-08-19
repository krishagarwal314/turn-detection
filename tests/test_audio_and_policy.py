import numpy as np

from turn_detection.audio import labels_for_clip, windows_for_audio
from turn_detection.constants import SAMPLE_RATE
from turn_detection.policy import ConsecutiveThresholdPolicy


def test_short_audio_is_one_padded_window_with_final_label():
    audio = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    assert len(windows_for_audio(audio)) == 1
    assert labels_for_clip(audio, True).tolist() == [1.0]
    assert labels_for_clip(audio, False).tolist() == [0.0]


def test_policy_requires_consecutive_frames():
    policy = ConsecutiveThresholdPolicy(threshold=0.9, consecutive_frames=2)
    assert [policy.update(value) for value in (0.91, 0.7, 0.95, 0.96)] == [False, False, False, True]
