#!/usr/bin/env python3
"""Minimal Gradio demo for an exported frozen-baseline checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from turn_detection.audio import load_mono_audio, windows_for_audio
from turn_detection.streaming import StreamingTurnDetector


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "best.pt")
    parser.add_argument("--share", action="store_true"); args = parser.parse_args()
    detector = StreamingTurnDetector(args.checkpoint)

    def predict(audio_path, threshold, consecutive):
        if not audio_path: return None, "Upload or record audio first."
        detector.policy.threshold = threshold; detector.policy.consecutive_frames = int(consecutive); detector.reset()
        audio = load_mono_audio(audio_path); results = [detector.step(window) for window in windows_for_audio(audio)]
        probabilities, events = zip(*results)
        times = [index * 0.2 for index in range(len(probabilities))]
        figure, axis = plt.subplots(figsize=(8, 3)); axis.plot(times, probabilities, marker="o", ms=3); axis.axhline(threshold, color="tab:red", ls="--")
        for time, event in zip(times, events):
            if event: axis.axvline(time, color="tab:green", lw=2)
        axis.set(xlabel="window end time (s, 200 ms hop)", ylabel="P(turn_end)", ylim=(0, 1)); figure.tight_layout()
        event_times = [f"{time:.1f}s" for time, event in zip(times, events) if event]
        return figure, ("END TURN at " + ", ".join(event_times)) if event_times else "No END TURN event under this policy."

    with gr.Blocks(title="Streaming Turn Detection") as app:
        gr.Markdown("# Whisper Tiny + GRU streaming turn detector\nEach point recomputes Whisper Tiny from a fresh one-second window; only the GRU carries history.")
        with gr.Row():
            audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
            with gr.Column():
                threshold = gr.Slider(0, 1, value=0.9, label="Threshold")
                consecutive = gr.Slider(1, 4, value=2, step=1, label="Consecutive frames")
                button = gr.Button("Run streaming simulation")
        plot = gr.Plot(); result = gr.Textbox(label="Policy decision"); button.click(predict, [audio, threshold, consecutive], [plot, result])
    app.launch(share=args.share)


if __name__ == "__main__": main()
