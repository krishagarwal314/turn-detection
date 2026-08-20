# Streaming Turn Detection — Whisper Tiny + GRU

A compact real-time turn-end detector for Indian English and Hinglish conversation. Every 200 ms it returns a raw `P(turn_end)` for the latest 1-second audio window. The model is designed for a voice-agent pipeline where the application, not the neural model, decides when to interrupt or hand the floor back.

## Architecture

```text
fresh 1.0 s raw audio window, every 200 ms
                    |
                    v
 Whisper Tiny encoder (frozen baseline; no shared encoder state)
                    |
           mean pool over time
                    |
                    v
       384-d embedding -----> GRU hidden state carried across windows
                                      |
                                      v
                             small MLP + sigmoid
                                      |
                                      v
                        raw P(turn_end) every 200 ms
                                      |
                          separate threshold/k-frame policy
                                      v
                                 END TURN event
```

Only the GRU remembers earlier audio. Whisper processes each window independently—there is no causal-attention modification, cross-window key/value cache, or reuse of Whisper activations. This is deliberately simpler and less efficient than true streaming/causal Whisper, but is reliable to implement within this project’s time budget. A production version should benchmark a causal acoustic encoder or incremental encoder cache before scaling traffic.

## Repository layout

```text
data/
  prepare_data.py             # language/synthetic filter, clip-level splits, reporting
  cache_embeddings.py         # frozen Whisper embeddings on Drive/local cache
  seed_hinglish_manifest.py   # 120 human-recording prompts; never generates TTS
  hinglish_recordings/        # put local WAV recordings here (ignored by git)
src/turn_detection/
  audio.py, model.py, policy.py, streaming.py
train.py                      # frozen encoder baseline + automatic resume
finetune.py                   # optional final-Whisper-layer fine-tuning experiment
evaluate_policy.py            # latency / false-interruption trade-off plot
demo.py                       # Gradio microphone or upload simulation
notebooks/train_colab.ipynb   # thin Colab wrapper around the scripts
```

## Data decisions

The primary source is [`pipecat-ai/smart-turn-data-v3.2-train`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train). It is a 271k-row, approximately 41 GB audio dataset. `prepare_data.py` streams it and applies the explicit language/synthetic policy: English is real-only, while Hindi synthetic clips may be allowed with `--allow-synthetic-languages hin`. It then stores retained, real, and synthetic counts, endpoint balance, and `midfiller`/`endfiller` balance in `filter_report.json`. Do not claim those values before running the filter—the dataset can change upstream.

The Pipecat pipeline retains `eng` and `hin` clips by default (`--languages eng hin`). Synthetic clips are rejected for English but may be allowed for Hindi with `--allow-synthetic-languages hin`; the report records real and synthetic counts separately. Other languages are excluded before audio is downloaded or cached. When `--max-pipecat-clips` is used, the cap is allocated equally across the requested languages. Train/validation/test splitting also stratifies by language, endpoint, and filler flags. The language field is one value per clip, so it does not represent code-switching inside an utterance. That is why the separate hand-recorded Hinglish set is useful for evaluating code-switching behavior.

Create the curated recording sheet once:

```bash
python3 data/seed_hinglish_manifest.py
```

It creates 120 short conversational prompts balanced between natural endpoints and mid-thought continuations, with fillers such as *um*, *actually*, and *matlab*. Record each prompt naturally with consenting human speakers; save the matching files as `data/hinglish_recordings/hinglish_001.wav`, etc. These recordings are intentionally absent from the repository: authentic human audio cannot be truthfully fabricated or replaced with TTS. After recordings exist, `prepare_data.py` automatically folds them into train/validation/test at the **clip** level and includes them in both validation and test.

### Clip-to-window labels

For a clip, generate 1-second windows at 200-ms hops; the final window is adjusted to end exactly at the clip end and short windows are zero padded. A true endpoint clip has a single positive label on its final window. A mid-utterance clip has only negatives.

Worked example: a 2.0-second endpoint clip gives windows ending at roughly `1.0, 1.2, 1.4, 1.6, 1.8, 2.0s` and labels `[0, 0, 0, 0, 0, 1]`. A 2.0-second non-endpoint clip gets `[0, 0, 0, 0, 0, 0]`. This provides sparse supervision—there is no label for a moderately likely pause—which is a known limitation.

Windows from a clip never cross splits. The split stratum includes endpoint, `midfiller`, and `endfiller` flags so filler cases are not accidentally diluted.

## Reproduce locally

Use Python 3.10 or 3.11 and a CUDA-enabled PyTorch build for serious runs.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"

# First do a smoke run; omit --max-pipecat-clips for a larger run after checking storage.
python data/prepare_data.py --max-pipecat-clips 500
python data/cache_embeddings.py --batch-size 32
python train.py --epochs 2 --checkpoint-dir checkpoints/smoke
pytest -q
```

For a full baseline, run the same commands without the sample cap, preferably on Colab. The data filtering script keeps raw and cached artifacts out of Git. Before a full run, inspect `data/processed/filter_report.json`; it is the authoritative count report required for the experiment write-up.

### Loss and experiments

`train.py` performs many-to-many sequence labeling and masks padded timesteps through `pack_padded_sequence`; padded entries are also excluded from loss. The default weighted BCE uses the training-window class ratio. Use `--pos-weight-scale` below 1.0 to trade some recall for precision, or use unweighted BCE with `--loss bce`. The alternate focal implementation is available with `--loss focal`.

```bash
# Frozen Whisper Tiny baseline: cached, fast iteration
python train.py --loss weighted_bce --epochs 12 --checkpoint-dir checkpoints/frozen_bce
python train.py --loss focal --epochs 12 --checkpoint-dir checkpoints/frozen_focal

# Optional low-LR, much slower experiment; only final Whisper encoder layer is unfrozen.
python finetune.py --unfreeze-last-layers 1 --epochs 4 --checkpoint-dir checkpoints/finetune
```

Every training run writes `last.pt` after each epoch and can also write it during an epoch with `--checkpoint-every-steps N`. A partial checkpoint contains the deterministic epoch ordering, next batch cursor, global step, optimizer/model state, and RNG state, so it resumes at the next unseen batch. `--max-train-steps N` is a controlled interruption switch for validating that path; `--step-trace trace.jsonl` records the clip IDs consumed by every optimizer step. `best.pt`, `history.json`, and `test_metrics.json` stay in the checkpoint directory. Metrics are precision/recall/F1 for positive `turn_end` windows plus confusion matrices, separated by `pipecat` and `hinglish`; no misleading aggregate accuracy is used as the primary metric.

### Results from the 3k Colab experiment

This was intentionally a small, compute-limited experiment rather than a claim of production accuracy. The retained dataset contained 5,000 real English clips and 5,000 synthetic Hindi clips; the actual training subset was 2,400 train, 300 validation, and 300 test clips. There were no recorded Hinglish clips available in this run.

The held-out Pipecat test results were:

| Experiment | Loss | Pipecat P/R/F1 | Hinglish P/R/F1 | Notes |
|---|---|---|---|---|
| Frozen Whisper + GRU | weighted BCE | 14.17% / 88.06% / 24.41% | not measured | recall-oriented baseline |
| Frozen Whisper + GRU | focal | 11.94% / 88.06% / 21.03% | not measured | more false positives than weighted BCE |
| Frozen Whisper + GRU | weighted BCE, positive-weight scale 0.25 | 17.98% / 86.57% / 29.78% | not measured | better precision/F1 trade-off |
| Frozen Whisper + GRU | weighted BCE, positive-weight scale 0.10 | 21.10% / 51.49% / 29.93% | not measured | precision-oriented operating point |

These are measured results from the small run, not projections. The model catches many endpoint windows but still produces too many false positives. With more genuine Hindi and Hinglish recordings, stronger labels, and more training compute, the hypothesis is that the audio-native approach should improve—but no future number is claimed here.

## Policy and demo

The policy is not learned: `ConsecutiveThresholdPolicy` triggers after a configurable number of consecutive probabilities above a threshold. This keeps latency/false interruption trade-offs outside the loss function. Typical settings to compare are `(0.5, 1)`, `(0.7, 2)`, and `(0.9, 3)`; higher thresholds/frame counts generally reduce false interruptions but increase latency or missed endpoints.

The policy script accepts stored per-window prediction records and saves the comparison chart:

```bash
python evaluate_policy.py --predictions path/to/test_window_predictions.json --output checkpoints/policy_tradeoff.png
python demo.py --checkpoint checkpoints/frozen_bce/best.pt
```

The demo accepts upload or microphone input, simulates streaming at 200-ms hops, plots raw probabilities, and marks policy events. It requires a frozen-baseline checkpoint because that path keeps Whisper external and stateless; fine-tuned model serving is intentionally left as a production packaging follow-up.

## Colab workflow (free GPU)

1. Create and push the GitHub repository using the commands below.
2. In GitHub, open `notebooks/train_colab.ipynb`, click **Open in Colab** (or upload it at [colab.research.google.com](https://colab.research.google.com)).
3. Choose **Runtime → Change runtime type → T4 GPU** (or GPU), then run cells from top to bottom.
4. The notebook mounts Drive and uses `/content/drive/MyDrive/turn_detection_artifacts` for raw audio, embedding cache, and checkpoints. A disconnect is safe: rerunning resumes from `last.pt` and reuses existing cache entries.

The dataset is large, so start with `--max-pipecat-clips 5000` to validate Drive capacity and throughput; remove it only after confirming your Drive quota is adequate.

The key notebook commands are:

```bash
# Fresh run after Drive mount
python data/prepare_data.py --output-dir "$ARTIFACTS/processed" --max-pipecat-clips 5000
python data/cache_embeddings.py --manifest-dir "$ARTIFACTS/processed" --cache-dir "$ARTIFACTS/cache" --batch-size 32
python train.py --cache-dir "$ARTIFACTS/cache" --checkpoint-dir "$ARTIFACTS/checkpoints/frozen_bce" --epochs 12

# After a disconnect: mount Drive again, then rerun the same training command.
# Or resume an explicit checkpoint:
python train.py --cache-dir "$ARTIFACTS/cache" --checkpoint-dir "$ARTIFACTS/checkpoints/frozen_bce" --resume "$ARTIFACTS/checkpoints/frozen_bce/last.pt" --epochs 12
```

## GitHub setup

The project is versioned in the `krishagarwal314/turn-detection` repository. For a fresh clone:

```bash
git clone https://github.com/krishagarwal314/turn-detection.git
```

Checkpoints, raw audio, and embedding caches are ignored; keep them on Drive during Colab training. If you later publish a small final checkpoint, use Git LFS or link to Drive rather than committing large binary artifacts normally.

## Known limitations / next steps

- Labels are clip-final-only and have at most one positive window, so they do not model ambiguous pauses well.
- The primary dataset is filtered at run time. This experiment deliberately used real English and synthetic Hindi because the available non-synthetic Hindi subset was insufficient; the report records that distinction explicitly.
- The included Hinglish manifest is a recording protocol, not fake audio. Metrics remain incomplete until those human recordings are collected.
- Recomputing Whisper for overlapping windows costs substantially more than a causal encoder or encoder cache.
- The included policy evaluation needs prediction records emitted from a test run; expanding it into a calibration sweep and per-language error analysis is a useful next iteration.
