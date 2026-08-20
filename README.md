# Streaming Turn Detection — Whisper Tiny + GRU

An audio-native turn-end detector for Indian English and Hindi/Hinglish conversation. Every 200 ms it returns a raw `P(turn_end)` for the latest 1-second audio window; a separate policy decides when that probability is strong enough to end the turn.

## Approach

Most other turn detectors are transcript-first: speech-to-text runs first, then a language model decides whether the user is finished. That is powerful, but it adds transcription latency and a second, often large, language model. I chose an audio-native route instead: Whisper Tiny's encoder is roughly 19M parameters on its own (about 39M for the full encoder-decoder), and the decoder is unnecessary when the output is only a turn probability.

Around 90% of the code was generated or edited with AI coding agents, while I iterated on the data policy, labels, experiments, and evaluation myself. I had a tight deadline and trained on a free Google Colab instance, so this run was about building and testing a sensible solution—not claiming production-level accuracy.

## Data preparation at a glance

Data quality was the main constraint, so preparation is explicit rather than hidden:

1. Stream the Pipecat dataset instead of loading its ~41 GB of audio into Colab RAM.
2. Keep English and Hindi rows only; reject synthetic English and allow synthetic Hindi explicitly (because the dataset did not have any real hindi rows).
3. Decode every clip to mono 16 kHz `float32` audio and write clip-level JSONL manifests.
4. Split by clip—not by window—to prevent overlapping windows from leaking across train/validation/test.
5. Balance the language cap and stratify where possible by language, endpoint, and filler flags.
6. Convert each clip into 1-second windows with a 200-ms hop; only the final window is positive for an endpoint clip.

The resulting labels are intentionally sparse because the source metadata tells us whether a clip ends a turn, not where every ambiguous pause occurs.

## Design decisions

**Why not use Whisper's cache as the sequence model?** Whisper processes each overlapping audio window independently in this baseline; it does not expose a reliable causal cross-window KV cache for this setup. Caching the resulting embeddings makes repeated experiments faster, but it does not create temporal context. A GRU consumes the ordered 384-dimensional encoder embeddings and carries the conversational state from one window to the next. That gives us context without pretending Whisper is streaming-causal.

**Why 1-second windows every 200 ms?** A one-second window contains enough local speech and silence to distinguish a boundary, while the 200-ms hop gives a responsive update rate. Short final windows are zero-padded so every encoder input has the same shape.

**What supervision was actually available?** The source metadata tells us only whether a clip ends at a turn boundary. It does not mark every ambiguous pause. Therefore an endpoint clip has one positive label—the final window—and all earlier windows are negative. A non-endpoint clip has only negative labels. This is honest but sparse supervision, and it explains both the severe class imbalance and the model's tendency to trade recall against false interruptions.

**Why keep the policy outside the model?** The neural model provides evidence at every window. The application may prefer fast responses or may strongly penalize false interruptions, so threshold and consecutive-frame requirements remain configurable instead of being hidden inside the loss.

## Architecture

```text
1.0 s audio window ──(200 ms hop)──> Whisper Tiny encoder
                                      (decoder removed)
                                             │
                                      mean-pool → 384-d
                                             │
                         ordered embeddings → GRU → MLP → P(turn_end)
                                                               │
                                             threshold + consecutive-frame policy
                                                               │
                                                            END TURN
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

## Data preparation details

The primary source is [`pipecat-ai/smart-turn-data-v3.2-train`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train), a roughly 271k-row, approximately 41 GB audio dataset. Preparation is streamed so Colab does not materialize the complete dataset in RAM.

The language decision was the most important practical compromise. Real English clips were available in useful volume. The Hindi rows we could obtain in the time window were synthetic, and the synthetic Hindi was often very formal, **shudh Hindi 😂**, closer to documentary or written narration than conversational Hinglish. That means it cannot honestly stand in for natural Hindi turn-taking. However, discarding Hindi entirely would make the requested Indian-language experiment impossible, so I kept real English and synthetic Hindi as a clearly labeled baseline. Synthetic English is still rejected. The report records the distinction instead of hiding it.

For the measured run, the filter retained 10,000 clips: 5,000 real `eng` and 5,000 synthetic `hin`. The actual training experiment used a smaller 3,000-clip subset: 2,400 train, 300 validation, and 300 test. No human Hinglish recordings were available, so Hinglish metrics are intentionally absent. A 120-prompt recording sheet is included for collecting that missing evaluation data rather than fabricating it.

Preparation does the following before training:

1. Streams the dataset and keeps only the requested language codes.
2. Rejects synthetic English and optionally allows synthetic Hindi explicitly.
3. Decodes audio with `soundfile`, converts it to mono `float32`, and resamples it to 16 kHz.
4. Saves one audio file per clip and records the source metadata in JSONL manifests.
5. Splits by clip, never by window, so overlapping windows from one recording cannot leak across train/validation/test.
6. Stratifies the split using language, endpoint status, and filler flags where the stratum has enough examples.
7. Treats null filler metadata as false instead of accidentally turning missing values into positives.

The authoritative evidence is the generated `filter_report.json`; counts are never hard-coded because the upstream dataset can change.

Create the curated recording sheet once:

```bash
python3 data/seed_hinglish_manifest.py
```

It creates 120 short conversational prompts balanced between natural endpoints and mid-thought continuations, with fillers such as *um*, *actually*, and *matlab*. Record each prompt naturally with consenting human speakers; save the matching files as `data/hinglish_recordings/hinglish_001.wav`, etc. These recordings are intentionally absent from the repository: authentic human audio cannot be truthfully fabricated or replaced with TTS. After recordings exist, `prepare_data.py` automatically folds them into train/validation/test at the **clip** level and includes them in both validation and test.

### Clip-to-window labels

For a clip, generate 1-second windows at 200-ms hops; the final window is adjusted to end exactly at the clip end and short windows are zero padded. A true endpoint clip has a single positive label on its final window. A mid-utterance clip has only negatives.

Worked example: a 2.0-second endpoint clip gives windows ending at roughly `1.0, 1.2, 1.4, 1.6, 1.8, 2.0s` and labels `[0, 0, 0, 0, 0, 1]`. A 2.0-second non-endpoint clip gets `[0, 0, 0, 0, 0, 0]`. This provides sparse supervision—there is no label for a moderately likely pause—which is a known limitation.

Windows from a clip never cross splits. The split stratum includes endpoint, `midfiller`, and `endfiller` flags so filler cases are not accidentally diluted.

### Feature extraction

Each 16-kHz waveform window is passed through `WhisperProcessor`, which converts the samples into Whisper's log-Mel input features. The frozen Whisper Tiny encoder produces a sequence of acoustic states; mean-pooling those states gives one 384-dimensional vector for that window. The cache stores a tensor shaped `[number_of_windows, 384]` plus the clip labels and metadata. Training then reads those vectors instead of repeatedly running Whisper, which is why caching is expensive once but makes the GRU experiments fast and reproducible. The cache is an optimization for iteration—it is not a substitute for temporal memory, which is the GRU's job.

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

### What the experiments taught us

- **Weighted BCE** strongly protected recall but produced too many false interruptions.
- **Focal loss** did not help on this small split; it reached lower precision and lower F1 than weighted BCE.
- **Reducing the positive weight** improved the operating point. A scale of `0.25` reached 17.98% precision, 86.57% recall, and 29.78% F1. A scale of `0.10` reached 21.10% precision, 51.49% recall, and 29.93% F1.
- **The policy is a product decision, not a magic fix.** Requiring consecutive high-probability windows reduces false interruptions, but it also misses more endpoints. The current policy evaluator is exploratory and its latency calculation still needs calibration before being reported as a final latency benchmark.

This is the depth of the solutioning: the project does not stop at a classifier score. It traces the data provenance, exposes the real/synthetic compromise, measures class imbalance, compares losses, adds precision-oriented weighting, saves deterministic mid-epoch checkpoints, and keeps the final interruption policy separate so its trade-offs are visible.

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
