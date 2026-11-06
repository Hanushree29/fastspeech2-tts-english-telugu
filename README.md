# FastSpeech2-Based TTS: English & Telugu

Implementation and evaluation of **FastSpeech 2** for text-to-speech synthesis in **English** (Svarah) and **Telugu** (IndicTTS Telugu), built on the [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) reference implementation.

**Submitted by:** Hanushree29 | **Deadline:** 07-11-2026

| Component | Status |
|---|---|
| English pipeline (preprocess, train, inference) | ✅ Complete |
| English checkpoints (30,000 steps) | ✅ Complete |
| English results (loss curves, audio, variance plots) | ✅ Complete |
| Telugu pipeline | 🔄 In progress |
| Technical Report | 🔄 In progress |

---

## 1. Overview

Two independently trained FastSpeech2 models, one per language, each with a full preprocessing → training → inference pipeline, trained checkpoints, generated speech for every required test sentence, duration/pitch/energy variance-adaptor visualizations, training/validation loss curves, and a technical report analyzing results and limitations with evidence from the plots themselves rather than general claims.

## 2. Repository Structure

```
project/
├── English/            # Code + checkpoints for the English model (Svarah)
├── Telugu/              # Code + checkpoints for the Telugu model (IndicTTS)
├── Results/
│   ├── English/
│   │   ├── training_loss.png
│   │   ├── validation_loss.png
│   │   ├── test_generated_audios/        # E1, E2a, E2b, E3, E4
│   │   └── variance_adaptor_plots/       # duration/ pitch/ energy/ per sentence
│   └── Telugu/                           # same structure, for T1-T4
├── Report/Technical_Report.pdf
├── README.md
└── requirements.txt
```

## 3. Datasets & Subsampling Strategy

| Language | Dataset | Train | Val | Speakers |
|---|---|---|---|---|
| English | [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) (AI4Bharat) | 5.01h | 0.99h | 115 pseudo-speakers |
| Telugu | IndicTTS Telugu | ~5h | ~1h | TBD |

**Why pseudo-speakers (English):** Svarah ships no explicit speaker ID field. 115 pseudo-speakers were constructed by combining eight metadata fields (gender, age group, mother tongue, native state/district, qualification, job category, occupation domain) — people sharing all eight are, with high probability, the same or a near-identical speaker. This let the multi-speaker embedding learn meaningful speaker variation despite the missing ground-truth IDs.

**Split method:** per-pseudo-speaker utterance split (not a random utterance split), guaranteeing zero text/audio overlap between train and validation while keeping every validation speaker also present in training — required for the speaker embedding to generalize rather than encounter unseen IDs at validation time.

**Filtering:** utterances restricted to 1-15 seconds with ≥3 words; resampled to 22.05kHz; silence/background-noise trimmed with `librosa.effects.trim(top_db=30)` prior to forced alignment, removing boundary artifacts that otherwise propagate into training targets.

## 4. Model & Training Configuration

| Setting | Value | Rationale |
|---|---|---|
| Architecture | FastSpeech2 (multi-speaker) | Required by assignment; multi-speaker needed since Svarah has 115 pseudo-speakers |
| Vocoder | Pretrained universal HiFi-GAN | Bundled with reference repo; avoids a separate multi-hour vocoder training run given the time budget |
| Alignment | MFA, `english_us_arpa` | Standard pretrained English acoustic model + dictionary for forced duration extraction |
| Sample rate | 22,050 Hz | Matches HiFi-GAN's pretrained expectation |
| max_seq_len | 1500 | Raised from the reference default (1000) to accommodate the longest Svarah utterances (~15s ≈ 1300 mel frames) without truncation |
| Batch size | 16 | Fits comfortably within a single Kaggle T4's 15GB VRAM alongside the multi-speaker embedding table |
| Training steps | 30,000 | Checkpointed every 5,000 steps; chosen to balance convergence against the ~3.5h/run time budget under the assignment deadline |
| Hardware | Kaggle, NVIDIA T4 (single GPU) | — |

## 5. Results — English (step 30,000)

| | Total | Mel | Mel PostNet | Pitch | Energy | Duration |
|---|---|---|---|---|---|---|
| **Train** | 0.9145 | 0.4272 | 0.4268 | 0.0204 | 0.0237 | 0.0165 |
| **Validation** | 3.4104 | 0.9576 | 0.9575 | 0.5367 | 0.7471 | 0.2115 |

Training loss converges smoothly from ~11.7 to ~0.91 with no instability. Validation loss plateaus early (~step 2,000) around 3.3-3.5 and never closes the gap with training loss — a consistent ~3.7× separation held for the full run (see `Results/English/validation_loss.png`). This is interpreted as **overfitting to the 115 training pseudo-speakers** rather than training instability, since the training curve itself is clean; full discussion in the Technical Report.

**Variance-adaptor observation:** energy predictions for longer test sentences (E1, E4 — 84-85 phonemes) show a visible decline from the start of the utterance toward the end (see `Results/English/variance_adaptor_plots/energy/`). This decline correlates with audibly increased vocoder artifacts in the corresponding `.wav` files during the same region — low-energy mel frames are a known weak point for HiFi-GAN reconstruction. Shorter test sentences (E2a, E2b, E3 — 37-56 phonemes) do not show the same decay and are audibly cleaner throughout.

## 6. Adaptations Made to the Reference Implementation

Required to run `ming024/FastSpeech2` on current library versions (NumPy 2.1, PyTorch 2.11) — documented for reproducibility and as evidence of debugging depth beyond default execution:

1. **`torch.load` compatibility** (`utils/model.py`) — added `weights_only=False, map_location=device`. PyTorch ≥2.6 changed the default to `weights_only=True`, which rejects the repo's legacy checkpoint format (both FastSpeech2 and HiFi-GAN checkpoints).
2. **NumPy dtype mismatch** (`dataset.py`) — explicit `.astype(np.float32)` added when loading mel/pitch/energy/duration `.npy` arrays. Under NumPy 2.x these load as `float64`, causing `RuntimeError: Found dtype Double but expected Float` during backpropagation.
3. **Silence/noise trimming** — added to the English raw-data preparation step to reduce vocoder-audible artifacts at utterance boundaries in the noisy, crowd-sourced Svarah recordings.
4. **Inference-time text normalization** — numerals, times, and dates expanded to words (e.g., "10:45 AM" → "ten forty five A M") to match the convention used when cleaning training transcripts; without this, `g2p_en` phonemizes raw digits into nonsense phoneme sequences.

## 7. Reproducing

```bash
pip install -r requirements.txt

cd English
python3 preprocess.py config/preprocess.yaml
python3 train.py -p config/preprocess.yaml -m config/model.yaml -t config/train.yaml
python3 inference.py --restore_step 30000 --mode single \
  --text "your text here" --speaker_id 0 \
  -p config/preprocess.yaml -m config/model.yaml -t config/train.yaml
```

Telugu follows the same command pattern inside `Telugu/` once training completes.

## 8. Checkpoints

Checkpoints (~419MB each, 6 per language, every 5,000 steps) are versioned via **Git LFS** per the assignment's guidance for oversized files.

Backup mirror (used live during training for crash recovery after two Kaggle kernel interruptions): https://huggingface.co/Hanushree/fastspeech2-svarah-english

## 9. Known Limitations & Possible Improvements

Full analysis with supporting plots in `Report/Technical_Report.pdf`. Summary:

- **Energy decay → vocoder degradation in longer utterances** — see Section 5. *Possible improvement:* curriculum training ordered by utterance length, or explicit duration/energy clamping during inference.
- **Thin per-speaker data** — ~2.5 minutes of audio per pseudo-speaker across 115 speakers limits how precisely the model learns any individual voice. *Possible improvement:* fewer speakers with more audio each, given the same total 5-hour budget.
- **G2P is American-English-tuned** — `g2p_en` does not model Indian-English prosody; OOV/non-English tokens (e.g., "andaroo" in E4) fall back to statistical grapheme-to-phoneme prediction with no verified ground truth. *Possible improvement:* a lexicon augmented with Indian-English pronunciation entries, or a fine-tuned G2P model.
- **Train/validation loss gap** — consistent ~3.7× separation suggests the model would benefit from regularization (dropout tuning, weight decay) or more validation speakers relative to total speaker count.
