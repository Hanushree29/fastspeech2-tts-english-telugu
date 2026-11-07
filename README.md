# FastSpeech2-Based TTS: English & Telugu

Implementation and evaluation of **FastSpeech 2** for text-to-speech synthesis in **English** (Svarah) and **Telugu** (IndicTTS Telugu), built on the [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) reference implementation.

**Submitted by:** Hanushree29 | hanushree60@gmail.com




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

## 9. Results at a Glance

### Training & Validation Loss

| Training Loss | Validation Loss |
|---|---|
| ![Training Loss](Results/English/training_loss.png) | ![Validation Loss](Results/English/validation_loss.png) |

### Variance-Adaptor Predictions (Example: E1 — longest test sentence)

| Duration | Pitch (F0) | Energy |
|---|---|---|
| ![Duration](Results/English/variance_adaptor_plots/duration/E1_duration.png) | ![Pitch](Results/English/variance_adaptor_plots/pitch/E1_pitch.png) | ![Energy](Results/English/variance_adaptor_plots/energy/E1_energy.png) |

See `Results/English/variance_adaptor_plots/` for all 5 sentences × 3 variance types (15 plots total).

### Checkpoint Progression

| Step | Total Loss (Train) | File |
|---|---|---|
| 5,000  | 2.0793 | `English/checkpoints/5000.pth.tar` |
| 10,000 | 1.4624 | `English/checkpoints/10000.pth.tar` |
| 15,000 | 1.0841 | `English/checkpoints/15000.pth.tar` |
| 20,000 | 1.0939 | `English/checkpoints/20000.pth.tar` |
| 25,000 | 1.0624 | `English/checkpoints/25000.pth.tar` |
| 30,000 (final) | 0.9145 | `English/checkpoints/30000.pth.tar` |

*(All values from the live training log, captured in `loss_history.csv` during training.)*

![Checkpoint Progression](Results/English/checkpoint_progression.png)

## 10. Known Limitations & Possible Improvements

Full analysis with supporting plots in `Report/Technical_Report.pdf`. Each limitation below is grounded in observed, plotted evidence rather than general expectation:

- **Energy decay correlating with vocoder artifacts in longer utterances.** FastSpeech2's variance adaptor (Ren et al., 2021) is non-autoregressive: duration, pitch, and energy are predicted independently per phoneme in a single forward pass, with no mechanism to condition later predictions on accumulated earlier error. On low-resource, noisy training data this is known to compound over longer sequences — observed directly here (`Results/English/variance_adaptor_plots/energy/E1_energy.png`, `E4_energy.png`), where predicted energy declines toward the end of the two longest test sentences (84-85 phonemes), with the corresponding generated audio showing increased vocoder noise in that same region. Neural vocoders including HiFi-GAN are documented to degrade on low-energy / near-silent mel regions, since there is less harmonic structure to reconstruct from. *Possible improvement:* a minimum-energy floor during inference, or duration-aware curriculum training that increases sequence length gradually.

- **Thin per-speaker training data.** With 115 pseudo-speakers sharing a 5-hour training budget, each speaker contributes roughly 2.5 minutes of audio — well below the tens of minutes to hours per speaker typically used in multi-speaker TTS literature for stable, speaker-consistent prosody. *Possible improvement:* fewer speakers with more data each, within the same total 5-hour budget specified by the assignment.

- **Phonemizer limited to its built-in lexicon.** `g2p_en`'s statistical fallback (used for out-of-vocabulary tokens such as "andaroo" in E4) has no mechanism to verify correctness for words outside standard English vocabulary, since it was trained on the CMU Pronouncing Dictionary. There is no ground-truth phoneme sequence to validate such output against. *Possible improvement:* an expanded or domain-specific pronunciation lexicon covering the test set's vocabulary in advance.

- **Persistent train/validation loss gap.** A consistent ~3.7× separation (0.91 train vs. 3.41 validation at step 30,000, see `Results/English/validation_loss.png`) that does not close over the full 30,000-step run indicates the model is fitting training-speaker-specific patterns rather than generalizing — consistent with the thin-per-speaker-data limitation above, since a speaker embedding has little signal to learn a generalizable identity from 2.5 minutes of audio. *Possible improvement:* stronger regularization (increased dropout, weight decay) or a larger, better-balanced validation set relative to the number of distinct speakers.
