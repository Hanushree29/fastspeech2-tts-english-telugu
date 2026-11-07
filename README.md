# Multilingual Neural Text-to-Speech

### FastSpeech 2 | English & Telugu

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-ee4c2c?logo=pytorch)
![Languages](https://img.shields.io/badge/Languages-English%20%7C%20Telugu-purple)
![Status](https://img.shields.io/badge/Project-Research%20Implementation-2ea44f)

**A multilingual Text-to-Speech research implementation based on FastSpeech 2, covering English and Telugu speech generation.**

This repository brings together language-specific data preparation, acoustic-model training, inference pipelines, model checkpoints, generated speech samples, and experiment visualizations.

**Implemented languages:**

| Language | Dataset | Implementation | Model checkpoints |
|---|---|---|---|
| English | AI4Bharat Svarah | [`English/`](English/) | [Hugging Face](https://huggingface.co/Hanushree/fastspeech2-svarah-english) |
| Telugu | IndicTTS | [`Telugu/`](Telugu/) | [Hugging Face](https://huggingface.co/Hanushree/fastspeech2-telugu-v2) |

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Repository Organization](#repository-organization)
- [Datasets and Language Pipelines](#datasets-and-language-pipelines)
- [Model Checkpoints](#model-checkpoints)
- [Training Experiments](#training-experiments)
- [Generated Speech Samples](#generated-speech-samples)
- [Variance Adaptor Analysis](#variance-adaptor-analysis)
- [Installation](#installation)
- [Reproducibility](#reproducibility)
- [Technical Report](#technical-report)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

Text-to-Speech (TTS) converts written language into natural-sounding speech. This project implements FastSpeech 2 for two languages, with separate data-processing and configuration pipelines.

FastSpeech 2 is a non-autoregressive acoustic model. It uses explicit duration, pitch, and energy representations to improve speech generation and provide control over important prosodic characteristics.

### Project objectives

- Develop separate English and Telugu TTS pipelines.
- Prepare language-specific text and speech data.
- Train FastSpeech 2 acoustic models using configurable parameters.
- Generate mel-spectrograms from input text.
- Convert predicted mel-spectrograms into audio using HiFi-GAN.
- Examine training behaviour and variance-adaptor predictions.

## Architecture

The overall text-to-speech pipeline is:

```mermaid
flowchart TD
    A[Input Text] --> B[Text Normalization and Processing]
    B --> C[Phoneme / Symbol Sequence]
    C --> D[FastSpeech 2 Text Encoder]
    D --> E[Variance Adaptor]
    E --> F[Duration Prediction]
    E --> G[Pitch Prediction]
    E --> H[Energy Prediction]
    F --> I[Length Regulator]
    G --> I
    H --> I
    I --> J[Mel-Spectrogram Decoder]
    J --> K[Predicted Mel-Spectrogram]
    K --> L[HiFi-GAN Vocoder]
    L --> M[Generated Speech]
```

### Main components

| Component | Responsibility |
|---|---|
| Text processing | Converts input text into the model's supported sequence representation. |
| Text encoder | Encodes linguistic input into hidden representations. |
| Duration predictor | Estimates the duration of input tokens or phonemes. |
| Pitch predictor | Models pitch-related information. |
| Energy predictor | Models speech energy. |
| Length regulator | Expands encoder representations according to predicted durations. |
| Mel decoder | Generates the acoustic mel-spectrogram. |
| HiFi-GAN | Converts mel-spectrograms into waveform audio. |

## Repository Organization

```text
fastspeech2-tts-english-telugu/
│
├── English/
│   ├── audio/
│   ├── checkpoints/
│   ├── config/
│   ├── hifigan/
│   ├── lexicon/
│   ├── model/
│   ├── text/
│   ├── utils/
│   ├── dataset.py
│   ├── preprocess.py
│   ├── train.py
│   ├── inference.py
│   └── synthesize.py
│
├── Telugu/
│   ├── audio/
│   ├── checkpoints/
│   ├── config/
│   ├── hifigan/
│   ├── lexicon/
│   ├── model/
│   ├── text/
│   ├── utils/
│   ├── dataset.py
│   ├── preprocess.py
│   ├── train.py
│   ├── inference.py
│   └── synthesize.py
│
├── Results/
│   ├── English/
│   │   ├── training_loss.png
│   │   ├── validation_loss.png
│   │   ├── test_generated_audios/
│   │   └── variance_adaptor_plots/
│   └── Telugu/
│       ├── training_loss.png
│       ├── validation_loss.png
│       ├── test_generated_audios/
│       └── variance_adaptor_plots/
│
├── Report/
├── requirements.txt
└── README.md
```

## Datasets and Language Pipelines

### English — Svarah

The English pipeline uses the AI4Bharat Svarah speech dataset. The language-specific preprocessing and training configuration are maintained under `English/`.

### Telugu — IndicTTS

The Telugu pipeline uses the IndicTTS dataset. Its preprocessing, language configuration, training code, and inference implementation are maintained separately under `Telugu/`.

Each language has its own configuration files. Refer to the respective `config/` directory for the actual preprocessing, model, and training settings.

## Model Checkpoints

Trained model checkpoints are hosted on Hugging Face to avoid GitHub's large-file limitations.

### English checkpoints

Model repository: [Hanushree/fastspeech2-svarah-english](https://huggingface.co/Hanushree/fastspeech2-svarah-english)

| Training step | Checkpoint |
|---:|---|
| 5,000 | [5000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-svarah-english/resolve/main/5000.pth.tar) |
| 10,000 | [10000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-svarah-english/resolve/main/10000.pth.tar) |
| 15,000 | [15000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-svarah-english/resolve/main/15000.pth.tar) |
| 20,000 | [20000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-svarah-english/resolve/main/20000.pth.tar) |
| 25,000 | [25000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-svarah-english/resolve/main/25000.pth.tar) |
| 30,000 | [30000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-svarah-english/resolve/main/30000.pth.tar) |

### Telugu checkpoints

Model repository: [Hanushree/fastspeech2-telugu-v2](https://huggingface.co/Hanushree/fastspeech2-telugu-v2)

| Training step | Checkpoint |
|---:|---|
| 1,000 | [1000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-telugu-v2/resolve/main/1000.pth.tar) |
| 2,000 | [2000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-telugu-v2/resolve/main/2000.pth.tar) |
| 3,000 | [3000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-telugu-v2/resolve/main/3000.pth.tar) |
| 4,000 | [4000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-telugu-v2/resolve/main/4000.pth.tar) |
| 5,000 | [5000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-telugu-v2/resolve/main/5000.pth.tar) |
| 6,000 | [6000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-telugu-v2/resolve/main/6000.pth.tar) |
| 7,000 | [7000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-telugu-v2/resolve/main/7000.pth.tar) |
| 8,000 | [8000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-telugu-v2/resolve/main/8000.pth.tar) |
| 9,000 | [9000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-telugu-v2/resolve/main/9000.pth.tar) |
| 10,000 | [10000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-telugu-v2/resolve/main/10000.pth.tar) |
| 11,000 | [11000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-telugu-v2/resolve/main/11000.pth.tar) |
| 12,000 | [12000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-telugu-v2/resolve/main/12000.pth.tar) |
| 13,000 | [13000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-telugu-v2/resolve/main/13000.pth.tar) |
| 14,000 | [14000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-telugu-v2/resolve/main/14000.pth.tar) |
| 15,000 | [15000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-telugu-v2/resolve/main/15000.pth.tar) |
| 16,000 | [16000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-telugu-v2/resolve/main/16000.pth.tar) |
| 17,000 | [17000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-telugu-v2/resolve/main/17000.pth.tar) |
| 18,000 | [18000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-telugu-v2/resolve/main/18000.pth.tar) |
| 19,000 | [19000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-telugu-v2/resolve/main/19000.pth.tar) |
| 20,000 | [20000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-telugu-v2/resolve/main/20000.pth.tar) |

See [`English/checkpoints/`](English/checkpoints/) and [`Telugu/checkpoints/`](Telugu/checkpoints/) for language-specific checkpoint documentation.

---

## Experimental Results and Visual Comparison

The experimental outputs are organized in parallel columns to facilitate direct inspection of the English and Telugu FastSpeech 2 implementations.

## 1. Training Performance

### Training Loss

<table width='100%'>
<thead><tr>
<th width='50%'>ENGLISH</th>
<th width='50%'>TELUGU</th>
</tr></thead><tbody>
<tr><td width='50%' valign='top'><div align='center'><b>Training Loss</b><br><br><a href='Results/English/training_loss.png'><img src='Results/English/training_loss.png' alt='Training Loss' width='100%'></a></div></td><td width='50%' valign='top'><div align='center'><b>Training Loss</b><br><br><a href='Results/Telugu/training_loss.png'><img src='Results/Telugu/training_loss.png' alt='Training Loss' width='100%'></a></div></td></tr>
</tbody></table>

### Validation Loss

<table width='100%'>
<thead><tr>
<th width='50%'>ENGLISH</th>
<th width='50%'>TELUGU</th>
</tr></thead><tbody>
<tr><td width='50%' valign='top'><div align='center'><b>Validation Loss</b><br><br><a href='Results/English/validation_loss.png'><img src='Results/English/validation_loss.png' alt='Validation Loss' width='100%'></a></div></td><td width='50%' valign='top'><div align='center'><b>Validation Loss</b><br><br><a href='Results/Telugu/validation_loss.png'><img src='Results/Telugu/validation_loss.png' alt='Validation Loss' width='100%'></a></div></td></tr>
</tbody></table>

### Additional Experiment Graphs

<table width='100%'>
<thead><tr>
<th width='50%'>ENGLISH</th>
<th width='50%'>TELUGU</th>
</tr></thead><tbody>
<tr><td width='50%' valign='top'><div align='center'><b>Checkpoint Progression</b><br><br><a href='Results/English/checkpoint_progression.png'><img src='Results/English/checkpoint_progression.png' alt='Checkpoint Progression' width='100%'></a></div></td><td width='50%' valign='top'><i>Not available</i></td></tr>
<tr><td width='50%' valign='top'><div align='center'><b>E1 Spectrogram</b><br><br><a href='Results/English/test_generated_audios/E1_spectrogram.png'><img src='Results/English/test_generated_audios/E1_spectrogram.png' alt='E1 Spectrogram' width='100%'></a></div></td><td width='50%' valign='top'><i>Not available</i></td></tr>
<tr><td width='50%' valign='top'><div align='center'><b>E2A Spectrogram</b><br><br><a href='Results/English/test_generated_audios/E2a_spectrogram.png'><img src='Results/English/test_generated_audios/E2a_spectrogram.png' alt='E2A Spectrogram' width='100%'></a></div></td><td width='50%' valign='top'><i>Not available</i></td></tr>
<tr><td width='50%' valign='top'><div align='center'><b>E2B Spectrogram</b><br><br><a href='Results/English/test_generated_audios/E2b_spectrogram.png'><img src='Results/English/test_generated_audios/E2b_spectrogram.png' alt='E2B Spectrogram' width='100%'></a></div></td><td width='50%' valign='top'><i>Not available</i></td></tr>
<tr><td width='50%' valign='top'><div align='center'><b>E3 Spectrogram</b><br><br><a href='Results/English/test_generated_audios/E3_spectrogram.png'><img src='Results/English/test_generated_audios/E3_spectrogram.png' alt='E3 Spectrogram' width='100%'></a></div></td><td width='50%' valign='top'><i>Not available</i></td></tr>
<tr><td width='50%' valign='top'><div align='center'><b>E4 Spectrogram</b><br><br><a href='Results/English/test_generated_audios/E4_spectrogram.png'><img src='Results/English/test_generated_audios/E4_spectrogram.png' alt='E4 Spectrogram' width='100%'></a></div></td><td width='50%' valign='top'><i>Not available</i></td></tr>
</tbody></table>

---

## 2. Variance Adaptor Analysis

The following plots present the variance-adaptor outputs for duration, pitch, and energy.

### Duration Prediction

<table width='100%'>
<thead><tr>
<th width='50%'>ENGLISH</th>
<th width='50%'>TELUGU</th>
</tr></thead><tbody>
<tr><td width='50%' valign='top'><div align='center'><b>E1 Duration</b><br><br><a href='Results/English/variance_adaptor_plots/duration/E1_duration.png'><img src='Results/English/variance_adaptor_plots/duration/E1_duration.png' alt='E1 Duration' width='100%'></a></div></td><td width='50%' valign='top'><i>Not available</i></td></tr>
<tr><td width='50%' valign='top'><div align='center'><b>E2A Duration</b><br><br><a href='Results/English/variance_adaptor_plots/duration/E2a_duration.png'><img src='Results/English/variance_adaptor_plots/duration/E2a_duration.png' alt='E2A Duration' width='100%'></a></div></td><td width='50%' valign='top'><i>Not available</i></td></tr>
<tr><td width='50%' valign='top'><div align='center'><b>E2B Duration</b><br><br><a href='Results/English/variance_adaptor_plots/duration/E2b_duration.png'><img src='Results/English/variance_adaptor_plots/duration/E2b_duration.png' alt='E2B Duration' width='100%'></a></div></td><td width='50%' valign='top'><i>Not available</i></td></tr>
<tr><td width='50%' valign='top'><div align='center'><b>E3 Duration</b><br><br><a href='Results/English/variance_adaptor_plots/duration/E3_duration.png'><img src='Results/English/variance_adaptor_plots/duration/E3_duration.png' alt='E3 Duration' width='100%'></a></div></td><td width='50%' valign='top'><i>Not available</i></td></tr>
<tr><td width='50%' valign='top'><div align='center'><b>E4 Duration</b><br><br><a href='Results/English/variance_adaptor_plots/duration/E4_duration.png'><img src='Results/English/variance_adaptor_plots/duration/E4_duration.png' alt='E4 Duration' width='100%'></a></div></td><td width='50%' valign='top'><i>Not available</i></td></tr>
</tbody></table>

### Pitch Prediction

<table width='100%'>
<thead><tr>
<th width='50%'>ENGLISH</th>
<th width='50%'>TELUGU</th>
</tr></thead><tbody>
<tr><td width='50%' valign='top'><div align='center'><b>E1 Pitch</b><br><br><a href='Results/English/variance_adaptor_plots/pitch/E1_pitch.png'><img src='Results/English/variance_adaptor_plots/pitch/E1_pitch.png' alt='E1 Pitch' width='100%'></a></div></td><td width='50%' valign='top'><i>Not available</i></td></tr>
<tr><td width='50%' valign='top'><div align='center'><b>E2A Pitch</b><br><br><a href='Results/English/variance_adaptor_plots/pitch/E2a_pitch.png'><img src='Results/English/variance_adaptor_plots/pitch/E2a_pitch.png' alt='E2A Pitch' width='100%'></a></div></td><td width='50%' valign='top'><i>Not available</i></td></tr>
<tr><td width='50%' valign='top'><div align='center'><b>E2B Pitch</b><br><br><a href='Results/English/variance_adaptor_plots/pitch/E2b_pitch.png'><img src='Results/English/variance_adaptor_plots/pitch/E2b_pitch.png' alt='E2B Pitch' width='100%'></a></div></td><td width='50%' valign='top'><i>Not available</i></td></tr>
<tr><td width='50%' valign='top'><div align='center'><b>E3 Pitch</b><br><br><a href='Results/English/variance_adaptor_plots/pitch/E3_pitch.png'><img src='Results/English/variance_adaptor_plots/pitch/E3_pitch.png' alt='E3 Pitch' width='100%'></a></div></td><td width='50%' valign='top'><i>Not available</i></td></tr>
<tr><td width='50%' valign='top'><div align='center'><b>E4 Pitch</b><br><br><a href='Results/English/variance_adaptor_plots/pitch/E4_pitch.png'><img src='Results/English/variance_adaptor_plots/pitch/E4_pitch.png' alt='E4 Pitch' width='100%'></a></div></td><td width='50%' valign='top'><i>Not available</i></td></tr>
</tbody></table>

### Energy Prediction

<table width='100%'>
<thead><tr>
<th width='50%'>ENGLISH</th>
<th width='50%'>TELUGU</th>
</tr></thead><tbody>
<tr><td width='50%' valign='top'><div align='center'><b>E1 Energy</b><br><br><a href='Results/English/variance_adaptor_plots/energy/E1_energy.png'><img src='Results/English/variance_adaptor_plots/energy/E1_energy.png' alt='E1 Energy' width='100%'></a></div></td><td width='50%' valign='top'><i>Not available</i></td></tr>
<tr><td width='50%' valign='top'><div align='center'><b>E2A Energy</b><br><br><a href='Results/English/variance_adaptor_plots/energy/E2a_energy.png'><img src='Results/English/variance_adaptor_plots/energy/E2a_energy.png' alt='E2A Energy' width='100%'></a></div></td><td width='50%' valign='top'><i>Not available</i></td></tr>
<tr><td width='50%' valign='top'><div align='center'><b>E2B Energy</b><br><br><a href='Results/English/variance_adaptor_plots/energy/E2b_energy.png'><img src='Results/English/variance_adaptor_plots/energy/E2b_energy.png' alt='E2B Energy' width='100%'></a></div></td><td width='50%' valign='top'><i>Not available</i></td></tr>
<tr><td width='50%' valign='top'><div align='center'><b>E3 Energy</b><br><br><a href='Results/English/variance_adaptor_plots/energy/E3_energy.png'><img src='Results/English/variance_adaptor_plots/energy/E3_energy.png' alt='E3 Energy' width='100%'></a></div></td><td width='50%' valign='top'><i>Not available</i></td></tr>
<tr><td width='50%' valign='top'><div align='center'><b>E4 Energy</b><br><br><a href='Results/English/variance_adaptor_plots/energy/E4_energy.png'><img src='Results/English/variance_adaptor_plots/energy/E4_energy.png' alt='E4 Energy' width='100%'></a></div></td><td width='50%' valign='top'><i>Not available</i></td></tr>
</tbody></table>

---

## 3. Generated Speech Samples

The audio outputs are grouped by language for convenient evaluation.

<table width='100%'>
<tr><th width='50%'>ENGLISH AUDIO</th><th width='50%'>TELUGU AUDIO</th></tr>
<tr><td><a href='Results/English/test_generated_audios/E1.wav'>▶ E1.wav</a></td><td>—</td></tr>
<tr><td><a href='Results/English/test_generated_audios/E2a.wav'>▶ E2a.wav</a></td><td>—</td></tr>
<tr><td><a href='Results/English/test_generated_audios/E2b.wav'>▶ E2b.wav</a></td><td>—</td></tr>
<tr><td><a href='Results/English/test_generated_audios/E3.wav'>▶ E3.wav</a></td><td>—</td></tr>
<tr><td><a href='Results/English/test_generated_audios/E4.wav'>▶ E4.wav</a></td><td>—</td></tr>
</table>

---

## 4. Result Inventory

| Result | English | Telugu |
|---|---:|---:|
| Training loss graphs | 1 | 1 |
| Validation loss graphs | 1 | 1 |
| Variance adaptor plots | 15 | 0 |
| Generated audio samples | 5 | 0 |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Hanushree29/fastspeech2-tts-english-telugu.git
cd fastspeech2-tts-english-telugu
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Use the Python and PyTorch versions compatible with the implementation and your hardware environment.

## Reproducibility

Each language implementation is maintained independently. For an experiment:

1. Prepare the corresponding language dataset.
2. Review the language-specific preprocessing configuration.
3. Run the preprocessing pipeline.
4. Review the model and training configuration.
5. Train or load the relevant model checkpoint.
6. Run inference using the language-specific pipeline.
7. Save generated audio and experiment visualizations under `Results/`.

Consult the configuration files in each language directory for the actual parameter values. Download links for model checkpoints are provided above.

## Technical Report

The `Report/` directory is reserved for the technical documentation of the implementation and experiments.

## Acknowledgements

- [FastSpeech 2](https://arxiv.org/abs/2006.04558) — Non-autoregressive speech synthesis architecture.
- [AI4Bharat](https://ai4bharat.org/) — Indian-language AI research and resources.
- [HiFi-GAN](https://github.com/jik876/hifi-gan) — Neural vocoder.

## Author

**Hanushree M**

[GitHub](https://github.com/Hanushree29)

---

<p align="center">
  <b>Multilingual FastSpeech 2 · English & Telugu</b><br>
  Research implementation and experimental outputs
</p>
