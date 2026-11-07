# Multilingual FastSpeech 2 Text-to-Speech System

### English (Svarah) and Telugu

A multilingual Text-to-Speech implementation using the FastSpeech 2 architecture.
This repository contains separate English and Telugu implementations,
preprocessing pipelines, configurations, checkpoints, inference code and results.

## Languages

| Language | Dataset | Directory |
|---|---|---|
| English | AI4Bharat Svarah | `English/` |
| Telugu | IndicTTS | `Telugu/` |

## Architecture

```text
Input Text
    |
Text Processing / Phoneme Conversion
    |
Text Encoder
    |
Variance Adaptor
(Duration, Pitch, Energy)
    |
Mel-Spectrogram Decoder
    |
HiFi-GAN Vocoder
    |
Generated Speech
```

## Repository Structure

```text
.
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
│   └── inference.py
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
│   └── inference.py
├── Results/
│   ├── English/
│   └── Telugu/
├── Report/
├── requirements.txt
└── README.md
```

## Model Checkpoints

Large model checkpoints are hosted on Hugging Face.
The language-specific checkpoint directories contain download links.

### English

[Hugging Face: Hanushree/fastspeech2-svarah-english](https://huggingface.co/Hanushree/fastspeech2-svarah-english)

| Step | Checkpoint |
|---:|---|
| 5,000 | [5000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-svarah-english/resolve/main/5000.pth.tar) |
| 10,000 | [10000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-svarah-english/resolve/main/10000.pth.tar) |
| 15,000 | [15000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-svarah-english/resolve/main/15000.pth.tar) |
| 20,000 | [20000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-svarah-english/resolve/main/20000.pth.tar) |
| 25,000 | [25000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-svarah-english/resolve/main/25000.pth.tar) |
| 30,000 | [30000.pth.tar](https://huggingface.co/Hanushree/fastspeech2-svarah-english/resolve/main/30000.pth.tar) |

### Telugu

[Hugging Face: Hanushree/fastspeech2-telugu-v2](https://huggingface.co/Hanushree/fastspeech2-telugu-v2)

| Step | Checkpoint |
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

## Training and Validation Graphs

### English

**Checkpoint Progression**

![checkpoint progression](Results/English/checkpoint_progression.png)

**E1 Spectrogram**

![E1 spectrogram](Results/English/test_generated_audios/E1_spectrogram.png)

**E2A Spectrogram**

![E2a spectrogram](Results/English/test_generated_audios/E2a_spectrogram.png)

**E2B Spectrogram**

![E2b spectrogram](Results/English/test_generated_audios/E2b_spectrogram.png)

**E3 Spectrogram**

![E3 spectrogram](Results/English/test_generated_audios/E3_spectrogram.png)

**E4 Spectrogram**

![E4 spectrogram](Results/English/test_generated_audios/E4_spectrogram.png)

**Training Loss**

![training loss](Results/English/training_loss.png)

**Validation Loss**

![validation loss](Results/English/validation_loss.png)

**E1 Duration**

![E1 duration](Results/English/variance_adaptor_plots/duration/E1_duration.png)

**E2A Duration**

![E2a duration](Results/English/variance_adaptor_plots/duration/E2a_duration.png)

**E2B Duration**

![E2b duration](Results/English/variance_adaptor_plots/duration/E2b_duration.png)

**E3 Duration**

![E3 duration](Results/English/variance_adaptor_plots/duration/E3_duration.png)

**E4 Duration**

![E4 duration](Results/English/variance_adaptor_plots/duration/E4_duration.png)

**E1 Energy**

![E1 energy](Results/English/variance_adaptor_plots/energy/E1_energy.png)

**E2A Energy**

![E2a energy](Results/English/variance_adaptor_plots/energy/E2a_energy.png)

**E2B Energy**

![E2b energy](Results/English/variance_adaptor_plots/energy/E2b_energy.png)

**E3 Energy**

![E3 energy](Results/English/variance_adaptor_plots/energy/E3_energy.png)

**E4 Energy**

![E4 energy](Results/English/variance_adaptor_plots/energy/E4_energy.png)

**E1 Pitch**

![E1 pitch](Results/English/variance_adaptor_plots/pitch/E1_pitch.png)

**E2A Pitch**

![E2a pitch](Results/English/variance_adaptor_plots/pitch/E2a_pitch.png)

**E2B Pitch**

![E2b pitch](Results/English/variance_adaptor_plots/pitch/E2b_pitch.png)

**E3 Pitch**

![E3 pitch](Results/English/variance_adaptor_plots/pitch/E3_pitch.png)

**E4 Pitch**

![E4 pitch](Results/English/variance_adaptor_plots/pitch/E4_pitch.png)

### Telugu

**Training Loss**

![training loss](Results/Telugu/training_loss.png)

**Validation Loss**

![validation loss](Results/Telugu/validation_loss.png)


## Generated Speech Samples

- [English audio samples](Results/English/test_generated_audios/)
- [Telugu audio samples](Results/Telugu/test_generated_audios/)

## Variance Adaptor Visualizations

The variance adaptor models duration, pitch and energy.

- `Results/English/variance_adaptor_plots/`
- `Results/Telugu/variance_adaptor_plots/`

## Installation

```bash
git clone https://github.com/Hanushree29/fastspeech2-tts-english-telugu.git
cd fastspeech2-tts-english-telugu
pip install -r requirements.txt
```

## Training and Inference

Use the respective language directory and its configuration files
for preprocessing, training and inference.
Download the required checkpoint using the links in the checkpoint README.

## Technical Report

See the `Report/` directory for the technical report and experiment analysis.

## Author

**Hanushree M**

[GitHub Profile](https://github.com/Hanushree29)

---

English and Telugu FastSpeech 2 implementations.
