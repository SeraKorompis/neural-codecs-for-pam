# Neural Audio Codecs for Passive Acoustic Monitoring

Code for the MSc AI Applications and Innovation dissertation:
**"AI Compression of Natural Soundscapes for Large-Scale Biodiversity Monitoring"**
Seraphina Korompis | Imperial College London | September 2026
Supervisor: Sarab Sethi, Co-supervisor: Sam Orchard



## Overview

This repository contains the pipeline for evaluating neural audio codecs (EnCodec and DAC)
alongside conventional codecs (MP3 and Opus) for passive acoustic monitoring (PAM) data.
The pipeline assesses:

- **Reconstruction quality** (SI-SNR and STFT distance) relative to published baselines
- **Downstream species detection performance** using pre-trained BirdNET and transfer-learned classifiers
- **Computational efficiency** (real-time factors on GPU and CPU)

Three bioacoustic datasets are evaluated:
- **AnuraSet:** tropical anurans, Brazil
- **Northeastern US Soundscapes:**  temperate birds, Ithaca NY, USA
- **Black-and-White Ruffed Lemur:**  tropical rainforest, Madagascar


## Repository Structure
```
├── src/ # Core reusable library modules
│ ├── compress.py # Codec compression pipeline
│ ├── evaluate.py # Reconstruction quality metrics
│ ├── codec_embeddings.py # BirdNET embedding extraction
│ ├── birdnet_eval.py # Pre-trained BirdNET evaluation
│ ├── figures.py # Figure generation utilities
│ └── utils.py # Shared utilities
│
├── scripts/
│ ├── reconstruction/ # Reconstruction quality evaluation
│ │ ├── run_benchmark.py # Run compression and evaluation for all datasets
│ │ ├── compress_anuraset_preprocessed.py # Separate running script without evaluation to obtained compressed data for transfer learned AnuraSet classifiers
│ │ ├── evaluate_recon_at_native_sr.py 
│ │ ├── compute_recon_metrics_rtf_agg.py # Computes RTF and reconstruction metrics
│ │ └── figures/ # Reconstruction quality figures
│ │
│ └── downstream/ # Downstream classification evaluation
│   ├── pretrained/ # Pre-trained BirdNET evaluation
│   ├── transfer_learned/ # Transfer-learned classifier pipeline
│   └── figures/ # Downstream performance figures
│
├── config.yaml.example # Example configuration file
└── README.md
```

## Installation

This project was developed and run on the Imperial College London CX3 HPC cluster.

### 1. Clone the repository

```bash
git clone https://github.com/SeraKorompis/neural-codecs-for-pam.git
cd neural-codecs-for-pam
```

### 2. Create conda environments

Two conda environments are used:

```bash
# For codec compression and reconstruction evaluation
conda create -n conda-codec-env python=3.10
conda activate conda-codec-env
pip install encodec descript-audio-codec pydub soundfile librosa torch torchaudio

# For BirdNET embedding extraction and classifier training
conda create -n conda-birdnet-env python=3.10
conda activate conda-birdnet-env
pip install birdnetlib soundfile librosa scikit-learn pandas numpy
```

### 3. Configure paths

```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml` to point to your data directories:

```yaml
paths:
  source_dir:   /path/to/source_audio
  results_dir:  /path/to/results
  recon_dir:    /path/to/reconstructed_audio
  figures_dir:  /path/to/figures
```

---

## Datasets

The following datasets are required and must be obtained separately:

| Dataset | Source |
|---------|--------|
| AnuraSet | [Zenodo](https://zenodo.org/record/8154908) |
| Northeastern US Soundscapes | [Zenodo](https://zenodo.org/record/7079124) |
| Black-and-White Ruffed Lemur | [Zenodo](https://zenodo.org/record/7540359) |

Once downloaded, datasets should be organised as follows:

```
source_dir/
├── anuraset/
│   ├── raw/
│   │   ├── raw_data/
│   │   └── strong_labels/
│   └── preprocessed/
│       └── metadata.csv
├── northeastern_us_soundscapes/
│   ├── soundscape_data/
│   ├── annotations.csv
│   └── species.csv
└── black_and_white_ruffed_lemur/
    ├── Audio1/
    └── Annotations1/
    ├── Audio3/
    └── Annotations3/
```

---

## Usage

### 1. Compress audio and evaluate reconstruction quality

```bash
python scripts/reconstruction/run_benchmark.py \
    --dataset anuraset --codec encodec --bitrate 24.0
```

### 2. Extract BirdNET embeddings

```bash
python scripts/downstream/transfer_learned/extract_birdnet_embeddings.py \
    --dataset northeastern
```

### 3. Train transfer-learned classifier

```bash
# Train on original audio
python scripts/downstream/transfer_learned/train_classifier.py \
    --dataset anuraset

# Train on compressed audio 
python scripts/downstream/transfer_learned/train_classifier.py \
    --dataset anuraset --train_source encodec --train_bitrate 24.0

# Northeastern US Soundscapes 
python scripts/downstream/transfer_learned/train_classifier.py \
    --dataset northeastern_ovr --train_source original
```
### 4. Evaluate classifier

```bash
python scripts/downstream/transfer_learned/evaluate_classifier.py \
    --dataset anuraset --source encodec --bitrate 24.0
```

### 5. Generate figures

```bash
python scripts/downstream/figures/plot_violin_with_spectrograms.py \
    --dataset anuraset
```