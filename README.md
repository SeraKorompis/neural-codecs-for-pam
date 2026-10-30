# Neural Audio Codecs for Passive Acoustic Monitoring

Code for the MSc AI Applications and Innovation dissertation:

**"AI Compression of Natural Soundscapes for Large-Scale Biodiversity Monitoring"**

Seraphina Korompis | Imperial College London | September 2026
Supervisor: Sarab Sethi, Co-supervisor: Sam Orchard



## Overview

This repository contains the pipeline for evaluating neural audio codecs (EnCodec and DAC)
alongside conventional codecs (MP3 and Opus) for passive acoustic monitoring (PAM) data.
The pipeline assesses:

- **Reconstruction quality** (SI-SNR and STFT distance) relative to published performances
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

Two environments are used. The codec environment runs compression and
reconstruction evaluation. The BirdNET environment runs embedding extraction
and classifier training.

#### Codec environment (Python 3.13)

First create the environment and install ffmpeg and libopus with conda.

```bash
conda create -y -n conda-codec-env python=3.13
conda activate conda-codec-env
conda deactivate && conda activate conda-codec-env
conda install -y -c conda-forge "ffmpeg=8.0.0=gpl*" libopus=1.6.1
```

Then install the Python packages in one of two ways.

**Option A: from the requirements file** (exact versions used in this study)

```bash
pip install -r requirements-codec.txt
```

**Option B: install the packages yourself**

```bash
pip install torch==2.12.0 torchaudio==2.11.0 encodec==0.1.1 \
    descript-audio-codec==1.0.0 librosa==0.11.0 pydub==0.25.1 soundfile==0.14.0 \
    numpy scipy pandas scikit-learn joblib matplotlib adjusttext tqdm pyyaml
```

#### BirdNET environment (Python 3.11)

```bash
conda create -y -n conda-birdnet-env python=3.11
conda activate conda-birdnet-env
conda deactivate && conda activate conda-birdnet-env
```

**Option A: from the requirements file**

```bash
pip install -r requirements-birdnet.txt
```

**Option B: install the packages yourself**

```bash
pip install birdnet==0.2.16 tensorflow==2.15.1 numpy==1.26.4 \
    scikit-learn pandas scipy joblib pyyaml
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
| AnuraSet | [Zenodo](https://zenodo.org/records/8342596) |
| Northeastern US Soundscapes | [Zenodo](https://zenodo.org/records/7018484) |
| Black-and-White Ruffed Lemur | [Zenodo](https://zenodo.org/records/6331594) |

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
    --dataset anuraset --codec encodec --bitrates 24.0 --device cuda --mode full
```

Run once per dataset, codec and bitrate. Datasets: `anuraset`, `northeastern_birds`,
`lemur_s4a`, `lemur_swift1`, `lemur_swift2`. Codecs and bitrates: `encodec` (1.5, 3.0,
6.0, 12.0, 24.0), `dac` (2, 3, 6, 9), `mp3` (8, 16, 24), `opus` (8, 14, 24). Use
`--device cpu` for MP3 and Opus, and `--mode test` for a 3-file subset (point
`config.yaml` at an empty output folder first).

On a PBS cluster, `bash scripts/reconstruction/jobs/submit_benchmarks.sh` submits all 75
combinations (`DRY_RUN=1` prints the commands). It assumes `$HOME/miniforge3` and an L40S
GPU, so edit it for other clusters.

Then merge, aggregate and plot:

```bash
python scripts/reconstruction/merge_metrics.py --mode full
python scripts/reconstruction/compute_recon_metrics_agg.py
python scripts/reconstruction/compute_recon_metrics_rtf_agg.py
python scripts/reconstruction/figures/plot_metrics_vs_bitrate.py
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
