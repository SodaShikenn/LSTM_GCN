# Railway Ticket OCR using LSTM-GCN

## Overview

LSTM-GCN model for Chinese railway ticket OCR and information extraction. Combines **LSTM** (temporal features) with **GCN** (spatial features) to classify text regions into 10 categories.

### Key Features

- **Self-created dataset**: 21 training + 10 test images from Internet
- **10 label categories**: ticket_num, stations, train_num, seat, date, price, name, etc.
- **3000-character vocabulary** with text normalization
- **Graph-based spatial modeling** of document layout

---

## Architecture

```text
Image → PaddleOCR → Text Regions → Graph → LSTM → GCN → Classification
```

**Components**:

- `process/ocr.py` - PaddleOCR v5 text extraction
- `process/graph.py` - Spatial graph construction (normalized adjacency matrix)
- `utils.py` - Text normalization (digits→'0', letters→'A') and data loading
- `process/other.py` - Vocabulary/label generation
- `model.py` - LSTM-GCN architecture
- `train.py` - Training with progress tracking
- `predict.py` - Inference on new tickets

---

## Installation

```bash
# Create environment
conda create -n lstm-gcn python=3.11
conda activate lstm-gcn

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: PyTorch, PaddlePaddle≥3.0, PaddleOCR≥2.8, NetworkX, tqdm

---

## Quick Start

### 1. OCR Extraction

```bash
cd process
python ocr.py  # Processes all images in input/imgs/train/ and test/
```

### 2. Manual Labeling

Add `label` column to CSV files in `output/train/csv/` and save to `output/train/csv_label/`

### 3. Generate Vocabulary & Labels

```bash
cd process
python other.py
```

### 4. Build Graph Structures

```bash
cd process
python graph.py
```

### 5. Train Model

```bash
python train.py
```

### 6. Run Inference

```bash
python predict.py
```

---

## Project Structure

```text
code/
├── config.py              # Configuration
├── utils.py               # Data utilities
├── model.py               # LSTM-GCN model
├── train.py               # Training script
├── predict.py             # Inference script
├── process/
│   ├── ocr.py            # OCR processing
│   ├── graph.py          # Graph construction
│   └── other.py          # Vocab/label generation
├── input/imgs/           # Input images
│   ├── train/            # 21 training images
│   └── test/             # 10 test images
└── output/               # Results
    ├── vocab.txt         # 3000 characters
    ├── label.txt         # 10 categories
    ├── train/            # Training data
    └── test/             # Test data
```

---

## Recommended Hyperparameters

For small dataset (21 samples):

```python
EMBEDDING_DIM = 128      # ↑ from 100
HIDDEN_DIM = 128         # ↑ from 64
LR = 5e-4                # ↓ from 1e-3
EPOCH = 100              # ↓ from 200
DROPOUT = 0.3            # Add regularization
WEIGHT_DECAY = 1e-5      # Add L2 penalty
```

## Recent Improvements

✅ Fixed isolated node removal bug (reverse-sorted indices)

✅ PaddleOCR v3+ compatibility (`.predict()` API)

✅ PyTorch 2.7+ compatibility (`weights_only=False`)

✅ Comprehensive documentation for all modules

✅ Optimized training with progress bars and accuracy tracking

✅ Warning suppression (softmax deprecation, PaddlePaddle ccache)

---

## References

- **GCN Paper**: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (Kipf & Welling, 2016)
- **PyGCN**: (https://github.com/tkipf/pygcn)

---

## Languages

- [English](README_en.md)
- [日本語](README.md)
