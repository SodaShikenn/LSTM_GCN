# Railway Ticket OCR and Information Extraction using LSTM-GCN

## 📋 Project Overview

This project implements **Graph Convolutional Networks (GCN)** combined with **LSTM** for railway ticket text recognition and information extraction.

### What is LSTM-GCN?

LSTM-GCN is a hybrid model architecture that combines:

- **Temporal Features**: Captured by LSTM from sequential text data
- **Spatial Features**: Captured by GCN from the graph structure of document layout

This network architecture demonstrates excellent performance on various document types including:

- Railway tickets
- ID cards and passports
- Driving licenses
- Invoices and receipts
- Various structured documents

---

## 🏗️ Architecture

```text
Input Image → OCR (PaddleOCR) → Text Extraction → Graph Construction
                                                           ↓
                                                    LSTM Feature
                                                           ↓
                                                    GCN Layers
                                                           ↓
                                                    Classification
```

### Model Components

1. **OCR Module** (`process/ocr.py`)
   - Uses PaddleOCR v5 for text detection and recognition
   - Default models: PP-OCRv5_server_det & PP-OCRv5_server_rec

2. **Graph Construction** (`process/graph.py`)
   - Builds spatial relationships between text regions
   - Creates adjacency matrix for GCN input

3. **LSTM-GCN Model**
   - LSTM extracts sequential features from text
   - GCN processes spatial relationships
   - Classifies each text region into predefined categories

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or 3.12
- CUDA-compatible GPU (recommended)
- Anaconda or virtualenv

### Installation

1. **Clone the repository**

   ```bash
   git clone git@github.com:SodaShikenn/LSTM_GCN.git
   ```

2. **Create virtual environment** (optional but recommended)

   ```bash
   conda create -n lstm-gcn python=3.11
   conda activate lstm-gcn
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

- `torch` - PyTorch for deep learning
- `paddlepaddle==3.0.0` - PaddlePaddle framework
- `paddleocr==2.10.0` - OCR engine
- `scikit-learn` - Machine learning utilities
- `networkx` - Graph processing
- `matplotlib` - Visualization

---

## 📊 Dataset

The dataset used in this project is **self-created** by collecting railway ticket images from the Internet. The dataset includes various types of Chinese railway tickets with different layouts and formats.

### Dataset Characteristics

- **Source**: Collected from publicly available Internet sources
- **Type**: Chinese railway ticket images
- **Format**: JPEG/PNG images
- **Annotations**: Text regions with category labels (ticket number, station names, date, price, etc.)

### Data Organization

- Training images are stored in `input/imgs/train/`
- Test images are stored in `input/imgs/test/`
- OCR results and annotations are saved in `output/csv/`

**Note**: Due to privacy considerations, the original dataset is not included in this repository. Users can collect their own railway ticket images or similar structured documents for training and testing.

---

## 📁 Project Structure

```text
code/
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── config.py             # Configuration settings
├── process/
│   ├── ocr.py           # OCR processing module
│   └── graph.py         # Graph construction module
├── input/               # Input images directory
│   └── imgs/
│       ├── train/       # Training images
│       └── test/        # Test images
└── output/              # Output results directory
    ├── csv/             # Extracted text data (CSV)
    └── imgs_marked/     # Annotated images
```

---

## 💻 Usage

### 1. OCR Text Extraction

Run OCR on images to extract text regions:

```python
from process.ocr import OCR

ocr = OCR()
ocr.scan(
    file_path="input/imgs/train/ticket.jpg",
    output_path="output/csv/ticket.csv",
    marked_path="output/imgs_marked/ticket.jpg"
)
```

### 2. Graph Construction

Build spatial graph from extracted text:

```python
from process.graph import Graph

graph = Graph()
graph_dict, loss_idx = graph.connect(csv_path="output/csv/ticket.csv")
adj_matrix = graph.get_adjacency_norm(graph_dict)
```

### 3. Training (TODO)

Details to be added for model training.

### 4. Inference (TODO)

Details to be added for model inference.

---

## ⚙️ Configuration

Edit `config.py` to customize:

- Model paths
- Input/output directories
- Hyperparameters
- Vocabulary and label mappings

---

## 🔧 Troubleshooting

### Common Issues

#### Issue 1: PaddleOCR API Errors

- **Error**: `TypeError: 'PaddleOCR' object is not callable`
- **Solution**: Use `ocr.predict()` instead of `ocr.ocr()` for PaddleOCR v3+

#### Issue 2: PyTorch Model Loading

- **Error**: `Weights only load failed`
- **Solution**: Add `weights_only=False` parameter to `torch.load()`

#### Issue 3: MKL Library Warnings

- **Warning**: `Intel MKL function load error`
- **Solution**: Usually harmless, can be ignored or install Intel MKL library

---

## 📚 References

### Papers

- **Semi-Supervised Classification with Graph Convolutional Networks**
  - Paper: [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
  - Thomas N. Kipf, Max Welling (2016)

### Code

- **PyGCN Implementation**: [https://github.com/tkipf/pygcn](https://github.com/tkipf/pygcn)

### Tutorials

- **GNN Introduction**: [https://distill.pub/2021/gnn-intro/](https://distill.pub/2021/gnn-intro/)
- **GCN Explained**: [https://ai.plainenglish.io/graph-convolutional-networks-gcn-baf337d5cb6b](https://ai.plainenglish.io/graph-convolutional-networks-gcn-baf337d5cb6b)

---

## 📝 TODO

- [ ] Add model training script
- [ ] Add inference/prediction script
- [ ] Add model evaluation metrics
- [ ] Add data preprocessing utilities
- [ ] Add visualization tools for graph structure
- [ ] Add batch processing support
- [ ] Add configuration file documentation
- [ ] Add performance benchmarks

---

## 📄 License

See [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

## 🌐 Languages

- [English](README.md)
- [日本語](README_ja.md)
