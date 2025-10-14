# Railway Ticket OCR and Information Extraction using LSTM-GCN

## 📋 Project Overview

This project implements **Graph Convolutional Networks (GCN)** combined with **LSTM** for railway ticket text recognition and information extraction. The system processes Chinese railway ticket images, extracts text regions using OCR, builds a spatial graph structure, and classifies each text region into predefined categories.

### What is LSTM-GCN?

LSTM-GCN is a hybrid model architecture that combines:

- **Temporal Features**: Captured by LSTM from sequential text data (character-level encoding)
- **Spatial Features**: Captured by GCN from the graph structure of document layout (spatial relationships between text regions)

This network architecture demonstrates excellent performance on various document types including:

- Railway tickets
- ID cards and passports
- Driver's licenses
- Invoices and receipts
- Various structured documents

---

## 🏗️ Architecture

```text
Input Image → OCR (PaddleOCR) → Text Extraction → Graph Construction
                                         ↓                  ↓
                                   CSV (x,y,text)    Adjacency Matrix
                                         ↓                  ↓
                                 Text Normalization → LSTM Encoding
                                                           ↓
                                                 GCN Layers (2 layers)
                                                           ↓
                                                    Classification
                                                           ↓
                                           Labels (ticket_num, date, price, etc.)
```

### Model Components

1. **OCR Module** (`process/ocr.py`)
   - Uses PaddleOCR v5 for text detection and recognition
   - Default models: PP-OCRv5_server_det & PP-OCRv5_server_rec
   - Outputs bounding boxes and recognized text

2. **Graph Construction** (`process/graph.py`)
   - Builds spatial relationships between text regions
   - Creates directed graph: each node connects to nearest right and bottom neighbors
   - Generates normalized adjacency matrix for GCN: D^-0.5 × A × D^-0.5
   - Identifies isolated nodes

3. **Text Processing** (`utils.py`)
   - Text normalization: digits→'0', letters→'A'
   - Character-to-ID encoding using vocabulary
   - Handles unknown characters with `<UNK>` token

4. **Data Preparation** (`process/other.py`)
   - Vocabulary generation from training data
   - Label mapping generation
   - Supports up to 3000 unique characters

5. **Configuration** (`config.py`)
   - Centralized settings for paths and hyperparameters
   - Vocabulary size: 3000
   - Label categories: 10 types

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or 3.12
- CUDA-compatible GPU (recommended)
- Anaconda or virtualenv

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd code
   ```

2. **Create virtual environment** (recommended)

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
- `paddlepaddle>=3.0.0` - PaddlePaddle framework
- `paddleocr>=2.8.0` - OCR engine
- `scikit-learn` - Machine learning utilities
- `networkx` - Graph processing
- `matplotlib` - Visualization
- `pandas` - Data manipulation
- `tqdm` - Progress bars

---

## 📊 Dataset

The dataset used in this project is **self-created** by collecting railway ticket images from the Internet. The dataset includes various types of Chinese railway tickets with different layouts and formats.

### Dataset Characteristics

- **Source**: Collected from publicly available Internet sources
- **Type**: Chinese railway ticket images
- **Format**: JPEG/PNG images
- **Size**: 21 training images, 10 test images
- **Annotations**: Text regions with category labels

### Label Categories (10 types)

1. `ticket_num` - Ticket number
2. `starting_station` - Departure station
3. `destination_station` - Arrival station
4. `train_num` - Train number
5. `seat_number` - Seat information
6. `date` - Travel date and time
7. `ticket_grade` - Ticket class (e.g., second class)
8. `ticket_price` - Price
9. `name` - Passenger name
10. `other` - Other information

### Data Organization

- Training images: `input/imgs/train/` (21 images)
- Test images: `input/imgs/test/` (10 images)
- OCR results: `output/train/csv/` and `output/test/csv/`
- Labeled data: `output/train/csv_label/` and `output/test/csv_label/`
- Graph structures: `output/train/graph/` and `output/test/graph/`
- Vocabulary: `output/vocab.txt` (3000 characters)
- Labels: `output/label.txt` (10 categories)

**Note**: Due to privacy considerations, the original dataset is not included in this repository. Users can collect their own railway ticket images or similar structured documents for training and testing.

---

## 📁 Project Structure

```text
code/
├── README.md              # Japanese documentation
├── README_en.md           # English documentation (this file)
├── requirements.txt       # Python dependencies
├── config.py             # Configuration settings
├── utils.py              # Utility functions (data loading, text processing)
├── process/
│   ├── ocr.py           # OCR processing module
│   ├── graph.py         # Graph construction module
│   └── other.py         # Vocabulary and label generation
├── input/               # Input images directory
│   └── imgs/
│       ├── train/       # Training images (21 files)
│       └── test/        # Test images (10 files)
└── output/              # Output results directory
    ├── vocab.txt        # Character vocabulary (3000 chars)
    ├── label.txt        # Label categories (10 types)
    ├── train/
    │   ├── csv/         # OCR results (coordinates + text)
    │   ├── csv_label/   # Labeled OCR results
    │   ├── imgs_marked/ # Annotated images
    │   └── graph/       # Graph structures (adjacency matrices)
    └── test/
        ├── csv/         # Test OCR results
        ├── csv_label/   # Labeled test results
        ├── imgs_marked/ # Annotated test images
        └── graph/       # Test graph structures
```

### Module Descriptions

#### `config.py` - Configuration

Centralized configuration for the entire project:

```python
ROOT_PATH = os.path.dirname(__file__)
TRAIN_CSV_DIR = ROOT_PATH + '/output/train/csv_label/'
TEST_CSV_DIR = ROOT_PATH + '/output/test/csv_label/'
VOCAB_SIZE = 3000
WORD_UNK = '<UNK>'
WORD_UNK_ID = 0
```

#### `utils.py` - Utility Functions

Core utilities for data processing:

- **File I/O**: `file_dump()`, `file_load()` - Pickle serialization
- **Text Preprocessing**: `text_replace()` - Normalizes digits→'0', letters→'A'
- **Vocabulary Management**: `get_vocab()` - Loads character-to-ID mappings
- **Label Management**: `get_label()` - Loads label-to-ID mappings
- **Data Loading**: `load_data()` - Converts text and labels to numeric IDs

**Key Feature**: Text normalization reduces feature variance by replacing variable content (like ID numbers) with canonical forms, significantly improving model accuracy.

#### `process/ocr.py` - OCR Processing

Handles text extraction from images:

- **PaddleOCR Integration**: Uses PP-OCRv5 for detection and recognition
- **Initialization**: `PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)`
- **API**: `ocr.predict()` method (compatible with PaddleOCR v3+)
- **Output Format**: CSV with columns (index, x1, y1, x2, y2, text)
- **Marked Images**: Generates annotated images with red bounding boxes
- **Batch Processing**: Processes all images in train/test directories

#### `process/graph.py` - Graph Construction

Builds spatial graph structure from text regions:

- **Node Connection Strategy**: Each text region connects to:
  - Nearest neighbor to the right (if exists)
  - Nearest neighbor below (if exists)
- **Distance Calculation**: Euclidean distance between bounding boxes
- **Adjacency Matrix**: Normalized form: D^-0.5 × (A + I) × D^-0.5
  - A: Adjacency matrix
  - I: Identity matrix (adds self-loops)
  - D: Degree matrix
- **Isolated Nodes**: Identifies and returns indices of disconnected nodes
- **Output**: Saves adjacency matrix and isolated node indices as `.pkl` files

#### `process/other.py` - Data Preparation

Generates vocabulary and labels from training data:

- **Vocabulary Generation** (`generate_vocab()`):
  - Extracts all unique characters from training CSV files
  - Applies text normalization before extraction
  - Limits to VOCAB_SIZE (3000) most frequent characters
  - Adds `<UNK>` token as first entry (ID: 0)
  - Saves to `output/vocab.txt`

- **Label Generation** (`generate_label()`):
  - Collects all unique labels from training data
  - Creates label-to-ID mapping
  - Saves to `output/label.txt`

---

## 💻 Usage

### Complete Workflow

#### Step 1: OCR Text Extraction

Run OCR on images to extract text regions:

**Single Image:**

```python
from process.ocr import OCR

ocr = OCR()
ocr.scan(
    file_path="input/imgs/train/ticket.jpg",
    output_path="output/train/csv/ticket.csv",
    marked_path="output/train/imgs_marked/ticket.jpg"
)
```

**Batch Processing:**

```bash
cd process
python ocr.py  # Processes all images in train/ and test/ directories
```

**Output**: CSV files with columns: `index, x1, y1, x2, y2, text`

#### Step 2: Manual Labeling

After OCR extraction, manually label the CSV files:

1. Open CSV files in `output/train/csv/` or `output/test/csv/`
2. Add a `label` column with appropriate category
3. Save labeled files to `output/train/csv_label/` or `output/test/csv_label/`

**Example CSV format:**

```csv
,x1,y1,x2,y2,text,label
0,165,104,248,166,003B052946,ticket_num
1,354,209,398,359,麻城北站,starting_station
2,1164,200,1398,1168,D3069汉口站,destination_station
3,258,505,640,259,2019年03月10日14：50开,date
4,277,659,777,279,￥41.0元,ticket_price
```

#### Step 3: Generate Vocabulary and Labels

Create vocabulary and label mappings from labeled data:

```bash
cd process
python other.py
```

**Output**:
- `output/vocab.txt` - Character-to-ID mappings (3000 entries)
- `output/label.txt` - Label-to-ID mappings (10 categories)

#### Step 4: Build Graph Structures

Generate graph structures for all labeled data:

```bash
cd process
python graph.py
```

**Output**: Adjacency matrices saved as `.pkl` files in:
- `output/train/graph/` - Training graph structures
- `output/test/graph/` - Test graph structures

Each `.pkl` file contains: `[adjacency_matrix, isolated_node_indices]`

#### Step 5: Data Loading for Training

Load preprocessed data for model training:

```python
from utils import load_data
from process.graph import Graph
import pickle

# Load text and labels
inputs, targets = load_data('output/train/csv_label/ticket.csv')

# Load graph structure
with open('output/train/graph/ticket.pkl', 'rb') as f:
    adj_matrix, loss_idx = pickle.load(f)

# Remove isolated nodes from inputs
for i in sorted(loss_idx, reverse=True):
    inputs.pop(i)
    targets.pop(i)

print(f"Nodes: {len(inputs)}, Adjacency: {adj_matrix.shape}")
```

#### Step 6: Training (TODO)

Model training script to be implemented. Will include:

- LSTM-GCN model definition
- Training loop with loss computation
- Model checkpoint saving
- Validation metrics

#### Step 7: Inference (TODO)

Inference script to be implemented. Will include:

- Model loading
- Prediction on new tickets
- Result visualization

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
import os

# Root directory
ROOT_PATH = os.path.dirname(__file__)

# Training data paths
TRAIN_CSV_DIR = ROOT_PATH + '/output/train/csv_label/'
TRAIN_GRAPH_DIR = ROOT_PATH + '/output/train/graph/'

# Test data paths
TEST_CSV_DIR = ROOT_PATH + '/output/test/csv_label/'
TEST_GRAPH_DIR = ROOT_PATH + '/output/test/graph/'

# Vocabulary settings
WORD_UNK = '<UNK>'        # Unknown word token
WORD_UNK_ID = 0           # ID for unknown words
VOCAB_SIZE = 3000         # Maximum vocabulary size

# Vocabulary and label file paths
VOCAB_PATH = ROOT_PATH + '/output/vocab.txt'
LABEL_PATH = ROOT_PATH + '/output/label.txt'
```

---

## 🔧 Troubleshooting

### Common Issues

#### Issue 1: PaddleOCR API Errors

- **Error**: `TypeError: 'PaddleOCR' object is not callable`
- **Cause**: Trying to call PaddleOCR object directly: `self.ocr(file_path, cls=False)`
- **Solution**: Use the `predict()` method: `self.ocr.predict(file_path)`
- **Fixed in**: `process/ocr.py` line 20

#### Issue 2: Deprecated 'cls' Parameter

- **Error**: `TypeError: PaddleOCR.predict() got an unexpected keyword argument 'cls'`
- **Cause**: PaddleOCR v3+ replaced `cls` parameter with `use_textline_orientation`
- **Solution**: Initialize with `PaddleOCR(use_textline_orientation=False)` instead of passing `cls=False` to method
- **Fixed in**: `process/ocr.py` lines 14-18

#### Issue 3: PyTorch Model Loading

- **Error**: `Weights only load failed`
- **Cause**: PyTorch 2.7+ defaults to `weights_only=True` for security
- **Solution**: Add `weights_only=False` parameter: `torch.load('model.pth', weights_only=False)`

#### Issue 4: MKL Library Warnings

- **Warning**: `Intel MKL function load error`
- **Impact**: Usually harmless, doesn't affect functionality
- **Solution**: Can be ignored or install Intel MKL library: `conda install mkl`

#### Issue 5: Missing CSV Label Column

- **Error**: `KeyError: 'label'` when running graph.py or data loading
- **Cause**: CSV files in `csv_label/` directories missing the `label` column
- **Solution**: Ensure manual labeling step (Step 2) is completed

#### Issue 6: Vocabulary Size Exceeded

- **Behavior**: Characters not found in vocabulary
- **Handling**: Unknown characters are automatically mapped to `<UNK>` (ID: 0)
- **Solution**: Increase `VOCAB_SIZE` in config.py if needed

#### Issue 7: Empty Graph Structures

- **Error**: Graph has no connections
- **Cause**: Text regions too far apart or no spatial overlap
- **Solution**: Check OCR extraction quality and bounding box coordinates

---

## 📚 References

### Papers

- **Semi-Supervised Classification with Graph Convolutional Networks**
  - Paper: [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
  - Thomas N. Kipf, Max Welling (2016)
  - Introduces spectral graph convolution for semi-supervised learning

### Code

- **PyGCN Implementation**: [https://github.com/tkipf/pygcn](https://github.com/tkipf/pygcn)
  - Reference implementation of GCN in PyTorch

### Tutorials

- **GNN Introduction**: [https://distill.pub/2021/gnn-intro/](https://distill.pub/2021/gnn-intro/)
  - Interactive introduction to Graph Neural Networks

- **GCN Explained**: [https://ai.plainenglish.io/graph-convolutional-networks-gcn-baf337d5cb6b](https://ai.plainenglish.io/graph-convolutional-networks-gcn-baf337d5cb6b)
  - Detailed explanation of GCN architecture

---

## 📝 TODO

### High Priority

- [ ] Implement LSTM-GCN model architecture
- [ ] Add training script with loss computation
- [ ] Add inference/prediction script
- [ ] Add model evaluation metrics (precision, recall, F1)

### Medium Priority

- [ ] Add data augmentation techniques
- [ ] Add visualization tools for graph structure
- [ ] Add automated data labeling tools
- [ ] Add cross-validation support
- [ ] Add model architecture diagram

### Low Priority

- [ ] Add batch processing support for inference
- [ ] Add performance benchmarks
- [ ] Add data statistics and analysis tools
- [ ] Add export to different formats (JSON, XML)
- [ ] Add web interface for demo

---

## 📄 License

See [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guide for Python code
- Add docstrings to all functions
- Update README if adding new features
- Test your changes before submitting

---

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

## 🌐 Languages

- [English](README_en.md)
- [日本語](README.md)

---

## 📈 Project Statistics

- **Dataset Size**: 31 images total (21 train + 10 test)
- **Label Categories**: 10 types
- **Vocabulary Size**: 3000 characters
- **Python Files**: 5 modules
- **Lines of Code**: ~500 lines

---

**Last Updated**: 2025-01-15
