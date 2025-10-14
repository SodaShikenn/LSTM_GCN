# LSTM-GCNを用いた鉄道切符OCR

> **プロジェクト状況**: ✅ **完了** - すべてのコンポーネントが実装され、動作しています

## 概要

中国の鉄道切符のOCRと情報抽出のためのLSTM-GCNモデル。**LSTM**（時系列特徴）と**GCN**（空間特徴）を組み合わせて、テキスト領域を10カテゴリに分類します。

### 主要機能

- **自作データセット**: インターネットから収集した訓練21枚 + テスト10枚
- **10ラベルカテゴリ**: 切符番号、駅名、列車番号、座席、日付、価格、名前など
- **3000文字の語彙** とテキスト正規化
- **グラフベースの空間モデリング** による文書レイアウト解析

---

## アーキテクチャ

```text
画像 → PaddleOCR → テキスト領域 → グラフ → LSTM → GCN → 分類
```

**コンポーネント**:

- `process/ocr.py` - PaddleOCR v5によるテキスト抽出
- `process/graph.py` - 空間グラフ構築（正規化隣接行列）
- `utils.py` - テキスト正規化（数字→'0'、文字→'A'）とデータロード
- `process/other.py` - 語彙/ラベル生成
- `model.py` - LSTM-GCNアーキテクチャ
- `train.py` - プログレストラッキング付き訓練
- `predict.py` - 新しい切符の推論

---

## インストール

```bash
# 環境作成
conda create -n lstm-gcn python=3.11
conda activate lstm-gcn

# 依存関係インストール
pip install -r requirements.txt
```

**必要なライブラリ**: PyTorch, PaddlePaddle≥3.0, PaddleOCR≥2.8, NetworkX, tqdm

---

## クイックスタート

### 1. OCR抽出

```bash
cd process
python ocr.py  # input/imgs/train/ と test/ の全画像を処理
```

### 2. 手動ラベリング

`output/train/csv/` のCSVファイルに `label` 列を追加し、`output/train/csv_label/` に保存

### 3. 語彙とラベルの生成

```bash
cd process
python other.py
```

### 4. グラフ構造の構築

```bash
cd process
python graph.py
```

### 5. モデル訓練

```bash
python train.py
```

### 6. 推論実行

```bash
python predict.py
```

---

## プロジェクト構造

```text
code/
├── config.py              # 設定
├── utils.py               # データユーティリティ
├── model.py               # LSTM-GCNモデル
├── train.py               # 訓練スクリプト
├── predict.py             # 推論スクリプト
├── process/
│   ├── ocr.py            # OCR処理
│   ├── graph.py          # グラフ構築
│   └── other.py          # 語彙/ラベル生成
├── input/imgs/           # 入力画像
│   ├── train/            # 訓練画像21枚
│   └── test/             # テスト画像10枚
└── output/               # 結果
    ├── vocab.txt         # 3000文字
    ├── label.txt         # 10カテゴリ
    ├── train/            # 訓練データ
    └── test/             # テストデータ
```

---

## 推奨ハイパーパラメータ

小規模データセット（21サンプル）の場合:

```python
EMBEDDING_DIM = 128      # 100から増加
HIDDEN_DIM = 128         # 64から増加
LR = 5e-4                # 1e-3から減少
EPOCH = 100              # 200から減少
DROPOUT = 0.3            # 正則化を追加
WEIGHT_DECAY = 1e-5      # L2ペナルティを追加
```

---

## トラブルシューティング

| 問題 | 解決策 |
|------|--------|
| `TypeError: 'PaddleOCR' object is not callable` | `ocr()` の代わりに `ocr.predict()` を使用 |
| `unexpected keyword argument 'cls'` | `use_textline_orientation=False` で初期化 |
| `Weights only load failed` | `torch.load()` に `weights_only=False` を追加 |
| `KeyError: 'label'` | 手動ラベリング（ステップ2）を完了 |
| 未知文字 | 自動的に `<UNK>` (ID: 0) にマッピング |

---

## 参考文献

- **GCN論文**: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (Kipf & Welling, 2016)
- **PyGCN**: (https://github.com/tkipf/pygcn)

---

## 言語

- [English](README_en.md)
- [日本語](README.md)
