# LSTM-GCNを用いた鉄道切符のOCRと情報抽出

## 📋 プロジェクト概要

本プロジェクトでは、**Graph Convolutional Networks (GCN)** と **LSTM** を組み合わせて、鉄道切符の文字認識と情報抽出を実現します。

### LSTM-GCNとは？

LSTM-GCNは、以下の2つの特徴を組み合わせたハイブリッドモデルアーキテクチャです：

- **時系列特徴（Temporal Features）**：LSTMによってシーケンシャルなテキストデータから抽出
- **空間特徴（Spatial Features）**：GCNによってドキュメントレイアウトのグラフ構造から抽出

このネットワークアーキテクチャは、以下のような様々な書類タイプにおいて優れた性能を発揮します：

- 鉄道切符
- 身分証明書やパスポート
- 運転免許証
- 請求書やレシート
- 各種構造化文書

---

## 🏗️ アーキテクチャ

```text
入力画像 → OCR (PaddleOCR) → テキスト抽出 → グラフ構築
                                                    ↓
                                              LSTM特徴量
                                                    ↓
                                                GCN層
                                                    ↓
                                                分類
```

### モデルコンポーネント

1. **OCRモジュール** (`process/ocr.py`)
   - PaddleOCR v5を使用してテキスト検出と認識を実行
   - デフォルトモデル：PP-OCRv5_server_det & PP-OCRv5_server_rec

2. **グラフ構築** (`process/graph.py`)
   - テキスト領域間の空間的関係を構築
   - GCN入力用の隣接行列を作成

3. **LSTM-GCNモデル**
   - LSTMがテキストからシーケンシャル特徴を抽出
   - GCNが空間的関係を処理
   - 各テキスト領域を事前定義されたカテゴリに分類

---

## 🚀 はじめに

### 前提条件

- Python 3.11 または 3.12
- CUDA対応GPU（推奨）
- AnacondaまたはVirtualenv

### インストール

1. **リポジトリのクローン**

   ```bash
   git clone <repository-url>
   cd code
   ```

2. **仮想環境の作成**（推奨）

   ```bash
   conda create -n lstm-gcn python=3.11
   conda activate lstm-gcn
   ```

3. **依存関係のインストール**

   ```bash
   pip install -r requirements.txt
   ```

### 依存ライブラリ

- `torch` - ディープラーニング用PyTorch
- `paddlepaddle==3.0.0` - PaddlePaddleフレームワーク
- `paddleocr==2.10.0` - OCRエンジン
- `scikit-learn` - 機械学習ユーティリティ
- `networkx` - グラフ処理
- `matplotlib` - 可視化

---

## 📊 データセット

本プロジェクトで使用されているデータセットは、インターネットから鉄道切符の画像を収集して**自作**したものです。データセットには、様々なレイアウトとフォーマットの中国の鉄道切符が含まれています。

### データセットの特徴

- **出典**：インターネット上で公開されているソースから収集
- **種類**：中国の鉄道切符画像
- **フォーマット**：JPEG/PNG画像
- **アノテーション**：カテゴリラベル付きのテキスト領域（切符番号、駅名、日付、価格など）

### データ構成

- 訓練画像は `input/imgs/train/` に格納
- テスト画像は `input/imgs/test/` に格納
- OCR結果とアノテーションは `output/csv/` に保存

**注意**：プライバシーの観点から、元のデータセットは本リポジトリには含まれていません。ユーザーは独自に鉄道切符画像または類似の構造化文書を収集して訓練・テストに使用できます。

---

## 📁 プロジェクト構造

```text
code/
├── README.md              # プロジェクトドキュメント
├── requirements.txt       # Python依存関係
├── config.py             # 設定ファイル
├── process/
│   ├── ocr.py           # OCR処理モジュール
│   └── graph.py         # グラフ構築モジュール
├── input/               # 入力画像ディレクトリ
│   └── imgs/
│       ├── train/       # 訓練画像
│       └── test/        # テスト画像
└── output/              # 出力結果ディレクトリ
    ├── csv/             # 抽出されたテキストデータ（CSV）
    └── imgs_marked/     # アノテーション付き画像
```

---

## 💻 使用方法

### 1. OCRテキスト抽出

画像からテキスト領域を抽出するOCRを実行：

```python
from process.ocr import OCR

ocr = OCR()
ocr.scan(
    file_path="input/imgs/train/ticket.jpg",
    output_path="output/csv/ticket.csv",
    marked_path="output/imgs_marked/ticket.jpg"
)
```

### 2. グラフ構築

抽出されたテキストから空間グラフを構築：

```python
from process.graph import Graph

graph = Graph()
graph_dict, loss_idx = graph.connect(csv_path="output/csv/ticket.csv")
adj_matrix = graph.get_adjacency_norm(graph_dict)
```

### 3. 訓練（TODO）

モデル訓練の詳細は後日追加予定。

### 4. 推論（TODO）

モデル推論の詳細は後日追加予定。

---

## ⚙️ 設定

`config.py` を編集してカスタマイズ：

- モデルのパス
- 入出力ディレクトリ
- ハイパーパラメータ
- 語彙とラベルのマッピング

---

## 🔧 トラブルシューティング

### よくある問題

#### 問題1：PaddleOCR APIエラー

- **エラー**：`TypeError: 'PaddleOCR' object is not callable`
- **解決策**：PaddleOCR v3+では `ocr.ocr()` の代わりに `ocr.predict()` を使用してください

#### 問題2：PyTorchモデル読み込みエラー

- **エラー**：`Weights only load failed`
- **解決策**：`torch.load()` に `weights_only=False` パラメータを追加してください

#### 問題3：MKLライブラリ警告

- **警告**：`Intel MKL function load error`
- **解決策**：通常は無害です。無視するか、Intel MKLライブラリをインストールしてください

---

## 📚 参考文献

### 論文

- **Semi-Supervised Classification with Graph Convolutional Networks**
  - 論文：[https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
  - Thomas N. Kipf, Max Welling (2016)

### コード

- **PyGCN実装**：[https://github.com/tkipf/pygcn](https://github.com/tkipf/pygcn)

### チュートリアル

- **GNN入門**：[https://distill.pub/2021/gnn-intro/](https://distill.pub/2021/gnn-intro/)
- **GCN解説**：[https://ai.plainenglish.io/graph-convolutional-networks-gcn-baf337d5cb6b](https://ai.plainenglish.io/graph-convolutional-networks-gcn-baf337d5cb6b)

---

## 📝 TODO

- [ ] モデル訓練スクリプトの追加
- [ ] 推論/予測スクリプトの追加
- [ ] モデル評価メトリクスの追加
- [ ] データ前処理ユーティリティの追加
- [ ] グラフ構造の可視化ツールの追加
- [ ] バッチ処理サポートの追加
- [ ] 設定ファイルドキュメントの追加
- [ ] パフォーマンスベンチマークの追加

---

## 📄 ライセンス

詳細は [LICENSE](LICENSE) ファイルをご覧ください。

---

## 🤝 コントリビューション

コントリビューションを歓迎します！以下の手順に従ってください：

1. リポジトリをフォーク
2. フィーチャーブランチを作成
3. 変更を加える
4. プルリクエストを送信

---

## 📧 お問い合わせ

質問や問題がある場合は、リポジトリにIssueを作成してください。

---

## 🌐 言語

- [English](README.md)
- [日本語](README_ja.md)
