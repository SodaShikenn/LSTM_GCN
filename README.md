# LSTM-GCNを用いた鉄道切符のOCRと情報抽出

## 📋 プロジェクト概要

本プロジェクトでは、**Graph Convolutional Networks (GCN)** と **LSTM** を組み合わせて、鉄道切符の文字認識と情報抽出を実現します。システムは中国の鉄道切符画像を処理し、OCRを使用してテキスト領域を抽出し、空間グラフ構造を構築し、各テキスト領域を事前定義されたカテゴリに分類します。

### LSTM-GCNとは？

LSTM-GCNは、以下の2つの特徴を組み合わせたハイブリッドモデルアーキテクチャです：

- **時系列特徴（Temporal Features）**：LSTMによって、シーケンシャルなテキストデータ（文字レベルエンコーディング）から抽出
- **空間特徴（Spatial Features）**：GCNによって、ドキュメントレイアウトのグラフ構造（テキスト領域間の空間的関係）から抽出

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
                                    ↓              ↓
                              CSV (x,y,text)  隣接行列
                                    ↓              ↓
                             テキスト正規化 → LSTMエンコーディング
                                                   ↓
                                            GCN層（2層）
                                                   ↓
                                                分類
                                                   ↓
                                ラベル（ticket_num, date, priceなど）
```

### モデルコンポーネント

1. **OCRモジュール** (`process/ocr.py`)
   - PaddleOCR v5を使用してテキスト検出と認識を実行
   - デフォルトモデル：PP-OCRv5_server_det & PP-OCRv5_server_rec
   - バウンディングボックスと認識されたテキストを出力

2. **グラフ構築** (`process/graph.py`)
   - テキスト領域間の空間的関係を構築
   - 有向グラフを作成：各ノードは最も近い右と下の隣接ノードに接続
   - GCN用の正規化隣接行列を生成：D^-0.5 × A × D^-0.5
   - 孤立ノードを識別

3. **テキスト処理** (`utils.py`)
   - テキスト正規化：数字→'0'、文字→'A'
   - 語彙を使用した文字からIDへのエンコーディング
   - `<UNK>`トークンで未知文字を処理

4. **データ準備** (`process/other.py`)
   - 訓練データからの語彙生成
   - ラベルマッピング生成
   - 最大3000の一意な文字をサポート

5. **設定** (`config.py`)
   - パスとハイパーパラメータの一元管理
   - 語彙サイズ：3000
   - ラベルカテゴリ：10種類

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
- `paddlepaddle>=3.0.0` - PaddlePaddleフレームワーク
- `paddleocr>=2.8.0` - OCRエンジン
- `scikit-learn` - 機械学習ユーティリティ
- `networkx` - グラフ処理
- `matplotlib` - 可視化
- `pandas` - データ操作
- `tqdm` - プログレスバー

---

## 📊 データセット

本プロジェクトで使用されているデータセットは、インターネットから鉄道切符の画像を収集して**自作**したものです。データセットには、様々なレイアウトとフォーマットの中国の鉄道切符が含まれています。

### データセットの特徴

- **出典**：インターネット上で公開されているソースから収集
- **種類**：中国の鉄道切符画像
- **フォーマット**：JPEG/PNG画像
- **サイズ**：訓練画像21枚、テスト画像10枚
- **アノテーション**：カテゴリラベル付きのテキスト領域

### ラベルカテゴリ（10種類）

1. `ticket_num` - 切符番号
2. `starting_station` - 出発駅
3. `destination_station` - 到着駅
4. `train_num` - 列車番号
5. `seat_number` - 座席情報
6. `date` - 旅行日時
7. `ticket_grade` - 切符クラス（例：二等座）
8. `ticket_price` - 価格
9. `name` - 乗客名
10. `other` - その他の情報

### データ構成

- 訓練画像：`input/imgs/train/`（21画像）
- テスト画像：`input/imgs/test/`（10画像）
- OCR結果：`output/train/csv/` および `output/test/csv/`
- ラベル付きデータ：`output/train/csv_label/` および `output/test/csv_label/`
- グラフ構造：`output/train/graph/` および `output/test/graph/`
- 語彙：`output/vocab.txt`（3000文字）
- ラベル：`output/label.txt`（10カテゴリ）

**注意**：プライバシーの観点から、元のデータセットは本リポジトリには含まれていません。ユーザーは独自に鉄道切符画像または類似の構造化文書を収集して訓練・テストに使用できます。

---

## 📁 プロジェクト構造

```text
code/
├── README.md              # 日本語ドキュメント（本ファイル）
├── README_en.md           # 英語ドキュメント
├── requirements.txt       # Python依存関係
├── config.py             # 設定ファイル
├── utils.py              # ユーティリティ関数（データロード、テキスト処理）
├── process/
│   ├── ocr.py           # OCR処理モジュール
│   ├── graph.py         # グラフ構築モジュール
│   └── other.py         # 語彙とラベル生成
├── input/               # 入力画像ディレクトリ
│   └── imgs/
│       ├── train/       # 訓練画像（21ファイル）
│       └── test/        # テスト画像（10ファイル）
└── output/              # 出力結果ディレクトリ
    ├── vocab.txt        # 文字語彙（3000文字）
    ├── label.txt        # ラベルカテゴリ（10種類）
    ├── train/
    │   ├── csv/         # OCR結果（座標 + テキスト）
    │   ├── csv_label/   # ラベル付きOCR結果
    │   ├── imgs_marked/ # アノテーション付き画像
    │   └── graph/       # グラフ構造（隣接行列）
    └── test/
        ├── csv/         # テストOCR結果
        ├── csv_label/   # ラベル付きテスト結果
        ├── imgs_marked/ # アノテーション付きテスト画像
        └── graph/       # テストグラフ構造
```

### モジュール説明

#### `config.py` - 設定

プロジェクト全体の一元設定：

```python
ROOT_PATH = os.path.dirname(__file__)
TRAIN_CSV_DIR = ROOT_PATH + '/output/train/csv_label/'
TEST_CSV_DIR = ROOT_PATH + '/output/test/csv_label/'
VOCAB_SIZE = 3000
WORD_UNK = '<UNK>'
WORD_UNK_ID = 0
```

#### `utils.py` - ユーティリティ関数

データ処理のコアユーティリティ：

- **ファイルI/O**：`file_dump()`, `file_load()` - Pickleシリアライゼーション
- **テキスト前処理**：`text_replace()` - 数字→'0'、文字→'A'に正規化
- **語彙管理**：`get_vocab()` - 文字からIDへのマッピングを読み込み
- **ラベル管理**：`get_label()` - ラベルからIDへのマッピングを読み込み
- **データロード**：`load_data()` - テキストとラベルを数値IDに変換

**主要機能**：テキスト正規化は、可変コンテンツ（ID番号など）を正規形式に置き換えることで特徴量の分散を減らし、モデルの精度を大幅に向上させます。

#### `process/ocr.py` - OCR処理

画像からのテキスト抽出を処理：

- **PaddleOCR統合**：PP-OCRv5を使用して検出と認識を実行
- **初期化**：`PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)`
- **API**：`ocr.predict()`メソッド（PaddleOCR v3+互換）
- **出力形式**：列（index, x1, y1, x2, y2, text）を持つCSV
- **マーク画像**：赤いバウンディングボックス付きのアノテーション画像を生成
- **バッチ処理**：train/testディレクトリ内のすべての画像を処理

#### `process/graph.py` - グラフ構築

テキスト領域から空間グラフ構造を構築：

- **ノード接続戦略**：各テキスト領域は以下に接続：
  - 右側の最も近い隣接ノード（存在する場合）
  - 下側の最も近い隣接ノード（存在する場合）
- **距離計算**：バウンディングボックス間のユークリッド距離
- **隣接行列**：正規化形式：D^-0.5 × (A + I) × D^-0.5
  - A：隣接行列
  - I：単位行列（自己ループを追加）
  - D：次数行列
- **孤立ノード**：切断されたノードのインデックスを識別して返す
- **出力**：隣接行列と孤立ノードインデックスを`.pkl`ファイルとして保存

#### `process/other.py` - データ準備

訓練データから語彙とラベルを生成：

- **語彙生成** (`generate_vocab()`):
  - 訓練CSVファイルからすべての一意な文字を抽出
  - 抽出前にテキスト正規化を適用
  - VOCAB_SIZE（3000）の最も頻度の高い文字に制限
  - 最初のエントリとして`<UNK>`トークンを追加（ID: 0）
  - `output/vocab.txt`に保存

- **ラベル生成** (`generate_label()`):
  - 訓練データからすべての一意なラベルを収集
  - ラベルからIDへのマッピングを作成
  - `output/label.txt`に保存

---

## 💻 使用方法

### 完全なワークフロー

#### ステップ1：OCRテキスト抽出

画像からテキスト領域を抽出するOCRを実行：

**単一画像：**

```python
from process.ocr import OCR

ocr = OCR()
ocr.scan(
    file_path="input/imgs/train/ticket.jpg",
    output_path="output/train/csv/ticket.csv",
    marked_path="output/train/imgs_marked/ticket.jpg"
)
```

**バッチ処理：**

```bash
cd process
python ocr.py  # train/とtest/ディレクトリ内のすべての画像を処理
```

**出力**：列を持つCSVファイル：`index, x1, y1, x2, y2, text`

#### ステップ2：手動ラベリング

OCR抽出後、CSVファイルに手動でラベルを付ける：

1. `output/train/csv/`または`output/test/csv/`内のCSVファイルを開く
2. 適切なカテゴリで`label`列を追加
3. ラベル付きファイルを`output/train/csv_label/`または`output/test/csv_label/`に保存

**CSVフォーマット例：**

```csv
,x1,y1,x2,y2,text,label
0,165,104,248,166,003B052946,ticket_num
1,354,209,398,359,麻城北站,starting_station
2,1164,200,1398,1168,D3069汉口站,destination_station
3,258,505,640,259,2019年03月10日14：50開,date
4,277,659,777,279,￥41.0元,ticket_price
```

#### ステップ3：語彙とラベルの生成

ラベル付きデータから語彙とラベルマッピングを作成：

```bash
cd process
python other.py
```

**出力**：
- `output/vocab.txt` - 文字からIDへのマッピング（3000エントリ）
- `output/label.txt` - ラベルからIDへのマッピング（10カテゴリ）

#### ステップ4：グラフ構造の構築

すべてのラベル付きデータのグラフ構造を生成：

```bash
cd process
python graph.py
```

**出力**：`.pkl`ファイルとして保存された隣接行列：
- `output/train/graph/` - 訓練グラフ構造
- `output/test/graph/` - テストグラフ構造

各`.pkl`ファイルの内容：`[隣接行列, 孤立ノードインデックス]`

#### ステップ5：訓練用データロード

モデル訓練用の前処理済みデータをロード：

```python
from utils import load_data
from process.graph import Graph
import pickle

# テキストとラベルをロード
inputs, targets = load_data('output/train/csv_label/ticket.csv')

# グラフ構造をロード
with open('output/train/graph/ticket.pkl', 'rb') as f:
    adj_matrix, loss_idx = pickle.load(f)

# 孤立ノードをinputsから削除
for i in sorted(loss_idx, reverse=True):
    inputs.pop(i)
    targets.pop(i)

print(f"ノード数: {len(inputs)}, 隣接行列: {adj_matrix.shape}")
```

#### ステップ6：訓練（TODO）

実装予定のモデル訓練スクリプト。含まれる内容：

- LSTM-GCNモデル定義
- 損失計算を含む訓練ループ
- モデルチェックポイント保存
- 検証メトリクス

#### ステップ7：推論（TODO）

実装予定の推論スクリプト。含まれる内容：

- モデルロード
- 新しい切符への予測
- 結果の可視化

---

## ⚙️ 設定

`config.py`を編集してカスタマイズ：

```python
import os

# ルートディレクトリ
ROOT_PATH = os.path.dirname(__file__)

# 訓練データパス
TRAIN_CSV_DIR = ROOT_PATH + '/output/train/csv_label/'
TRAIN_GRAPH_DIR = ROOT_PATH + '/output/train/graph/'

# テストデータパス
TEST_CSV_DIR = ROOT_PATH + '/output/test/csv_label/'
TEST_GRAPH_DIR = ROOT_PATH + '/output/test/graph/'

# 語彙設定
WORD_UNK = '<UNK>'        # 未知語トークン
WORD_UNK_ID = 0           # 未知語のID
VOCAB_SIZE = 3000         # 最大語彙サイズ

# 語彙とラベルファイルパス
VOCAB_PATH = ROOT_PATH + '/output/vocab.txt'
LABEL_PATH = ROOT_PATH + '/output/label.txt'
```

---

## 🔧 トラブルシューティング

### よくある問題

#### 問題1：PaddleOCR APIエラー

- **エラー**：`TypeError: 'PaddleOCR' object is not callable`
- **原因**：PaddleOCRオブジェクトを直接呼び出そうとしている：`self.ocr(file_path, cls=False)`
- **解決策**：`predict()`メソッドを使用：`self.ocr.predict(file_path)`
- **修正箇所**：`process/ocr.py` 20行目

#### 問題2：非推奨の'cls'パラメータ

- **エラー**：`TypeError: PaddleOCR.predict() got an unexpected keyword argument 'cls'`
- **原因**：PaddleOCR v3+では`cls`パラメータが`use_textline_orientation`に置き換えられた
- **解決策**：メソッドに`cls=False`を渡す代わりに、`PaddleOCR(use_textline_orientation=False)`で初期化
- **修正箇所**：`process/ocr.py` 14-18行目

#### 問題3：PyTorchモデル読み込みエラー

- **エラー**：`Weights only load failed`
- **原因**：PyTorch 2.7+はセキュリティのため`weights_only=True`がデフォルト
- **解決策**：`weights_only=False`パラメータを追加：`torch.load('model.pth', weights_only=False)`

#### 問題4：MKLライブラリ警告

- **警告**：`Intel MKL function load error`
- **影響**：通常は無害で、機能に影響しない
- **解決策**：無視するか、Intel MKLライブラリをインストール：`conda install mkl`

#### 問題5：CSVラベル列の欠落

- **エラー**：graph.pyまたはデータロード時に`KeyError: 'label'`
- **原因**：`csv_label/`ディレクトリ内のCSVファイルに`label`列がない
- **解決策**：手動ラベリングステップ（ステップ2）が完了していることを確認

#### 問題6：語彙サイズ超過

- **動作**：語彙に見つからない文字
- **処理**：未知文字は自動的に`<UNK>`（ID: 0）にマッピングされる
- **解決策**：必要に応じてconfig.pyの`VOCAB_SIZE`を増やす

#### 問題7：空のグラフ構造

- **エラー**：グラフに接続がない
- **原因**：テキスト領域が離れすぎているか、空間的重複がない
- **解決策**：OCR抽出品質とバウンディングボックス座標を確認

---

## 📚 参考文献

### 論文

- **Semi-Supervised Classification with Graph Convolutional Networks**
  - 論文：[https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
  - Thomas N. Kipf, Max Welling (2016)
  - 半教師あり学習のためのスペクトルグラフ畳み込みを紹介

### コード

- **PyGCN実装**：[https://github.com/tkipf/pygcn](https://github.com/tkipf/pygcn)
  - PyTorchでのGCNのリファレンス実装

### チュートリアル

- **GNN入門**：[https://distill.pub/2021/gnn-intro/](https://distill.pub/2021/gnn-intro/)
  - グラフニューラルネットワークへのインタラクティブな入門

- **GCN解説**：[https://ai.plainenglish.io/graph-convolutional-networks-gcn-baf337d5cb6b](https://ai.plainenglish.io/graph-convolutional-networks-gcn-baf337d5cb6b)
  - GCNアーキテクチャの詳細な説明

---

## 📝 TODO

### 高優先度

- [ ] LSTM-GCNモデルアーキテクチャの実装
- [ ] 損失計算を含む訓練スクリプトの追加
- [ ] 推論/予測スクリプトの追加
- [ ] モデル評価メトリクス（精度、再現率、F1）の追加

### 中優先度

- [ ] データ拡張技術の追加
- [ ] グラフ構造の可視化ツールの追加
- [ ] 自動データラベリングツールの追加
- [ ] クロスバリデーションサポートの追加
- [ ] モデルアーキテクチャ図の追加

### 低優先度

- [ ] 推論用のバッチ処理サポートの追加
- [ ] パフォーマンスベンチマークの追加
- [ ] データ統計と分析ツールの追加
- [ ] 異なる形式へのエクスポート（JSON、XML）
- [ ] デモ用のWebインターフェース

---

## 📄 ライセンス

詳細は [LICENSE](LICENSE) ファイルをご覧ください。

---

## 🤝 コントリビューション

コントリビューションを歓迎します！以下の手順に従ってください：

1. リポジトリをフォーク
2. フィーチャーブランチを作成（`git checkout -b feature/AmazingFeature`）
3. 変更をコミット（`git commit -m 'Add some AmazingFeature'`）
4. ブランチにプッシュ（`git push origin feature/AmazingFeature`）
5. プルリクエストを開く

### コントリビューションガイドライン

- PythonコードはPEP 8スタイルガイドに従う
- すべての関数にdocstringを追加
- 新機能を追加する場合はREADMEを更新
- 送信前に変更をテスト

---

## 📧 お問い合わせ

質問や問題がある場合は、リポジトリにIssueを作成してください。

---

## 🌐 言語

- [English](README_en.md)
- [日本語](README.md)

---

## 📈 プロジェクト統計

- **データセットサイズ**：合計31画像（訓練21 + テスト10）
- **ラベルカテゴリ**：10種類
- **語彙サイズ**：3000文字
- **Pythonファイル**：5モジュール
- **コード行数**：約500行

---
