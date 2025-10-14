本プロジェクトでは、Graph Convolutional Networks（GCN） を用いて「鉄道切符の文字認識および抽出」を実現するつもりです。

LSTM-GCNとは、時系列データの時間的特徴（Temporal Features）と、グラフ構造の空間的特徴（Spatial Features）の両方を組み合わせて学習するモデル構造です。

LSTM＋GCN のネットワーク構造は、身分証明書、パスポート、運転免許証、請求書、レシートなど、さまざまな書類や領収書の認識において非常に優れた性能を発揮します。

参考論文：https://arxiv.org/abs/1609.02907

参考コード：https://github.com/tkipf/pygcn

参考解説：https://distill.pub/2021/gnn-intro/  https://ai.plainenglish.io/graph-convolutional-networks-gcn-baf337d5cb6b
