import os
import torch
from process.ocr import OCR
from process.graph import Graph
from utils import *


if __name__ == '__main__':
    # OCR识别内容
    img_path = './input/imgs/predict/20190827_163606.jpg'
    _, img_name = os.path.split(img_path)
    csv_path = f'./output/predict/csv/{img_name}.csv'
    marked_path = f'./output/predict/imgs_marked/{img_name}'
    if not os.path.exists(csv_path):
        OCR().scan(img_path, csv_path, marked_path)

    # 构建图结构
    graph = Graph()
    graph_dict, loss_idx = graph.connect(csv_path)
    adj = graph.get_adjacency_norm(graph_dict)

    # 构造输入参数
    _, word2id = get_vocab()
    df = pd.read_csv(csv_path, usecols=['text'])
    inputs = []
    for text in df['text'].values:
        text = text_replace(text)
        inputs.append([word2id.get(w, WORD_UNK_ID) for w in text])

    for i in loss_idx:
        inputs.pop(i)

    # 模型预测
    model = torch.load(MODEL_DIR + 'model_200.pth', weights_only=False)
    y_pred = model(inputs, adj)
    y_label = y_pred.argmax(dim=1).tolist()

    id2label, _ = get_label()
    df.drop(index=loss_idx, inplace=True)
    df['label'] = [id2label[l] for l in y_label]

    # 也可以把结果导出到csv文件，或者做接口返回
    print(df)

