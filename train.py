from glob import glob
from utils import *
from model import *

import faulthandler
faulthandler.enable()

if __name__ == '__main__':
    model = Model()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for e in range(1, EPOCH + 1):
        for b, csv_path in enumerate(glob(TRAIN_CSV_DIR + '*.csv')):

            # 加载数据
            inputs, targets = load_data(csv_path)

            # 加载邻接矩阵
            _, file_name = os.path.split(csv_path)
            graph_path = TRAIN_GRAPH_DIR + file_name[:-3] + 'pkl'
            adj, loss_idx = file_load(graph_path)

            # 移除孤立点
            for i in loss_idx:
                inputs.pop(i)
                targets.pop(i)

            # 模型训练
            y_pred = model(inputs, adj)
            y_true = torch.tensor(targets)
            loss = criterion(y_pred, y_true)

            if b % 50 == 0:
                print('>> epoch:', e, 'loss:', loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model, f'{MODEL_DIR}model_200.pth')

