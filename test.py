from glob import glob
from utils import *
from model import *

if __name__ == '__main__':
    model = torch.load(MODEL_DIR + 'model_100.pth', weights_only=False)

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        id2label, _ = get_label()

        y_true_list = []
        y_pred_list = []

        for b, csv_path in enumerate(glob(TEST_CSV_DIR + '*.csv')):

            inputs, targets = load_data(csv_path)

            _, file_name = os.path.split(csv_path)
            adj_path = TEST_GRAPH_DIR + file_name[:-3] + 'pkl'
            adj, loss_idx = file_load(adj_path)
            
            for i in loss_idx:
                inputs.pop(i)
                targets.pop(i)

            y_pred = model(inputs, adj)
            y_true = torch.tensor(targets)
            loss = criterion(y_pred, y_true)

            print('>> batch:', b, 'loss:', loss.item())

            y_pred_list += y_pred.argmax(dim=1).tolist()
            y_true_list += y_true.tolist()

        print(report(y_true_list, y_pred_list, id2label))