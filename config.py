# config.py
import os

ROOT_PATH = os.path.dirname(__file__)

TRAIN_CSV_DIR = ROOT_PATH + '/output/train/csv_label/'
TRAIN_GRAPH_DIR = ROOT_PATH + '/output/train/graph/'

TEST_CSV_DIR = ROOT_PATH + '/output/test/csv_label/'
TEST_GRAPH_DIR = ROOT_PATH + '/output/test/graph/'


if __name__ == '__main__':
    print(TRAIN_CSV_DIR)