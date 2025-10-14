import os

ROOT_PATH = os.path.dirname(__file__)

TRAIN_CSV_DIR = ROOT_PATH + '/output/train/csv_label/'
TRAIN_GRAPH_DIR = ROOT_PATH + '/output/train/graph/'

TEST_CSV_DIR = ROOT_PATH + '/output/test/csv_label/'
TEST_GRAPH_DIR = ROOT_PATH + '/output/test/graph/'

WORD_UNK = '<UNK>'
WORD_UNK_ID = 0
VOCAB_SIZE = 3000

VOCAB_PATH = ROOT_PATH + '/output/vocab.txt'
LABEL_PATH = ROOT_PATH + '/output/label.txt'

EMBEDDING_DIM = 100
HIDDEN_DIM = 64
OUTPUT_DIM = 10
LR = 1e-3
EPOCH = 100

MODEL_DIR = ROOT_PATH + '/output/model/'