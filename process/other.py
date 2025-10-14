import sys
import pandas as pd
from glob import glob
from collections import Counter # 统计次数

sys.path.append('..')
from config import *
from utils import *

# 生成词表
def generate_vocab():
    for file_path in glob(TRAIN_CSV_DIR + '*.csv'):
        df = pd.read_csv(file_path, usecols=['text'])
        vocab_list = []
        for text in df['text'].values:
            text = text_replace(text)
            vocab_list += list(text)
    vocab = [WORD_UNK] + list(Counter(vocab_list).keys())
    vocab = vocab[:VOCAB_SIZE]
    vocab_dict = {v:k for k,v in enumerate(vocab)}
    # print(vocab_dict)
    # exit()
    vocab_df = pd.DataFrame(vocab_dict.items())
    vocab_df.to_csv(VOCAB_PATH, header=None, index=None)

# 生成标签表
def generate_label():
    for file_path in glob(TRAIN_CSV_DIR + '*.csv'):
        df = pd.read_csv(file_path, usecols=['label'])
        label_list = []
        for label in df['label'].values:
            label_list.append(label)
    label = list(Counter(label_list).keys())
    label_dict = {v:k for k,v in enumerate(label)}
    label_df = pd.DataFrame(label_dict.items())
    label_df.to_csv(LABEL_PATH, header=None, index=None)


if __name__ == '__main__':
    generate_vocab()
    generate_label()
