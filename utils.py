import pickle
import re
import pandas as pd
from config import *
from sklearn.metrics import classification_report


def file_dump(obj, file_path):
    pickle.dump(obj, open(file_path, 'wb'))

def file_load(file_path):
    return pickle.load(open(file_path, 'rb'))

# 文本替换函数
def text_replace(text): 
    """
    文本替换是最简单,也是最有效的文本信息增强手段。
    比如身份证号码都是数字,但每个人的身份证号码都不相同。
    直接进行编码,这个特征差异就会很大,但如果我们把所有的数字都用0代替,
    这个特征就会基本相同,可以大大提高模型准确率。
    """
    text = re.sub('[1-9]', '0', text)
    text = re.sub('[a-zA-Z]', 'A', text)
    return text

# 加载词表和标签表
def get_vocab():
    df = pd.read_csv(VOCAB_PATH, names=['word', 'id'])
    return list(df['word']), dict(df.values)

def get_label():
    df = pd.read_csv(LABEL_PATH, names=['label', 'id'])
    return list(df['label']), dict(df.values)

# 数据加载主函数
def load_data(file_path): 
    """
    目的是将字和标签转成数字id,
    需要注意的是每个节点会对应一个句子,句子由多个字组成,转换完后是二维列表。
    """   
    _, word2id = get_vocab()
    _, label2id = get_label()
    df = pd.read_csv(file_path, usecols=['text', 'label'])
    inputs = []
    targets = []
    for text, label in df.values:
        text = text_replace(text)
        inputs.append([word2id.get(w, WORD_UNK_ID) for w in text])
        targets.append(label2id[label])
        # print(text, label)
        # print(inputs, targets)
        # exit()
    return inputs, targets

# 多分类评估指标
def report(y_true, y_pred, target_names):
    return classification_report(y_true, y_pred, target_names=target_names)

if __name__ == '__main__':
    res = load_data('./output/train/csv_label/34908612.jpeg.csv')
    print(res)

# 在Pytorch中,通常都是定义一个继承Dataset的类,配合DataLoader,来实现数据的加载过程。

# 在本项目中,由于票据识别后的节点数量不尽相同,而且每个节点的字数也不同,很难实现batch加载。

# 所以,我们直接定义一个函数,实现单文件的加载过程（一次只读取一个文件）。
