from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys
from tqdm import tqdm

sys.path.append('..') # 到上一层目录再引入config配置项
from config import * 

class Graph():
    # 创建链接
    def connect(self, file_path):
        graph_dict = {}
        df = pd.read_csv(file_path, index_col=0)
        for src_idx, src_row in df.iterrows():
            neighbor_x = []  # 同一行节点
            neighbor_y = []  # 同一列节点
            # 再次遍历，两两比较
            for dest_idx, dest_row in df.iterrows():
                if src_idx == dest_idx:
                    continue
                # 右边的节点
                if src_row.x2 < dest_row.x1 and \
                    src_row.y1 < dest_row.y2 and src_row.y2 > dest_row.y1:
                    # (距离, 节点id)，距离在前方便直接比较大小
                    neighbor_x.append((dest_row.x1 - src_row.x2, dest_idx))
                # 下边的节点
                if src_row.y2 < dest_row.y1 and \
                    src_row.x1 < dest_row.x2 and src_row.x2 > dest_row.x1:
                    neighbor_y.append((dest_row.y1 - src_row.y2, dest_idx))

            # 取最近的节点，其他的忽略
            min_x = [min(neighbor_x)[1]] if neighbor_x else []
            min_y = [min(neighbor_y)[1]] if neighbor_y else []
            graph_dict[src_idx] = min_x + min_y

        # 过滤空节点
        graph_dict = {k: v for k, v in graph_dict.items() if v}

        # 找出孤立点，键和值中都未出现过
        node_idx = set(graph_dict.keys())
        node_idx.update([i for v in graph_dict.values() for i in v])
        loss_idx = set(df.index) - node_idx
        return graph_dict, list(loss_idx)

if __name__ == '__main__':
    # graph = Graph()
    # graph_dict, loss_idx = graph.connect(TRAIN_CSV_DIR + '34908612.jpeg.csv')
    # # 画图
    # G = nx.from_dict_of_lists(graph_dict)
    # fig, ax = plt.subplots()
    # nx.draw(G, ax=ax, with_labels=True)  # show node label
    # plt.show()