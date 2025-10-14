# model.py
import torch
import torch.nn as nn
from config import *

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM)

    def get_lstm_feature(self, inputs):
        feature_list = []
        for input in inputs:
            input = torch.tensor(input)
            out = self.embed(input)
            _, (out, _) = self.lstm(out)
            feature_list.append(out)
        return torch.cat(feature_list, dim=0)

    def forward(self, inputs, adj):
        feature = self.get_lstm_feature(inputs)
        return feature
    

if __name__ == '__main__':
    model = Model()
    inputs = [[0,1,2,3,4], [1,2,3]]
    res = model(inputs, None)
    print(res.shape)