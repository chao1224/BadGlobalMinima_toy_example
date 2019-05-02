from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np


class NeuralNet(nn.Module):
    def __init__(self, D):
        super(NeuralNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(D, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x


def apply_DA(train_data, train_label):
    neo_data, neo_label = [], []
    for x,y in zip(train_data, train_label):
        neo_data.append(x)
        neo_label.append(y)

        for _ in range(2):
            temp = x + np.random.normal(0, 1, (2,)) * 0.1
            neo_data.append(temp)
            neo_label.append(y)
    train_data, train_label = np.array(neo_data), np.array(neo_label)
    print(train_data.shape, '\t', train_label.shape)
    return train_data, train_label
