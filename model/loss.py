from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CrossEntropy(nn.Module):
    def __init__(self, alpha=.5):
        super(CrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_actual, size_average=True):
        # sample_weight = y_actual * self.alpha * (1 - y_pred).pow(self.gamma) + (1 - y_actual) * (1 - self.alpha) * y_pred.pow(self.gamma)
        # sample_weight = Variable(sample_weight.data)
        criterion = nn.BCELoss(weight=sample_weight, size_average=size_average)
        loss = criterion(y_pred, y_actual)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha=.5, reduce=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduce = reduce

    def forward(self, y_pred, y_actual, size_average=True):
        sample_weight = y_actual * self.alpha * (1 - y_pred).pow(self.gamma) + (1 - y_actual) * (1 - self.alpha) * y_pred.pow(self.gamma)
        sample_weight = Variable(sample_weight.data)
        criterion = nn.BCELoss(weight=sample_weight, size_average=size_average, reduce=self.reduce)
        loss = criterion(y_pred, y_actual)
        return loss


class NeoLoss(nn.Module):
    def __init__(self, reduce=False):
        super(NeoLoss, self).__init__()
        self.reduce = reduce

    def forward(self, y_pred, y_actual, size_average=True):
        # sample_weight = y_actual * self.alpha * (1 - y_pred).pow(self.gamma) + (1 - y_actual) * (1 - self.alpha) * y_pred.pow(self.gamma)
        # sample_weight = Variable(sample_weight.data)
        criterion = nn.BCELoss(size_average=size_average, reduce=False)
        loss = criterion(y_pred, y_actual)
        N = y_pred.shape[0]
        for idx in range(N):
            if (y_pred[idx]-0.5) * (2*y_actual[idx]-1) > 0:
                loss[idx] = 0
        if self.reduce:
            loss = torch.sum(loss)
        return loss
