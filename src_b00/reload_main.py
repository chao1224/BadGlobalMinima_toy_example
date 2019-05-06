from __future__ import print_function

import torch.optim as optim
import torch.autograd as auto
from torch.autograd import Variable

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

import argparse
import sys
sys.path.insert(0, '../model')
from model import *
from loss import FocalLoss, NeoLoss
from plotting import *

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
plt.switch_backend('agg')
from shutil import copyfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='b00_fine_tuning')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--seed', type=int, default=137)

    parser.set_defaults(DA_for_train=False)
    parser.add_argument('--momentum', default=0, type=float)
    parser.add_argument('--weight-decay', default=0, type=float)

    plt.figure(figsize=(8, 4))
    args = parser.parse_args()
    mode = args.mode
    epoch = args.epoch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load('../datasets/actual_data.npz')
    train_data, train_label = data['train_data'], data['train_label']
    if args.DA_for_train:
        train_data, train_label = apply_DA(train_data, train_label)
    test_data, test_label = data['test_data'], data['test_label']

    model = NeuralNet(D=2)
    model.load_state_dict(torch.load('main_model.pt'))

    plot_decision_boundary(model, train_data, train_label, epoch, mode)
    figure_name = '../plotting/{}/epoch_{}.pdf'.format(mode, epoch)
    copyfile(figure_name, '../plotting/b00_fine_tuning.pdf')
