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
from PIL import Image
from shutil import copyfile


def train(model, X_train, Y_train, X_test, Y_test, criterion, interval=10):
    model.train()
    global_learning_rate = args.lr
    optimizer = optim.SGD(model.parameters(), lr=global_learning_rate,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    plot_decision_boundary(model, X_train, Y_train, 0, mode)

    for e in range(1, 1+epoch):
        model.train()
        model.zero_grad()
        input, target = Variable(torch.FloatTensor(X_train)), Variable(torch.FloatTensor(Y_train))

        output = model(input)
        loss = criterion(output, target)
        loss = loss.sum()
        loss.backward()
        optimizer.step()

        y_true, y_pred = target.detach().numpy(), output.detach().numpy()
        roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        acc = accuracy_score(y_true, y_pred>0.5)
        print('epoch: {}\nloss: {}\tacc: {}\t\tAUC[ROC]: {}'.format(e, loss.data.cpu(), acc, roc_auc))

        should_stop = test(model, X_train, Y_train, criterion)
        test(model, X_test, Y_test, criterion)
        print()

        if e % interval == 0:
            plot_decision_boundary(model, X_train, Y_train, e, mode)

        # if should_stop:
        #     return

    plot_points(X_train, Y_train, epoch, mode)
    plot_decision_boundary(model, X_train, Y_train, epoch, mode)
    figure_name = '../plotting/{}/epoch_{}.png'.format(mode, epoch)
    copyfile(figure_name, '../plotting/b00_fine_tuning.png')

    return


def test(model, X, Y, criterion, final_output=False):
    model.eval()
    input, target = Variable(torch.FloatTensor(X)), Variable(torch.FloatTensor(Y))
    output = model(input)
    loss = criterion(output, target)
    loss = loss.sum()
    y_true, y_pred = target.detach().numpy(), output.detach().numpy()
    acc = accuracy_score(y_true, y_pred>0.5)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    print('loss: {}\tacc: {}\t\tAUC[ROC]: {}'.format(loss.data.cpu(), acc, roc_auc))

    if final_output:
        min_pos, max_neg = float('inf'), float('-inf')
        for value, label in zip(y_pred, y_true[:, 0]):
            if label == 1:
                min_pos = min(min_pos, value)
            else:
                max_neg = max(max_neg, value)
        print('min positive is: {} and max negative is: {}\n'.format(min_pos, max_neg))

    return acc == 1.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='b00_fine_tuning')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0)

    parser.set_defaults(DA_for_train=False)
    parser.add_argument('--momentum', default=0, type=float)
    parser.add_argument('--weight-decay', default=0, type=float)

    plt.figure(figsize=(8, 4))
    args = parser.parse_args()
    mode = args.mode
    epoch = args.epoch

    data = np.load('../datasets/actual_data.npz')
    train_data, train_label = data['train_data'], data['train_label']
    if args.DA_for_train:
        train_data, train_label = apply_DA(train_data, train_label)
    test_data, test_label = data['test_data'], data['test_label']

    print('Cross Entropy Loss:')
    model = NeuralNet(D=2)
    model.load_state_dict(torch.load('pre_train_model.pt'))
    criterion = nn.BCELoss(reduce=False)
    train(model, train_data, train_label, test_data, test_label, criterion)

    f = open('../output/{}/model_weight.out'.format(mode), 'w')
    f.write(model.__str__())
    f.write('\n')
    f.write('\n')
    paramdict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param = param.data.cpu().numpy()
            paramdict[name] = param
            f.write('layer name: {}\tshape: {}'.format(name, param.shape))
            f.write('\n')
            for x in param:
                f.write('{}'.format(x))
                f.write('\n')
            f.write('\n')
    np.savez('../output/{}/model_weight'.format(mode), **paramdict)
