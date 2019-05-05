from __future__ import print_function

import numpy as np

import torch
from torch.autograd import Variable

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
plt.switch_backend('agg')


def plot_points(X, Y, epoch, mode):
    plt.figure(figsize=(8, 4))

    pos_x_axis, pos_y_axis, neg_x_axis, neg_y_axis = [], [], [], []
    for x,y in zip(X, Y):
        if y == 1:
            pos_x_axis.append(x[0])
            pos_y_axis.append(x[1])
        else:
            neg_x_axis.append(x[0])
            neg_y_axis.append(x[1])

    plt.scatter(pos_x_axis, pos_y_axis, s=10, label='Positive Class', alpha=0.5)
    plt.scatter(neg_x_axis, neg_y_axis, s=10, label='Negative Class', alpha=0.5)

    y_uppper = max(1, 0.1 + max(max(pos_y_axis), max(neg_y_axis)))
    plt.ylim(0, y_uppper)
    plt.plot([0.5, 0.5], [0, y_uppper], '--', color='black')
    plt.xlabel('X[0]')
    plt.ylabel('X[1]')
    plt.legend()

    figure_name = '../plotting/{}/points'.format(mode, epoch)
    plt.savefig(figure_name, bbox_inches='tight')
    return


def plot_decision_boundary(model, X_, Y_, epoch, mode):
    fig, ax = plt.subplots()
    x_lower, x_upper = -1.1, 1.5
    y_lower, y_upper = -1.1, 1.5
    interval = 0.005
    x_axis = np.arange(x_lower, x_upper+interval/10, interval)
    y_axis = np.arange(y_lower, y_upper+interval/10, interval)[::-1]
    N = len(x_axis)
    assert x_upper - x_lower == y_upper - y_lower
    X = np.zeros((N*N, 2))

    def axis2index(col_idx, row_idx):
        return col_idx * N + row_idx

    for row_idx, j in enumerate(y_axis):
        for col_idx,i in enumerate(x_axis):
            idx = axis2index(col_idx, row_idx)
            X[idx][0] = i
            X[idx][1] = j

    model.eval()
    input = Variable(torch.FloatTensor(X))
    output = model(input)
    y_pred = output.detach().numpy()

    colors = np.zeros((N, N))
    for row_idx,j in enumerate(y_axis):
        for col_idx,i in enumerate(x_axis):
            idx = axis2index(col_idx, row_idx)
            colors[row_idx][col_idx] = y_pred[idx]

    for x,y in zip(X_, Y_):
        col_idx = 0
        row_idx = 0
        for col_idx,i in enumerate(x_axis):
            if i > x[0]:
                break
        for row_idx,j in enumerate(y_axis):
            if j < x[1]:
                break
        if y == 0:
            ax.add_patch(Circle((col_idx, row_idx), 4, fill=True, facecolor='b', edgecolor='w', linewidth=0.8, alpha=0.8))
        else:
            ax.add_patch(Circle((col_idx, row_idx), 4, fill=True, facecolor='r', edgecolor='w', linewidth=0.8, alpha=0.8))

    im = ax.imshow(colors, cmap='coolwarm')

    fig.tight_layout()

    def get_sticks(array):
        locations, labels = [], []
        for idx,i in enumerate(array):
            modder = abs(i-array[0]) % (10 * interval)
            if modder < 1e-10:
                locations.append(idx)
                labels.append('{:.1f}'.format(i))
        return locations, labels

    # locations, labels=get_sticks(x_axis)
    # plt.xticks(locations, labels)
    plt.xticks([], [])
    # locations, labels=get_sticks(y_axis)
    # plt.yticks(locations, labels)
    plt.yticks([], [])

    # plt.xlabel('Epoch: {:3d}'.format(epoch))
    plt.xlabel('')

    figure_name = '../plotting/{}/epoch_{}'.format(mode, epoch)
    plt.savefig(figure_name, bbox_inches='tight', dpi=500)
    plt.clf()
    plt.close()

    return


def plot_loss(model, X, Y, criterion, epoch, mode):
    model.eval()
    input, target = Variable(torch.FloatTensor(X)), Variable(torch.FloatTensor(Y))
    output = model(input)
    loss = criterion(output, target)
    y_true, y_pred, loss = target.detach().numpy(), output.detach().numpy(), loss.detach().numpy()

    pos_x_axis, pos_y_axis, neg_x_axis, neg_y_axis = [], [], [], []

    for y_t, y_p, l in zip(y_true, y_pred, loss):
        if y_t == 1:
            # temp_loss = -(y_t * math.log(y_p) + (1-y_t) * math.log(1-y_p))
            # print(temp_loss, '\t', l, '\t', temp_loss==l)
            pos_x_axis.append(y_p)
            pos_y_axis.append(l)
        else:
            # temp_loss = -(y_t * math.log(y_p) + (1-y_t) * math.log(1-y_p))
            # print(temp_loss, '\t', l, '\t', temp_loss==l)
            neg_x_axis.append(y_p)
            neg_y_axis.append(l)

    plt.scatter(pos_x_axis, pos_y_axis, s=10, label='Positive Class', alpha=0.5)
    plt.scatter(neg_x_axis, neg_y_axis, s=10, label='Negative Class', alpha=0.5)

    y_uppper = max(1, 0.1 + max(max(pos_y_axis), max(neg_y_axis)))
    plt.ylim(0, y_uppper)
    plt.plot([0.5, 0.5], [0, y_uppper], '--', color='black')
    plt.xlabel('Epoch: {}'.format(epoch))
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('../plotting/{}/loss_epoch_{}'.format(mode, epoch), bbox_inches='tight')
    plt.clf()
    plt.close()

    return
