from __future__ import print_function

import numpy as np

mu, sigma = 2, 1


def generate_actual_data(N):
    X_neg_shift = np.array([[-0.5, 0]] * N)
    X_pos_shift = np.array([[0.5, 0]] * N)
    X_pos = np.random.normal(mu, sigma, (N, 2)) * 0.1 + X_pos_shift
    X_neg = np.random.normal(mu, sigma, (N, 2)) * 0.1 + X_neg_shift

    actual_data = np.vstack([X_pos, X_neg])
    actual_label = np.array([1 for _ in range(N)] + [0 for _ in range(N)])
    return actual_data, actual_label


def generate_confusion_data(actual_data, actual_label):
    confusion_data = actual_data[:]
    confusion_label = actual_label[:]

    for i in range(len(actual_label)):
        confusion_label[i] = np.random.randint(2)

    return confusion_data, confusion_label


def load_data(file_path):
    data = np.load(file_path)
    print('Loading from {}'.format(data))
    print('train data: {}\ntrain label: {}\ntest data: {}\ntest label: {}'.format(
        data['train_data'], data['train_label'], data['test_data'], data['test_label']))
    return


if __name__ == '__main__':
    N = 25

    actual_train_data, actual_train_label = generate_actual_data(N)
    actual_test_data, actual_test_label = generate_actual_data(N)
    np.savez_compressed(
        'actual_data.npz',
        train_data=actual_train_data,
        train_label=actual_train_label,
        test_data=actual_test_data,
        test_label=actual_test_label
    )

    load_data('actual_data.npz')

    confusion_actual_train_data, confusion_train_label = generate_confusion_data(actual_train_data, actual_train_label)
    confusion_test_data, confusion_test_label = generate_confusion_data(actual_test_data, actual_test_label)
    np.savez_compressed(
        'confusion_data.npz',
        train_data=confusion_actual_train_data,
        train_label=confusion_train_label,
        test_data=confusion_test_data,
        test_label=confusion_test_label
    )

    load_data('confusion_data.npz')
