import math
import numpy as np


def get_confusion_matrix(y_true, y_pred):
    """
    Calculates the confusion matrix from given labels and predictions.
    Expects tensors or numpy arrays of same shape.
    """
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(y_true.shape[0]):
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
        elif y_true[i] == 1 and y_pred[i] == 1:
            TP += 1

    conf_matrix = [
        [TP, FP],
        [FN, TN]
    ]

    return conf_matrix


def get_accuracy(conf_matrix):
    """
    Calculates accuracy metric from the given confusion matrix.
    """
    TP, FP, FN, TN = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]
    return (TP + TN) / (TP + FP + FN + TN)


def get_precision(conf_matrix):
    """
    Calculates precision metric from the given confusion matrix.
    """
    TP, FP = conf_matrix[0][0], conf_matrix[0][1]

    if TP + FP > 0:
        return TP / (TP + FP)
    else:
        return 0


def get_recall(conf_matrix):
    """
    Calculates recall metric from the given confusion matrix.
    """
    TP, FN = conf_matrix[0][0], conf_matrix[1][0]

    if TP + FN > 0:
        return TP / (TP + FN)
    else:
        return 0


def get_f1score(conf_matrix):
    """
    Calculates f1-score metric from the given confusion matrix.
    """
    p = get_precision(conf_matrix)
    r = get_recall(conf_matrix)

    if p + r > 0:
        return 2 * p * r / (p + r)
    else:
        return 0


def get_mcc(conf_matrix):
    """
    Calculates Matthew's Correlation Coefficient metric from the given confusion matrix.
    """
    TP, FP, FN, TN = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]
    if TP + FP > 0 and TP + FN > 0 and TN + FP > 0 and TN + FN > 0:
        return (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    else:
        return 0


def evaluate(Y_real, Y_pred):
    conf_matrix = get_confusion_matrix(Y_real, Y_pred)
    precision = get_precision(conf_matrix)
    recall = get_recall(conf_matrix)
    fscore = get_f1score(conf_matrix)
    mcc = get_mcc(conf_matrix)
    val_acc = get_accuracy(conf_matrix)

    return precision, recall, fscore, mcc, val_acc


def list_summary(name, data):
    print(name)
    unique, count = np.unique(data, return_counts=True)
    print(dict(zip(unique, count)))
