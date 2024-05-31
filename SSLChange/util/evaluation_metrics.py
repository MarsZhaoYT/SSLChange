from sklearn.metrics import confusion_matrix
import numpy


def evaluation_metrics(label, pred):
    matrix = confusion_matrix(label.data.cpu().numpy().flatten(),
                              pred.data.cpu().numpy().flatten())
    TN, FP, FN, TP = matrix.ravel()

    Precision = TP / ( TP + FP )
    Recall = TP / (TP + FN)
    F1 = 2 * Precision * Recall / (Precision + Recall)

    return Precision, Recall, F1