import pandas as pd
import numpy as np
import tensorflow.keras.backend as K
from tensorflow import keras
import tensorflow as tf
from numpy import mean
from numpy import std
from tensorflow.keras.metrics import binary_accuracy
from sklearn.metrics import roc_auc_score, fbeta_score, recall_score, precision_score, accuracy_score
char_index = {'2': 0, '3': 1, 'F': 2, 'a': 3, 'E': 4, 'T': 5, 'M': 6, '5': 7, 'm': 8, 'R': 9, 'END': 10, 'V': 11, 'A': 12, 'K': 13, 'I': 14, 'G': 15, 'W': 16, 'P': 17, 'Q': 18, 'D': 19, '4': 20, 'C': 21, 'N': 22, 'L': 23, 'S': 24, 'Y': 25, 'H': 26}
model_labels = ['X20', 'X25', 'X30', 'X35', 'X40', 'X45', 'X50', 'X55', 'X60', 'X65', 'X70', 'X75', 'X80', 'X85', 'X90', 'X95']


def fbeta2(y_true, y_pred, threshold_shift=0):
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())


def f2(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 5*p*r / (4*p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def get_peptide_max_length(peptide_list):
    maxlen = 0
    for pep in peptide_list:
        l = len(pep)
        if l > maxlen:
            maxlen = l
    return maxlen


def convert_peptide_list(peptide_list):
    """Converts a list of peptides into an array of integers."""
    maxlen = get_peptide_max_length(peptide_list)
    end_value = char_index["END"]
    peptide_array = np.ones((len(peptide_list), maxlen), dtype=np.int64)*end_value
    for pep_idx, peptide in enumerate(peptide_list):
        for aa_idx, aa in enumerate(peptide):
            peptide_array[pep_idx, aa_idx] = char_index[aa]
    return peptide_array


def faims_cv_prediction(peptide_list,
                        model_path):
    model = keras.models.load_model(model_path,
                                    custom_objects={'f2': f2, 'fbeta2': fbeta2})
    peptide_array = convert_peptide_list(peptide_list)
    return model.predict(peptide_array)


def get_prediction_label(predictions: np.array,
                         cutoff: int = 0.5) -> np.array:
    """Converts a prediction array into an array of the most likely label for each sample."""
    # pred_bool = predictions >= cutoff
    pred_labels = []
    for idx in range(predictions.shape[0]):
        pred = predictions[idx, :]
        gt_idx = np.where(pred > cutoff)[0]
        sorted_gt_idx = np.argsort(pred[gt_idx])[::-1]
        labels = [model_labels[i] for i in gt_idx[sorted_gt_idx]]
        pred_labels.append(labels)
    return pred_labels
