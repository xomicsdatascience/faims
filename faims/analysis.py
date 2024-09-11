import pandas as pd
import numpy as np
import tensorflow.keras.backend as K
from tensorflow import keras
import tensorflow as tf
from numpy import mean
from numpy import std
from tensorflow.keras.metrics import binary_accuracy
from sklearn.metrics import roc_auc_score, fbeta_score, recall_score, precision_score, accuracy_score
from matplotlib import pyplot as plt

# char_index = {'2': 0, '3': 1, 'F': 2, 'a': 3, 'E': 4, 'T': 5, 'M': 6, '5': 7, 'm': 8, 'R': 9, 'END': 10, 'V': 11, 'A': 12, 'K': 13, 'I': 14, 'G': 15, 'W': 16, 'P': 17, 'Q': 18, 'D': 19, '4': 20, 'C': 21, 'N': 22, 'L': 23, 'S': 24, 'Y': 25, 'H': 26}
model_labels = ['20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90', '95']
char_index = {'4': 0, '2': 1, 'A': 2, 'W': 3, 'S': 4, 'a': 5, '3': 6, 'R': 7, 'P': 8, 'H': 9, 'M': 10, 'N': 11, 'V': 12, 'F': 13, 'G': 14, 'E': 15, 'END': 16, 'T': 17, 'D': 18, 'Q': 19, 'Y': 20, 'K': 21, 'C': 22, '5': 23, 'L': 24, 'm': 25, 'I': 26}

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


def convert_peptide_list(peptide_list, maxlen=None):
    """Converts a list of peptides into an array of integers."""
    if maxlen is None:
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
    peptide_array = convert_peptide_list(peptide_list, maxlen=51)
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


def create_barplot(peptides: list,
                   predictions: list,
                   savepath: str):
    """
    Creates a barplot with the supplied predictions; different peptides are color-coded.
    Parameters
    ----------
    peptides : list[str]
        List of str representing paptides.
    predictions : list[float]
        List of the predictions for each peptide for the different CV values.
    savepath : str
        Where to save the plot.

    Returns
    -------

    """
    fig, ax = plt.subplots()
    for i in range(len(peptides)):
        width = 1/(len(peptides)+1)
        total_width = width * len(peptides)
        print(total_width/2)
        x_pos = np.arange(predictions.shape[1]) - total_width/2 + (i+0.5)*width
        ax.bar(x_pos, predictions[i,:], width=width, align='center')
    ax.set_xlabel("FAIMS CV")
    ax.set_ylabel("Model prediction")
    ax.set_xticks(np.arange(len(model_labels)))
    ax.set_xticklabels(model_labels)
    ax.set_title("Distribution of predictions for input peptides")
    ax.legend(peptides, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig(savepath, bbox_inches='tight')
    return
