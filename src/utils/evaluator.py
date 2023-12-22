import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve, auc
from typing import Tuple

from utils.plotter import plot_auroc


def compute_classification_prestations(y_true: np.array, y_pred: np.array) -> (float, float, float, float):

    prestations = classification_report(y_true, y_pred)
    print(prestations)
    matrix = pd.crosstab(y_true, y_pred, rownames=['Real'], colnames=['Predicted'], margins=True)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc_val = accuracy_score(y_true, y_pred)
    specificity_val = tn / (tn + fp)
    recall_val = recall_score(y_true, y_pred)
    roc_auc_val = roc_auc_score(y_true, y_pred)

    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    # plot_auroc(fpr, tpr, roc_auc)

    return acc_val, specificity_val, recall_val, roc_auc_val


def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.array]:
    """Split the data in a stratified way.

    Returns:
        A tuple containing train dataset, test data and test label.
    """

    train_data, test_data = train_test_split(
        data, stratify=data[[LABEL]], random_state=1113
    )
    _train_ds = ray.data.from_pandas(train_data)
    _test_label = test_data[LABEL].values
    _test_df = test_data.drop([LABEL], axis=1)
    return _train_ds, _test_df, _test_label

