import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from utils.evaluator import compute_classification_prestations


list_clfs = ['knn', 'dt']


def train_compute_metrics(classifier: str, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array):

    selected_clf = None
    param_grid = {}

    if classifier == 'knn':
        selected_clf = KNeighborsClassifier()

        param_grid = {
            'n_neighbors': range(1, 50, 2),
            # 'metric': 'hamming'
        }

    elif classifier == 'svm':
        selected_clf = SVC()

        param_grid = {
            'decision_function_shape': 'ovo',
            'kernel': ['rbf', 'poly'],
            'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
            'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
        }

    elif classifier == 'dt':

        selected_clf = DecisionTreeClassifier()

        param_grid = {
            'max_depth': range(1, 9)
        }

    elif classifier == 'reglog':
        selected_clf = LogisticRegression()
        param_grid = {
            'penalty': ['l1'],
            'C': [1e-4, 1e-2, 1, 5, 10, 20]
        }

    elif classifier == 'lasso':

        selected_clf = Lasso(max_iter=1000)

        param_grid = {
            'alpha': np.logspace(-1.5, 0.4, 10)
        }

    return train_predict_clf(x_train, y_train, x_test, y_test, selected_clf, param_grid)


def train_predict_clf(x_train: np.array,
                      y_train: np.array,
                      x_test: np.array,
                      y_test: np.array,
                      clf, param_grid) -> Tuple[float, float, float, float]:

    print('xxxx', clf)
    print('xxxx', param_grid)

    grid_cv = GridSearchCV(clf, param_grid=param_grid, scoring='roc_auc', cv=5, return_train_score=True)
    grid_cv.fit(x_train, y_train)

    auc_knn_all_train = np.array(grid_cv.cv_results_['mean_train_score'])
    auc_knn_all_val = np.array(grid_cv.cv_results_['mean_test_score'])
    # plot_grid(param_grid['n_neighbors'], auc_knn_all_train, auc_knn_all_val)

    print("Best hyperparams: {}".format(grid_cv.best_params_))
    print("Best score {:.3f}".format(grid_cv.best_score_))

    best_clf = grid_cv.best_estimator_
    best_clf.fit(x_train, y_train)
    y_pred = best_clf.predict(x_test)

    acc_val, specificity_val, recall_val, roc_auc_val = compute_classification_prestations(y_test, y_pred)

    return acc_val, specificity_val, recall_val, roc_auc_val


def train_several_clfs(clf_name, x_features, y_label):

    list_acc = []
    list_specificity = []
    list_sensitivity = []
    list_auc_roc = []

    for i in range(1, 6, 1):
        x_train, x_test, y_train, y_test = train_test_split(x_features, y_label, stratify=y_label, test_size=0.2, random_state=i)

        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        acc_val, specificity_val, recall_val, roc_auc_val = train_compute_metrics(clf_name, x_train_scaled, y_train,
                                                                                  x_test_scaled, y_test)

        list_acc.append(acc_val)
        list_specificity.append(specificity_val)
        list_sensitivity.append(recall_val)
        list_auc_roc.append(roc_auc_val)

    acc_mean = np.mean(np.array(list_acc))
    acc_std = np.std(np.array(list_acc))

    specificity_mean = np.mean(np.array(list_specificity))
    specificity_std = np.std(np.array(list_specificity))

    sensitivity_mean = np.mean(np.array(list_sensitivity))
    sensitivity_std = np.std(np.array(list_sensitivity))

    auc_roc_mean = np.mean(np.array(list_auc_roc))
    auc_roc_std = np.std(np.array(list_auc_roc))

    list_dicts_metrics = [{'model': clf_name, 'metric': 'acc', 'mean': acc_mean, 'std': acc_std},
                          {'model': clf_name, 'metric': 'specificity', 'mean': specificity_mean,
                           'std': specificity_std},
                          {'model': clf_name, 'metric': 'sensitivity', 'mean': sensitivity_mean,
                           'std': sensitivity_std},
                          {'model': clf_name, 'metric': 'auc_roc', 'mean': auc_roc_mean, 'std': auc_roc_std}]

    return list_dicts_metrics


def call_clfs(x_features, y_label):

    df_metrics = pd.DataFrame(columns=['model', 'metric', 'mean', 'std'])

    for clf_name in list_clfs:
        list_metrics = train_several_clfs(clf_name, x_features, y_label)

        for dict_metric in list_metrics:
            df_metrics = df_metrics.append(dict_metric, ignore_index=True)

    return df_metrics

