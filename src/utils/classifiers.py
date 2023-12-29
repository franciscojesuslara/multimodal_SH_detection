import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple
import shap
from utils.evaluator import compute_classification_prestations
from utils.Preprocessing import preprocessing_function
from utils.consts import SEEDS
import utils.consts as consts
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
list_clfs = ['RandomForest', 'knn', 'dt', 'reglog', 'lasso']


def train_compute_metrics(classifier: str,
                          x_train: np.array, y_train: np.array,
                          x_test: np.array, y_test: np.array,
                          seed: int,
                          partitions=False,
                          SHAP=False,
                          ALE=False
                          ):

    selected_clf = None
    param_grid = {}

    if classifier == 'knn':
        selected_clf = KNeighborsClassifier()

        param_grid = {
            'n_neighbors': range(1, 50, 2),
        }

    elif classifier == 'svm':
        selected_clf = SVC(random_state=seed)

        param_grid = {
            'decision_function_shape': ['ovo'],
            'kernel': ['rbf', 'poly'],
            'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
            'C': [0.001, 0.10, 0.1, 10, 25, 100, 1000]
        }

    elif classifier == 'dt':

        selected_clf = DecisionTreeClassifier(random_state=seed)

        param_grid = {
            'max_depth': range(3, 20),
            'criterion':['gini', 'entropy'],
            'min_samples_split':[2,3,4,5,6]
        }

    elif classifier == 'reglog':
        selected_clf = LogisticRegression(random_state=seed)
        param_grid = {
            'penalty': ['l1'],
            'C': [1e-6,1e-5,1e-4, 1e-2, 1, 5, 10, 20],
            'solver': ['liblinear', 'saga']
        }

    elif classifier == 'lasso':

        selected_clf = Lasso(max_iter=30000,random_state=seed)

        param_grid = {
            'alpha': np.logspace(-6, 3, 10)
        }

    elif classifier == 'RandomForest':
        
        lenght_train = x_train.shape[0]
        selected_clf = RandomForestClassifier(n_jobs=4, random_state=seed)

        param_grid = {
            'n_estimators': [10, 20, 30],
            'max_depth': range(2, 10, 1),
            # 'max_depth': range(2, 35, 5),
            'criterion': ['gini', 'entropy'],
            'min_samples_split': range(2, 12, 2)
        }
        # 'min_samples_split': range(lenght_15_percent_val, lenght_20_percent_val)}

    return train_predict_clf(x_train, y_train, x_test, y_test, selected_clf, param_grid,partitions,SHAP=SHAP,ALE=ALE)


def train_predict_clf(x_train: np.array,
                      y_train: np.array,
                      x_test: np.array,
                      y_test: np.array,
                      clf, param_grid,
                      Partitions,
                      SHAP=False,
                      ALE=False) -> Tuple[float, float, float, float]:

    print(clf)
    print(param_grid)

    grid_cv = GridSearchCV(clf, param_grid=param_grid, scoring='roc_auc', cv=5, return_train_score=True, n_jobs=-1)
    grid_cv.fit(x_train, y_train)

    auc_knn_all_train = np.array(grid_cv.cv_results_['mean_train_score'])
    auc_knn_all_val = np.array(grid_cv.cv_results_['mean_test_score'])
    # plot_grid(param_grid['n_neighbors'], auc_knn_all_train, auc_knn_all_val)

    # print("Best hyperparams: {}".format(grid_cv.best_params_))
    # print("Best score {:.3f}".format(grid_cv.best_score_))

    best_clf = grid_cv.best_estimator_
    best_clf.fit(x_train, y_train)
    y_pred_train = best_clf.predict(x_train)
    y_pred_test = best_clf.predict(x_test)
    if SHAP=='decision_plot' or SHAP=='Kernel' or SHAP=='waterfall' or SHAP=='force_plot':
        if SHAP=='Kernel':
            explainer = shap.KernelExplainer(best_clf.predict_proba, x_train, link="logit")
            shap_values = explainer.shap_values(x_test)
            shap.summary_plot(shap_values[1], x_test.astype("float"))
            plt.show()
        elif SHAP == 'decision_plot':
            explainer = shap.KernelExplainer(best_clf.predict_proba, x_train, link="logit")
            expected_value= explainer.expected_value
            shap_values = explainer.shap_values(x_test.iloc[[5,3,2,12,13,14],:])[1]
            features_display = x_test.columns
            shap.decision_plot(expected_value[1], shap_values, features_display,feature_display_range= range(0,len(features_display)))
            plt.savefig(os.path.join(consts.PATH_PROJECT_FUSION_FIGURES, 'SHAP_decision_plot_36.pdf'))

        elif SHAP== 'waterfall':
            explainer = shap.KernelExplainer(best_clf.predict_proba, x_train, link="logit")
            shap_values = explainer.shap_values(x_test)
            shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], np.asarray(shap_values)[0,0,:],
                                                   x_test.iloc[0], x_test.columns, max_display=len(x_test.columns),show=False)
            plt.savefig(os.path.join(consts.PATH_PROJECT_FUSION_FIGURES, 'SHAP_waterfall.pdf'))

        elif SHAP== 'force_plot':
            explainer = shap.KernelExplainer(best_clf.predict_proba, x_train, link="logit")
            shap_values = explainer.shap_values(x_test.iloc[[3],:])
            x=pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED,'test_36.csv'))
            x=x[x_test.columns]
            shap.force_plot(explainer.expected_value[1],
                            shap_values[1],
                            x.iloc[[3],:],
                            feature_names=x_test.columns.values,
                            # out_names='CVD risk',
                            link="identity",
                            figsize=(18, 5),
                            text_rotation=30,
                            matplotlib=True,
                            show=True )
            # shap.plots.force(explainer.expected_value[1],np.asarray(shap_values)[1,1,:],
            #                  feature_names=x_test.columns,matplotlib=True,show=False)
            # plt.savefig(os.path.join(consts.PATH_PROJECT_FUSION_FIGURES, 'SHAP_force_plot9.pdf'))
        # path_shap_exp = os.path.join(consts.PATH_PROJECT_METRICS,
        #                               'explainer_shap_{}.bz2'.format(clf))
        # pickle.dump(explainer, open(str(path_shap_exp), 'wb'))

        return shap_values[1]
    elif ALE:
        return best_clf
    else:
        for i in range(len(y_pred_test)):
            if y_pred_test[i] >= 0.5:
                y_pred_test[i] = 1.0
            else:
                y_pred_test[i] = 0.0

        for i in range(len(y_pred_train)):
            if y_pred_train[i] >= 0.5:
                y_pred_train[i] = 1.0
            else:
                y_pred_train[i] = 0.0
        print('Performance in test: ')
        acc_val, specificity_val, recall_val, roc_auc_val = compute_classification_prestations(y_test, y_pred_test)

        if Partitions:
            return y_pred_test, y_pred_train
        else:
            return acc_val, specificity_val, recall_val, roc_auc_val


def train_several_clfs(clf_name, x_features, y_label,bbdd_name,features,test_s,tfidf):

    list_acc = []
    list_specificity = []
    list_sensitivity = []
    list_auc_roc = []

    for i in range(len(SEEDS)):

        x_train_scaled, x_test_scaled, y_train, y_test = preprocessing_function(x_features, y_label,
                                                                                SEEDS[i], bbdd_name,
                                                                                test_s, tfidf)

        
        if len(features) == len(x_features.columns):

            acc_val, specificity_val, recall_val, roc_auc_val = train_compute_metrics(clf_name,
                                                                                      x_train_scaled, y_train,
                                                                                      x_test_scaled, y_test, SEEDS[i])


        else:
            acc_val, specificity_val, recall_val, roc_auc_val = train_compute_metrics(clf_name,
                                                                                      x_train_scaled[features], y_train,
                                                                                      x_test_scaled[features], y_test, SEEDS[i])
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
    # pd.DataFrame(np.asarray(IDS).T,columns=['ID1','ID2','ID3','ID4','ID5']).to_csv(os.path.join(consts.PATH_PROJECT_REPORTS ,'ID.csv'))
    return list_dicts_metrics


def call_clfs(x_features, y_label, bbdd_name, features, test_s, tfidf=20):

    df_metrics = pd.DataFrame(columns=['model', 'metric', 'mean', 'std'])

    for clf_name in list_clfs:
        list_metrics = train_several_clfs(clf_name, x_features, y_label,bbdd_name,features,test_s,tfidf)

        for dict_metric in list_metrics:
            # df_aux = {'model': clf_name, 'metric': 0, 'mean': 0, 'std': 0}f_metrics = df_metrics.append(dict_metric, ignore_index=True)
            df_metrics = pd.concat([df_metrics, pd.DataFrame(dict_metric, index=[0])])
    return df_metrics


def train_best_clfs(clf_name, x_features, y_label, bbdd_name, features, test_s,tfidf=10):
    list_pred_train = []
    list_pred_test = []
    list_y_train = []
    list_y_test = []
    for i in SEEDS:
        print(i)
        x_train_scaled, x_test_scaled, y_train, y_test = preprocessing_function(x_features, y_label, i, bbdd_name, test_s,tfidf)
        if len(features) == len(x_features.columns):
            y_pred_test, y_pred_train = train_compute_metrics(clf_name, x_train_scaled, y_train,
                                                              x_test_scaled, y_test, i,partitions=True)
        else:
            y_pred_test, y_pred_train = train_compute_metrics(clf_name, x_train_scaled[features], y_train,
                                                              x_test_scaled[features], y_test, i,partitions=True)
        list_pred_train.append(y_pred_train)
        list_pred_test.append(y_pred_test)
        list_y_test.append(y_test)
        list_y_train.append(y_train)

    return list_pred_train, list_pred_test, list_y_train, list_y_test


def call_best_clfs(x_features, y_label, bbdd_name, features, list_clfs, test_s,tfidf=10):
    for clf_name in list_clfs:
        list_pred_train, list_pred_test, list_y_train, list_y_test = train_best_clfs(clf_name, x_features, y_label,
                                                                                     bbdd_name, features, test_s,tfidf)

    return list_pred_train, list_pred_test, list_y_train, list_y_test


def train_several_clfs_fusion(clf_name, train_databases, test_databases, features,SHAP=False):
    list_acc = []
    list_specificity = []
    list_sensitivity = []
    list_auc_roc = []

    for i, j in enumerate(train_databases):
        x_train = j.drop(['label'], axis=1)
        x_test = test_databases[i].drop(['label'], axis=1)
        y_train = j['label']
        y_test = test_databases[i]['label']
        if len(features) == 0:

            acc_val, specificity_val, recall_val, roc_auc_val = train_compute_metrics(clf_name, x_train, y_train,
                                                                               x_test, y_test,seed=SEEDS[i],SHAP=SHAP)
        else:
            acc_val, specificity_val, recall_val, roc_auc_val = train_compute_metrics(clf_name, x_train[features],
                                                                                      y_train, x_test[features], y_test,seed=SEEDS[i],SHAP=SHAP)

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


def call_models_fusion(train_databases, test_databases,
                list_clfs=['RandomForest', 'knn', 'dt', 'svm', 'reglog', 'lasso'], features=[],SHAP=False):
    df_metrics = pd.DataFrame(columns=['model', 'metric', 'mean', 'std'])

    for clf_name in list_clfs:
        list_metrics = train_several_clfs_fusion(clf_name, train_databases, test_databases, features,SHAP=SHAP)

        for dict_metric in list_metrics:
            # df_aux = {'model': clf_name, 'metric': 0, 'mean': 0, 'std': 0}
            df_metrics = pd.concat([df_metrics, pd.DataFrame(dict_metric, index=[0])])
    return df_metrics



