import numpy as np
import pandas as pd
import os
import warnings
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from utils.evaluator import compute_classification_prestations
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import utils.consts as consts
from utils.Embedding_text import Embedded_text


def main_function(df_data, var_name='raw_medcon', encoding='tfidf', classifiers='lasso', ngrams=1, embedding_size=50,
                  Reduction='None',kernelKPCA='rbf',FS=False, path=''):
    y_label=df_data['label_encoded']
    df_acc = pd.DataFrame([[0] * len(classifiers)] * 5, columns=classifiers)
    df_recall = pd.DataFrame([[0] * len(classifiers)] * 5, columns=classifiers)
    df_roc = pd.DataFrame([[0] * len(classifiers)] * 5, columns=classifiers)
    df_spec = pd.DataFrame([[0] * len(classifiers)] * 5, columns=classifiers)
    list_mae_test = []
    list_mae_train = []
    for idx in range(len(consts.SEEDS)):
        x_train, x_test, y_train, y_test = Embedded_text(df_data, var_name, encoding,
                      ngrams, embedding_size,
                      Reduction, consts.SEEDS[idx], 0.2, y_label, kernelKPCA, FS,
                      path)
        # df_train = pd.concat([x_train, y_train.reset_index(drop=True)], axis=1)
        # df_test = pd.concat([x_test, y_test.reset_index(drop=True)], axis=1)
        # df_train.to_csv(os.path.join(path, 'train'+str(idx)+'.csv'))
        # df_test.to_csv(os.path.join(path, 'test'+str(idx)+'.csv'))
        print('Partition: ', idx)
        for i in classifiers:
            if i == 'lasso':
                print(i, '----', encoding, '---')
                model_lasso = Lasso(max_iter=5000, random_state=0)

                hyperparameter_space = {
                    'alpha': np.logspace(-6, 3, 10)
                }
                grid_cv = GridSearchCV(estimator=model_lasso, param_grid=hyperparameter_space, scoring='roc_auc',
                                       cv=5, n_jobs=-1)
                grid_cv.fit(x_train, y_train)

                clf_model = grid_cv.best_estimator_
                clf_model.fit(x_train, y_train)
                y_pred = clf_model.predict(x_test)

                for l in range(len(y_pred)):
                    if y_pred[l] >= 0.5:
                        y_pred[l] = 1.0
                    else:
                        y_pred[l] = 0.0
                acc_val, specificity_val, recall_val, roc_auc_val = compute_classification_prestations(y_test, y_pred)
                df_acc[i].iloc[idx ] = acc_val
                df_recall[i].iloc[idx ] = recall_val
                df_roc[i].iloc[idx] = roc_auc_val
                df_spec[i].iloc[idx] = specificity_val
            elif i == 'Random Forest':
                print(i, '----', encoding, '---')

                model = RandomForestClassifier(n_jobs=4, random_state=0)
                model_param_grid = {
                    'n_estimators': [10,20,30],
                    'max_depth': range(2, 10),
                    'criterion': ['gini', 'entropy'],
                    'min_samples_split': range(2, 12,2)}

                grid_cv = GridSearchCV(model, param_grid=model_param_grid, cv=5, scoring='roc_auc')
                grid_cv.fit(x_train, y_train)
                print(grid_cv.best_params_)
                model_clf = grid_cv.best_estimator_

                y_pred = model_clf.predict(x_test)
                acc_val, specificity_val, recall_val, roc_auc_val = compute_classification_prestations(y_test, y_pred)
                df_acc[i].iloc[idx] = acc_val
                df_recall[i].iloc[idx] = recall_val
                df_roc[i].iloc[idx] = roc_auc_val
                df_spec[i].iloc[idx] = specificity_val
            elif i== 'knn':
                selected_clf = KNeighborsClassifier()
                param_grid = {
                    'n_neighbors': range(1, 50, 2),
                    # 'metric': 'hamming'
                }
                grid_cv = GridSearchCV(selected_clf, param_grid=param_grid, cv=5, scoring='roc_auc')
                grid_cv.fit(x_train, y_train)
                print(grid_cv.best_params_)
                mlp_clf = grid_cv.best_estimator_

                y_pred = mlp_clf.predict(x_test)
                acc_val, specificity_val, recall_val, roc_auc_val = compute_classification_prestations(y_test, y_pred)
                df_acc[i].iloc[idx] = acc_val
                df_recall[i].iloc[idx] = recall_val
                df_roc[i].iloc[idx] = roc_auc_val
                df_spec[i].iloc[idx] = specificity_val

            elif i == 'reglog':
                selected_clf = LogisticRegression()
                param_grid = {
                    'penalty': ['l1' ],
                    'C': [1e-6, 1e-5, 1e-4, 1e-2, 1, 5, 10, 20],
                    'solver': ['liblinear', 'saga']}

                grid_cv = GridSearchCV(selected_clf, param_grid=param_grid , cv=5, scoring='roc_auc')
                grid_cv.fit(x_train, y_train)
                print(grid_cv.best_params_)
                mlp_clf = grid_cv.best_estimator_

                y_pred = mlp_clf.predict(x_test)
                acc_val, specificity_val, recall_val, roc_auc_val = compute_classification_prestations(y_test, y_pred)
                df_acc[i].iloc[idx] = acc_val
                df_recall[i].iloc[idx] = recall_val
                df_roc[i].iloc[idx] = roc_auc_val
                df_spec[i].iloc[idx] = specificity_val

            elif i == 'svm':
                print(i, '----', encoding, '---')

                model_clf_generic = SVC(max_iter=100000, random_state=0)

                hyperparameter_space = {
            'decision_function_shape': ['ovo'],
            'kernel': ['rbf', 'poly'],
            'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
            'C': [0.001, 0.10, 0.1, 10, 25, 100, 1000]
        }
                grid_cv = GridSearchCV(estimator=model_clf_generic, param_grid=hyperparameter_space, scoring='roc_auc',
                                       cv=5)
                grid_cv.fit(x_train, y_train)

                clf_model = grid_cv.best_estimator_
                clf_model.fit(x_train, y_train)

                y_pred = clf_model.predict(x_test)
                acc_val, specificity_val, recall_val, roc_auc_val = compute_classification_prestations(y_test, y_pred)
                df_acc[i].iloc[idx ] = acc_val
                df_recall[i].iloc[idx ] = recall_val
                df_roc[i].iloc[idx ] = roc_auc_val
                df_spec[i].iloc[idx ] = specificity_val
            elif i == 'dt':
                print(i, '----', encoding, '---')

                tuned_parameters = {
            'max_depth': range(3, 20),
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 3, 4, 5, 6]
        }
                model_tree = DecisionTreeClassifier(random_state=0)

                grid_cv = GridSearchCV(estimator=model_tree, param_grid=tuned_parameters, scoring='roc_auc', cv=5)
                grid_cv.fit(x_train, y_train)
                clf_model = grid_cv.best_estimator_

                clf_model.fit(x_train, y_train)
                y_pred = clf_model.predict(x_test)
                acc_val, specificity_val, recall_val, roc_auc_val = compute_classification_prestations(y_test, y_pred)

                df_acc[i].iloc[idx ] = acc_val
                df_recall[i].iloc[idx ] = recall_val
                df_roc[i].iloc[idx ] = roc_auc_val
                df_spec[i].iloc[idx] = specificity_val

    acc = df_acc.mean().tolist()
    acc_std = df_acc.std().tolist()

    recall = df_recall.mean().tolist()
    recall_std = df_recall.std().tolist()

    roc = df_roc.mean().tolist()
    roc_std = df_roc.std().tolist()

    spec = df_spec.mean().tolist()
    spec_std = df_spec.std().tolist()
    if len(list_mae_train) > 1:
        return acc, acc_std, recall, recall_std, roc, roc_std, spec, spec_std, np.mean(list_mae_test), np.mean(
            list_mae_train)
    else:
        return acc, acc_std, recall, recall_std, roc, roc_std, spec, spec_std, 0, 0

def Embeddings(Database, var_name, classifier,metrics,encoding, ngrams=1, Reduction_model='None', path='',kernel_KPCA='rbf',FS=False):
    mat = [0] * len(metrics)
    mat2 = []
    for e in range(len(classifier)):
        mat2.append(mat)
    ROC = pd.DataFrame(data=mat2, columns=metrics)
    ACC = pd.DataFrame(data=mat2, columns=metrics)
    REC = pd.DataFrame(data=mat2, columns=metrics)
    SPEC = pd.DataFrame(data=mat2, columns=metrics)
    for t in encoding:
        list_error_test = []
        list_error_train = []
        for i in metrics:
            print('--' + str(i) + '--')
            acc, acc_std, recall, recall_std, roc, roc_std, spec, spec_std, error_test, error_train = main_function(
                Database, var_name, encoding=t, classifiers=classifier, ngrams=ngrams, embedding_size=int(i),
                Reduction=Reduction_model,kernelKPCA=kernel_KPCA,FS=FS,path= path)
            ROC[i] = np.asarray([roc, roc_std]).T.tolist()
            ACC[i] = np.asarray([acc, acc_std]).T.tolist()
            REC[i] = np.asarray([recall, recall_std]).T.tolist()
            SPEC[i] = np.asarray([spec, spec_std]).T.tolist()
            list_error_test.append(error_test)
            list_error_train.append(error_train)
        p= str(ngrams)+'GRAM'
        ROC1 = ROC.T
        ROC1.columns = classifier
        ROC1 = ROC1.T

        ACC1 = ACC.T
        ACC1.columns = classifier
        ACC1 = ACC1.T

        REC1 = REC.T
        REC1.columns = classifier
        REC1 = REC1.T

        SPEC1 = SPEC.T
        SPEC1.columns = classifier
        SPEC1 = SPEC1.T
        if type(FS)== int:
            p=p+'_FS'+str(FS)
        if  Reduction_model != 'KPCA':
            ROC1.to_csv(os.path.join(path, t, p + '_' + Reduction_model + '_ROC.csv'))
            ACC1.to_csv(os.path.join(path, t, p + '_' + Reduction_model + '_ACC.csv'))
            REC1.to_csv(os.path.join(path, t, p + '_' + Reduction_model + '_REC.csv'))
            SPEC1.to_csv(os.path.join(path, t, p + '_' + Reduction_model + '_SPEC.csv'))
            if Reduction_model == 'AE' or Reduction_model == 'SAE':
                df = pd.DataFrame(columns=metrics).T
                df['test'] = list_error_test
                df['train'] = list_error_train
                df = df.T
                df.to_csvos.path.join(path, t , 'Reconstruction_error' + '_' + Reduction_model + '.csv')
        else:
            ROC1.to_csv(os.path.join(path, t, p + '_' + Reduction_model+ kernel_KPCA + '_ROC.csv'))
            ACC1.to_csv(os.path.join(path, t, p + '_' + Reduction_model +kernel_KPCA+ '_ACC.csv'))
            REC1.to_csv(os.path.join(path, t, p + '_' + Reduction_model +kernel_KPCA+ '_REC.csv'))
            SPEC1.to_csv(os.path.join(path, t, p + '_' + Reduction_model +kernel_KPCA + '_SPEC.csv'))