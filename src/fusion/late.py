import os
import numpy as np
import pandas as pd
from utils.FS_relief import relief_fusion,relief_plot
import utils.consts as consts
from utils.evaluator import compute_classification_prestations
from utils.check_patients import get_patients_id
from utils.FS_bbdd import relief_bbdd
from utils.classifiers import call_best_clfs,call_models_fusion
import math


def late_fusion(databases_list,
                classifiers=['knn'],
                partition=0.8,
                features_selected=[],
                paths=consts.PATH_PROJECT_FUSION_FIGURES):
    x = len(databases_list) + 1
    # patients = get_patients_id(databases_list)

    patients = get_patients_id(
        ['Medications', 'Conditions', 'Fear', 'BTOTSCORE', 'BSample', 'Attitude', 'Lifestyle', 'MOCA', 'Depression',
         'Signal', 'Unaware'])
    z = math.ceil(partition * len(patients))
    print('num. patients: ', len(patients))
    y = len(patients)-z

    databases_list.append('label')
    df_train1 = pd.DataFrame([[0] * x] * y, columns=databases_list)
    df_train2 = pd.DataFrame([[0] * x] * y, columns=databases_list)
    df_train3 = pd.DataFrame([[0] * x] * y, columns=databases_list)
    df_train4 = pd.DataFrame([[0] * x] * y, columns=databases_list)
    df_train5 = pd.DataFrame([[0] * x] * y, columns=databases_list)

    df_test1 = pd.DataFrame([[0] * x] * z, columns=databases_list)
    df_test2 = pd.DataFrame([[0] * x] * z, columns=databases_list)
    df_test3 = pd.DataFrame([[0] * x] * z, columns=databases_list)
    df_test4 = pd.DataFrame([[0] * x] * z, columns=databases_list)
    df_test5 = pd.DataFrame([[0] * x] * z, columns=databases_list)

    for j, e in enumerate(databases_list[:-1]):
        if e == 'Attitude':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_attitude.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            # df1 = df1.sort_values('PtID')

        elif e == 'BMedChart':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_BMedChart.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            # df1 = df1.sort_values('PtID')

        elif e == 'BTOTSCORE':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_BTOTSCORE.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            # df1 = df1.sort_values('PtID')

        elif e == 'Depression':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_depression.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            # df1 = df1.sort_values('PtID')

        elif e == 'Fear':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_fear.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            # df1 = df1.sort_values('PtID')

        elif e == 'Lifestyle':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_lifestyle.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            # df1 = df1.sort_values('PtID')

        elif e == 'Conditions':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TEXT, 'bbdd_medconditions2.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            # df1 = df1.sort_values('PtID')

        elif e == 'Medications':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TEXT, 'bbdd_medications2.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            # df1 = df1.sort_values('PtID')

        elif e == 'Conditions_hot':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TEXT, 'bbdd_medconditions.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            # df1 = df1.sort_values('PtID')

        elif e == 'Medications_hot':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TEXT, 'bbdd_medications.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            # df1 = df1.sort_values('PtID')

        elif e == 'MOCA':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_MOCA.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            # df1 = df1.sort_values('PtID')

        elif e == 'Unaware':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_unaware.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            # df1 = df1.sort_values('PtID')

        elif e == 'BSample':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_BSample.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            # df1 = df1.sort_values('PtID')

        elif e == 'Signal':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_SIGNAL, 'gSAX.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            # df1 = df1.sort_values('PtID')

        print('***', e, '***')

        Y = df1['label_encoded']
        X = df1.drop(['label_encoded', 'PtID'], axis=1)

        if len(features_selected) > 0:
            features= relief_bbdd(X, Y, e, test_s=0.2,FS=features_selected[j], path=paths)

            list_pred_train, list_pred_test, list_y_train, list_y_test = call_best_clfs(X, Y, e, features,
                                                                                        [classifiers[j]], z,300)

        else:
            list_pred_train, list_pred_test, list_y_train, list_y_test = call_best_clfs(X, Y, e, X.columns,
                                                                                        [classifiers[j]], z)
        df_train1[e] = list_pred_train[0]
        df_train2[e] = list_pred_train[1]
        df_train3[e] = list_pred_train[2]
        df_train4[e] = list_pred_train[3]
        df_train5[e] = list_pred_train[4]

        df_train1['label'] = list_y_train[0].reset_index().iloc[:, 1]
        df_train2['label'] = list_y_train[1].reset_index().iloc[:, 1]
        df_train3['label'] = list_y_train[2].reset_index().iloc[:, 1]
        df_train4['label'] = list_y_train[3].reset_index().iloc[:, 1]
        df_train5['label'] = list_y_train[4].reset_index().iloc[:, 1]

        df_test1[e] = list_pred_test[0]
        df_test2[e] = list_pred_test[1]
        df_test3[e] = list_pred_test[2]
        df_test4[e] = list_pred_test[3]
        df_test5[e] = list_pred_test[4]

        df_test1['label'] = list_y_test[0].reset_index().iloc[:, 1]
        df_test2['label'] = list_y_test[1].reset_index().iloc[:, 1]
        df_test3['label'] = list_y_test[2].reset_index().iloc[:, 1]
        df_test4['label'] = list_y_test[3].reset_index().iloc[:, 1]
        df_test5['label'] = list_y_test[4].reset_index().iloc[:, 1]
        train = [df_train1, df_train2, df_train3, df_train4, df_train5]
        test = [df_test1, df_test2, df_test3, df_test4, df_test5]

    return train, test

def Average_classfier(df,features=[]):
    if len(features) == 0:
        features= df[0].drop(['label'], axis=1).columns
    df_evaluation = pd.DataFrame([[0] * 4] * len(df), columns=['ROC', 'ACC', 'REC', 'SPEC'])
    for j, e in enumerate(df):
        y_test = e['label']
        X = e.drop(['label'], axis=1)
        X=X[features]
        y_pred = []
        for i in range(len(X)):
            y_pred.append(X.iloc[i].mode()[0])

        acc_val, specificity_val, recall_val, roc_auc_val = compute_classification_prestations(y_test, y_pred)

        df_evaluation['ACC'].iloc[j] = np.round(acc_val, 3)
        df_evaluation['REC'].iloc[j] = np.round(recall_val, 3)
        df_evaluation['ROC'].iloc[j] = np.round(roc_auc_val, 3)
        df_evaluation['SPEC'].iloc[j] = np.round(specificity_val, 3)

    ROC = [np.round(df_evaluation['ROC'].mean(), 3), np.round(df_evaluation['ROC'].std(), 3)]
    REC = [np.round(df_evaluation['REC'].mean(), 3), np.round(df_evaluation['REC'].std(), 3)]
    ACC = [np.round(df_evaluation['ACC'].mean(), 3), np.round(df_evaluation['ACC'].std(), 3)]
    SPEC = [np.round(df_evaluation['SPEC'].mean(), 3), np.round(df_evaluation['SPEC'].std(), 3)]

    return ROC, REC,ACC, SPEC

def Meta_classifier(train, test, meta='Average', FS=False, names='Single'):
    if meta == 'Average':
        if FS== False:
            ROC,REC,ACC,SPEC=Average_classfier(train, )
            ROC1, REC1, ACC1, SPEC1 = Average_classfier(test)

            Average_performance = pd.DataFrame([[ROC, ACC, REC, SPEC], [ROC1, ACC1, REC1, SPEC1]],
                                               columns=['ROC', 'ACC', 'REC', 'SPEC'])
            Average_performance = Average_performance.T
            Average_performance.columns = ['Train_perfromance', 'Test_perfromance']
            df = Average_performance.T
            df.to_csv(os.path.join(consts.PATH_PROJECT_FUSION_METRICS, str('LATE') + str(meta) + names + '.csv'))
            return df
        else:
            if type(FS)!= int:
                df1=relief_fusion(train,len(train[0].columns)-1)
                relief_plot(df1, consts.PATH_PROJECT_FUSION_FIGURES, 'LATE_' + str(len(train[0].columns) - 1))
            else:
                df_score=relief_fusion(train,FS)
                features1 = df_score['names'][:FS]
                ROC, REC, ACC, SPEC = Average_classfier(train, features=features1)
                ROC1, REC1, ACC1, SPEC1 = Average_classfier(test, features=features1)

                Average_performance = pd.DataFrame([[ROC, ACC, REC, SPEC], [ROC1, ACC1, REC1, SPEC1]],
                                                   columns=['ROC', 'ACC', 'REC', 'SPEC'])
                Average_performance = Average_performance.T
                Average_performance.columns = ['Train_perfromance', 'Test_perfromance']
                df = Average_performance.T
                df.to_csv(os.path.join(consts.PATH_PROJECT_FUSION_METRICS, str('LATE') + str(meta) + names + '.csv'))
                return df, features1
    else:
        if FS == False:
            df = call_models_fusion(train, test)
            df.to_csv(os.path.join(consts.PATH_PROJECT_FUSION_METRICS, str('LATE') + str(meta) + names + '.csv'))
            return df
        else:
            if type(FS) != int:
                df1= relief_fusion(train, len(train[0].columns) - 1)
                relief_plot(df1, consts.PATH_PROJECT_FUSION_FIGURES, 'LATE_' + str(len(train[0].columns) - 1))
            else:
                df_score = relief_fusion(train, FS)
                features1 = df_score['names'][:FS]
                df = call_models_fusion(train, test, features=features1)
                df.to_csv(os.path.join(consts.PATH_PROJECT_FUSION_METRICS,str('LATE')+str(meta)+ names+ '.csv'))
                return df, features1


def save_partitionsID(train, test):
    list3 = []
    for e in train:
        u = e['PtID']
        u.columns = ['ID', 'ID2', 'ID3', 'ID4', 'ID5', 'ID6', 'ID7', 'ID8']
        list3.append(u['ID'])

    train1 = pd.DataFrame(np.asarray(list3).T, columns=['PtID1', 'PtID2', 'PtID3', 'PtID4', 'PtI5'])
    train1.to_csv('trainID.csv')

    list3 = []
    for e in test:
        u = e['PtID']
        u.columns = ['ID', 'ID2', 'ID3', 'ID4', 'ID5', 'ID6', 'ID7', 'ID8']
        list3.append(u['ID'])

    test1 = pd.DataFrame(np.asarray(list3).T, columns=['PtID1', 'PtID2', 'PtID3', 'PtID4', 'PtI5'])
    test1.to_csv('testID.csv')
    return train1, test1


