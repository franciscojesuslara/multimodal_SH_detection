import pandas as pd
from utils.check_patients import get_patients_id
import utils.consts as consts
import os
from utils.FS_bbdd import relief_bbdd
from utils.classifiers import call_clfs


def tabular_classification(databases_list, features_selected=[], paths=''):
    patients = get_patients_id(
        ['Medications', 'Conditions', 'Fear', 'BTOTSCORE', 'BSample', 'Attitude', 'Lifestyle', 'MOCA', 'Depression',
         'Signal', 'Unaware'])

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

        print('***', e, '***')
        Y = df1['label_encoded']
        X = df1.drop(['label_encoded', 'PtID'], axis=1)
        if len(features_selected) > 0:
            features = relief_bbdd(X, Y, e, test_s=0.2, FS=features_selected[j], path=paths)
            relief_bbdd(X, Y, e, test_s=0.2, FS='plot', path=paths)
            df_metrics = call_clfs(X, Y, e, features, 0.2)
            df_metrics.to_csv(os.path.join(paths, e+'_FS.csv'))
        else:
            df_metrics = call_clfs(X, Y, e, X.columns, 0.2)
        df_metrics.to_csv(os.path.join(paths, e+'.csv'))
