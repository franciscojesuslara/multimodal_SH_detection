import os
import numpy as np
import pandas as pd
import utils.consts as cons


def get_patients_id(databases_list=['Unaware', 'Fear', 'BTOTSCORE', 'BSample', 'Attitude', 'Lifestyle',
                                    'MOCA', 'Depression', 'Conditions', 'Medications', 'Signal']):
    df_final = []
    for e in databases_list:
        if e == 'Attitude':
            df1 = pd.read_csv(os.path.join(cons.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_attitude.csv'))
            df_final.append(df1)
        elif e == 'BMedChart':
            df1 = pd.read_csv(os.path.join(cons.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_BMedChart.csv'))
            df_final.append(df1)
        elif e == 'BTOTSCORE':
            df1 = pd.read_csv(os.path.join(cons.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_BTOTSCORE.csv'))
            df_final.append(df1)
        elif e == 'Depression':
            df1 = pd.read_csv(os.path.join(cons.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_depression.csv'))
            df_final.append(df1)
        elif e == 'Fear':
            df1 = pd.read_csv(os.path.join(cons.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_fear.csv'))
            df_final.append(df1)
        elif e == 'Lifestyle':
            df1 = pd.read_csv(os.path.join(cons.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_lifestyle.csv'))
            df_final.append(df1)
        elif e == 'Conditions':
            df1 = pd.read_csv(os.path.join(cons.PATH_PROJECT_DATA_PREPROCESSED_TEXT, 'bbdd_medconditions2.csv'))
            df_final.append(df1)
        elif e == 'Medications':
            df1 = pd.read_csv(os.path.join(cons.PATH_PROJECT_DATA_PREPROCESSED_TEXT, 'bbdd_medications2.csv'))
            df_final.append(df1)
        elif e == 'MOCA':
            df1 = pd.read_csv(os.path.join(cons.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_MOCA.csv'))
            df_final.append(df1)
        elif e == 'Unaware':
            df1 = pd.read_csv(os.path.join(cons.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_unaware.csv'))
            df_final.append(df1)
        elif e == 'BSample':
            df1 = pd.read_csv(os.path.join(cons.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_BSample.csv'))
            df_final.append(df1)
        elif e == 'Signal':
            df1 = pd.read_csv(os.path.join(cons.PATH_PROJECT_DATA_PREPROCESSED_SIGNAL, 'Time_series_Text.csv'))
            df_final.append(df1)

    list1 = []
    for e in df_final:
        for i in e['PtID']:
            list1.append(i)
    p = set(list1)
    id = []
    id_removed = []
    n_removed = []
    for e in p:
        if list1.count(e) == len(databases_list):
            id.append(e)
        else:
            id_removed.append(e)
            n_removed.append(list1.count(e))

    patients = pd.DataFrame(id, columns=['PtID'])

    # print('Number of patients selected: ', len(patients))
    # print('Patientes removed: ', pd.DataFrame(np.asarray([id_removed, n_removed]).T, columns=['ID','Appearances']))

    return patients


