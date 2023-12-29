import utils.consts as consts
import pandas as pd
import os
from utils.check_patients import get_patients_id
from text.embeddings_functions import Embeddings


# medical conditions
def conditions_unimodal():
    path1= os.path.join(consts.PATH_PROJECT_TEXT_METRICS, 'Conditions')
    df = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TEXT, 'bbdd_medconditions2.csv'))
    Base_de_datos = ['Attitude', 'BTOTSCORE', 'Depression', 'Fear', 'Lifestyle', 'Conditions', 'Medications', 'MOCA', 'Unaware', 'BSample', 'Signal']
    patients = get_patients_id(Base_de_datos)
    df = patients.merge(df, on=['PtID']).sort_values(by=['PtID'])


    # Embeddings(Database=df, var_name='medcon1',classifier= ['dt'], metrics=['30'], encoding=['tfidf'], path=path1, FS=False)
    # Embeddings(Database=df, var_name='medcon1',classifier= ['dt'], metrics=['70'], encoding=['w2vec'],  path=path1, FS=False)
    Embeddings(Database=df, var_name='medcon1', classifier=['knn'], metrics=['70'], encoding=['Longformer'], Reduction_model='KPCA', path=path1, FS=False)
    Embeddings(Database=df, var_name='medcon1', classifier=['lasso'], metrics=['50'], encoding=['Clinical'], Reduction_model='PCA', path=path1, FS=False)

    # Conditions with FS
    Embeddings(Database=df, var_name='medcon1', classifier=['knn'], metrics=['50'], encoding=['Clinical'], Reduction_model='PCA', path=path1, FS=3)

def medications_unimodal():
    path1 = os.path.join(consts.PATH_PROJECT_TEXT_METRICS, 'Medications')
    df = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TEXT, 'bbdd_medications2.csv'))
    Base_de_datos = ['Attitude', 'BTOTSCORE', 'Depression', 'Fear', 'Lifestyle', 'Conditions', 'Medications', 'MOCA',
                     'Unaware', 'BSample', 'Signal']
    patients = get_patients_id(Base_de_datos)
    df = patients.merge(df , on=['PtID']).sort_values(by=['PtID'])

    # Embeddings(Database=df, var_name='medications3',classifier= ['Random Forest'], metrics=['30'], encoding=['tfidf'], path=path1, FS=False)
    # Embeddings(Database=df, var_name='medications3',classifier= ['knn'], metrics=['10'], encoding=['w2vec'],  path=path1, FS=False)
    # Embeddings(Database=df, var_name='medications3',classifier= ['knn'], metrics=['50'], encoding=['Longformer'],Reduction_model='PCA', path=path1, FS=False)
    # Embeddings(Database=df, var_name='medications3',classifier= ['knn'], metrics=['10'], encoding=['Clinical'], Reduction_model='PCA', path=path1, FS=False)
    ## Medications with FS
    Embeddings(Database=df, var_name='medications3',classifier= ['knn'], metrics=['10'], encoding=['Clinical'], Reduction_model='PCA', path=path1, FS=2)