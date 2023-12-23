import pandas as pd
import os
from utils.Interpretability import call_clf_interpretability_SHAP
from fusion.late import late_fusion, Meta_classifier
from fusion.early import early_fusion, classifier_early
import utils.consts as consts
import numpy as np

def early_fusion_approaches():
    ### EARLY FUSION###
    print('** EARLY FUSION **')
    databases_list = ['Unaware', 'Fear', 'Medications', 'Signal', 'BTOTSCORE', 'Conditions', 'MOCA', 'Lifestyle', 'BSample',
                      'Attitude', 'Depression']
    train, test = early_fusion(databases_list, partition=0.2)
    classifier_early(train, test)

    ### EARLY FUSION Pre FS###
    print('** EARLY FUSION Pre FS**')

    FS_list = [9, 7, 2, 9, 11, 3, 4, 6, 4, 5, 4]
    train, test = early_fusion(databases_list, partition=0.2, FS=FS_list)
    classifier_early(train, test)

    ### EARLY FUSION Post FS###
    print('** EARLY FUSION Post FS** ')

    train, test = early_fusion(databases_list, partition=0.2)
    classifier_early(train, test, FS=18)

    ### EARLY FUSION Double###
    print('** EARLY FUSION Double**')

    FS_list = [9, 7, 2, 8, 11, 3, 4, 6, 4, 5, 4]
    train, test = early_fusion(databases_list, partition=0.2, FS=FS_list)
    df1 = classifier_early(train, test, FS=25)

    ###EARLY FUSION IA###
    print('** EARLY FUSION IA **')
    databases_list = ['Unaware', 'Fear', 'Medications', 'Signal', 'BTOTSCORE', 'Conditions', 'MOCA', 'Lifestyle']
    for e in range(2, len(databases_list) + 1):
        train, test = early_fusion(databases_list[0:e], partition=0.2)
        classifier_early(train, test)
    #
    # ### EARLY-IA-POST-FS ###
    print('** EARLY FUSION IA-Fs-Post**')
    FS_list = [16, 18, 18, 40, 22, 23, 23]
    for e in range(2, len(databases_list) + 1):
        train, test = early_fusion(databases_list[0:e], partition=0.2)
        classifier_early(train, test, FS=FS_list[e - 2])

    ###EARLY-IA-PRE-FS###
    print('** EARLY FUSION IA-PRE-FS**')
    FS_list = [9, 7, 2, 8, 11, 3, 4, 6]
    for e in range(2, len(databases_list) + 1):
        train, test = early_fusion(databases_list[0:e], partition=0.2, FS=FS_list[0:e])
        if e== 5:
            u = call_clf_interpretability_SHAP(train, test, 'knn')
        classifier_early(train, test)

### LATE FUSION  CREATION TRAIN TEST###
def late_train_test_creation():
    databases_list = ['Unaware', 'Fear', 'Medications', 'Signal', 'BTOTSCORE', 'Conditions', 'MOCA', 'Lifestyle', 'BSample',
                      'Attitude', 'Depression']
    clf_NoFS = ['lasso', 'knn', 'knn', 'knn', 'reglog', 'lasso', 'reglog', 'knn', 'lasso', 'knn', 'svm']
    print('** creating train test for normal late fusion **')
    train, test = late_fusion(databases_list,
                              classifiers=clf_NoFS,
                              partition=0.2,
                              paths=consts.PATH_PROJECT_FUSION_METRICS)
    for e in range(len(train)):
        train[e].to_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_FUSION, 'Late_Train_NoFS_' + str(e) + '.csv'),
                        index=False)

    for e in range(len(test)):
        test[e].to_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_FUSION, 'Late_Test_NoFS' + str(e) + '.csv'),
                       index=False)

    ### LATE FUSION PRE FS CREATION TRAIN TEST###
    databases_list = ['Unaware', 'Fear', 'Medications', 'Signal', 'BTOTSCORE', 'Conditions', 'MOCA', 'Lifestyle',
                      'BSample','Attitude', 'Depression']
    clf_FS = ['lasso', 'knn', 'reglog', 'knn', 'knn', 'svm', 'knn', 'knn', 'lasso', 'lasso', 'lasso']
    FS = [9, 7, 2, 8, 11, 3, 4, 6, 4, 5, 4]
    print('** creating train test for Pre Fs late fusion **')
    train, test = late_fusion(databases_list,
                              classifiers=clf_FS,
                              partition=0.2,
                              features_selected=FS,
                              paths=consts.PATH_PROJECT_FUSION_METRICS)
    for e in range(len(train)):
        train[e].to_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_FUSION, 'Late_Train_FS_' + str(e) + '.csv'),
                        index=False)

    for e in range(len(test)):
        test[e].to_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_FUSION, 'Late_Test_FS' + str(e) + '.csv'),
                       index=False)


def READ_LATE(database_list, FS=False):
    if 'label' not in database_list:
        database_list.append('label')
    train = []
    test = []
    for e in range(5):
        if FS:
            df = pd.read_csv(
                os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_FUSION, 'Late_Train_FS_' + str(e) + '.csv'))
            train.append(df[database_list])
            df = pd.read_csv(
                os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_FUSION, 'Late_Test_FS' + str(e) + '.csv'))
            test.append(df[database_list])
        else:
            df = pd.read_csv(
                os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_FUSION, 'Late_Train_NoFS_' + str(e) + '.csv'))
            train.append(df[database_list])
            df = pd.read_csv(
                os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_FUSION, 'Late_Test_NoFS' + str(e) + '.csv'))
            test.append(df[database_list])
    return train, test

def late_fusion_main():
    ###LATE FUSION APPROACHES###

    databases_list = ['Unaware', 'Fear', 'Medications', 'Signal', 'BTOTSCORE', 'Conditions', 'MOCA', 'Lifestyle',
                       'BSample', 'Depression', 'label']
    train, test = READ_LATE(databases_list)

    ###LATE FUSION ###
    print('** LATE FUSION **')
    N_A_N = Meta_classifier(train, test, meta='Average', FS=False, names='_Total_noFS')
    N_A_N = Meta_classifier(train, test, meta='clf', FS=False, names='_Total_noFS')

    print('** LATE FUSION IA **')
    ###LATE FUSION IA ###
    for e in range(2, 8):
        train, test = READ_LATE(databases_list[:e])
        N_A_N = Meta_classifier(train, test, meta='Average', FS=False, names=str(e) + '_noFS')
        N_A_N = Meta_classifier(train, test, meta='clf', FS=False, names=str(e) + '_noFS')


    ###LATE FUSION POST-FS###

    print('** LATE FUSION POST-FS **')
    N_A_N = Meta_classifier(train, test, meta='Average', FS=3, names='TOTAL_postFS')
    N_A_N = Meta_classifier(train, test, meta='clf', FS=3, names='TOTAL_postFS')

    ###LATE FUSION IA PRE FS###
    print('** LATE FUSION IA PRE FS **')
    for e in range(2, 8):
        train, test = READ_LATE(databases_list[:e], FS=True)
        # u=call_clf_interpretability(train, test,'knn')
        N_A_N = Meta_classifier(train, test, meta='Average', FS=False, names=str(e) + '_preFS')
        N_A_N = Meta_classifier(train, test, meta='classifier', FS=False, names=str(e) + '_preFS')

    ###LATE FUSION DOUBLE FS###
    print('** LATE FUSION DOUBLE FS **')
    train, test = READ_LATE(databases_list, FS=True)
    N_A_N = Meta_classifier(train, test, meta='Average', FS=5, names='5_doubleFS')
    N_A_N = Meta_classifier(train, test, meta='clf', FS=5, names='5_doubleFS')

    ###LATE FUSION Pre FS###
    print('LATE FUSION Pre FS**')
    train, test = READ_LATE(databases_list, FS=True)
    N_A_N = Meta_classifier(train, test, meta='clf', FS=False, names='Total_preFS')
    N_A_N = Meta_classifier(train, test, meta='Average', FS=False, names='Total_preFS')
