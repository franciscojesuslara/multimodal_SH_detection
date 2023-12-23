from skrebate import ReliefF
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

from utils.Preprocessing import preprocessing_function
warnings.filterwarnings('ignore')
import nltk

import utils.consts as consts
nltk.download('stopwords')
nltk.download('punkt')


def relief_plot(df_score,path_figure,name):

    if len(df_score['score_sum']) > 25:
        df_score=df_score.iloc[0:25]
    fig = plt.figure(figsize=(25, 15))
    fig.clf()
    ax = fig.subplots(1, 1)
    plt.bar(df_score['names'], df_score['score_sum'], yerr=df_score['std'], align='center', alpha=0.8, ecolor='black', capsize=3)
    plt.ylabel('Importance', fontname='serif', fontsize=60)
    plt.xticks(fontname='serif', fontsize=45)
    plt.yticks(fontname='serif', fontsize=45)
    # ax.set_title('RELIEF IMPORTANCE')
    fig.autofmt_xdate(rotation=45)
    # Save the figure and show
    plt.tight_layout()
    plt.savefig(os.path.join(path_figure, str(name)+ '.png'))
    plt.show()

    # fig = plt.figure(figsize=(20, 6))
    #
    # sns.barplot(y=df_score['score_sum'], x=df_score['names'])
    # plt.title('FEATURE IMPORTANCE')
    # plt.xlabel('Features')
    # plt.ylabel('Importance')
    # plt.tight_layout()
    # fig.autofmt_xdate(rotation=45)
    # plt.savefig(os.path.join(path_figure, str(name)+ '.png'))
    # plt.show()

    #
    # fig = plt.figure(figsize=(20, 6))
    # if len(df_score['score_sum'])> 100:
    #     df_score=df_score.iloc[0:100]
    # sns.barplot(y=df_score['score_sum'], x=df_score['names'])
    #
    # plt.title('FEATURE IMPORTANCE')
    # plt.xlabel('Features')
    # plt.ylabel('Importance')
    # plt.tight_layout()
    # fig.autofmt_xdate(rotation=45)
    # plt.savefig(os.path.join(path_figure, str(name)+ '.png'))
    # plt.show()

    list_diff = [0]
    count = 0
    for e in range(1, len(df_score['score_sum']), 1):
        p = df_score['score_sum'].iloc[e - 1] - df_score['score_sum'].iloc[e]
        count = count + p
        list_diff.append(count)

    fig = plt.figure(figsize=(20, 6))
    plt.plot(df_score['names'], list_diff)
    plt.title('FEATURE IMPORTANCE DIFFERENCE')
    plt.xlabel('Features')
    plt.ylabel('Difference')
    plt.tight_layout()
    fig.autofmt_xdate(rotation=45)
    plt.savefig(os.path.join(path_figure, 'diff_' + str(name)+'.png'))
    plt.show()

def sum_score(df_score):
    df_score.columns = ['names', 'score0', 'score1', 'score2', 'score3', 'score4']
    val=['score0', 'score1', 'score2', 'score3', 'score4']
    df_score['std']=df_score[val].astype(float).T.std()
    df_score['score_sum']=df_score[val].astype(float).T.mean()
    df_score.sort_values(by=['score_sum'], ascending=False, inplace=True)
    return df_score

def relief_FS(X,Y,relief,i):
    relief.fit(X.values.astype(float), Y.values.astype(float))
    x_selected_features, list_selected_features_indices, scores_sorted = relief.transform(X.values)
    column = 'score' + str(i)
    df = pd.DataFrame(
        np.asarray([list(X.iloc[:, list(list_selected_features_indices)].columns),
                    scores_sorted.astype(float)]).T,
        columns=['names', column])
    return df

def relief_bbdd(x_features, y_label,bbdd_name,test_s=0.2,FS=True,path=''):
    if 'PtID' in x_features.columns:
        x_features.drop(['PtID'], axis=1, inplace=True)
    for i in range(len(consts.SEEDS)):
        x_train_scaled, x_test_scaled, y_train, y_test= preprocessing_function(x_features, y_label, consts.SEEDS[i], bbdd_name, test_s,295)
        # x_train_scaled.to_csv('Early_ID'+str(consts.SEEDS))
        if type(FS)!= int:
            relief = ReliefF(n_features_to_select=len(x_train_scaled.columns))
        else:
            relief = ReliefF(n_features_to_select=FS)
        df = relief_FS(x_train_scaled, y_train, relief, i)
        try:
            df_score = df_score.merge(df, on=['names'])
        except:
            df_score = df

    df_score = sum_score(df_score)

    if type(FS) != int:
        relief_plot(df_score, path, bbdd_name)
        plt.close()
    else:
        df_score.to_csv(os.path.join(path, 'FS_' + bbdd_name + '.csv'))
        return df_score['names'][:FS]