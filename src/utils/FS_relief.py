from skrebate import ReliefF
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from text.Autoencoders import SAE_dim_reduction, AE_dim_reduction
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import nltk
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import utils.consts as consts
nltk.download('stopwords')
nltk.download('punkt')
from utils.train_neural_text_models import w2vec_encoding,fasttext_encoding
def relief_plot(df_score,path_figure,name):
    if len(df_score['score_sum']) > 100:
        df_score=df_score.iloc[0:100]
    fig, ax = plt.subplots()
    ax.bar(range(len(df_score['names'])), df_score['score_sum'], yerr=df_score['std'], align='center', alpha=0.8, ecolor='black', capsize=3)
    ax.set_ylabel('Importance')
    ax.set_xticks(range(len(df_score['names'])))
    ax.set_xticklabels(df_score['names'])
    ax.set_title('RELIEF IMPORTANCE')
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
    relief.fit(X.values, Y.values)
    x_selected_features, list_selected_features_indices, scores_sorted = relief.transform(X.values)
    column = 'score' + str(i)
    df = pd.DataFrame(
        np.asarray([list(X.iloc[:, list(list_selected_features_indices)].columns),
                    scores_sorted.astype(float)]).T,
        columns=['names', column])
    return df

def relief_fusion(train,n_features):
    for j, e in enumerate(train):
        X = e.drop(['label'], axis=1)
        Y = e['label']
        relief = ReliefF(n_features_to_select=n_features)
        df=relief_FS(X, Y, relief, j)
        try:
            df_score = df_score.merge(df, on=['names'])
        except:
            df_score = df
    df_score = sum_score(df_score)
    return df_score




def FS_text_fuction(dataframe, varname, y_label, encoding, Reduction, n_grams, FS, kernel_KPCA, embeddingsize, test_s, path):
    seed_value = 100
    bow_vectorizer = CountVectorizer(ngram_range=(1, n_grams), max_features=embeddingsize)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, n_grams), max_features=embeddingsize)
    for idx in range(len(consts.SEEDS)):
        x_train, x_test, y_train, y_test = train_test_split(dataframe, y_label, stratify=y_label,
                                                            random_state=consts.SEEDS[idx], test_size=test_s)
        if encoding == 'tfidf':
            m_tfidf_sparse = tfidf_vectorizer.fit_transform(x_train.values)
            x_train = m_tfidf_sparse.toarray()

        elif encoding == 'mbow':
            m_bow_sparse = bow_vectorizer.fit_transform(x_train)
            x_train = m_bow_sparse.toarray()

        elif encoding == 'w2vec':
            x_train,x_test= w2vec_encoding(x_train, x_test, embeddingsize, seed_value)

        elif encoding == 'fasttext':
            x_train, x_test = fasttext_encoding(x_train,x_test,n_grams, embeddingsize)
        else:
            if Reduction == 'PCA':
                mo = PCA(n_components=embeddingsize, random_state=0)
                mo = mo.fit(x_train)
                x_train = mo.transform(x_train)
            elif Reduction == 'AE':
                x_train, x_test, mae_train, mae_test = AE_dim_reduction(x_train, x_test, Emb_size=embeddingsize)

            elif Reduction == 'SAE':
                x_train, x_test, mae_train, mae_test = SAE_dim_reduction(x_train, x_test, Emb_size=embeddingsize,)

            elif Reduction == 'KPCA':
                mo = KernelPCA(n_components=embeddingsize, kernel=kernel_KPCA, random_state=0)
                mo = mo.fit(x_train)
                x_train = mo.transform(x_train)
        names = []
        for e in range(embeddingsize):
            names.append(str(varname) + '_' + str(e))
        x_train = pd.DataFrame(x_train, columns=names)
        if type(FS) != int:
            relief = ReliefF(n_features_to_select=len(x_train.columns))
        else:
            relief = ReliefF(n_features_to_select=FS)
        df = relief_FS(x_train, y_train, relief, idx)
        try:
            df_score = df_score.merge(df, on=['names'])
        except:
            df_score = df
    df_score = sum_score(df_score)
    if type(FS) != int:
        relief_plot(df_score, path, varname+encoding)
    else:
        df_score.to_csv(os.path.join(path,'FS_'+varname+'.csv'))
        return df_score['names'][:FS]