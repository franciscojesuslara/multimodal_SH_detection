import pandas as pd
import esax.get_motif as mot
import esax.get_subsequences as subs
from utils.plotter import lollipop_plot
from utils.classifiers import *
from utils.check_patients import get_patients_id
from sklearn.model_selection import train_test_split
from utils.FS_bbdd import relief_bbdd


def get_ecdf(data):
    """
    This method provides the empirical cumulative distribution function (ECDF) of a time series.

    :param data: a numeric vector representing the univariate time series
    :type data: pandas.Series
    :return: ECDF function of the time series
    :rtype: (x,y) -> x and y numpy.ndarrays
    """
    ecdf = calculate_ecdf(data)
    return ecdf


def calculate_ecdf(data):
    """
    This method calculates the empirical cumulative distribution function (ECDF) of a time series.
    Warning: This method is equal to stats::ecdf in R. The ECDF in
    statsmodels.distributions.empirical_distribution.ECDF does not calculate the same ECDF as stats::ecdf does.

    :param data: numeric vector representing the univariate time series
    :type: pandas.Series
    :return: ECDF for the time series as tuple of numpy.ndarrays
    :rtype: (x,y) -> x and y numpy.ndarrays
    """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n + 1) / n
    return x, y

def creation_quartiles(data,breaks):
    ecdf = get_ecdf(data)
    ecdf_df = pd.DataFrame()
    ecdf_df["x"] = ecdf[0]
    ecdf_df["y"] = ecdf[1]
    qq = np.linspace(start=0, stop=1, num=breaks + 1)
    # Store the percentiles
    per = np.quantile(ecdf_df["x"], qq)
    # Use only unique percentiles for the alphabet distribution
    per = np.unique(per)
    # Add the minimum as the lowest letter
    minimum = min(data)
    per[0] = minimum
    return per


def preprocessing_signal(df, e):
    u = df[df['patient_id'] == e]
    u.index = pd.to_datetime(u['new'])
    u = u['mean']
    u = u.resample('10T', offset='10s').mean().interpolate(method='slinear')
    u = u.dropna()
    return u


def Application_ESAX(signal_preprocessed, wl, vocab, ws, break_points):
    u = signal_preprocessed
    u.index = pd.arrays.DatetimeArray(u.index, dtype=np.dtype("<M8[ns]"), freq=None, copy=False)
    ts_subs, startpoints, indexes_subs = subs.get_subsequences(u, resolution=ws)
    # try:
    found_motifs = mot.get_motifs(u, ts_subs, breaks=vocab, word_length=wl, num_iterations=50,
                                  mask_size=2, mdr=2.5, cr1=5, cr2=1.5,
                                  per=break_points
                                  )
    # except:
    #     print('Error')

    if wl == 3:
        ESAX = found_motifs['ts_sax_df'][0] + found_motifs['ts_sax_df'][1] + found_motifs['ts_sax_df'][2]
    elif wl == 4:
        ESAX = found_motifs['ts_sax_df'][0] + found_motifs['ts_sax_df'][1] + found_motifs['ts_sax_df'][2] + \
               found_motifs['ts_sax_df'][3]
    elif wl == 5:
        ESAX = found_motifs['ts_sax_df'][0] + found_motifs['ts_sax_df'][1] + found_motifs['ts_sax_df'][2] + \
               found_motifs['ts_sax_df'][3] + found_motifs['ts_sax_df'][4]
    ESAX = ESAX.reset_index(drop=True)
    return ESAX,found_motifs

def Counting_wors_hypo(ESAX,found_motifs,ID,seed):
    Hypo = []
    for i in found_motifs['ts_subs']:
        if min(i) < 70:
            Hypo.append(1)
        else:
            Hypo.append(0)
    ESAX = pd.concat([ESAX, pd.Series(Hypo)], axis=1)
    ESAX.columns = ['Word', 'Hypo']
    p = ESAX.groupby('Word').sum()
    LEN = []
    WORD = []
    for q in set(ESAX['Word']):
        LEN.append(len(ESAX[ESAX['Word'] == q]))
        WORD.append(q)
    data = pd.DataFrame(np.asarray([LEN, WORD]).T, columns=['Number', 'Word'])
    data1 = p.merge(data, on='Word')
    data1.to_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_SIGNAL, str(ID) + '_seed_'+str(seed)+'.csv'))
    return ESAX


def Main_ESAX(df, wl, vocab, ws, test_s=0.2, seed=1):
    emb = []
    Quartiles = []
    valores = []
    pd1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED, 'BPtRoster.txt'), sep='|')
    pd1['BCaseControlStatus'] = pd1['BCaseControlStatus'].replace(['Case'], 0).replace(['Control'], 1)
    pd1['label_encoded'] = pd1['BCaseControlStatus']
    pd1.drop(['RecID', 'BCaseControlStatus'], axis=1, inplace=True)
    patients = get_patients_id()
    data = patients.merge(pd1, on=['PtID'])
    X_train, X_test, Y_train, Y_test = train_test_split(data['PtID'], data['label_encoded'],
                                                        test_size=test_s,
                                                        random_state=seed,
                                                        stratify=data['label_encoded'])
    for e in X_train.values:
        u = preprocessing_signal(df, e)
        valores.extend(u.values)
    break_points = creation_quartiles(valores, vocab)

    for e in patients['PtID']:
        u = preprocessing_signal(df, e)
        ESAX, found_motifs = Application_ESAX(u, wl, vocab, ws, break_points)
        ESAX = Counting_wors_hypo(ESAX, found_motifs, e, seed)
        text = ' '.join(ESAX['Word'].tolist())
        emb.append(text)
    data = pd.DataFrame(emb, columns=['text' + str(seed)])
    data['PtID'] = patients['PtID']
    data = data.merge(pd1, on='PtID')
    return data


def Contar_hypo(seed):
    Base_de_datos = ['Unaware', 'Fear', 'BTOTSCORE', 'BSample', 'Attitude', 'Lifestyle', 'MOCA',
                     'Depression', 'Conditions', 'Medications', 'Signal']
    pd1=pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED, 'BPtRoster.txt'), sep='|')
    pd1['BCaseControlStatus'] = pd1['BCaseControlStatus'].replace(['Case'], 0).replace(['Control'], 1)
    pd1['label_encoded'] = pd1['BCaseControlStatus']
    pd1.drop(['RecID', 'BCaseControlStatus'], axis=1, inplace=True)
    patients = get_patients_id(Base_de_datos)
    data = patients.merge(pd1, on=['PtID'])
    df_hypo = pd.DataFrame()
    df_no_hypo = pd.DataFrame()
    for e, j in enumerate(data['label_encoded']):
        if j == 0:
            df = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_SIGNAL,
                                          str(data['PtID'].iloc[e]) + '_seed_'+str(seed)+'.csv'))
            df_hypo = pd.concat([df, df_hypo])
        else:
            df = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_SIGNAL,
                                          str(data['PtID'].iloc[e]) + '_seed_'+str(seed)+'.csv'))
            df_no_hypo = pd.concat([df, df_no_hypo])
    df_no_hypo = df_no_hypo.drop(['Unnamed: 0'], axis=1)
    df_hypo = df_hypo.drop(['Unnamed: 0'], axis=1)
    df_hypo = df_hypo.groupby(by='Word').sum()
    df_no_hypo = df_no_hypo.groupby(by='Word').sum()
    df_hypo['Percent_hypo'] = round(100*df_hypo['Hypo']/df_hypo['Number'],2)
    df_no_hypo['Percent_no_hypo'] = round(100*df_no_hypo['Hypo']/df_no_hypo['Number'],2)
    df_hypo.columns = ['Episodes of Hypo in severe', 'Number in severe', 'Percent episodes of hypo in severe']
    df_no_hypo.columns =['Episodes of Hypo in control', 'Number in control', 'Percent episodes of hypo in control']
    df_hypo_final = df_hypo.join(df_no_hypo)
    print(len(df_hypo_final))
    df_hypo_final['Difference in Number'] = df_hypo_final['Number in severe'] - df_hypo_final['Number in control']
    df_hypo_final['Difference in percentage'] = abs(df_hypo_final['Percent episodes of hypo in severe'] - df_hypo_final[
        'Percent episodes of hypo in control'])
    df_hypo_final = df_hypo_final.fillna(0).sort_values(by=['Difference in percentage', 'Difference in Number'],
                                                        ascending=False)

    return df_hypo_final


def gSAX_application(df, window_size, length_word, vocabulary, tfidf_size):
    for w in window_size:
        for l in length_word:
            if l <= w:
                for v in vocabulary:
                    data_final = pd.DataFrame()
                    for s in consts.SEEDS:
                        data = Main_ESAX(df, l, v, w, seed=s)
                        try:
                            data2 = data.drop(['label_encoded'], axis=1)
                            data_final = data_final.merge(data2, on=['PtID'])
                        except KeyError:
                            data_final = data

                        df_hypo_final = Contar_hypo(s)
                        df_hypo_final.astype(float).to_csv(
                            os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_SIGNAL,
                                         'TS_analysis_' + str(w) + '_' + str(
                                             l) + '_' + str(v) +'_'+ str(s) + '.csv'))
                    data_final.to_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_SIGNAL, 'gSAX.csv'))
                    Y = data_final['label_encoded']
                    X = data_final.drop(['label_encoded','PtID'], axis=1)
                    for t in tfidf_size:
                        # if t < len(df_hypo_final):
                        print('window_size : ', w, 'length_word : ', l, 'vocabulary : ', v, 'tfidf_size : ', t)
                        metrics = call_clfs(X, Y, 'Signal', X.columns, 0.2, tfidf=t)
                        # print('AUC_ROC MAX:', metrics[metrics['metric'] == 'auc_roc']['mean'].max())
                        # if metrics[metrics['metric'] == 'auc_roc']['mean'].max() > 0.6:
                        #     print(metrics[metrics['metric'] == 'auc_roc']['mean'].max())
                        metrics.to_csv(os.path.join(consts.PATH_PROJECT_REPORTS, 'time_series',
                                                        'Metrics_' + str(t) + '_' + str(w) + '_' + str(
                                                            l) + '_' + str(v) + '.csv'))


def Signal_FS(data_final):

    u = relief_bbdd(data_final, data_final['label_encoded'], 'Signal', path=consts.PATH_PROJECT_REPORTS_SIGNAL, FS=True)
    plt.close()
    u = relief_bbdd(data_final, data_final['label_encoded'], 'Signal', path=consts.PATH_PROJECT_REPORTS_SIGNAL, FS=5)

    o = call_clfs(data_final, data_final['label_encoded'], 'Signal', u, 0.2, tfidf=100)

    o.to_csv(os.path.join(consts.PATH_PROJECT_REPORTS_SIGNAL, 'Metrics_FS_'+str(8)+'_'+str(6)+'_'+str(3)+'_'+str(10)+'.csv'))


def plot_freq_TS():
    # df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_SIGNAL,'Signal_Train_Partition0.csv'))
    # df2 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_SIGNAL,'Signal_Train_Partition2.csv'))
    # df3 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_SIGNAL,'Signal_Train_Partition10.csv'))
    # df4 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_SIGNAL,'Signal_Train_Partition36.csv'))
    # df5 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_SIGNAL,'Signal_Train_Partition64.csv'))
    #
    # names = df1.columns.tolist() + df2.columns.tolist() + df3.columns.tolist() + df4.columns.tolist() + df5.columns.tolist()
    # names = list(set(names))

    P = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_SIGNAL,
                                  'TS_analysis_6_3_10_36.csv'))

    P['Difference in Number'] = P['Number in severe'] - P['Number in control']
    P['abs Difference in Number'] = abs(P['Difference in Number'] )
    P = P.sort_values(by=['abs Difference in Number'])
    df = P[-10:]
    x = [e.upper() for e in df['Word']]
    y = df['Difference in Number']
    lollipop_plot(x, y, 'blue')
