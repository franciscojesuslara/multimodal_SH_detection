import os
import utils.consts as consts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.classifiers import train_compute_metrics
from utils.consts import SEEDS
import shap
from PyALE import ale
import math
def call_clf_interpretability_SHAP(train_databases, test_databases,
                clfs, features=[],SHAP='Kernel',seeds=36,path=consts.PATH_PROJECT_FUSION_FIGURES):

    if SHAP== 'Kernel':
        list_SHAP = train_interpretability_SHAP(clfs, train_databases, test_databases, features,SHAP=SHAP)
        list2=[]
        for e in list_SHAP:
            list2.extend(e)

        i=pd.concat([test_databases[0].drop('label',axis=1),test_databases[1].drop('label',axis=1),test_databases[2].drop('label',axis=1),test_databases[3].drop('label',axis=1),test_databases[4].drop('label',axis=1)])
        dataframe = pd.DataFrame(list2, columns=train_databases[0].columns[:-1])
        plt.close()
        shap.summary_plot(np.asarray(list2),i.astype('float'),max_display=len(dataframe.columns))
        plt.savefig(os.path.join(path, 'SHAP_Early_bee.pdf'))
        # o=abs(dataframe).mean()
        # o1 = abs(dataframe).std()
        # df = pd.DataFrame(columns=['mean', 'std'])
        # df['mean'] = o.values
        # df['std'] = o1.values
        # df.index = o.index
        # df=df.sort_values(by=['mean'],ascending=False)
        # fig, ax = plt.subplots()
        # ax.barh(df.index, xerr=df['std'],width=df['mean'],capsize=3)
        # ax.invert_yaxis()
        # fig.tight_layout()
        # plt.show()
        # plt.savefig(os.path.join(path, 'SHAP_Early.pdf'))
        return  dataframe
    elif SHAP== 'decision_plot'or SHAP == 'waterfall' or SHAP== 'force_plot':

        x_train = train_databases.drop(['label'], axis=1)
        x_test = test_databases.drop(['label'], axis=1)
        y_train = train_databases['label']
        y_test = test_databases['label']

        if len(features) == 0:

            SHAP_values = train_compute_metrics(clfs, x_train, y_train,
                                                x_test, y_test, seeds, SHAP=SHAP)
        else:
            SHAP_values = train_compute_metrics(clfs, x_train[features], y_train, x_test[features], y_test,
                                                seed=seeds, SHAP=SHAP)

def train_interpretability_SHAP(clf_name, train_databases, test_databases, features,SHAP=False):
    SHAP_list=[]
    if SHAP == 'Kernel':
        for i, j in enumerate(train_databases):
            x_train = j.drop(['label'], axis=1)
            x_test = test_databases[i].drop(['label'], axis=1)
            y_train = j['label']
            y_test = test_databases[i]['label']

            if len(features) == 0:
                SHAP_values = train_compute_metrics(clf_name, x_train, y_train,
                                                    x_test, y_test, SEEDS[i], SHAP=SHAP)
            else:
                SHAP_values = train_compute_metrics(clf_name, x_train[features], y_train, x_test[features], y_test,
                                                    seed=SEEDS[i],SHAP=SHAP)
            SHAP_list.append(SHAP_values)
        return SHAP_list


def interpretability_ALE(train,test,clf_name,Partition_seed ,features=[],numerical_features=[],categorical_features=[], continuous_features=[]):
    x_train = train.drop(['label'], axis=1)
    x_test = test.drop(['label'], axis=1)
    y_train = train['label']
    y_test = test['label']
    if len(features) == 0:
        model = train_compute_metrics(clf_name, x_train, y_train,
                                            x_test, y_test, seed=Partition_seed,ALE= True)
    else:
        model = train_compute_metrics(clf_name, x_train[features], y_train, x_test[features], y_test,
                                            seed=Partition_seed, ALE=True)
    n, c = get_categorical_numerical_names(x_train)
    for e in x_train.columns[0:15]:
        if e in n:
            ale_eff = ale(
            X=x_test, model=model, feature=[e], grid_size=50, include_CI=True, C=0.95, feature_type='discrete')
        elif e in c:
            ale_eff = ale(
                X=x_test, model=model, feature=[e], grid_size=50, include_CI=True, C=0.95, feature_type='continuous')



def get_categorical_numerical_names(df_data: pd.DataFrame) -> (list, list):

    df_info = identify_type_features(df_data)
    list_numerical_vars = list(df_info[df_info['type'] == 'c'].index)
    list_categorical_vars = list(df_info[df_info['type'] == 'd'].index)

    return list_categorical_vars, list_numerical_vars


def identify_type_features(df, discrete_threshold=10, debug=False):
    """
    Categorize every feature/column of df as discrete or continuous according to whether or not the unique responses
    are numeric and, if they are numeric, whether or not there are fewer unique renponses than discrete_threshold.
    Return a dataframe with a row for each feature and columns for the type, the count of unique responses, and the
    count of string, number or null/nan responses.
    """
    counts = []
    string_counts = []
    float_counts = []
    null_counts = []
    types = []
    for col in df.columns:
        responses = df[col].unique()
        counts.append(len(responses))
        string_count, float_count, null_count = 0, 0, 0
        for value in responses:
            try:
                val = float(value)
                if not math.isnan(val):
                    float_count += 1
                else:
                    null_count += 1
            except:
                try:
                    val = str(value)
                    string_count += 1
                except:
                    print('Error: Unexpected value', value, 'for feature', col)

        string_counts.append(string_count)
        float_counts.append(float_count)
        null_counts.append(null_count)
        types.append('d' if len(responses) < discrete_threshold or string_count > 0 else 'c')

    feature_info = pd.DataFrame({'count': counts,
                                 'string_count': string_counts,
                                 'float_count': float_counts,
                                 'null_count': null_counts,
                                 'type': types}, index=df.columns)
    if debug:
        print(f'Counted {sum(feature_info["type"] == "d")} discrete features and {sum(feature_info["type"] == "c")} continuous features')

    return feature_info







