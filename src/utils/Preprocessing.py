
from sklearn.impute import KNNImputer
from statistics import mode
from statistics import median
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.Embedding_text import Embedded_text


def preprocessing_function(x_features, y_label, i, bbdd_name, test_s, tfidf=10):

    def var_categ_train(variable, X_train):

        data = X_train
        enc = OneHotEncoder(handle_unknown='ignore')
        enc_data = pd.DataFrame(enc.fit_transform(X_train[[variable]]).toarray())
        X_train[variable].value_counts()  # Identifity categories
        enc_data.describe()
        # Solo incluye categorías que tengan datos
        # aunque no tengan datos
        X_train = pd.concat([X_train, pd.get_dummies(X_train[variable], prefix=variable)], axis=1)
        return X_train

    def var_categ_test(variable, X_test):
        data = X_test
        enc = OneHotEncoder(handle_unknown='ignore')
        enc_data = pd.DataFrame(enc.fit_transform(X_test[[variable]]).toarray())
        X_test[variable].value_counts()  # Identifity categories
        enc_data.describe()
        # Solo incluye categorías que tengan datos
        # aunque no tengan datos
        X_test = pd.concat([X_test, pd.get_dummies(X_test[variable], prefix=variable)], axis=1)
        return X_test

    if bbdd_name == 'Attitude':
        # x_features.drop(['PtID','label'], axis=1, inplace=True)
        X_train, X_test, Y_train, Y_test = train_test_split(x_features, y_label, stratify=y_label, test_size=test_s,
                                                            random_state=i)

        X_train["DealHypoEp"].replace(np.nan, mode(X_train["DealHypoEp"]), inplace=True)
        X_train["UndertreatHypo"].replace(np.nan, mode(X_train["UndertreatHypo"]), inplace=True)
        X_train["HighBGDamage"].replace(np.nan, mode(X_train["HighBGDamage"]), inplace=True)
        X_train["FreqHypoDamage"].replace(np.nan, mode(X_train["FreqHypoDamage"]), inplace=True)
        X_train["DangersHighBG"].replace(np.nan, mode(X_train["DangersHighBG"]), inplace=True)

        X_test["DealHypoEp"].replace(np.nan, mode(X_train["DealHypoEp"]), inplace=True)
        X_test["UndertreatHypo"].replace(np.nan, mode(X_train["UndertreatHypo"]), inplace=True)
        X_test["HighBGDamage"].replace(np.nan, mode(X_train["HighBGDamage"]), inplace=True)
        X_test["FreqHypoDamage"].replace(np.nan, mode(X_train["FreqHypoDamage"]), inplace=True)
        X_test["DangersHighBG"].replace(np.nan, mode(X_train["DangersHighBG"]), inplace=True)

        imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

        imputer.fit(X_train[['HighBGLevTakeAction', 'LowBGLevTakeAction', 'PtCurrA1cGoal']])

        # transform the dataset
        X_train[['HighBGLevTakeAction', 'LowBGLevTakeAction', 'PtCurrA1cGoal']] = imputer.transform(
            X_train[['HighBGLevTakeAction', 'LowBGLevTakeAction', 'PtCurrA1cGoal']])
        X_test[['HighBGLevTakeAction', 'LowBGLevTakeAction', 'PtCurrA1cGoal']] = imputer.transform(
            X_test[['HighBGLevTakeAction', 'LowBGLevTakeAction', 'PtCurrA1cGoal']])

        X_train = var_categ_train("DealHypoEp", X_train)
        X_train = var_categ_train("UndertreatHypo", X_train)
        X_train = var_categ_train("HighBGDamage", X_train)
        X_train = var_categ_train("FreqHypoDamage", X_train)
        X_train = var_categ_train("DangersHighBG", X_train)

        X_train.drop(['DealHypoEp'], axis=1, inplace=True)
        X_train.drop(['UndertreatHypo'], axis=1, inplace=True)
        X_train.drop(['HighBGDamage'], axis=1, inplace=True)
        X_train.drop(['FreqHypoDamage'], axis=1, inplace=True)
        X_train.drop(['DangersHighBG'], axis=1, inplace=True)

        X_test = var_categ_test("DealHypoEp", X_test)
        X_test = var_categ_test("UndertreatHypo", X_test)
        X_test = var_categ_test("HighBGDamage", X_test)
        X_test = var_categ_test("FreqHypoDamage", X_test)
        X_test = var_categ_test("DangersHighBG", X_test)

        X_test.drop(['DealHypoEp'], axis=1, inplace=True)
        X_test.drop(['UndertreatHypo'], axis=1, inplace=True)
        X_test.drop(['HighBGDamage'], axis=1, inplace=True)
        X_test.drop(['FreqHypoDamage'], axis=1, inplace=True)
        X_test.drop(['DangersHighBG'], axis=1, inplace=True)

        scaler = MinMaxScaler()
        # scaler = StandardScaler()

        Cat = ['DealHypoEp_Agree', 'DealHypoEp_Disagree', 'DealHypoEp_Neutral', 'UndertreatHypo_Agree',
               'UndertreatHypo_Disagree', 'UndertreatHypo_Neutral', 'HighBGDamage_Agree', 'HighBGDamage_Disagree',
               'HighBGDamage_Neutral', 'FreqHypoDamage_Agree', 'FreqHypoDamage_Disagree', 'FreqHypoDamage_Neutral',
               'DangersHighBG_Agree', 'DangersHighBG_Disagree', 'DangersHighBG_Neutral']

        scaled_train = X_train.drop(Cat, axis=1)
        scaled_test = X_test.drop(Cat, axis=1)

        scaler.fit(scaled_train)
        scaled_train[scaled_train.columns] = scaler.transform(scaled_train)
        scaled_test[scaled_test.columns] = scaler.transform(scaled_test)

        scaled_train[Cat] = X_train[Cat]
        scaled_test[Cat] = X_test[Cat]

        X_train = scaled_train
        X_test = scaled_test


    elif bbdd_name == 'Lifestyle':
        data = x_features
        data["Gender"] = data["Gender"].replace(["F", "M"], [0, 1])
        data["Race"] = data["Race"].replace(
            ["Asian", "American/ Alaskan", "American Indian/Alaskan Native", "Unknown/not reported",
             "Black/African American"], ["Other", "Other", "Other", "Other", "Other"])
        data["Race"] = data["Race"].replace(["White", "Other"], [0, 1])
        data["Ethnicity"] = data["Ethnicity"].replace(["UNK", "H/L"], ["Other", "Other"])
        data["Ethnicity"] = data["Ethnicity"].replace(["NH/NL", "Other"], [0, 1])
        enc = OneHotEncoder(handle_unknown='ignore')
        enc_data = pd.DataFrame(enc.fit_transform(data[['EduLevel']]).toarray())
        data['EduLevel'].value_counts()  # Identifity categories
        Cateogy_onehot = pd.get_dummies(data.EduLevel, prefix='EduLevel')
        data = pd.concat([data, pd.get_dummies(data['EduLevel'], prefix='EduLevel')], axis=1)
        enc_data = pd.DataFrame(enc.fit_transform(data[['AnnualInc']]).toarray())
        data['AnnualInc'].value_counts()  # Identifity categories
        Cateogy_onehot = pd.get_dummies(data.AnnualInc, prefix='AnnualInc')
        data = pd.concat([data, pd.get_dummies(data['AnnualInc'], prefix='AnnualInc')], axis=1)
        data.drop(["EduLevel", "AnnualInc"], axis=1, inplace=True)
        data["Race"] = data["Race"].replace(["A", "American/ Alaskan", "AI/ AN", "UNK", "B/AM"],
                                            ["Other", "Other", "Other", "Other", "Other"])
        data["Race"] = data["Race"].replace(["W", "Other"], [0, 1])
        X_train, X_test, Y_train, Y_test = train_test_split(data, y_label, stratify=y_label, test_size=test_s,
                                                            random_state=i)


    elif bbdd_name == 'Depression':
        data = x_features
        data = data.replace('Yes', 0)
        data = data.replace('No', 1)
        X_train, X_test, Y_train, Y_test = train_test_split(data, y_label, stratify=y_label, test_size=test_s,
                                                            random_state=i)



    elif bbdd_name == 'Unaware':

        data_input_new = x_features

        val = []
        for e in data_input_new["FeelSympLowBG"]:
            if len(e) <= 10:
                if '<' in e:
                    val.append('0-40')

                else:
                    val.append(e)

            else:
                val.append('0')

        val1 = [mode(val) if x == '0' else x for x in val]

        data_input_new["FeelSympLowBG"] = val1
        data_input_new["LowBGLostSymp"] = x_features["LowBGLostSymp"].replace(["Yes", "No"], [1, 0])

        enc = OneHotEncoder(handle_unknown='ignore')

        enc_data = pd.DataFrame(enc.fit_transform(data_input_new[['ExtentSympLowBG']]).toarray())
        data_input_new['ExtentSympLowBG'].value_counts()  # Identifity categories
        Cateogy_onehot = pd.get_dummies(data_input_new.ExtentSympLowBG, prefix='ExtentSympLowBG')
        data_input_new = pd.concat(
            [data_input_new, pd.get_dummies(data_input_new['ExtentSympLowBG'], prefix='ExtentSympLowBG')], axis=1)

        enc_data = pd.DataFrame(enc.fit_transform(data_input_new[['LowBGSympCat']]).toarray())
        #         data_input_new['LowBGSympCat'].value_counts() # Identifity categories
        Cateogy_onehot = pd.get_dummies(data_input_new.ExtentSympLowBG, prefix='LowBGSympCat')
        data_input_new = pd.concat(
            [data_input_new, pd.get_dummies(data_input_new['LowBGSympCat'], prefix='LowBGSympCat')], axis=1)

        enc_data = pd.DataFrame(enc.fit_transform(data_input_new[['Bel70PastMonWSymp']]).toarray())
        data_input_new['Bel70PastMonWSymp'].value_counts()  # Identifity categories
        Cateogy_onehot = pd.get_dummies(data_input_new.ExtentSympLowBG, prefix='Bel70PastMonWSymp')
        data_input_new = pd.concat(
            [data_input_new, pd.get_dummies(data_input_new['Bel70PastMonWSymp'], prefix='Bel70PastMonWSymp')], axis=1)

        # enc_data = pd.DataFrame(enc.fit_transform(data_input_new[['ModHypoEpPast6Mon']]).toarray())
        # data_input_new['ModHypoEpPast6Mon'].value_counts()  # Identifity categories
        # Cateogy_onehot = pd.get_dummies(data_input_new.ExtentSympLowBG, prefix='ModHypoEpPast6Mon')
        # data_input_new = pd.concat(
        #     [data_input_new, pd.get_dummies(data_input_new['ModHypoEpPast6Mon'], prefix='ModHypoEpPast6Mon')], axis=1)

        enc_data = pd.DataFrame(enc.fit_transform(data_input_new[['Bel70PastMonNoSymp']]).toarray())
        data_input_new['Bel70PastMonNoSymp'].value_counts()  # Identifity categories
        Cateogy_onehot = pd.get_dummies(data_input_new.ExtentSympLowBG, prefix='Bel70PastMonNoSymp')
        data_input_new = pd.concat(
            [data_input_new, pd.get_dummies(data_input_new['Bel70PastMonNoSymp'], prefix='Bel70PastMonNoSymp')], axis=1)

        enc_data = pd.DataFrame(enc.fit_transform(data_input_new[['FeelSympLowBG']]).toarray())
        data_input_new['FeelSympLowBG'].value_counts()  # Identifity categories
        Cateogy_onehot = pd.get_dummies(data_input_new.ExtentSympLowBG, prefix='FeelSympLowBG')
        data_input_new = pd.concat(
            [data_input_new, pd.get_dummies(data_input_new['FeelSympLowBG'], prefix='FeelSympLowBG')], axis=1)

        #         enc_data = pd.DataFrame(enc.fit_transform(data_input_new[['SevHypoEpPastYear']]).toarray())
        #         data_input_new['SevHypoEpPastYear'].value_counts() # Identifity categories
        #         Cateogy_onehot = pd.get_dummies(data_input_new.SevHypoEpPastYear, prefix='SevHypoEpPastYear')
        #         data_input_new = pd.concat([data_input_new,pd.get_dummies(data_input_new['SevHypoEpPastYear'], prefix='SevHypoEpPastYear')],axis=1)

        data_input_new = data_input_new.drop(
            ["Bel70PastMonNoSymp", "SevHypoEpPastYear", "FeelSympLowBG", "FeelSympLowBG", "ModHypoEpPast6Mon",
             "Bel70PastMonWSymp", "LowBGSympCat", "ExtentSympLowBG"], axis=1)
        X_train, X_test, Y_train, Y_test = train_test_split(data_input_new, y_label, stratify=y_label, test_size=test_s,
                                                            random_state=i)
    elif bbdd_name == 'Signal':


        X_train, X_test, Y_train, Y_test = train_test_split(x_features, y_label, test_size=test_s, random_state=i,
                                                            stratify=y_label)

        tfidf_vectorizer = TfidfVectorizer(max_features=tfidf)
        m_tfidf_sparse = tfidf_vectorizer.fit_transform(X_train['text'+str(i)])
        X_train = m_tfidf_sparse.toarray()
        m_tfidf_sparse = tfidf_vectorizer.transform(X_test['text'+str(i)])
        X_test = m_tfidf_sparse.toarray()
        m = pd.DataFrame(tfidf_vectorizer.vocabulary_, index=[0]).sort_values(by=[0], axis=1)
        X_train = pd.DataFrame(X_train, columns=m.columns)
        X_test = pd.DataFrame(X_test, columns=m.columns)

        # scaler = MinMaxScaler()
        # # scaler = StandardScaler()
        # scaler.fit(X_train)
        # X_train = scaler.transform(X_train)
        # X_test = scaler.transform(X_test)

        X_train.to_csv('Signal_Train_Partition' + str(i) + '.csv', index=False)
        # X_test.to_csv('Signal_Test_Partition' + str(i) + '.csv', index=False)

    elif bbdd_name == 'Conditions':

        X_train, X_test, Y_train, Y_test = Embedded_text(x_features, var_name='medcon1', encoding='Clinical', ngrams=1,
                                                         embedding_size=50,
                                                         Reduction='PCA', seed=i, label=y_label)


        # scaler = MinMaxScaler()
        # scaler = StandardScaler()
        # scaler.fit(X_train)
        # X_train = scaler.transform(X_train)
        # X_test = scaler.transform(X_test)

    elif bbdd_name == 'Medications':
        X_train, X_test, Y_train, Y_test = Embedded_text(x_features, var_name='medications3', encoding='Clinical', ngrams=1,
                                                         embedding_size=10,
                                                         Reduction='PCA', seed=i, label=y_label)
        # scaler = MinMaxScaler()
        # scaler = StandardScaler()
        # scaler.fit(X_train)
        # X_train = scaler.transform(X_train)
        # X_test = scaler.transform(X_test)

    elif bbdd_name == 'Medications_hot':
        list1 = []
        data = x_features
        for e in x_features['medications3']:
            list1 = list1 + e.split()
        features = list(set(list1))

        mat = [0] * len(features)
        mat2 = []
        for e in range(len(data)):
            mat2.append(mat)

        Dataframe = pd.DataFrame(data=mat2, columns=features)

        for e in range(len(data)):
            for p in features:
                if p in data['medications3'].iloc[e].split():
                    Dataframe[p].iloc[e] = 1

        X_train, X_test, Y_train, Y_test = train_test_split(Dataframe, y_label, stratify=y_label, test_size=test_s,
                                                            random_state=i)

    elif bbdd_name == 'Conditions_hot':
        list1 = []
        data = x_features
        for e in x_features['medcon1']:
            list1 = list1 + e.split()
        features = list(set(list1))

        mat = [0] * len(features)
        mat2 = []
        for e in range(len(data)):
            mat2.append(mat)

        Dataframe = pd.DataFrame(data=mat2, columns=features)

        for e in range(len(data)):
            for p in features:
                if p in data['medcon1'].iloc[e].split():
                    Dataframe[p].iloc[e] = 1

        X_train, X_test, Y_train, Y_test = train_test_split(Dataframe, y_label, stratify=y_label, test_size=test_s,
                                                            random_state=i)

    elif bbdd_name == 'Fear':
        X_train, X_test, Y_train, Y_test = train_test_split(x_features, y_label, stratify=y_label, test_size=test_s,
                                                            random_state=i)
        scaler = MinMaxScaler()
        # scaler = StandardScaler()
        scaled_train = X_train
        scaled_test = X_test

        scaler.fit(scaled_train)
        scaled_train = scaler.transform(scaled_train)
        scaled_test = scaler.transform(scaled_test)
        X_train = pd.DataFrame(scaled_train,columns=X_train.columns)
        X_test = pd.DataFrame(scaled_test,columns=X_train.columns)
    elif bbdd_name == 'MOCA':

        data = pd.concat([x_features, y_label], axis=1)
        data['MoCAPtEff'] = data['MoCAPtEff'].replace(['Good'], 1).replace(['Questionable'], 0).replace(['Poor'], 0)
        data.drop(['MoCANotDoneReas'], axis=1, inplace=True)
        data.drop(['MoCANotDone'], axis=1, inplace=True)
        # data.drop(['MoCAOrient'], axis=1, inplace=True)
        # data=data.drop(data[data['MoCAOrient'].isna()].index)

        X = data.drop(['label_encoded'], axis=1)
        Y = data['label_encoded']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_s, random_state=i,
                                                            stratify=data.label_encoded)
        X_train["MoCAPtEff"].replace(np.nan, mode(X_train["MoCAPtEff"]), inplace=True)
        X_test["MoCAPtEff"].replace(np.nan, mode(X_train["MoCAPtEff"]), inplace=True)

        scaler = MinMaxScaler()
        # scaler = StandardScaler()
        scaled_train = X_train
        scaled_test = X_test

        scaler.fit(scaled_train)
        scaled_train = scaler.transform(scaled_train)
        scaled_test = scaler.transform(scaled_test)
        X_train = scaled_train
        X_test = scaled_test

        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)

        X_train = X_train.drop(['Unnamed: 0'], axis=1)
        X_test = X_test.drop(['Unnamed: 0'], axis=1)

    elif bbdd_name == 'BSample':
        X_train, X_test, Y_train, Y_test = train_test_split(x_features, y_label, test_size=test_s, random_state=i,
                                                            stratify=y_label)
        imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

        imputer.fit(X_train)

        # transform the dataset
        train1 = imputer.transform(X_train)
        test1 = imputer.transform(X_test)

        X_train = pd.DataFrame(train1, columns=x_features.columns)
        X_test = pd.DataFrame(test1, columns=x_features.columns)

        scaler = MinMaxScaler()
        #scaler = StandardScaler()
        scaled_train = X_train
        scaled_test = X_test
        scaler.fit(scaled_train)
        scaled_train = scaler.transform(scaled_train)
        scaled_test = scaler.transform(scaled_test)
        X_train = scaled_train
        X_test = scaled_test
        X_train = pd.DataFrame(X_train, columns=x_features.columns)
        X_test = pd.DataFrame(X_test, columns=x_features.columns)


    elif bbdd_name == 'BMedChart':
        data = x_features
        X_train, X_test, Y_train, Y_test = train_test_split(data, y_label, test_size=test_s, random_state=i,
                                                            stratify=y_label)
        X_train["LastFoodIntakeHrs"].replace(np.nan, mode(X_train["LastFoodIntakeHrs"]), inplace=True)
        X_test["LastFoodIntakeHrs"].replace(np.nan, mode(X_train["LastFoodIntakeHrs"]), inplace=True)

        X_train["LastFoodIntakeCarbs"].replace(np.nan, median(X_train["LastFoodIntakeCarbs"].dropna()), inplace=True)
        X_test["LastFoodIntakeCarbs"].replace(np.nan, median(X_train["LastFoodIntakeCarbs"].dropna()), inplace=True)

        imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

        imputer.fit(X_train[['Weight_mod', 'Height_mod']])

        # transform the dataset
        train1 = imputer.transform(X_train[['Weight_mod', 'Height_mod']])
        test1 = imputer.transform(X_test[['Weight_mod', 'Height_mod']])
        train1 = pd.DataFrame(train1, columns=['Weight_mod', 'Height_mod'])
        test1 = pd.DataFrame(test1, columns=['Weight_mod', 'Height_mod'])
        X_train = X_train.reset_index()
        train1 = train1.reset_index()
        X_test = X_test.reset_index()
        test1 = test1.reset_index()
        X_train = pd.concat(
            [X_train[["LastFoodIntakeCarbs", 'LastFoodIntakeHrs']], train1[['Weight_mod', 'Height_mod']]], axis=1)
        X_test = pd.concat([X_test[["LastFoodIntakeCarbs", 'LastFoodIntakeHrs']], test1[['Weight_mod', 'Height_mod']]],
                           axis=1)

        X_train = var_categ_train("LastFoodIntakeHrs", X_train)
        X_train.drop(['LastFoodIntakeHrs'], axis=1, inplace=True)

        X_test = var_categ_test("LastFoodIntakeHrs", X_test)
        X_test.drop(['LastFoodIntakeHrs'], axis=1, inplace=True)

        scaler = MinMaxScaler()
        #scaler = StandardScaler()
        Cat = ['LastFoodIntakeHrs_0-<4 hours prior', 'LastFoodIntakeHrs_4-<8 hours prior',
               'LastFoodIntakeHrs_8 or more hours prior']

        scaled_train = X_train.drop(Cat, axis=1)
        try:
            scaled_test = X_test.drop(Cat, axis=1)
        except:
            X_test['LastFoodIntakeHrs_8 or more hours prior'] = [0] * test_s
            scaled_test = X_test.drop(Cat, axis=1)

        scaler.fit(scaled_train)
        scaled_train[scaled_train.columns] = scaler.transform(scaled_train)
        scaled_test[scaled_test.columns] = scaler.transform(scaled_test)
        Cat2 = ['LastFoodIntakeHrs_0-4 hours prior', 'LastFoodIntakeHrs_4-8 hours prior',
                'LastFoodIntakeHrs_8 or more hours prior']
        scaled_train[Cat2] = X_train[Cat]
        scaled_test[Cat2] = X_test[Cat]
        X_train = scaled_train
        X_test = scaled_test




    elif bbdd_name == 'BTOTSCORE':
        data = x_features
        p = np.where(np.asarray(data.isna().sum()) > 0)
        features2 = data.columns[p]

        data = var_categ_train('ReadCardLowLine', data)
        data = data.drop(['ReadCardLowLine'], axis=1)
        Cat = ['GrPegDomHand', 'ReadCardCorrLens', 'ReadCardLowLine_20/125',
               'ReadCardLowLine_20/16', 'ReadCardLowLine_20/20',
               'ReadCardLowLine_20/200', 'ReadCardLowLine_20/25',
               'ReadCardLowLine_20/32', 'ReadCardLowLine_20/40',
               'ReadCardLowLine_20/50', 'ReadCardLowLine_20/63',
               'ReadCardLowLine_20/80']

        X_train, X_test, Y_train, Y_test = train_test_split(data, y_label, test_size=test_s, random_state=i,
                                                            stratify=y_label)

        imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

        imputer.fit(X_train[features2])

        # transform the dataset
        train1 = imputer.transform(X_train[features2])
        test1 = imputer.transform(X_test[features2])

        train1 = pd.DataFrame(train1, columns=features2)
        test1 = pd.DataFrame(test1, columns=features2)

        p = np.where(np.asarray(data.isna().sum()) == 0)
        features3 = data.columns[p]
        X_train = X_train.reset_index(drop=True)
        train1 = train1.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        test1 = test1.reset_index(drop=True)
        X_train = pd.concat([X_train[features3], train1[features2]], axis=1)
        X_test = pd.concat([X_test[features3], test1[features2]], axis=1)

        scaler = MinMaxScaler()
        #scaler = StandardScaler()
        scaled_train = X_train.drop(Cat, axis=1)
        scaled_test = X_test.drop(Cat, axis=1)

        scaler.fit(scaled_train)
        scaled_train[scaled_train.columns] = scaler.transform(scaled_train)
        scaled_test[scaled_test.columns] = scaler.transform(scaled_test)

        scaled_train[Cat] = X_train[Cat]
        scaled_test[Cat] = X_test[Cat]
        X_train = scaled_train
        X_test = scaled_test
        X_train = X_train.drop(['Unnamed: 0'], axis=1)
        X_test = X_test.drop(['Unnamed: 0'], axis=1)
    X_train=X_train.replace({True:1,False:0})
    X_test = X_test.replace({True: 1, False: 0})

    return X_train, X_test, Y_train, Y_test  # ,ID