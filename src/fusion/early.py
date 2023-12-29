import os
import numpy as np
import pandas as pd
from statistics import mode, median
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')
from utils.FS_bbdd import relief_plot,relief_bbdd
from utils.Embedding_text import Embedded_text
from utils.classifiers import call_models_fusion
from utils.check_patients import get_patients_id
import utils.consts as consts
from  utils.FS_relief import relief_fusion
import math
def main_early_fusion(x_features, y_label, names, test_s,tfidf=10):
    list_x_train = []
    list_x_test = []
    list_y_train = []
    list_y_test = []

    for i in consts.SEEDS:
        x_train, x_test,Y_train, Y_test = Preprocessing_function(x_features, y_label,i,names,test_s,tfidf)
        list_x_train.append(x_train)
        list_x_test.append(x_test)
        list_y_test.append(Y_test)
        list_y_train.append(Y_train)

    return list_x_train,list_x_test,list_y_train,list_y_test
    
    
def Preprocessing_function(x_features, y_label,i,names,test_s,tfidf):
    def var_categ_train(variable,X_train):
        
          data= X_train
          enc = OneHotEncoder(handle_unknown='ignore')
          enc_data = pd.DataFrame(enc.fit_transform(X_train[[variable]]).toarray())
          X_train[variable].value_counts() # Identifity categories
          enc_data.describe()
          #Solo incluye categorías que tengan datos
          #aunque no tengan datos
          X_train = pd.concat([X_train,pd.get_dummies(X_train[variable], prefix=variable)],axis=1)
          return X_train
            
    def var_categ_test(variable,X_test):
          data= X_test
          enc = OneHotEncoder(handle_unknown='ignore')
          enc_data = pd.DataFrame(enc.fit_transform(X_test[[variable]]).toarray())
          X_test[variable].value_counts() # Identifity categories
          enc_data.describe()
          #Solo incluye categorías que tengan datos
          #aunque no tengan datos
          X_test = pd.concat([X_test,pd.get_dummies(X_test[variable], prefix=variable)],axis=1)
          return X_test
        
    scaling_df_x_train=pd.DataFrame()
    no_scaling_df_x_train= pd.DataFrame()
    scaling_df_x_test=pd.DataFrame()
    no_scaling_df_x_test= pd.DataFrame()
    for j,e in enumerate(x_features):
        if 'PtID' in e.columns:
            e=e.drop(['PtID'],axis=1)
        if names[j]=='Attitude':
            print('Attitude')
            #x_features.drop(['PtID','label'], axis=1, inplace=True)
            X_train, X_test, Y_train, Y_test = train_test_split(e, y_label, stratify=y_label, test_size=test_s, random_state=i)

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

            imputer.fit(X_train[['HighBGLevTakeAction','LowBGLevTakeAction','PtCurrA1cGoal']])

            # transform the dataset
            X_train[['HighBGLevTakeAction','LowBGLevTakeAction','PtCurrA1cGoal']]= imputer.transform(X_train[['HighBGLevTakeAction','LowBGLevTakeAction','PtCurrA1cGoal']])
            X_test[['HighBGLevTakeAction','LowBGLevTakeAction','PtCurrA1cGoal']]= imputer.transform(X_test[['HighBGLevTakeAction','LowBGLevTakeAction','PtCurrA1cGoal']])


            X_train=var_categ_train("DealHypoEp",X_train)
            X_train=var_categ_train("UndertreatHypo",X_train)
            X_train=var_categ_train("HighBGDamage",X_train)
            X_train=var_categ_train("FreqHypoDamage",X_train)
            X_train=var_categ_train("DangersHighBG",X_train)

            X_train.drop(['DealHypoEp'], axis=1, inplace=True)
            X_train.drop(['UndertreatHypo'], axis=1, inplace=True) 
            X_train.drop(['HighBGDamage'], axis=1, inplace=True) 
            X_train.drop(['FreqHypoDamage'], axis=1, inplace=True) 
            X_train.drop(['DangersHighBG'], axis=1, inplace=True) 


            X_test=var_categ_test("DealHypoEp",X_test)
            X_test=var_categ_test("UndertreatHypo",X_test)
            X_test=var_categ_test("HighBGDamage",X_test)
            X_test=var_categ_test("FreqHypoDamage",X_test)
            X_test=var_categ_test("DangersHighBG",X_test)

            X_test.drop(['DealHypoEp'], axis=1, inplace=True)
            X_test.drop(['UndertreatHypo'], axis=1, inplace=True) 
            X_test.drop(['HighBGDamage'], axis=1, inplace=True) 
            X_test.drop(['FreqHypoDamage'], axis=1, inplace=True) 
            X_test.drop(['DangersHighBG'], axis=1, inplace=True) 


            Cat=['DealHypoEp_Agree','DealHypoEp_Disagree','DealHypoEp_Neutral', 'UndertreatHypo_Agree','UndertreatHypo_Disagree','UndertreatHypo_Neutral','HighBGDamage_Agree','HighBGDamage_Disagree','HighBGDamage_Neutral','FreqHypoDamage_Agree','FreqHypoDamage_Disagree','FreqHypoDamage_Neutral','DangersHighBG_Agree','DangersHighBG_Disagree','DangersHighBG_Neutral']
            
            
            scaling_df_x_train=pd.concat([scaling_df_x_train,X_train.drop(Cat,axis=1).reset_index(drop=True)],axis='columns')
            no_scaling_df_x_train=pd.concat([no_scaling_df_x_train,X_train[Cat].reset_index(drop=True)],axis='columns')
            scaling_df_x_test=pd.concat([scaling_df_x_test,X_test.drop(Cat,axis=1).reset_index(drop=True)],axis='columns')
            no_scaling_df_x_test=pd.concat([no_scaling_df_x_test,X_test[Cat].reset_index(drop=True)],axis='columns')
            
            
            

        elif names[j]=='Lifestyle':
            print('Lifestyle')
            data= e
            data["Gender"] = data["Gender"].replace(["F", "M"],[0, 1])
            data["Race"] = data["Race"].replace(["Asian", "American/ Alaskan", "American Indian/Alaskan Native", "Unknown/not reported", "Black/African American"],["Other","Other","Other", "Other", "Other"])
            data["Race"] = data["Race"].replace(["White", "Other"],[0, 1])
            data["Ethnicity"] = data["Ethnicity"].replace(["UNK", "H/L"],["Other", "Other"])
            data["Ethnicity"] = data["Ethnicity"].replace(["NH/NL", "Other"],[0, 1])
            enc = OneHotEncoder(handle_unknown='ignore')
            enc_data = pd.DataFrame(enc.fit_transform(data[['EduLevel']]).toarray())
            data['EduLevel'].value_counts() # Identifity categories
            Cateogy_onehot = pd.get_dummies(data.EduLevel, prefix='EduLevel')
            data = pd.concat([data,pd.get_dummies(data['EduLevel'], prefix='EduLevel')],axis=1)
            enc_data = pd.DataFrame(enc.fit_transform(data[['AnnualInc']]).toarray())
            data['AnnualInc'].value_counts() # Identifity categories
            Cateogy_onehot = pd.get_dummies(data.AnnualInc, prefix='AnnualInc')
            data = pd.concat([data,pd.get_dummies(data['AnnualInc'], prefix='AnnualInc')],axis=1)
            data.drop(["EduLevel", "AnnualInc"], axis = 1, inplace = True)
            data["Race"] = data["Race"].replace(["A", "American/ Alaskan","AI/ AN", "UNK", "B/AM"],["Other","Other","Other", "Other", "Other"])
            data["Race"] = data["Race"].replace(["W", "Other"],[0, 1])
            X_train,X_test,Y_train,Y_test= train_test_split(data, y_label, stratify=y_label, test_size=test_s, random_state=i)
            
            
            no_scaling_df_x_train=pd.concat([no_scaling_df_x_train,X_train.reset_index(drop=True)],axis='columns')
            
            no_scaling_df_x_test=pd.concat([no_scaling_df_x_test,X_test.reset_index(drop=True)],axis='columns')


        elif names[j]=='Depression':
            print('Depression')
            data= e
            data=data.replace('Yes',0)
            data=data.replace('No',1)
            X_train,X_test,Y_train,Y_test= train_test_split(data,y_label ,stratify=y_label, test_size=test_s, random_state=i)
            no_scaling_df_x_train=pd.concat([no_scaling_df_x_train,X_train.reset_index(drop=True)],axis='columns')
            
            no_scaling_df_x_test=pd.concat([no_scaling_df_x_test,X_test.reset_index(drop=True)],axis='columns')

        elif names[j]=='Unaware':  
            print('Unaware')
            data_input_new = e
            val=[]
            for e in data_input_new["FeelSympLowBG"]:
                if len(e)<= 10:
                    if '<' in e:
                        val.append('0-40')
                        
                    else:    
                        val.append(e)
                    
                else:
                    val.append('0')
                    
            val1=[mode(val) if x=='0' else x for x in val]
       
            data_input_new["FeelSympLowBG"]=val1        
                    
                
            data_input_new["LowBGLostSymp"] = data_input_new ["LowBGLostSymp"].replace(["Yes", "No"],[1, 0])

            enc = OneHotEncoder(handle_unknown='ignore')

            enc_data = pd.DataFrame(enc.fit_transform(data_input_new[['ExtentSympLowBG']]).toarray())
            data_input_new['ExtentSympLowBG'].value_counts() # Identifity categories
            Cateogy_onehot = pd.get_dummies(data_input_new.ExtentSympLowBG, prefix='ExtentSympLowBG')
            data_input_new = pd.concat([data_input_new,pd.get_dummies(data_input_new['ExtentSympLowBG'], prefix='ExtentSympLowBG')],axis=1)
            
            
            enc_data = pd.DataFrame(enc.fit_transform(data_input_new[['FeelSympLowBG']]).toarray())
            data_input_new['FeelSympLowBG'].value_counts() # Identifity categories
            Cateogy_onehot = pd.get_dummies(data_input_new.ExtentSympLowBG, prefix='FeelSympLowBG')
            data_input_new = pd.concat([data_input_new,pd.get_dummies(data_input_new['FeelSympLowBG'], prefix='FeelSympLowBG')],axis=1)

            enc_data = pd.DataFrame(enc.fit_transform(data_input_new[['LowBGSympCat']]).toarray())
            data_input_new['LowBGSympCat'].value_counts() # Identifity categories
            Cateogy_onehot = pd.get_dummies(data_input_new.ExtentSympLowBG, prefix='LowBGSympCat')
            data_input_new = pd.concat([data_input_new,pd.get_dummies(data_input_new['LowBGSympCat'], prefix='LowBGSympCat')],axis=1)

            enc_data = pd.DataFrame(enc.fit_transform(data_input_new[['Bel70PastMonWSymp']]).toarray())
            data_input_new['Bel70PastMonWSymp'].value_counts() # Identifity categories
            Cateogy_onehot = pd.get_dummies(data_input_new.ExtentSympLowBG, prefix='Bel70PastMonWSymp')
            data_input_new = pd.concat([data_input_new,pd.get_dummies(data_input_new['Bel70PastMonWSymp'], prefix='Bel70PastMonWSymp')],axis=1)

            # enc_data = pd.DataFrame(enc.fit_transform(data_input_new[['ModHypoEpPast6Mon']]).toarray())
            # data_input_new['ModHypoEpPast6Mon'].value_counts() # Identifity categories
            # Cateogy_onehot = pd.get_dummies(data_input_new.ExtentSympLowBG, prefix='ModHypoEpPast6Mon')
            # data_input_new = pd.concat([data_input_new,pd.get_dummies(data_input_new['ModHypoEpPast6Mon'], prefix='ModHypoEpPast6Mon')],axis=1)

            enc_data = pd.DataFrame(enc.fit_transform(data_input_new[['Bel70PastMonNoSymp']]).toarray())
            data_input_new['Bel70PastMonNoSymp'].value_counts() # Identifity categories
            Cateogy_onehot = pd.get_dummies(data_input_new.ExtentSympLowBG, prefix='Bel70PastMonNoSymp')
            data_input_new = pd.concat([data_input_new,pd.get_dummies(data_input_new['Bel70PastMonNoSymp'], prefix='Bel70PastMonNoSymp')],axis=1)

#             enc_data = pd.DataFrame(enc.fit_transform(data_input_new[['SevHypoEpPastYear']]).toarray())
#             data_input_new['SevHypoEpPastYear'].value_counts() # Identifity categories
#             Cateogy_onehot = pd.get_dummies(data_input_new.SevHypoEpPastYear, prefix='SevHypoEpPastYear')
#             data_input_new = pd.concat([data_input_new,pd.get_dummies(data_input_new['SevHypoEpPastYear'], prefix='SevHypoEpPastYear')],axis=1)

            data_input_new = data_input_new.drop(["Bel70PastMonNoSymp","SevHypoEpPastYear", "FeelSympLowBG", "ModHypoEpPast6Mon", "Bel70PastMonWSymp", "LowBGSympCat", "ExtentSympLowBG"], axis = 1)
            
            X_train,X_test,Y_train,Y_test= train_test_split(data_input_new, y_label, stratify=y_label, test_size=test_s, random_state=i)
            
            
            no_scaling_df_x_train=pd.concat([no_scaling_df_x_train,X_train.reset_index(drop=True)],axis='columns')
            
            no_scaling_df_x_test=pd.concat([no_scaling_df_x_test,X_test.reset_index(drop=True)],axis='columns')

        elif names[j] == 'Signal':
            print('Time series')
            X_train, X_test, Y_train, Y_test = train_test_split(e, y_label, test_size=test_s, random_state=i,
                                                                stratify=y_label)

            tfidf_vectorizer = TfidfVectorizer(max_features=tfidf)
            m_tfidf_sparse = tfidf_vectorizer.fit_transform(X_train['text'+str(i)])
            X_train = m_tfidf_sparse.toarray()
            m_tfidf_sparse = tfidf_vectorizer.transform(X_test['text'+str(i)])
            X_test = m_tfidf_sparse.toarray()
            m = pd.DataFrame(tfidf_vectorizer.vocabulary_, index=[0]).sort_values(by=[0], axis=1)
            X_train = pd.DataFrame(X_train, columns=m.columns)
            X_test = pd.DataFrame(X_test, columns=m.columns)

            scaling_df_x_train = pd.concat([scaling_df_x_train, X_train.reset_index(drop=True)], axis='columns')

            scaling_df_x_test = pd.concat([scaling_df_x_test, X_test.reset_index(drop=True)], axis='columns')
            # no_scaling_df_x_train = pd.concat([no_scaling_df_x_train, X_train.reset_index(drop=True)], axis='columns')
            #
            # no_scaling_df_x_test = pd.concat([no_scaling_df_x_test, X_test.reset_index(drop=True)], axis='columns')

        elif names[j] == 'Conditions':
            print('Conditions')
            X_train, X_test, Y_train, Y_test = Embedded_text (e, var_name='medcon1', encoding='Clinical',ngrams=1, embedding_size=50,
                  Reduction='KPCA',seed=i,label=y_label)


            scaling_df_x_train = pd.concat([scaling_df_x_train, X_train.reset_index(drop=True)], axis='columns')

            scaling_df_x_test = pd.concat([scaling_df_x_test, X_test.reset_index(drop=True)], axis='columns')

            # no_scaling_df_x_train = pd.concat([no_scaling_df_x_train, X_train.reset_index(drop=True)], axis='columns')
            #
            # no_scaling_df_x_test = pd.concat([no_scaling_df_x_test, X_test.reset_index(drop=True)], axis='columns')

        elif names[j] == 'Medications':
            print('Medications')
            X_train, X_test, Y_train, Y_test = Embedded_text(e, var_name='medications3', encoding='Clinical', ngrams=1,
                                                             embedding_size=10,
                                                             Reduction='PCA', seed=i,label=y_label)



            scaling_df_x_train = pd.concat([scaling_df_x_train, X_train.reset_index(drop=True)], axis='columns')

            scaling_df_x_test = pd.concat([scaling_df_x_test, X_test.reset_index(drop=True)], axis='columns')

            # no_scaling_df_x_train = pd.concat([no_scaling_df_x_train, X_train.reset_index(drop=True)], axis='columns')
            #
            # no_scaling_df_x_test = pd.concat([no_scaling_df_x_test, X_test.reset_index(drop=True)], axis='columns')

        elif names[j] == 'Medications_hot':
            list1 = []
            data = e
            for u in data['medications3']:
                list1 = list1 + u.split()
            features = list(set(list1))

            mat = [0] * len(features)
            mat2 = []
            for u in range(len(data)):
                mat2.append(mat)

            Dataframe = pd.DataFrame(data=mat2, columns=features)

            for u in range(len(data)):
                for p in features:
                    if p in data['medications3'].iloc[u].split():
                        Dataframe[p].iloc[u] = 1

            X_train, X_test, Y_train, Y_test = train_test_split(Dataframe, y_label, stratify=y_label, test_size=test_s,
                                                                random_state=i)
            no_scaling_df_x_train = pd.concat([no_scaling_df_x_train, X_train.reset_index(drop=True)], axis='columns')

            no_scaling_df_x_test = pd.concat([no_scaling_df_x_test, X_test.reset_index(drop=True)], axis='columns')
        elif names[j] == 'Conditions_hot':
            list1 = []
            data = x_features
            for u in data['medcon1']:
                list1 = list1 + u.split()
            features = list(set(list1))

            mat = [0] * len(features)
            mat2 = []
            for u in range(len(data)):
                mat2.append(mat)

            Dataframe = pd.DataFrame(data=mat2, columns=features)

            for u in range(len(data)):
                for p in features:
                    if p in data['medcon1'].iloc[u].split():
                        Dataframe[p].iloc[u] = 1

            X_train, X_test, Y_train, Y_test = train_test_split(Dataframe, y_label, stratify=y_label, test_size=test_s,
                                                                random_state=i)
            no_scaling_df_x_train = pd.concat([no_scaling_df_x_train, X_train.reset_index(drop=True)], axis='columns')

            no_scaling_df_x_test = pd.concat([no_scaling_df_x_test, X_test.reset_index(drop=True)], axis='columns')

        elif names[j]=='Fear':
            print('Fear')
            X_train,X_test,Y_train,Y_test= train_test_split(e, y_label, stratify=y_label, test_size=test_s, random_state=i)
            
            # no_scaling_df_x_train=pd.concat([no_scaling_df_x_train,X_train.reset_index(drop=True)],axis='columns')
            #
            # no_scaling_df_x_test=pd.concat([no_scaling_df_x_test,X_test.reset_index(drop=True)],axis='columns')

            scaling_df_x_train=pd.concat([scaling_df_x_train,X_train.reset_index(drop=True)],axis='columns')

            scaling_df_x_test=pd.concat([scaling_df_x_test,X_test.reset_index(drop=True)],axis='columns')

        elif names[j]=='MOCA':
            print('MOCA')

            #data.drop(['MoCAOrient'], axis=1, inplace=True)
            #data=data.drop(data[data['MoCAOrient'].isna()].index)

            X =e
            Y= y_label
            X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=test_s,random_state=i,stratify=y_label)

            X_train['MoCAPtEff'] = X_train['MoCAPtEff'].replace(['Good'], 1).replace(['Questionable'], 0).replace(['Poor'], 0)
            X_train.drop(['MoCANotDoneReas'], axis=1, inplace=True)
            X_train.drop(['MoCANotDone'], axis=1, inplace=True)

            X_test['MoCAPtEff'] = X_test['MoCAPtEff'].replace(['Good'], 1).replace(['Questionable'], 0).replace(['Poor'], 0)
            X_test.drop(['MoCANotDoneReas'], axis=1, inplace=True)
            X_test.drop(['MoCANotDone'], axis=1, inplace=True)

            X_train["MoCAPtEff"].replace(np.nan, mode(X_train["MoCAPtEff"]), inplace=True)
            X_test["MoCAPtEff"].replace(np.nan, mode(X_train["MoCAPtEff"]), inplace=True)
            
            X_train=X_train.drop(['Unnamed: 0'],axis=1)
            X_test=X_test.drop(['Unnamed: 0'],axis=1)
                

            no_scaling_df_x_train=pd.concat([no_scaling_df_x_train,X_train.reset_index(drop=True)],axis='columns')
            
            no_scaling_df_x_test=pd.concat([no_scaling_df_x_test,X_test.reset_index(drop=True)],axis='columns')


        elif names[j]=='BSample':
            print('BSample')
            X_train,X_test,Y_train,Y_test= train_test_split(e,y_label,test_size=test_s,random_state=i,stratify=y_label)
            imputer = KNNImputer(n_neighbors=5, weights='uniform',metric='nan_euclidean')

            imputer.fit(X_train)

            # transform the dataset
            train1= imputer.transform(X_train)
            test1= imputer.transform(X_test)

            X_train= pd.DataFrame(train1, columns= e.columns)
            X_test= pd.DataFrame(test1, columns= e.columns)

            
            scaling_df_x_train=pd.concat([scaling_df_x_train, X_train.reset_index(drop=True)],axis='columns')
            
            scaling_df_x_test=pd.concat([scaling_df_x_test, X_test.reset_index(drop=True)],axis='columns')
            


        elif names[j]=='BMedChart':
            print('BMedChart')
            data=e     
            X_train,X_test,Y_train,Y_test= train_test_split(data,y_label,test_size=test_s,random_state=i,stratify=y_label)
            X_train["LastFoodIntakeHrs"].replace(np.nan, mode(X_train["LastFoodIntakeHrs"]), inplace=True)
            X_test["LastFoodIntakeHrs"].replace(np.nan, mode(X_train["LastFoodIntakeHrs"]), inplace=True)

            X_train["LastFoodIntakeCarbs"].replace(np.nan, median(X_train["LastFoodIntakeCarbs"].dropna()), inplace=True)
            X_test["LastFoodIntakeCarbs"].replace(np.nan, median(X_train["LastFoodIntakeCarbs"].dropna()), inplace=True)

            imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

            imputer.fit(X_train[['Weight_mod','Height_mod']])

            # transform the dataset
            train1 = imputer.transform(X_train[['Weight_mod','Height_mod']])
            test1 = imputer.transform(X_test[['Weight_mod','Height_mod']])
            train1=pd.DataFrame(train1,columns=['Weight_mod','Height_mod'])
            test1=pd.DataFrame(test1,columns=['Weight_mod','Height_mod'])
            X_train=X_train.reset_index()
            train1=train1.reset_index()
            X_test=X_test.reset_index()
            test1=test1.reset_index()
            X_train=pd.concat([X_train[["LastFoodIntakeCarbs", 'LastFoodIntakeHrs']], train1[['Weight_mod','Height_mod']]],axis=1)
            X_test=pd.concat([X_test[["LastFoodIntakeCarbs", 'LastFoodIntakeHrs']], test1[['Weight_mod','Height_mod']]],axis=1)

            X_train=var_categ_train("LastFoodIntakeHrs",X_train)
            X_train.drop(['LastFoodIntakeHrs'], axis=1, inplace=True)

            X_test=var_categ_test("LastFoodIntakeHrs",X_test)
            X_test.drop(['LastFoodIntakeHrs'], axis=1, inplace=True)


            
            Cat=['LastFoodIntakeHrs_0-<4 hours prior', 'LastFoodIntakeHrs_4-<8 hours prior', 'LastFoodIntakeHrs_8 or more hours prior']
            
            
            
            X_train2=X_train[Cat]
            co=[]
            for m in X_train2.columns:
                if '<' in m:
                    co.append(m.replace('<','_'))
                else:
                    co.append(m)
                
            X_train2.columns=co
            try:
                X_test2=X_test[Cat]
            except:
                X_test['LastFoodIntakeHrs_8 or more hours prior']=[0]*test_s
                X_test2=X_test[Cat]
                
            X_test2.columns=co

            scaling_df_x_train=pd.concat([scaling_df_x_train,X_train.drop(Cat,axis=1)].reset_index(drop=True),axis='columns')
            no_scaling_df_x_train=pd.concat([no_scaling_df_x_train,X_train2].reset_index(drop=True),axis='columns')
            
            try:
                scaling_df_x_test=pd.concat([scaling_df_x_test,X_test.drop(Cat,axis=1)].reset_index(drop=True),axis='columns')
                no_scaling_df_x_test=pd.concat([no_scaling_df_x_test,X_test2].reset_index(drop=True),axis='columns')
            except:
                X_test['LastFoodIntakeHrs_8 or more hours prior']=[0]*test_s
                scaling_df_x_test=pd.concat([scaling_df_x_test,X_test.reset_index(drop=True)],axis='columns')
                no_scaling_df_x_test=pd.concat([no_scaling_df_x_test,X_test.reset_index(drop=True)],axis='columns')


        elif names[j]=='BTOTSCORE':
            print('BTOTSCORE')
            data=e
            p=np.where(np.asarray(data.isna().sum()) > 0)
            features2= data.columns[p]


            data=var_categ_train('ReadCardLowLine',data)  
            data=data.drop(['ReadCardLowLine'],axis=1)
            Cat=['GrPegDomHand','ReadCardCorrLens','ReadCardLowLine_20/125',
           'ReadCardLowLine_20/16', 'ReadCardLowLine_20/20',
           'ReadCardLowLine_20/200', 'ReadCardLowLine_20/25',
           'ReadCardLowLine_20/32', 'ReadCardLowLine_20/40',
           'ReadCardLowLine_20/50', 'ReadCardLowLine_20/63',
           'ReadCardLowLine_20/80']

            X_train,X_test,Y_train,Y_test= train_test_split(data,y_label,test_size=test_s,random_state=i,stratify=y_label)

            imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

            imputer.fit(X_train[features2])

            # transform the dataset
            train1 = imputer.transform(X_train[features2])
            test1 = imputer.transform(X_test[features2])

            train1=pd.DataFrame(train1,columns=features2)
            test1=pd.DataFrame(test1,columns=features2)

            p=np.where(np.asarray(data.isna().sum()) == 0)
            features3= data.columns[p]
            X_train=X_train.reset_index(drop=True)
            train1=train1.reset_index(drop=True)
            X_test=X_test.reset_index(drop=True)
            test1=test1.reset_index(drop=True)
            X_train=pd.concat([X_train[features3],train1[features2]],axis=1)
            X_test=pd.concat([X_test[features3],test1[features2]],axis=1)
            X_train=X_train.drop(['Unnamed: 0'],axis=1)
            X_test=X_test.drop(['Unnamed: 0'],axis=1)
            scaling_df_x_train=pd.concat([scaling_df_x_train,X_train.drop(Cat,axis=1).reset_index(drop=True)],axis='columns')
            no_scaling_df_x_train=pd.concat([no_scaling_df_x_train,X_train[Cat].reset_index(drop=True)],axis='columns')
            scaling_df_x_test=pd.concat([scaling_df_x_test,X_test.drop(Cat,axis=1).reset_index(drop=True)],axis='columns')
            no_scaling_df_x_test=pd.concat([no_scaling_df_x_test,X_test[Cat].reset_index(drop=True)],axis='columns')
            

    try:
        scaler = MinMaxScaler()
        #scaler = StandardScaler()
        columns=scaling_df_x_train.columns
        scaler.fit(scaling_df_x_train)
        scaling_df_x_train[columns] = scaler.transform(scaling_df_x_train)
        scaling_df_x_test[columns] = scaler.transform(scaling_df_x_test)
        x_train =pd.concat([scaling_df_x_train,no_scaling_df_x_train],axis='columns')
        x_test =pd.concat([scaling_df_x_test,no_scaling_df_x_test],axis='columns')
        
    except:
        x_train =pd.concat([scaling_df_x_train,no_scaling_df_x_train],axis='columns')
        x_test =pd.concat([scaling_df_x_test,no_scaling_df_x_test],axis='columns')

    x_train =x_train.replace({True:1, False:0})
    x_test = x_test.replace({True: 1, False: 0})
    return x_train, x_test,Y_train, Y_test


def early_fusion(databases_list, partition=0.2, FS=[]):
    # patients = get_patients_id(databases_list)
    patients = get_patients_id(['Medications','Conditions','Fear','BTOTSCORE','BSample','Attitude', 'Lifestyle', 'MOCA','Depression','Signal','Unaware'])
    z = math.ceil(partition * len(patients))
    list_data = []
    features_final = []
    for j, e in enumerate(databases_list):

        if e == 'Attitude':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_attitude.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            df1 = df1.sort_values('PtID')

        elif e == 'BMedChart':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_BMedChart.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            df1 = df1.sort_values('PtID')

        elif e == 'BTOTSCORE':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_BTOTSCORE.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            df1 = df1.sort_values('PtID')

        elif e == 'Depression':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_depression.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            df1 = df1.sort_values('PtID')

        elif e == 'Fear':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_fear.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            df1 = df1.sort_values('PtID')

        elif e == 'Lifestyle':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_lifestyle.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            df1 = df1.sort_values('PtID')

        elif e == 'Conditions':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TEXT, 'bbdd_medconditions2.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            df1 = df1.sort_values('PtID')

        elif e == 'Medications':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TEXT, 'bbdd_medications2.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            df1 = df1.sort_values('PtID')

        elif e == 'Conditions_hot':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TEXT, 'bbdd_medconditions.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            df1 = df1.sort_values('PtID')

        elif e == 'Medications_hot':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TEXT, 'bbdd_medications.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            df1 = df1.sort_values('PtID')

        elif e == 'MOCA':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_MOCA.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            df1 = df1.sort_values('PtID')

        elif e == 'Unaware':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_unaware.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            df1 = df1.sort_values('PtID')

        elif e == 'BSample':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TABULAR, 'bbdd_BSample.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            df1 = df1.sort_values('PtID')

        elif e == 'Signal':
            df1 = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_SIGNAL, 'gSAX.csv'))
            df1 = patients.merge(df1, on=['PtID'])
            df1 = df1.sort_values('PtID')

        Y = df1['label_encoded']
        X = df1.drop(['label_encoded'], axis=1)
        list_data.append(X)
        if len(FS) > 0:
            features=relief_bbdd(X, Y, e, test_s=z,FS=FS[j],path=consts.PATH_PROJECT_TABULAR_METRICS)
            for u in list(features):
                features_final.append(u)
            list_pred_train, list_pred_test, list_y_train, list_y_test = main_early_fusion(list_data, Y, databases_list,
                                                                                           z, 300)
        else:
            list_pred_train, list_pred_test, list_y_train, list_y_test = main_early_fusion(list_data, Y, databases_list, z,10)
    if len(FS) > 0:
        print(len(features_final))
        df_train1 = list_pred_train[0][features_final]
        df_train2 = list_pred_train[1][features_final]
        df_train3 = list_pred_train[2][features_final]
        df_train4 = list_pred_train[3][features_final]
        df_train5 = list_pred_train[4][features_final]

        df_test1 = list_pred_test[0][features_final]
        df_test2 = list_pred_test[1][features_final]
        df_test3 = list_pred_test[2][features_final]
        df_test4 = list_pred_test[3][features_final]
        df_test5 = list_pred_test[4][features_final]


    else:
        df_train1 = list_pred_train[0]
        df_train2 = list_pred_train[1]
        df_train3 = list_pred_train[2]
        df_train4 = list_pred_train[3]
        df_train5 = list_pred_train[4]

        df_test1 = list_pred_test[0]
        df_test2 = list_pred_test[1]
        df_test3 = list_pred_test[2]
        df_test4 = list_pred_test[3]
        df_test5 = list_pred_test[4]

    df_train1['label'] = list_y_train[0].reset_index().iloc[:, 1]
    df_train2['label'] = list_y_train[1].reset_index().iloc[:, 1]
    df_train3['label'] = list_y_train[2].reset_index().iloc[:, 1]
    df_train4['label'] = list_y_train[3].reset_index().iloc[:, 1]
    df_train5['label'] = list_y_train[4].reset_index().iloc[:, 1]

    df_test1['label'] = list_y_test[0].reset_index().iloc[:, 1]
    df_test2['label'] = list_y_test[1].reset_index().iloc[:, 1]
    df_test3['label'] = list_y_test[2].reset_index().iloc[:, 1]
    df_test4['label'] = list_y_test[3].reset_index().iloc[:, 1]
    df_test5['label'] = list_y_test[4].reset_index().iloc[:, 1]

    train = [df_train1, df_train2, df_train3, df_train4, df_train5]
    test = [df_test1, df_test2, df_test3, df_test4, df_test5]
    return train, test
list_clfs = ['RandomForest', 'knn', 'dt', 'svm', 'reglog', 'lasso']
def classifier_early(train, test, clfs=list_clfs, FS=False,save_path=consts.PATH_PROJECT_FUSION_METRICS):
    if FS==True:
        df_score= relief_fusion(train,len(train[0].columns)-1)
        relief_plot(df_score, consts.PATH_PROJECT_FUSION_FIGURES, 'Early_'+str(len(train[0].columns)-1))
    elif type(FS)== int :
        df_score = relief_fusion(train, FS)
        features_selected=df_score['names'][:FS]
        df=call_models_fusion(train, test, list_clfs=clfs,features=features_selected)
        df.to_csv(os.path.join(save_path, 'Early_' + str(len(train[0].columns) - 1) + '_' + str(FS) + '.csv'))
        return df
    else:
        df = call_models_fusion(train, test, list_clfs=clfs)
        df.to_csv(os.path.join(save_path,'Early_'+str(len(train[0].columns)-1)+'_'+str(FS)+'.csv'))
        return df

