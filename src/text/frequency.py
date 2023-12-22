import pandas as pd
import utils.consts as cons
import os
from utils.check_patients import  get_patients_id
def frequency_text_bbdds():
    Base_de_datos = ['Attitude', 'BTOTSCORE', 'Depression', 'Fear', 'Lifestyle', 'Conditions', 'Medications', 'MOCA', 'Unaware', 'BSample', 'Signal']
    patients = get_patients_id(Base_de_datos)
    cond=pd.read_csv(os.path.join(cons.PATH_PROJECT_DATA_PREPROCESSED_TEXT,'bbdd_medconditions.csv'))
    cond = patients.merge(cond , on=['PtID']).sort_values(by=['PtID'])
    cond=cond[['medcon1','label_encoded']]
    med=pd.read_csv(os.path.join(cons.PATH_PROJECT_DATA_PREPROCESSED_TEXT,'bbdd_medications.csv'))
    med = patients.merge(med , on=['PtID']).sort_values(by=['PtID'])
    med=med[['medications3','label_encoded']]
    med_1=med[med['label_encoded']== 1]
    med_0=med[med['label_encoded']== 0]
    cond_1=cond[cond['label_encoded']== 1]
    cond_0=cond[cond['label_encoded']== 0]

    list_med_1=[]
    list_med_0=[]
    list_cond_1=[]
    list_cond_0=[]
    for e in cond_1['medcon1']:
        try:
            for i in e.split(' '):
                list_cond_1.append(i)
        except:
            continue
    for e in cond_0['medcon1']:
        try:
            for i in e.split(' '):
                list_cond_0.append(i)
        except:
            continue
    for e in med_1['medications3']:
        try:
            for i in e.split(' '):
                list_med_1.append(i)
        except:
            continue
    for e in med_0['medications3']:
        try:
            for i in e.split(' '):
                list_med_0.append(i)
        except:
            continue
    df_cond_1=pd.DataFrame(list_cond_1,columns=['word_1'])
    df_cond_0=pd.DataFrame(list_cond_0,columns=['word_0'])
    df_med_1=pd.DataFrame(list_med_1,columns=['word_1'])
    df_med_0=pd.DataFrame(list_med_0,columns=['word_0'])


    df_cond_1=df_cond_1.value_counts().reset_index()
    df_cond_1.columns=['word','cond_1']
    df_cond_0=df_cond_0.value_counts().reset_index()
    df_cond_0.columns=['word','cond_0']
    df_med_1=df_med_1.value_counts().reset_index()
    df_med_1.columns=['word','med_1']
    df_med_0=df_med_0.value_counts().reset_index()
    df_med_0.columns=['word','med_0']



    cond = df_cond_1.merge(df_cond_0,on = ['word'],how='outer').fillna(0)
    med = df_med_1.merge(df_med_0,on = ['word'],how='outer').fillna(0)

    cond['Difference']=cond['cond_0']-cond['cond_1']
    med['Difference']=med['med_0']-med['med_1']
    cond['Difference_abs']=abs(cond['cond_0']-cond['cond_1'])
    med['Difference_abs']=abs(med['med_0']-med['med_1'])
    cond=cond.sort_values(by=['Difference_abs'],ascending=False)
    med=med.sort_values(by=['Difference_abs'],ascending=False)
    cond.to_csv(os.path.join(cons.PATH_PROJECT_TEXT_METRICS,'Freq_cond.csv'))
    med.to_csv(os.path.join(cons.PATH_PROJECT_TEXT_METRICS,'Freq_med.csv'))

#
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(3, figsize=(20, 50))
# ax[0].set_title('Freq. Case')
# ax[0].bar( x=cond['word'],height=cond['cond_0'])
# plt.xticks(rotation=90)
# ax[1].set_title('Freq. Control')
# ax[1].bar( x=cond['word'],height=cond['cond_1'])
# plt.xticks(rotation=90)
# ax[2].set_title('Difference')
# ax[2].bar( x=cond['word'],height=cond['Difference'])
# ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=50, ha='right')
# ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=50, ha='right')
# ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=50, ha='right')
# plt.show()
# path1= os.path.join(cons.PATH_PROJECT_TEXT_METRICS, 'Conditions')
# plt.savefig(str(os.path.join(path1, 'Freq_Conditions.png')),
#                 bbox_inches='tight')
#
# plt.close()
# fig, ax = plt.subplots(3, figsize=(20, 50))
# ax[0].set_title('Freq. Case')
# ax[0].bar( x=med['word'],height=med['med_0'])
# plt.xticks(rotation=90)
# ax[1].set_title('Freq. Control')
# ax[1].bar( x=med['word'],height=med['med_1'])
# plt.xticks(rotation=90)
# ax[2].set_title('Difference')
# ax[2].bar( x=med['word'],height=med['Difference'])
# ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=50, ha='right')
# ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=50, ha='right')
# ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=50, ha='right')
# plt.show()
# path1= os.path.join(cons.PATH_PROJECT_TEXT_METRICS, 'Medications')
# plt.savefig(str(os.path.join(path1, 'Freq_Medications.png')),
#                 bbox_inches='tight')
#
# def lollipop_plot (x,y,color,name):
#     fig = plt.figure(figsize=(40, 20))
#     plt.hlines(y=x, xmin = 0, xmax =y, color=color)
#     plt.plot(y, x, "o",ms=15,color =color)
#     plt.xlabel('Difference in the number of ocurrences in case vs control',fontname='serif', fontsize=30)
#     plt.ylabel(name,fontname='serif', fontsize=30)
#     plt.xticks(fontname='serif', fontsize=25)
#     plt.yticks(fontname='serif', fontsize=25)
#     plt.grid(axis='x', ls='-', lw=2, alpha=0.9)
#     plt.grid(axis='y', ls=':', lw=2, alpha=0.9)
#     fig.tight_layout()
#     fig.show()
#
# plt.close()
# path1= os.path.join(cons.PATH_PROJECT_TEXT_METRICS, 'Conditions')
# lollipop_plot (cond['word'][:10],cond['Difference'][:10],'seagreen','Conditions')
# plt.savefig(str(os.path.join(path1, 'Freq_Conditions2.pdf')),
#                 bbox_inches='tight')
# plt.close()
# path1= os.path.join(cons.PATH_PROJECT_TEXT_METRICS, 'Medications')
# lollipop_plot (med['word'][:10],med['Difference'][:10],'coral','Medications')
# plt.savefig(str(os.path.join(path1, 'Freq_Medications2.pdf')),
#                 bbox_inches='tight')
