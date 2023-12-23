import matplotlib.pyplot as plt
import utils.consts as consts
import pandas as pd
import os
from utils.plotter import lollipop_plot
from text.text_unimodal import conditions_unimodal,medications_unimodal
from fusion.fusion_functions import early_fusion_approaches,late_train_test_creation,late_fusion_main
from ts.timeseries import CGM_preprocessing
from utils.FS_bbdd import relief_bbdd
from src.utils.classifiers import call_clfs
from src.Tabular_unimodal import tabular_classification
from text.frequency import frequency_text_bbdds
from ts.Series_representation_esax_2 import gSAX_application,Signal_FS,plot_freq_TS

### UNIMODAL CLASSIFICATION APPROACH##

##TABULAR DATA##
databases_list=[]
features_selected=[]
path=consts.PATH_PROJECT_TABULAR_METRICS
databases_list = ['Unaware', 'Fear', 'BTOTSCORE','MOCA', 'Lifestyle', 'BSample',
                      'Attitude', 'Depression']
FS = [9, 7, 2, 8, 11, 3, 4, 6, 4, 5, 4]
tabular_classification(databases_list,paths=path)
tabular_classification(databases_list,features_selected=FS,paths=path)
# #
# ##CGM##
# CGM_preprocessing()
# df = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_SIGNAL, 'Time_series_CGM.csv'))
# gSAX_application(df,[3],[3],[7],[80])
# gSAX_application(df,[6],[5],[10],[80])
# gSAX_application(df,[6],[3],[10],[10])
#
# data_final= pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_SIGNAL, 'gSAX.csv'))
# Signal_FS(data_final)
# ##TEXT##
#CONDITIONS
# conditions_unimodal()
# #MEDICATIONS
# medications_unimodal()
#
# ### FUSION APPROACHES##
## EARLY FUSION
# early_fusion_approaches()
## LATE FUSION
# late_train_test_creation()
# late_fusion_main()

## PLOTS
# plot_freq_TS()


# frequency_text_bbdds()

# path1= os.path.join(consts.PATH_PROJECT_TEXT_METRICS, 'Conditions')
# cond= pd.read_csv(os.path.join(consts.PATH_PROJECT_TEXT_METRICS,'Freq_cond.csv'))
# lollipop_plot (cond['word'][:10],cond['Difference'][:10],'seagreen','Conditions')
# plt.savefig(str(os.path.join(path1, 'Freq_Conditions2.pdf')),
#                 bbox_inches='tight')
#
# plt.close()
# path1= os.path.join(consts.PATH_PROJECT_TEXT_METRICS, 'Medications')
# med= pd.read_csv(os.path.join(consts.PATH_PROJECT_TEXT_METRICS,'Freq_med.csv'))
# lollipop_plot (med['word'][:10],med['Difference'][:10],'coral','Medications')
# plt.savefig(str(os.path.join(path1, 'Freq_Medications2.pdf')),
#                 bbox_inches='tight')