import os
import time

import pandas as pd
import matplotlib.pyplot as plt
from utils.plotter import lollipop_plot
from text.text_unimodal import conditions_unimodal, medications_unimodal
from fusion.fusion_functions import early_fusion_approaches, late_train_test_creation, late_fusion_main
from ts.timeseries import CGM_preprocessing
import utils.consts as consts
from utils.FS_bbdd import relief_bbdd
from utils.classifiers import call_clfs
from Tabular_unimodal import tabular_classification
from text.frequency import frequency_text_bbdds
from ts.Series_representation_esax_2 import gSAX_application, Signal_FS, plot_freq_TS
import argparse
import coloredlogs
import logging

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def parse_arguments(parser):
    parser.add_argument('--type_data', default='unimodal', type=str)
    parser.add_argument('--type_modality', default='time_series', type=str)
    parser.add_argument('--type_fusion', default='early', type=str)
    parser.add_argument('--preprocessing_data', default=False, type=bool)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='Obtain results.')
args = parse_arguments(parser)


if args.type_data == 'unimodal':
    if args.type_modality == 'tabular':
        logger.info('Models trained for tabular data')
        features_selected = []
        databases_list = ['Unaware', 'Fear', 'BTOTSCORE', 'MOCA', 'Lifestyle',
                          'BSample', 'Attitude', 'Depression']

        FS = [9, 7, 2, 8, 11, 3, 4, 6, 4, 5, 4]

        tabular_classification(databases_list, paths=consts.PATH_PROJECT_TABULAR_METRICS)
        tabular_classification(databases_list, features_selected=FS, paths=consts.PATH_PROJECT_TABULAR_METRICS)

    elif args.type_modality == 'time_series':

        df = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_SIGNAL, 'Time_series_CGM.csv'))
        gSAX_application(df, [3], [3], [7], [80])
        gSAX_application(df, [6], [5], [10], [80])
        gSAX_application(df, [6], [3], [10], [10])

        data_final = pd.read_csv(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_SIGNAL, 'gSAX.csv'))
        Signal_FS(data_final)
    elif args.type_modality == 'text':
        # obtain results using medical conditions
        conditions_unimodal()
        # obtain results using medications
        medications_unimodal()

# Models using multimodal data
elif args.type_data == 'multimodal':
    if args.type_fusion == 'early': # early fusion
        early_fusion_approaches()
    elif args.type_fusion == 'late': # late fusion
        late_train_test_creation()
        late_fusion_main()

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