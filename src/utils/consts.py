from pathlib import Path

SEEDS = [2, 64, 0, 10, 36]

PATH_PROJECT_DIR = Path(__file__).resolve().parents[2]
PATH_PROJECT_DATA = Path.joinpath(PATH_PROJECT_DIR, 'data')
PATH_PROJECT_DATA_RAW = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw')
PATH_PROJECT_DATA_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'preprocessed')
PATH_PROJECT_DATA_PREPROCESSED_SIGNAL = Path.joinpath(PATH_PROJECT_DIR, 'data', 'preprocessed', 'time_series')
PATH_PROJECT_DATA_PREPROCESSED_TEXT = Path.joinpath(PATH_PROJECT_DIR, 'data', 'preprocessed', 'text')
PATH_PROJECT_DATA_PREPROCESSED_TABULAR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'preprocessed', 'tabular')
PATH_PROJECT_DATA_PREPROCESSED_FUSION = Path.joinpath(PATH_PROJECT_DIR, 'data', 'preprocessed', 'fusion')
PATH_PROJECT_MODELS = Path.joinpath(PATH_PROJECT_DIR, 'models')
PATH_PROJECT_REPORTS = Path.joinpath(PATH_PROJECT_DIR, 'reports')
PATH_PROJECT_FIGURES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'figures')
PATH_PROJECT_TABULAR_METRICS = Path.joinpath(PATH_PROJECT_DIR, 'reports','tabular','metrics')
PATH_PROJECT_TABULAR_FIGURES = Path.joinpath(PATH_PROJECT_DIR, 'reports','tabular','figures')
PATH_PROJECT_REPORTS_SIGNAL = Path.joinpath(PATH_PROJECT_DIR, 'reports','time_series','metrics')
PATH_PROJECT_TEXT_METRICS = Path.joinpath(PATH_PROJECT_DIR, 'reports','text')
PATH_PROJECT_FS_METRICS = Path.joinpath(PATH_PROJECT_DIR, 'reports','FS')
PATH_PROJECT_FUSION_METRICS = Path.joinpath(PATH_PROJECT_DIR, 'reports','fusion','metrics')
PATH_PROJECT_FUSION_FIGURES = Path.joinpath(PATH_PROJECT_DIR, 'reports','fusion','figures')

BBDD_HYPO = 'hypo'
BBDD_ID_HYPO_LIFESTYLE = 'BDemoLifeDiabHxMgmt'
BBDD_HYPO_LIFESTYLE = 'lifestyle'
BBDD_ID_HYPO_DEPRESSION = 'BGeriDepressScale'
BBDD_HYPO_DEPRESSION = 'depression'
BBDD_ID_HYPO_ATTITUDE = 'BBGAttitudeScale'
BBDD_HYPO_ATTITUDE = 'attitude'
BBDD_ID_HYPO_FEAR = 'BHypoFearSurvey'
BBDD_HYPO_FEAR = 'fear'
BBDD_HYPO_CGM = 'BDataCGM'
BBDD_ID_HYPO_UNAWARE = 'BHypoUnawareSurvey'
BBDD_HYPO_UNAWARE = 'unaware'
BBDD_HYPO_LABEL = 'BPtRoster'
