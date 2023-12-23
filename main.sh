#!/bin/bash

python src/train.py --type_data='unimodal' --type_modality='tabular'
python src/train.py --type_data='unimodal' --type_modality='time_series'
python src/train.py --type_data='unimodal' --type_modality='text'
python src/train.py --type_data='multimodal' --type_fusion='early'
python src/train.py --type_data='multimodal' --type_fusion='late'

exit