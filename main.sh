#!/bin/bash

python src/main.py --type_data='unimodal' --type_modality='tabular'
python src/main.py --type_data='unimodal' --type_modality='time_series'
python src/main.py --type_data='unimodal' --type_modality='text'
python src/main.py --type_data='multimodal' --type_fusion='early'
python src/main.py --type_data='multimodal' --type_fusion='late'

exit