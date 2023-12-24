Multimodal severe hypoglycemia detection
====

## Installation and configuration

Install libraries for project.
```console
pip install -r requirements.txt 
pip install git+https://github.com/cdchushig/scikit-rebate.git@1efbe530a46835c86f2e50f17342541a3085be9c
```

If have any issue with skrebate, please install the following modified version:
```console
pip install git+https://github.com/cdchushig/scikit-rebate.git@1efbe530a46835c86f2e50f17342541a3085be9c
```


## Download data and copy to project
Preprocessed data are available in:

[Link with datasets](https://urjc-my.sharepoint.com/:f:/g/personal/francisco_lara_urjc_es/EvAc3WtMD5FGpkOC0iszAE8B2Oe6tQhfQiwbj-tN_rTiYQ?e=Z2vctl&xsdata=MDV8MDJ8fGEzMTMwNGNkYjAxMTRlMjcyNmJkMDhkYzAzYTczZGQzfDVmODRjNGVhMzcwZDRiOWU4MzBjNzU2ZjhiZjFiNTFmfDB8MHw2MzgzODkyNjQwMTE0Mzc3NTd8VW5rbm93bnxWR1ZoYlhOVFpXTjFjbWwwZVZObGNuWnBZMlY4ZXlKV0lqb2lNQzR3TGpBd01EQWlMQ0pRSWpvaVYybHVNeklpTENKQlRpSTZJazkwYUdWeUlpd2lWMVFpT2pFeGZRPT18MXxMMk5vWVhSekx6RTVPak0xTldFeU56VTBMVFV5TlRndE5ESXhOeTFpWldaaUxXSmxZVEF6TnpnNFlUZ3haRjgzTkRCak1tVTRPUzFpTXpRNUxUUXlNVE10T0RGaVlpMDBNRGRrT0RZeU1EQXhNbVJBZFc1eExtZGliQzV6Y0dGalpYTXZiV1Z6YzJGblpYTXZNVGN3TXpNeU9UWXdNREV4T0E9PXw3MjlmOGZmOWY0MTY0Nzc5MjZiZDA4ZGMwM2E3M2RkM3w1OTM1YTFlZTNhNjE0ZGM2YTdlNDI1OWIzZmY5N2M2Yg%3D%3D&sdata=WUpjWkJuVFR3S05ObkhJeGdhRFI3dkZ0QnNTalhiK3M2UHYva3NvcGhxWT0%3D)

After downloading data, you have to put folders and files in data/preprocessed

## To obtain results of models using single-modality data

For results using tabular data:
```console
python src/train.py --type_data='unimodal' --type_modality='tabular'
```

For results using time series:
```console
python src/train.py --type_data='unimodal' --type_modality='time_series'
```

For results using text:
```console
python src/train.py --type_data='unimodal' --type_modality='text'
```

## To obtain results of models using multi-modality data

For results with early fusion:
```console
python src/train.py --type_data='multimodal' --type_fusion='early'
```

For results with late fusion:
```console
python src/train.py --type_data='multimodal' --type_fusion='late'
```

