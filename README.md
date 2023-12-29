Multimodal severe hypoglycemia detection
====

## Clone and download files of repository

To dowload the source code, you can clone it from the Github repository.
```console
git clone git@github.com:franciscojesuslara/multimodal_SH_detection.git
```

## Installation and configuration

Before installing libraries, ensuring that a Python virtual environment is activated (using conda o virtualenv). To install Python libraries run: 

```console
pip install -r requirements.txt 
```

If have any issue with skrebate, please install the following modified version:
```console
pip install git+https://github.com/cdchushig/scikit-rebate.git@1efbe530a46835c86f2e50f17342541a3085be9c
```

## Download data and copy to project

A further description of the original datasets is available in the paper: "Severe Hypoglycemia in Older Adults with Type 1 Diabetes: A Study to Identify Factors Associated with the Occurrence of Severe Hypoglycemia in Older Adults with T1D".

The original datasets can be download in the following link:

[Link to original datasets](https://public.jaeb.org/datasets/diabetes)

Raw data and preprocessed data have been uploaded in Onedrive folder. The link for both raw and preprocessed datasets is:

[Link to raw and preprocessed datasets](https://urjc-my.sharepoint.com/:f:/g/personal/francisco_lara_urjc_es/EvAc3WtMD5FGpkOC0iszAE8B2Oe6tQhfQiwbj-tN_rTiYQ?e=Z2vctl&xsdata=MDV8MDJ8fGEzMTMwNGNkYjAxMTRlMjcyNmJkMDhkYzAzYTczZGQzfDVmODRjNGVhMzcwZDRiOWU4MzBjNzU2ZjhiZjFiNTFmfDB8MHw2MzgzODkyNjQwMTE0Mzc3NTd8VW5rbm93bnxWR1ZoYlhOVFpXTjFjbWwwZVZObGNuWnBZMlY4ZXlKV0lqb2lNQzR3TGpBd01EQWlMQ0pRSWpvaVYybHVNeklpTENKQlRpSTZJazkwYUdWeUlpd2lWMVFpT2pFeGZRPT18MXxMMk5vWVhSekx6RTVPak0xTldFeU56VTBMVFV5TlRndE5ESXhOeTFpWldaaUxXSmxZVEF6TnpnNFlUZ3haRjgzTkRCak1tVTRPUzFpTXpRNUxUUXlNVE10T0RGaVlpMDBNRGRrT0RZeU1EQXhNbVJBZFc1eExtZGliQzV6Y0dGalpYTXZiV1Z6YzJGblpYTXZNVGN3TXpNeU9UWXdNREV4T0E9PXw3MjlmOGZmOWY0MTY0Nzc5MjZiZDA4ZGMwM2E3M2RkM3w1OTM1YTFlZTNhNjE0ZGM2YTdlNDI1OWIzZmY5N2M2Yg%3D%3D&sdata=WUpjWkJuVFR3S05ObkhJeGdhRFI3dkZ0QnNTalhiK3M2UHYva3NvcGhxWT0%3D)

To replicate results, download datasets of preprocessed folder. Please, after downloading data, you have to put folders and files in data/preprocessed.  

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

