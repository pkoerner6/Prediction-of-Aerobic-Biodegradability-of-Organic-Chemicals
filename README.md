# Data Quality Matters: Enhancing Machine Learning Models for Predicting Aerobic Biodegradability
This is the code base for the paper "..." written by Paulina Körner, Dr. Juliane Glüge, Dr. Stefan Glüge and Prof. Dr. Martin Scheringer. 

## Abstract
...

## Setup
To run the code, Python3 is required. 
Three virtual environments are needed to run all scripts. 
The environment used to run most of the files can be installed like this:
```
python3 -m main_venv
source main_venv/bin/activate
pip install -r requirements.txt
```
To create the second environment, which is required to run the downloaded models from Hunag and Zhang [2022], run the following code:
```
python3 -m Huang_venv
source Huang_venv/bin/activate
pip install -r requirements_huang_zhang_replication.txt
```
The third environment is only required if one wants to use MolGpKa to add pKa and $\alpha$ values to a data frame:
```
python3 -m molgpka_venv
source molgpka_venv/bin/activate
pip install -r requirements_molgpka.txt
```

## Overview of the scripts

### processing_functions.py
Contains all kinds of functions used in the other scripts. 

### ml_functions.py
Contains functions for creating, validating, and testing machine learning models. 

### xml_parse.py
Running this file requires you to download the REACH study results dossiers from the [IUCLID website](https://iuclid6.echa.europa.eu). 
The code is largely based on code by Gluege et al. and can be used to retrieve information on biodegradation screening tests. 
The output of this file is used to create the additional testing set used in the Huang_Zhang.py file. 
It also creates the data frame that will be used to remove read-across studies. 

### data_processing.py
This file carries out the steps described in the SMILES-Retrieval-Pipeline to create the $\text{Curated}_\text{S}$ and the $\text{Curated}_\text{SCS}$ datasets. 

### add_pka_values.py
To run this file, the molgpka_venv needs to be activated. 
It can be used to add pKa and $\alpha$ values to datasets. 

### Huang_Zhang_replicated.py
To run this file, the Huang_venv needs to be activated.
The purpose of this file is to run the models presented by Huang and Zhang [2022], replicate the models, and test the models on the additional testing set. 

### creating_datasets.py
This file creates all curated datasets and saves them to the dataframes folder. 
This file needs to be run before running the following scripts. 

### curated_data.py
This file trains XGBoost models on the curated datasets. The train and test sets can be selected.

### curated_data_analysis.py
This file carries out the analysis of the curated datasets after label validation with the BIOWIN™️ models. 

### other_features.py
This file is used to test the impact of other feature creation methods (RDK fingerprints, Morgan fingerprints, and features created using the pretrained model Molformer). 

### improved_models.py
This file is used to run LazyPredict and to carry out the hyperparameter tuning. 

### applicability_domain.py
This file can be run to define the applicability domain of the models trained on the curated datasets. 
