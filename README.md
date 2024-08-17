# Critical Insights into Data Curation and Label Noise for Accurate Prediction of Aerobic Biodegradability of Organic Chemicals
This is the code base for the paper "Critical Insights into Data Curation and Label Noise for Accurate Prediction of Aerobic Biodegradability of Organic Chemicals" written by Paulina Körner, Dr. Juliane Glüge, Dr. Stefan Glüge and Prof. Dr. Martin Scheringer. 

## Abstract
The focus of the current study is to enhance state-of-the-art Machine Learning (ML) models that can predict the aerobic biodegradability of organic chemicals through a data-centric approach. To do that, an already existing dataset that was previously used to train ML models was analyzed for mismatching chemical identifiers and data leakage between test and training set and the detected errors were corrected. Chemicals with high variance between study results were removed. An XGBoost was trained on the dataset and compared to a XGBoost that was trained on a dataset where certain substances were excluded. The results show that despite comprehensive data curation, only marginal improvement was observed in the classification model’s performance. This was attributed to three potential reasons: 1) a significant number of data labels were noisy, 2) the features could not sufficiently represent the chemicals, and/or 3) the model struggled to learn and generalize effectively. All three potential reasons were examined, but only removing data points with possibly noisy labels by performing label noise filtering using other predictive models increased the classification model’s balanced accuracy from 80.9% to 94.2%. While no indications were found that label noise filtering removed difficult-to-learn substances, this possibility cannot be entirely ruled out.


## Setup
To run the code, Python3 is required. 
Three virtual environments are needed to run all scripts. 
The environment used to run most of the files can be installed like this:
```
python3 -m venv main_venv
source main_venv/bin/activate
pip install -r requirements.txt
```
To create the second environment, which is required to run the downloaded models from Hunag and Zhang [2022], run the following code:
```
python3 -m venv Huang_venv
source Huang_venv/bin/activate
pip install -r requirements_huang_zhang_replication.txt
```
The third environment is only required if one wants to use MolGpKa to add pKa and $\alpha$ values to a data frame:
```
python3 -m venv molgpka_venv
source molgpka_venv/bin/activate
pip install -r requirements_molgpka.txt
```

## Overview of the scripts

### processing_functions.py
Contains all kinds of functions used in the other scripts. 

### ml_functions.py
Contains functions for creating, validating, and testing machine learning models. 

### data_processing.py
This file carries out the steps described in the SMILES-Retrieval-Pipeline to create the $\text{Curated}_\text{S}$ and the $\text{Curated}_\text{SCS}$ datasets. It uses the dataset iuclid_echa.csv which includes information on biodegradation screening tests from REACH. The code for retrieving this data will be published seperately.

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
