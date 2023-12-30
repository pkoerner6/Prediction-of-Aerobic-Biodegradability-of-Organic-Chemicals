# Data Quality Matters: Enhancing Machine Learning Models for Predicting Aerobic Biodegradability
This is the code base for the master's thesis written by Paulina Körner and supervised by Dr. Juliane Glüge and Prof. Dr. Martin Scheringer. 

## Abstract
The field of risk assessment of chemicals has attracted substantial interest over the years. 
However, despite significant research efforts, creating a scalable system that would allow testing of all chemicals released to the environment has proven difficult.
A promising direction for addressing this crucial challenge is leveraging the capabilities of machine learning (ML) models, which are increasingly used to predict the activity and characteristics of chemicals.
Analyzing chemicals requires a model to develop a deep understanding of chemical structures and activity relationships, which inherently relies on a substantial amount of training data. 
At the same time, creating a large, clean training dataset capturing a wide range of chemicals is very difficult due to the resource-intensive nature of biodegradability screening tests.

In this master's thesis, a data-centric approach was taken to improve state-of-the-art ML models to predict the aerobic biodegradability of organic chemicals. 
This approach involved data curation, emphasizing the importance of matching chemical identifiers and structural representations and of reliable data labels. 
Following the dataset improvement, model selection was carried out using LazyPredict, and the applicability domain (AD) was defined. 
The research findings provide valuable insights into several critical aspects of predictive modeling for aerobic biodegradability. 
Contrary to initial expectations, the curation of CAS RN™️-SMILES pairings did not improve model performance. 
However, validating the labels associated with data points using two predictive models resulted in substantial performance enhancements. 
The obtained classifier and regressor achieved a mean accuracy of 95.7 ± 0.4\% and a mean $\text{R}^2$ score of 0.74 ± 0.01, which is higher than any other model published thus far. 
Remarkably, these improvements were achieved despite a reduction in the dataset's size.
Furthermore, the reduction in dataset size did not lead to a narrowing of the AD for the models, showing that they can be reliably applied for predicting aerobic biodegradability across a wide range of organic chemicals.
The insights acquired from this research contribute to the advancement of predictive modeling for chemical safety assessment.

## Setup
To run the code, Python3 is required. 
Three virtual environments are needed to run all scripts. 
The environment used to run most of the files can be installed like this:
```
python3 -m thesis_venv
source thesis_venv/bin/activate
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

### Huang_Zhang.py
To run this file, the Huang_venv needs to be activated.
The purpose of this file is to run the models presented by Huang and Zhang [2022], replicate the models, and test the models on the additional testing set. 

### data_processing.py
This file carries out the steps described in the SMILES-Retrieval-Pipeline to create the $\text{Curated}_\text{S}$ and the $\text{Curated}_\text{SCS}$ datasets. 

### add_pka_values.py
To run this file, the molgpka_venv needs to be activated. 
It can be used to add pKa and $\alpha$ values to datasets. 

### creating_datasets.py
This file creates all curated datasets and saves them to the dataframes folder. 
This file needs to be run before running the following scripts. 

### improved_data.py
This file trains XGBoost models on the curated datasets. 
It can be run for classification and regression models and can be run to either create the Data-Impact-Plots or the SMILES-Impact-Plot. To create the Data-Impact-Plots, the argument progress_or_comparison needs to be set to "progress," and to create the SMILES-Impact-Plot, it needs to be set to "comparison."

### improved_data_analysis.py
This file carries out the analysis of the curated datasets after label validation with the BIOWIN™️ models. 

### improved_data_analysis.py
This file is used to run LazyPredict and to carry out the hyperparameter tuning. 
When running this file on Euler, the argument "njobs" should be set to 12. 

### applicability_domain.py
This file can be run to define the applicability domain of the models trained on the $\text{Curated}_\text{FINAL}$ dataset. 
