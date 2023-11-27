

import argparse
import pandas as pd
import numpy as np
import structlog
import sys
import os
import statistics
import matplotlib.pyplot as pl
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor
from pyADA import ApplicabilityDomain
import shap
import pickle

log = structlog.get_logger()
from typing import List, Dict, Tuple


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from processing_functions import create_input_classification
from processing_functions import get_maccs_names
from processing_functions import column_to_structure
from ml_functions import get_lazy_xgbc_parameters
from ml_functions import get_balanced_data_adasyn
from ml_functions import get_best_classifier_readded

parser = argparse.ArgumentParser()
parser.add_argument(
    "--random_seed",
    type=int,
    default=42,
)
args = parser.parse_args()



def explain_classifier_with_shap():
    log.info("Start loading model")
    classifier, x = get_best_classifier_readded(train_new=False, random_seed=args.random_seed) 
    log.info("Finished loading model")
    explainer = shap.TreeExplainer(classifier)
    log.info("Finished creating explainer")
    shap_values = explainer.shap_values(x)
    log.info("Finished creating shap_values")

    # log.info("force_plot for first data point")
    # shap.force_plot(explainer.expected_value, shap_values[0, :], x.iloc[0, :])

    maccs_names = get_maccs_names()
    # log.info("Explaining single feature for Halogens (134)")
    # shap.dependence_plot(ind=134, shap_values=shap_values, features=x, feature_names=maccs_names, alpha=0.8) # interaction_index

    log.info("Summary plot")
    shap.summary_plot(
      shap_values=shap_values, 
      features=x, 
      feature_names=maccs_names,
      max_display=40,
      show=False,
    )
    pl.savefig("figures/shap_summary.png")

    df_x = pd.DataFrame(data=x)
    column_to_structure_maccs = column_to_structure()
    df_x.rename(columns = column_to_structure_maccs, inplace = True)

    feature_names = df_x.columns
    rf_resultX = pd.DataFrame(shap_values, columns=feature_names)
    vals = np.abs(rf_resultX.values).mean(0)
    shap_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name','feature_importance_vals'])
    shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)

    shap_importance.to_csv("datasets/shap_importance.csv")



def explain_prediction_of_removed_with_shap():
    df_class_removed = pd.read_csv(
        "datasets/curated_data/class_curated_scs_biowin_readded_removed.csv", index_col=0
    )
    x_removed = create_input_classification(df_class_removed, include_speciation=False)
    df_x = pd.DataFrame(data=x_removed)
    column_to_structure_maccs = column_to_structure()
    df_x.rename(columns = column_to_structure_maccs, inplace = True)

    classifier, x_train = get_best_classifier_readded(train_new=False, random_seed=args.random_seed) 
    log.info("Finished loading model")
    df_x_train = pd.DataFrame(data=x_train)
    col_to_maccs_names = column_to_structure()
    df_x_train.rename(columns=col_to_maccs_names, inplace=True)

    explainer = shap.Explainer(classifier)
    log.info("Finished creating explainer")
    shap_values = explainer(df_x_train)
    log.info("Finished creating shap_values for prediction")

    maccs_names = get_maccs_names()
    log.info("Beginning plot")
    shap.plots.bar(
        shap_values=shap_values,
        show = True,
        max_display=20,
    ) 
    pl.savefig("figures/shap_explainer_bar.png")
    log.info("Finished first plot")

    shap.plots.bar(shap_values[0]) # For the first observation
    pl.savefig("figures/shap_explainer_first_element_removed.png")




if __name__ == "__main__":
    # explain_classifier_with_shap()

    explain_prediction_of_removed_with_shap()

