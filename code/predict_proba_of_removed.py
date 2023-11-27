

import argparse
import pandas as pd
import numpy as np
import structlog
import sys
import os
import statistics
import shap
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem

log = structlog.get_logger()
from typing import List, Dict, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from processing_functions import create_input_classification
from processing_functions import get_maccs_names
from processing_functions import column_to_structure
from processing_functions import openbabel_convert
from processing_functions import remove_smiles_with_incorrect_format
from processing_functions import load_checked_organics6
from ml_functions import get_best_classifier_readded


parser = argparse.ArgumentParser()
parser.add_argument(
    "--random_seed",
    type=int,
    default=42,
)
parser.add_argument(
    "--train_new",
    default=False,
    action=argparse.BooleanOptionalAction,
)
args = parser.parse_args()




def get_removed_data():
    df_class_removed = pd.read_csv(
        "datasets/curated_data/class_curated_scs_biowin_readded_removed.csv", index_col=0
    )
    df_class_removed = df_class_removed[df_class_removed["label"] != df_class_removed["prediction_class"]]
    df_class_removed.reset_index(inplace=True, drop=True)
    return df_class_removed


def make_prediction():
    classifier, x = get_best_classifier_readded(train_new=args.train_new, random_seed=args.random_seed) 

    df_class_removed = get_removed_data()
    x_removed = create_input_classification(df_class_removed, include_speciation=False)
    
    df_class_removed["prediction_of_classifier_trained_on_final"] = classifier.predict(x_removed)
    df_class_removed[["prediction_proba_nrb", "prediction_proba_rb"]] = classifier.predict_proba(x_removed)
    df_class_removed.drop(columns=["y_true"], inplace=True)
    df_class_removed.to_csv("datasets/curated_data/class_curated_scs_biowin_readded_removed_predicted.csv")


def analyse_predictions():
    df = pd.read_csv("datasets/curated_data/class_curated_scs_biowin_readded_removed_predicted.csv", index_col=0)
    log.info("Number of removed substances: ", removed=len(df))
    df_final_pred_agrees_with_label = df[df["label"] == df["prediction_of_classifier_trained_on_final"]]
    log.info("Substances for which the prediction of the final model agrees with label: ", agrees_with_label=len(df_final_pred_agrees_with_label))

    average_pred_confidence_nrb = df_final_pred_agrees_with_label[df_final_pred_agrees_with_label["prediction_of_classifier_trained_on_final"] == 0]
    log.info("Average prediciton confidence that NRB when Final agrees with Test: ", nrb=average_pred_confidence_nrb["prediction_proba_nrb"].mean())
    average_pred_confidence_rb = df_final_pred_agrees_with_label[df_final_pred_agrees_with_label["prediction_of_classifier_trained_on_final"] == 1]
    log.info("Average prediciton confidence that RB when Final agrees with Test: ", rb=average_pred_confidence_rb["prediction_proba_rb"].mean())

    df_final_pred_does_not_agree_with_label = df[df["label"] != df["prediction_of_classifier_trained_on_final"]]
    average_pred_confidence_nrb = df_final_pred_does_not_agree_with_label[df_final_pred_does_not_agree_with_label["prediction_of_classifier_trained_on_final"] == 0]
    log.info("Average prediciton confidence that NRB when Final agrees with Test: ", nrb=average_pred_confidence_nrb["prediction_proba_nrb"].mean())
    average_pred_confidence_rb = df_final_pred_does_not_agree_with_label[df_final_pred_does_not_agree_with_label["prediction_of_classifier_trained_on_final"] == 1]
    log.info("Average prediciton confidence that RB when Final agrees with Test: ", rb=average_pred_confidence_rb["prediction_proba_rb"].mean())


def explain_classifier_with_shap(df_removed: pd.DataFrame) -> pd.DataFrame:
    log.info("Start loading model")
    classifier, _ = get_best_classifier_readded(train_new=args.train_new, random_seed=args.random_seed) 
    log.info("Finished loading model")
    explainer = shap.TreeExplainer(classifier)
    log.info("Finished creating explainer")

    column_to_structure_maccs = column_to_structure()
    df_removed.rename(columns = column_to_structure_maccs, inplace = True)

    x = create_input_classification(df_removed, include_speciation=False)
    shap_values = explainer.shap_values(x)
    log.info("Finished creating shap_values")

    df_x = pd.DataFrame(data=x)
    column_to_structure_maccs = column_to_structure()
    df_x.rename(columns = column_to_structure_maccs, inplace = True)

    num_top_features_shap = 5

    top_features_shap_rb = []
    top_feature_vals_shap_rb = []
    top_feature_vals_rb = []
    top_features_shap_nrb = []
    top_feature_vals_shap_nrb = []
    top_feature_vals_nrb = []
    for i in range(len(df_x)):
        shap_values_i = shap_values[i]
        
        top_feature_indices_rb = shap_values_i.argsort()[::-1][:num_top_features_shap]
        top_feature_indices_nrb = shap_values_i.argsort()[:num_top_features_shap]
        
        top_feature_names_shap_rb = df_x.columns[top_feature_indices_rb].tolist()
        top_feature_values_shap_rb = shap_values_i[top_feature_indices_rb].tolist()
        top_feature_values_rb = df_x.iloc[i, top_feature_indices_rb].tolist()
        top_feature_names_shap_nrb = df_x.columns[top_feature_indices_nrb].tolist()
        top_feature_values_shap_nrb = shap_values_i[top_feature_indices_nrb].tolist()
        top_feature_values_nrb = df_x.iloc[i, top_feature_indices_nrb].tolist()
        
        top_features_shap_rb.append(top_feature_names_shap_rb)
        top_feature_vals_shap_rb.append(top_feature_values_shap_rb)
        top_feature_vals_rb.append(top_feature_values_rb)
        top_features_shap_nrb.append(top_feature_names_shap_nrb)
        top_feature_vals_shap_nrb.append(top_feature_values_shap_nrb)
        top_feature_vals_nrb.append(top_feature_values_nrb)

    df_removed["top_features_shap_rb"] = top_features_shap_rb
    df_removed["top_feature_vals_shap_rb"] = top_feature_vals_shap_rb
    df_removed["top_feature_vals_rb"] = top_feature_vals_rb
    df_removed["top_features_shap_nrb"] = top_features_shap_nrb
    df_removed["top_feature_vals_shap_nrb"] = top_feature_vals_shap_nrb
    df_removed["top_feature_vals_nrb"] = top_feature_vals_nrb
    return df_removed


def check_if_registered_under_reach(df: pd.DataFrame) -> pd.DataFrame:
    df_checked = load_checked_organics6()
    df_checked.to_csv("datasets/df_checked.csv")

    def check_if_in_echa(row):
        cas = row["cas"]
        df_cas = df_checked[df_checked["cas"] == cas]
        if len(df_cas)>0: 
            return 1
        return 0
    df["registered_under_REACH_2021"] = df.apply(check_if_in_echa, axis=1)
    return df


def process_data() -> pd.DataFrame:
    df = pd.read_csv("datasets/curated_data/class_curated_scs_biowin_readded_removed_predicted.csv", index_col=0)
    df = df[["cas", "smiles", "inchi_from_smiles", "label", "miti_linear_label", "miti_non_linear_label", "prediction_class", "prediction_of_classifier_trained_on_final", "prediction_proba_nrb", "prediction_proba_rb"]]
    df_reg = pd.read_csv("datasets/data_processing/reg_curated_scs_no_metal.csv", index_col=0)
    df_reg = remove_smiles_with_incorrect_format(df=df_reg, col_name_smiles="smiles")
    df_reg = openbabel_convert(
        df=df_reg,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )
    lunghini = pd.read_csv("datasets/external_data/lunghini_added_cas.csv", index_col=0)
    print(lunghini.columns)

    def find_count_percentage(row):
        inchi_from_smiles = row["inchi_from_smiles"]
        cas = row["cas"]
        lunghini_match = lunghini[(lunghini["inchi_from_smiles"]==inchi_from_smiles) | (lunghini["cas_pubchem"].str.contains(cas))]
        in_lunghini = 1 if (len(lunghini_match)>0) else 0
        lunghini_label = None
        if in_lunghini:
            lunghini_label = lunghini_match.y_true.values[0]
        df_sub_reg = df_reg[df_reg["inchi_from_smiles"]==inchi_from_smiles]
        df_sub_reg_in = df_sub_reg[
            ((df_sub_reg['time_day']==28.0) & (df_sub_reg["endpoint"]=="ready")) |
            ((df_sub_reg['time_day']>28.0) & (df_sub_reg["biodegradation_percent"]<0.7) & (df_sub_reg["principle"]=="DOC Die Away")) | 
            ((df_sub_reg['time_day']>28.0) & (df_sub_reg["biodegradation_percent"]<0.6) & (df_sub_reg["principle"]!="DOC Die Away")) | 
            ((df_sub_reg['time_day']<28.0) & (df_sub_reg["endpoint"]=="ready") & (df_sub_reg["biodegradation_percent"]>0.7) & (df_sub_reg["principle"]=="DOC Die Away")) | 
            ((df_sub_reg['time_day']<28.0) & (df_sub_reg["endpoint"]=="ready") & (df_sub_reg["biodegradation_percent"]>0.6) & (df_sub_reg["principle"]!="DOC Die Away"))
            ]
        percentagesin = list(df_sub_reg_in.biodegradation_percent.values)
        timein = list(df_sub_reg_in.time_day.values)
        endpointin = list(df_sub_reg_in.endpoint.values)
        guidelinein = list(df_sub_reg_in.guideline.values)
        principlein = list(df_sub_reg_in.principle.values)

        df_sub_reg_not_in = df_sub_reg[~df_sub_reg.index.isin(df_sub_reg_in.index.values)]
        assert len(df_sub_reg_in) + len(df_sub_reg_not_in) == len(df_sub_reg)
        df_sub_reg_not_in.sort_values(by=['time_day'], ascending=False, inplace=True)
        percentages = list(df_sub_reg_not_in.biodegradation_percent.values)
        time = list(df_sub_reg_not_in.time_day.values)
        endpoint = list(df_sub_reg_not_in.endpoint.values)
        guideline = list(df_sub_reg_not_in.guideline.values)
        principle = list(df_sub_reg_not_in.principle.values)
        return pd.Series([percentagesin, timein, endpointin, guidelinein, principlein, percentages, time, endpoint, guideline, principle, lunghini_label])

    df[["biodeg_percentage_included", "time_included", "endpoint_included", "guideline_included", "principle_included", "biodeg_percentage_other", "time_day_other", "endpoint_other", "guideline_other", "principle_other", "lunghini_label_other"]] = df.apply(find_count_percentage, axis=1)

    df["molecular_weight"] = [ExactMolWt(Chem.MolFromSmiles(smiles)) for smiles in df["smiles"]]

    df = check_if_registered_under_reach(df)

    df = explain_classifier_with_shap(df)

    df_reordered = df[[
        "cas", 
        "smiles", 
        "inchi_from_smiles", 
        "registered_under_REACH_2021", 
        "molecular_weight", 
        "label", 
        "miti_linear_label", 
        "miti_non_linear_label", 
        "prediction_class", 
        "prediction_of_classifier_trained_on_final", 
        "prediction_proba_nrb", 
        "prediction_proba_rb", 
        "biodeg_percentage_included", 
        "time_included", 
        "endpoint_included", 
        "guideline_included", 
        "principle_included", 
        "biodeg_percentage_other", 
        "time_day_other", 
        "endpoint_other", 
        "guideline_other", 
        "principle_other", 
        "lunghini_label_other", 
        "top_features_shap_rb", 
        "top_feature_vals_shap_rb", 
        "top_feature_vals_rb", 
        "top_features_shap_nrb", 
        "top_feature_vals_shap_nrb", 
        "top_feature_vals_nrb"
    ]].copy()
    df_reordered.rename(columns = {'prediction_class':'prediction_third_classifier'}, inplace = True)
    df_reordered.to_csv("datasets/curated_data/class_curated_scs_biowin_readded_removed_predicted.csv")
    df_reordered_str = df_reordered.copy()
    for col in ["biodeg_percentage_included", "time_included", "endpoint_included", "guideline_included", "principle_included", "biodeg_percentage_other", "time_day_other", "endpoint_other", "guideline_other", "principle_other", "top_features_shap_rb", "top_feature_vals_shap_rb", "top_feature_vals_rb", "top_features_shap_nrb", "top_feature_vals_shap_nrb", "top_feature_vals_nrb"]:
        df_reordered_str[col] = [', '.join(map(str, l)) for l in df_reordered_str[col]]
    df_reordered_str.to_excel("datasets/curated_data/class_curated_scs_biowin_readded_removed_predicted.xlsx")
    return df_reordered


if __name__ == "__main__":
    make_prediction()
    analyse_predictions()
    df_removed = process_data()

