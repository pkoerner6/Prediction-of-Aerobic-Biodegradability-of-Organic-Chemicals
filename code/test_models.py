import argparse
import pandas as pd
import numpy as np
import structlog
import sys
import os

log = structlog.get_logger()
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, XGBRegressor
from rdkit.Chem import PandasTools
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from biodegradation.processing_functions import create_input_classification
from biodegradation.processing_functions import create_input_regression
from biodegradation.processing_functions import load_external_class_test_dfs_additional_to_Huang_readded
from biodegradation.processing_functions import load_external_reg_test_dfs_additional_to_Huang_readded
from biodegradation.processing_functions import load_class_data_paper
from biodegradation.processing_functions import load_regression_df
from biodegradation.ml_functions import train_class_model_on_all_data
from biodegradation.ml_functions import train_reg_model_on_all_data
from biodegradation.ml_functions import get_class_results
from biodegradation.ml_functions import plot_regression_error
from biodegradation.ml_functions import get_lazy_xgbr_parameters
from biodegradation.ml_functions import get_lazy_xgbc_parameters


parser = argparse.ArgumentParser()

parser.add_argument(
    "--random_seed",
    type=int,
    default=42,
    help="Select the random seed",
)
args = parser.parse_args()


def analyse_composition_of_additional_datasets():
    (
        test_datasets,
        test_datasets_env,
        test_datasets_in_ad,
        test_datasets_env_in_ad,
    ) = load_external_class_test_dfs_additional_to_Huang_readded()
    (
        echa_additional_reg,
        echa_additional_reg_env,
        echa_additional_reg_in_ad,
        echa_additional_reg_env_in_ad,
    ) = load_external_reg_test_dfs_additional_to_Huang_readded()

    test_datasets["echa_additional_reg"] = echa_additional_reg
    test_datasets_env["echa_additional_reg_env"] = echa_additional_reg_env
    test_datasets_in_ad["echa_additional_reg_in_ad"] = echa_additional_reg_in_ad
    test_datasets_env_in_ad["echa_additional_reg_env_in_ad"] = echa_additional_reg_env_in_ad

    test_dict_name_to_dict = {
        "test_datasets": test_datasets,
        "test_datasets_env": test_datasets_env,
        "test_datasets_in_ad": test_datasets_in_ad,
        "test_datasets_env_in_ad": test_datasets_env_in_ad,
    }

    for test_dict_name, test_datasets in test_dict_name_to_dict.items():
        log.info(f"Analysing {test_dict_name}")
        for df_name, df in test_datasets.items():
            log.info(f"\n {df_name}")
            with_env = "_env" if "env" in test_dict_name else ""
            df.to_csv(f"biodegradation/dataframes/external_test_set_{df_name}{with_env}.csv")

            # Molecular weight
            df["molecular_weight"] = [ExactMolWt(Chem.MolFromSmiles(smiles)) for smiles in df["smiles"]]
            groupamount: List[int] = []
            bins = [0, 250, 500, 750, 1000, 2000]
            bin_names: List[str] = []
            df_grouped = df.groupby(pd.cut(df["molecular_weight"], bins=bins, include_lowest=True))[
                "molecular_weight"
            ].count()
            for idx, index in enumerate(df_grouped.index):
                groupamount.append(df_grouped[index])
                label = f"{int(index.left)} to {int(index.right)}"
                if idx == 0:  # need to change label because the first bin includes the left
                    label = f"0 to {int(index.right)}"
                if label not in bin_names:
                    bin_names.append(label)
            for i, label in enumerate(bin_names):
                if len(df) > 0:
                    group_percent = (groupamount[i] / len(df)) * 100
                    log.info(f"Molecular weight {label} Da", weight="{:.1f}%".format(group_percent))
                else:
                    log.warn("The dataset has no entries")

            # Halogens
            entries_with_f = len(df[df["smiles"].str.contains("F")])
            entries_with_br = len(df[df["smiles"].str.contains("Br")])
            entries_with_cl = len(df[df["smiles"].str.contains("Cl")])
            entries_with_all_halogens = len(
                df[
                    (df["smiles"].str.contains("F"))
                    | (df["smiles"].str.contains("Br"))
                    | (df["smiles"].str.contains("Cl"))
                ]
            )
            if len(df) > 0:
                log.info(
                    f"Relative number of halogens",
                    F="{:.1f}%".format((entries_with_f / len(df)) * 100),
                    Br="{:.1f}%".format((entries_with_br / len(df)) * 100),
                    Cl="{:.1f}%".format((entries_with_cl / len(df)) * 100),
                    all="{:.1f}%".format((entries_with_all_halogens / len(df)) * 100),
                )
            else:
                log.warn("The dataset has no entries")

            # Label distribution
            if "reg" in df_name:
                groupamount: List[int] = []
                bins = [0, 20, 40, 60, 80, 100]
                bin_names: List[str] = []
                df["biodegradation_percent"] = df["biodegradation_percent"] * 100
                df_grouped = df.groupby(pd.cut(df["biodegradation_percent"], bins=bins, include_lowest=True))[
                    "biodegradation_percent"
                ].count()
                for idx, index in enumerate(df_grouped.index):
                    groupamount.append(df_grouped[index])
                    label = f"{int(index.left)} to {int(index.right)}"
                    if idx == 0:  # need to change label because the first bin includes the left
                        label = f"0 to {int(index.right)}"
                    if label not in bin_names:
                        bin_names.append(label)
                for i, label in enumerate(bin_names):
                    if len(df) > 0:
                        group_percent = (groupamount[i] / len(df)) * 100
                        log.info(f"Biodegradation {label}", biodeg="{:.1f}%".format(group_percent))
                    else:
                        log.warn("The dataset has no entries")
            else:
                if len(df) > 0:
                    nrb = df[df["label"] == 0]
                    rb = df[df["label"] == 1]
                    log.info(
                        f"Relative classes",
                        NRB="{:.1f}%".format((len(nrb) / len(df)) * 100),
                        RB="{:.1f}%".format((len(rb) / len(df)) * 100),
                    )
                else:
                    log.warn("The dataset has no entries")
    return


def validate_results_classifier_on_additional_data() -> None:
    log.info("\n \n Validating classifiers on additional data")
    _, _, df_class = load_class_data_paper()
    df_readded = pd.read_csv(
        "biodegradation/dataframes/improved_data/class_improved_env_biowin_both_readded.csv", index_col=0
    )
    datasets_train = {"df_paper": df_class, "df_readded": df_readded}

    opera = pd.read_csv("biodegradation/dataframes/external_test_set_OPERA_data.csv", index_col=0)
    opera_env = pd.read_csv("biodegradation/dataframes/external_test_set_OPERA_data_env.csv", index_col=0)
    opera_in_ad = pd.read_csv("biodegradation/dataframes/external_test_set_OPERA_data_in_ad.csv", index_col=0)
    opera_env_in_ad = pd.read_csv("biodegradation/dataframes/external_test_set_OPERA_data_in_ad_env.csv", index_col=0)
    biowin_original_data = pd.read_csv(
        "biodegradation/dataframes/external_test_set_biowin_original_data.csv", index_col=0
    )
    biowin_original_data_env = pd.read_csv(
        "biodegradation/dataframes/external_test_set_biowin_original_data_env.csv", index_col=0
    )
    biowin_original_data_in_ad = pd.read_csv(
        "biodegradation/dataframes/external_test_set_biowin_original_data_in_ad.csv", index_col=0
    )
    biowin_original_data_env_in_ad = pd.read_csv(
        "biodegradation/dataframes/external_test_set_biowin_original_data_in_ad_env.csv", index_col=0
    )
    echa_additional = pd.read_csv("biodegradation/dataframes/external_test_set_echa_additional.csv", index_col=0)
    echa_additional_env = pd.read_csv(
        "biodegradation/dataframes/external_test_set_echa_additional_env.csv", index_col=0
    )
    echa_additional_in_ad = pd.read_csv(
        "biodegradation/dataframes/external_test_set_echa_additional_in_ad.csv", index_col=0
    )
    echa_additional_env_in_ad = pd.read_csv(
        "biodegradation/dataframes/external_test_set_echa_additional_in_ad_env.csv", index_col=0
    )

    test_datasets = {
        "OPERA_data": opera,
        "biowin_original_data": biowin_original_data,
        "echa_additional": echa_additional,
    }
    test_datasets_env = {
        "OPERA_data": opera_env,
        "biowin_original_data": biowin_original_data_env,
        "echa_additional": echa_additional_env,
    }
    test_datasets_in_ad = {
        "OPERA_data": opera_in_ad,
        "biowin_original_data": biowin_original_data_in_ad,
        "echa_additional": echa_additional_in_ad,
    }
    test_datasets_env_in_ad = {
        "OPERA_data": opera_env_in_ad,
        "biowin_original_data": biowin_original_data_env_in_ad,
        "echa_additional": echa_additional_env_in_ad,
    }

    test_datasets_not_and_in_ad = {
        "not_in_ad": [test_datasets, test_datasets_env],
        "in_ad": [test_datasets_in_ad, test_datasets_env_in_ad],
    }

    for ad_type, datasets in test_datasets_not_and_in_ad.items():
        log.info(f"\n Validating datasets {ad_type}")
        for (df_test_name, df_test), (df_test_env_name, df_test_env) in zip(datasets[0].items(), datasets[1].items()):
            if ad_type == "in_ad":
                df_test.to_csv(f"biodegradation/dataframes/external_test_{df_test_name}_in_ad_class.csv")
            if df_test_name == "OPERA_data":
                log.info(" \n Validate on additional OPERA data")
            elif df_test_name == "biowin_original_data":
                log.info(" \n Validate on BIOWIN1 and BIOWIN2 data")
            elif df_test_name == "echa_additional":
                log.info(" \n Validate on additional ECHA data")

            if len(df_test) == 0:
                log.warn("Dataset has no entries")
                continue

            df_test["label"] = df_test["label"].astype(int)
            df_test_env["label"] = df_test_env["label"].astype(int)

            for df_name, df in datasets_train.items():
                log.info(f"Dataset that classifier was trained on: {df_name}", entries=len(df))
                test_name = df_test_name if df_name == "df_paper" else df_test_env_name
                test = df_test if df_name == "df_paper" else df_test_env

                best_params = get_lazy_xgbc_parameters()
                model = XGBClassifier(**best_params)
                model_class = train_class_model_on_all_data(
                    df=df,
                    random_seed=args.random_seed,
                    use_adasyn=True,
                    include_speciation=False,
                    model_with_best_hyperparams=model,
                )
                x = create_input_classification(test, include_speciation=False)
                test[f"{df_name}_prediction_class"] = model_class.predict(x)

                accuracy, f1, sensitivity, specificity = get_class_results(
                    true=test.label, pred=test[f"{df_name}_prediction_class"]
                )
                log.info(f"Accuracy when classifier was trained on {df_name}", accuracy="{:.1f}".format(accuracy * 100))
                log.info(f"F1 when classifier was trained on {df_name}", f1="{:.1f}".format(f1 * 100))
                log.info(
                    f"Sensitivity when classifier was trained on {df_name}",
                    sensitivity="{:.1f}".format(sensitivity * 100),
                )
                log.info(
                    f"Specificity when classifier was trained on {df_name}",
                    specificity="{:.1f}".format(specificity * 100),
                )

            if test_name == "biowin_original_data":

                def label_biowin(row):
                    biowin1 = row["biowin1"]
                    biowin2 = row["biowin2"]
                    biowin1_label = 0
                    biowin2_label = 0
                    if biowin1 > 0.5:
                        biowin1_label = 1
                    if biowin2 > 0.5:
                        biowin2_label = 1
                    return pd.Series([biowin1_label, biowin2_label])

                test[["biowin1_label", "biowin2_label"]] = test.apply(label_biowin, axis=1)

            test.to_csv(f"biodegradation/dataframes/{test_name}_{ad_type}_class_finetuned_predicted.csv")


def run_regressor(test: pd.DataFrame, test_name: str, train_name: str, df: pd.DataFrame):
    best_params = get_lazy_xgbr_parameters()
    model = XGBRegressor(**best_params)
    model_reg = train_reg_model_on_all_data(
        df=df,
        random_seed=args.random_seed,
        include_speciation=False,
        model_with_best_hyperparams=model,
    )
    x = create_input_regression(test, include_speciation=False)
    test["prediction"] = model_reg.predict(x)

    rmse = mean_squared_error(test.biodegradation_percent, test["prediction"], squared=False)
    mae = mean_absolute_error(test.biodegradation_percent, test["prediction"])
    r2 = r2_score(test.biodegradation_percent, test["prediction"])

    log.info(f"RMSE when regressor was trained on {train_name}", rmse="{:.2f}".format(rmse))
    log.info(f"MAE when regressor was trained on {train_name}", mae="{:.2f}".format(mae))
    log.info(f"R2 when regressor was trained on {train_name}", r2="{:.2f}".format(r2))

    test.to_csv(f"biodegradation/dataframes/{test_name}_predicted_by_model_trained_on_{train_name}.csv")
    plot_regression_error(
        df=test, figure_name=f"regressor_trained_on_{train_name}_tested_on_{test_name}", figure_title=""
    )
    return test


def validate_results_regressor_on_additional_data() -> None:
    reg = load_regression_df()
    df_readded = pd.read_csv(
        "biodegradation/dataframes/improved_data/reg_improved_env_biowin_both_readded.csv", index_col=0
    )

    datasets_train = {"df_paper": reg, "df_readded": df_readded}

    df_echa_additional = pd.read_csv(f"biodegradation/dataframes/external_test_set_echa_additional_reg.csv")
    df_echa_additional_env = pd.read_csv(f"biodegradation/dataframes/external_test_set_echa_additional_reg_env.csv")
    df_echa_additional_in_ad = pd.read_csv(f"biodegradation/dataframes/external_test_set_echa_additional_reg_in_ad.csv")
    df_echa_additional_env_in_ad = pd.read_csv(
        f"biodegradation/dataframes/external_test_set_echa_additional_reg_env_in_ad.csv"
    )

    for df_name, df in datasets_train.items():
        log.info("Dataset that classifier was trained on", name=df_name, entries=len(df))

        test = df_echa_additional if df_name == "df_paper" else df_echa_additional_env
        if df_name == "df_paper":
            log.info("All data")
            test = run_regressor(test=df_echa_additional, test_name="df_echa_additional_reg", train_name=df_name, df=df)
            log.info("Only data in AD")
            test = run_regressor(
                test=df_echa_additional_in_ad, test_name="df_echa_additional_in_ad_reg", train_name=df_name, df=df
            )
        else:
            log.info("All data")
            test = run_regressor(
                test=df_echa_additional_env, test_name="df_echa_additional_env_reg", train_name=df_name, df=df
            )
            log.info("Only data in AD")
            test = run_regressor(
                test=df_echa_additional_env_in_ad,
                test_name="df_echa_additional_env_in_ad_reg",
                train_name=df_name,
                df=df,
            )


if __name__ == "__main__":
    analyse_composition_of_additional_datasets()
    validate_results_classifier_on_additional_data()
    validate_results_regressor_on_additional_data()
