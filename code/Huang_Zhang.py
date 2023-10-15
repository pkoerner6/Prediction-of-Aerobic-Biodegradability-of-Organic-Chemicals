""" Replication of the paper by Huang and Zhang """
""" Requirements are in requirements_huang_zhang_replication.tex"""

import argparse
import pickle
from typing import List
import sys
import os
import pandas as pd
import structlog
from rdkit import RDLogger
from xgboost import XGBClassifier, XGBRegressor

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from biodegradation.processing_functions import openbabel_convert
from biodegradation.processing_functions import remove_smiles_with_incorrect_format
from biodegradation.processing_functions import load_class_data_paper
from biodegradation.processing_functions import load_regression_df
from biodegradation.processing_functions import create_input_regression
from biodegradation.processing_functions import create_input_classification
from biodegradation.processing_functions import load_and_process_echa_additional
from biodegradation.processing_functions import convert_regression_df_to_input
from biodegradation.ml_functions import print_class_results
from biodegradation.ml_functions import run_XGBClassifier_Huang_Zhang
from biodegradation.ml_functions import run_XGBRegressor_Huang_Zhang
from biodegradation.ml_functions import get_class_results
from biodegradation.ml_functions import analyze_regression_results_and_plot
from biodegradation.ml_functions import plot_classification_bar
from biodegradation.ml_functions import get_Huang_Zhang_regression_parameters
from biodegradation.ml_functions import get_balanced_data_adasyn


RDLogger.DisableLog("rdApp.*")  # Disable warnings from rdkit
log = structlog.get_logger()

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    type=str,
    choices=["classification", "regression", "both"],
    default="both",
    help="Training modus: either classification or regression",
)
parser.add_argument(
    "--data_info",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Whether to print data information",
)
parser.add_argument(
    "--model_selection",
    type=str,
    choices=["retrained", "paper", "both"],
    help="Which model to use",
)
parser.add_argument("--random_state", type=int, default=42)
parser.add_argument("--nsplits", type=int, default=5)
parser.add_argument(
    "--reg_data",
    type=str,
    choices=["paper", "both"],
    default="paper",
    help="Which regression dataset to use",
)
parser.add_argument(
    "--class_data",
    type=str,
    choices=["paper", "both"],
    default="paper",
    help="Which classification dataset to use",
)
parser.add_argument(
    "--chemical_speciation_included",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Whether to include chemical speciation",
)
parser.add_argument(
    "--run_all",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="",
)
args = parser.parse_args()


def print_composition_of_dataframe(df: pd.DataFrame, inputs: List[str]) -> None:
    def print_number_of_entries(df: pd.DataFrame, input: str) -> None:
        for element in df[input].unique():
            log.info(
                f"Entries with {input} {element}",
                entries=len(df[df[input] == element]),
            )

    for input in inputs:
        print_number_of_entries(df, input)


def info_dataset(df: pd.DataFrame, dataset_type: str, inchi_col_name: str) -> None:
    log.info("Number of datapoints in dataset", datapoints=len(df))
    log.info("Unique CAS Numbers in dataset", unique_cas=df["cas"].nunique())
    log.info("Unique SMILES in dataset", unique_smiles=df["smiles"].nunique())
    df_correct_smiles = remove_smiles_with_incorrect_format(df=df, col_name_smiles="smiles")
    df_inchi = openbabel_convert(
        df=df_correct_smiles,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )
    log.info(
        "Unique InChI in dataset",
        unique_inchi_huang_zhang=df_inchi[inchi_col_name].nunique(),
    )
    type_to_inputs = {
        "regression": ["reliability", "endpoint", "principle", "guideline"],
        "classification": ["y_true"],
    }
    print_composition_of_dataframe(df=df, inputs=type_to_inputs[dataset_type])


def load_model(model_path: str):
    model = pickle.load(open(model_path, "rb"))
    return model


def run_xgbregressor_on_additional_echa_data(
    model,
    figure_name: str,
    figure_title: str,
    include_speciation: bool,
) -> None:
    df_additional, _, _, _ = load_and_process_echa_additional()
    log.info("Entries in additional regression echa data", df_additional_regression=len(df_additional))
    if args.data_info:
        log.info("Analysing additional ECHA data for regression")
        info_dataset(df_additional, dataset_type="regression", inchi_col_name="inchi_from_smiles")
    x_input = create_input_regression(df=df_additional, include_speciation=include_speciation)
    df_additional["prediction"] = model.predict(x_input)
    analyze_regression_results_and_plot(
        df=df_additional,
        figure_name=figure_name,
        figure_title=figure_title,
    )


def run_xgbclassifier_on_additional_echa_data(
    model,
    figure_name: str,
    figure_title: str,
    include_speciation: bool,
) -> None:
    _, _, df_additional, _ = load_and_process_echa_additional()
    log.info("Number of additional datapoints for classification: ", entries=len(df_additional))

    if args.data_info:
        log.info("Analysing additional ECHA data for classification")
        info_dataset(
            df_additional,
            dataset_type="classification",
            inchi_col_name="inchi_from_smiles",
        )
    x_input = create_input_classification(df_additional, include_speciation=include_speciation)
    df_additional["prediction"] = model.predict(x_input)
    accuracy, f1, sensitivity, specificity = get_class_results(
        df_additional["y_true"].to_numpy(), df_additional["prediction"].to_numpy()
    )
    accuracy = print_class_results(
        accuracy=accuracy,
        sensitivity=sensitivity,
        specificity=specificity,
        f1=f1,
    )
    plot_classification_bar(
        df=df_additional,
        accuracy=accuracy,
        figure_name=f"{figure_name}_tested_on_additional",
        figure_title=f"{figure_title} on external ECHA data",
    )


def create_regression_input(df: pd.DataFrame, include_speciation: bool) -> pd.DataFrame:
    if not include_speciation:
        df = df[
            [
                "cas",
                "smiles",
                "reliability",
                "endpoint",
                "guideline",
                "principle",
                "time_day",
                "biodegradation_percent",
            ]
        ]
    df = convert_regression_df_to_input(df=df)
    return df


def train_regressor_on_all_data_and_test_on_additional(
    df: pd.DataFrame, include_speciation: bool, figure_name: str, figure_title: str
) -> None:
    params = get_Huang_Zhang_regression_parameters()
    model = XGBRegressor(**params)
    x = create_input_regression(df=df, include_speciation=include_speciation)
    y = df["biodegradation_percent"]
    model.fit(x, y)
    run_xgbregressor_on_additional_echa_data(
        model,
        figure_name=figure_name,
        figure_title=figure_title,
        include_speciation=include_speciation,
    )


def train_classifier_on_all_data_and_test_on_additional(
    df: pd.DataFrame,
    random_seed: int,
    include_speciation: bool,
    figure_name: str,
    figure_title: str,
) -> None:
    model = XGBClassifier()
    x = create_input_classification(df, include_speciation=include_speciation)
    y = df["y_true"]
    x_balanced, y_balanced = get_balanced_data_adasyn(random_seed=random_seed, x=x, y=y)
    model.fit(x_balanced, y_balanced)
    run_xgbclassifier_on_additional_echa_data(
        model=model,
        figure_name=figure_name,
        figure_title=figure_title,
        include_speciation=include_speciation,
    )


def run_regression(include_speciation: bool) -> None:
    if (args.reg_data == "paper") or (args.reg_data == "both"):
        df_paper_reg = load_regression_df()
        df_paper_reg = create_regression_input(df=df_paper_reg, include_speciation=include_speciation)

        if not include_speciation and ((args.model_selection == "paper") or (args.model_selection == "both")):
            log.info(" \n Loaded paper regression model tested on additional data")
            model = load_model(
                model_path="biodegradation/Huang_Zhang_Model_files_and_sample_codes/AB_XGBRegression_model_21.pkl"
            )  # Requires _reggboost version 1.4.0
            run_xgbregressor_on_additional_echa_data(
                model,
                figure_name="loaded_paper_regression_model_tested_on_additional",
                figure_title="Paper regression model on external ECHA data",
                include_speciation=False,  # True does not work! Interesting because that means the provided model was trained without the extra features
            )
        if (args.model_selection == "retrained") or (args.model_selection == "both"):
            log.info(
                f" \n Newly trained regression model trained on paper data, {args.nsplits}-fold, tested on paper test dataset"
            )
            run_XGBRegressor_Huang_Zhang(
                df=df_paper_reg,
                column_for_grouping="smiles",
                random_seed=args.random_state,
                nsplits=args.nsplits,
                include_speciation=include_speciation,
                fixed_testset=False,
                df_smallest=df_paper_reg,
                dataset_name="df_regression_paper",
            )
            log.info(" \n Newly trained regression model trained on all paper data, tested on additional data")
            train_regressor_on_all_data_and_test_on_additional(
                df=df_paper_reg,
                include_speciation=include_speciation,
                figure_name="retrained_regressor_paper_data_tested_on_additional",
                figure_title="XGBRegressor trained on paper data tested on external ECHA data",
            )
        if args.data_info:
            log.info(" \n Analysing paper regression data")
            info_dataset(
                df_paper_reg,
                dataset_type="regression",
                inchi_col_name="inchi_from_smiles",
            )


def create_paper_classification(include_speciation: bool) -> pd.DataFrame:
    _, _, df_class = load_class_data_paper()
    if not include_speciation:
        df_class = df_class[["name", "name_type", "cas", "smiles", "y_true"]]
    return df_class


def run_classification(include_speciation: bool) -> None:
    if (args.class_data == "paper") or (args.class_data == "both"):
        df_paper_class = create_paper_classification(include_speciation=include_speciation)

        if not include_speciation and ((args.model_selection == "paper") or (args.model_selection == "both")):
            log.info(" \n Paper loaded classification model, tested on additional data")
            model = load_model(
                "biodegradation/Huang_Zhang_Model_files_and_sample_codes/AB_XGBClassification_model_21.pkl"
            )  # Requires xgboost version 1.2.0
            run_xgbclassifier_on_additional_echa_data(
                model=model,
                figure_name="loaded_paper_classification_model",
                figure_title="Paper classification model on external ECHA data",
                include_speciation=False,
            )
        if (args.model_selection == "retrained") or (args.model_selection == "both"):
            log.info(
                f" \n Newly trained classification model trained on paper data, {args.nsplits}-fold, tested on test set"
            )
            run_XGBClassifier_Huang_Zhang(
                df=df_paper_class,
                random_seed=args.random_state,
                nsplits=args.nsplits,
                use_adasyn=True,
                include_speciation=include_speciation,
                fixed_testset=False,
                df_smallest=df_paper_class,
                dataset_name="df_classification_paper",
            )
            log.info(" \n Newly trained classification model trained on all paper data, tested on additional data")
            train_classifier_on_all_data_and_test_on_additional(
                df=df_paper_class,
                random_seed=args.random_state,
                include_speciation=include_speciation,
                figure_name="classification_model_retrained_on_paper_data_tested",
                figure_title="Classification model retrained on paper data tested on additional",
            )
        if args.data_info:
            log.info(" \n Analysing paper classification data")
            info_dataset(
                df_paper_class,
                dataset_type="classification",
                inchi_col_name="inchi_from_smiles",
            )


if __name__ == "__main__":
    include_speciation = args.chemical_speciation_included

    if (args.mode == "regression") or (args.mode == "both"):
        run_regression(
            include_speciation=include_speciation,
        )
    if (args.mode == "classification") or (args.mode == "both"):
        run_classification(include_speciation=include_speciation)