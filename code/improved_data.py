"""XGBoost models with hyperparameters from HUang and Zhang will be trained on different improved datasets"""
"""Prior to this file, the files data_processing.py and creating_datasets.py need to be run"""

import pandas as pd
import numpy as np
import structlog
from typing import List, Tuple
import matplotlib.pyplot as plt
import sys
import os
import argparse
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from processing_functions import get_class_datasets
from processing_functions import get_regression_datasets
from processing_functions import get_comparison_datasets_regression
from processing_functions import get_comparison_datasets_classification
from processing_functions import create_input_classification
from processing_functions import remove_smiles_with_incorrect_format
from processing_functions import openbabel_convert
from processing_functions import get_inchi_main_layer
from ml_functions import run_XGBClassifier_Huang_Zhang
from ml_functions import run_XGBRegressor_Huang_Zhang
from ml_functions import train_XGBClassifier_Huang_Zhang_on_all_data
from ml_functions import get_class_results
from ml_functions import print_class_results
from ml_functions import get_balanced_data_adasyn

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
    "--nsplits",
    type=int,
    default=5,
    help="Number of KFold splits",
)
parser.add_argument(
    "--random_seed",
    type=int,
    default=42,
)
parser.add_argument(
    "--train_new",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Whether to train the models again or use previous results to just plot",
)
parser.add_argument(
    "--fixed_testset",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Whether to a fixed testset created from the smallest dataset",
)
parser.add_argument(
    "--progress_or_comparison",
    type=str,
    help="Whether to run and plot all dfs in the order of the progress made or to run dfs so that improvement from smiles-cas improvement and env. smiles is shown",
)
args = parser.parse_args()


def plot_results_with_stanard_deviation(
    all_data: List[np.ndarray],
    labels: List[str],
    colors: List[str],
    nsplits: int,
    title: str,
    mode: str,
    seed: int,
    plot_with_paper: bool,
    save_ending: str,
) -> None:

    plt.figure(figsize=(15, 4))

    bplot = plt.boxplot(all_data, vert=True, patch_artist=True, labels=labels, meanline=True, showmeans=True)

    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)

    if plot_with_paper:
        plt.plot(1, np.mean(all_data[0]), marker="o", markersize=14)

    plt.xlabel("Datasets", fontsize=22)
    ylabel = f"{title} (%)" if title == "Accuracy" else title
    plt.ylabel(ylabel, fontsize=22)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.grid(axis="y")

    fixed = "_fixed_testset" if args.fixed_testset else ""
    title = "R2" if title == "$\mathregular{R^{2}}$" else title
    plt.savefig(
        f"figures/{nsplits}fold_improved_{title}_seed{seed}_paper_hyperparameter_{save_ending}{fixed}.png"
    )
    plt.close()


def plot_results(
    all_data: List[np.ndarray],
    labels: List[str],
    colors: List[str],
    title: str,
    seed: int,
    save_ending: str,
) -> None:
    
    labels = [
        "Huang-Dataset \n reported",
        "Huang-Dataset \n replicated",
        "$\mathregular{Curated_{SCS}}$",
        "$\mathregular{Curated_{BIOWIN}}$",
        "$\mathregular{Curated_{FINAL}}$",
    ]
    colors = [
        "grey",
        "plum",
        "royalblue",
        "lightgreen",
        "seagreen",
    ]

    plt.figure(figsize=(12, 4))

    for position in range(len(colors)):
        plt.plot(position+1, all_data[position], marker="o", markersize=14, color=colors[position])

    plt.xlabel("Datasets", fontsize=22)
    ylabel = f"{title} (%)" if title == "Accuracy" else title
    plt.ylabel(ylabel, fontsize=22)

    plt.xticks(ticks=[1, 2, 3, 4, 5], fontsize=18, labels=labels)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.grid(axis="y")

    fixed = "_fixed_testset" if args.fixed_testset else ""
    title = "R2" if title == "$\mathregular{R^{2}}$" else title
    plt.savefig(
        f"figures/{title}_seed{seed}_paper_hyperparameter_{save_ending}{fixed}.png"
    )
    plt.close()


def train_regression_models(
    comparison_or_progress: str,
    column_for_grouping: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

    if comparison_or_progress == "progress":
        datasets = get_regression_datasets()
        df_smallest = datasets["df_curated_scs_biowin_readded"].copy()
    if comparison_or_progress == "comparison":
        datasets = get_comparison_datasets_regression(mode="regression", include_speciation=True)
        df_smallest = datasets["df_curated_scs_biowin"].copy()
    rmse: List[np.ndarray] = [np.asarray([])] * len(datasets)
    mae: List[np.ndarray] = [np.asarray([])] * len(datasets)
    r2: List[np.ndarray] = [np.asarray([])] * len(datasets)
    mse: List[np.ndarray] = [np.asarray([])] * len(datasets)

    for indx, (dataset_name, dataset) in enumerate(datasets.items()):
        log.info(f"Entries in {dataset_name}", entries=len(dataset))

        speciation = dataset_name == "df_paper_with_speciation"

        if (
            (dataset_name == "df_paper")
            or (dataset_name == "df_paper_biowin")
        ):
            column_for_grouping = "smiles"  # in the paper they group by smiles, not cas
        lst_rmse_paper, lst_mae_paper, lst_r2_paper, lst_mse_paper = run_XGBRegressor_Huang_Zhang(
            df=dataset,
            column_for_grouping=column_for_grouping,
            random_seed=args.random_seed,
            nsplits=args.nsplits,
            include_speciation=speciation,
            fixed_testset=args.fixed_testset,
            df_smallest=df_smallest,
            dataset_name=dataset_name,
        )
        rmse[indx] = np.asarray(lst_rmse_paper)
        mae[indx] = np.asarray(lst_mae_paper)
        r2[indx] = np.asarray(lst_r2_paper)
        mse[indx] = np.asarray(lst_mse_paper)
    return rmse, mae, r2, mse


def train_classification_models(
    comparison_or_progress: str,
    with_lunghini: bool,
    use_adasyn: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

    if comparison_or_progress == "progress":
        datasets = get_class_datasets()
        df_smallest = datasets["df_curated_scs_biowin_readded"].copy() # TODO df_curated_scs_biowin_readded, df_curated_scs
    if comparison_or_progress == "comparison":
        datasets = get_comparison_datasets_classification(
            mode="classification",
            include_speciation=True,
            with_lunghini=with_lunghini,
            create_new=True,
        )
        df_smallest = datasets["df_curated_scs_biowin"].copy() 
    accuracy: List[np.ndarray] = [np.asarray([])] * len(datasets)
    f1: List[np.ndarray] = [np.asarray([])] * len(datasets)
    sensitivity: List[np.ndarray] = [np.asarray([])] * len(datasets)
    specificity: List[np.ndarray] = [np.asarray([])] * len(datasets)

    for indx, (dataset_name, dataset) in enumerate(datasets.items()):
        log.info(f"Entries in {dataset_name}", entries=len(dataset))

        speciation = True if dataset_name == "df_paper_with_speciation" else False

        lst_accu_paper, lst_f1_paper, lst_sensitivity_paper, lst_specificity_paper = run_XGBClassifier_Huang_Zhang(
            df=dataset,
            random_seed=args.random_seed,
            nsplits=args.nsplits,
            use_adasyn=use_adasyn,
            include_speciation=speciation,
            fixed_testset=args.fixed_testset,
            df_smallest=df_smallest,
            dataset_name=dataset_name,
        )
        accuracy[indx] = np.asarray(lst_accu_paper)
        f1[indx] = np.asarray(lst_f1_paper)
        sensitivity[indx] = np.asarray(lst_sensitivity_paper)
        specificity[indx] = np.asarray(lst_specificity_paper)
    return accuracy, f1, sensitivity, specificity


def train_classification_models_test_set_multiple(
    use_adasyn: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

    datasets = get_class_datasets()
    test_set = pd.read_csv("datasets/curated_data/class_curated_scs_multiple.csv", index_col=0)

    accuracy: List[np.ndarray] = [np.asarray([])] * len(datasets)
    f1: List[np.ndarray] = [np.asarray([])] * len(datasets)
    sensitivity: List[np.ndarray] = [np.asarray([])] * len(datasets)
    specificity: List[np.ndarray] = [np.asarray([])] * len(datasets)

    # test an all of test set
    for indx, (dataset_name, dataset) in enumerate(datasets.items()):
        print("") # TODO
        log.info(f"Entries in {dataset_name}", entries=len(dataset))

        # remove substances from test set from training set
        df_smiles_correct = remove_smiles_with_incorrect_format(df=dataset, col_name_smiles="smiles")
        dataset = openbabel_convert(
            df=df_smiles_correct,
            input_type="smiles",
            column_name_input="smiles",
            output_type="inchi",
        )
        df_test_all = test_set.copy()
        df_train = dataset[~dataset["cas"].isin(df_test_all["cas"])]
        df_train = df_train[~df_train["inchi_from_smiles"].isin(df_test_all["inchi_from_smiles"])]
        if len(df_test_all) > len(dataset)-len(df_train):
            df_test_all = get_inchi_main_layer(df=df_test_all, inchi_col="inchi_from_smiles", layers=3) 
            df_train = get_inchi_main_layer(df=df_train, inchi_col="inchi_from_smiles", layers=3)
            df_train = df_train[~df_train["inchi_from_smiles_main_layer"].isin(df_test_all["inchi_from_smiles_main_layer"])]
        log.info("Entries train set, test set, dataset before - after removing test set", df_train=len(df_train), df_test=len(df_test_all), before_minus_after=len(dataset)-len(df_train))
        assert len(df_test_all) <= len(dataset)-len(df_train)

        model = train_XGBClassifier_Huang_Zhang_on_all_data(df=df_train, random_seed=args.random_seed, use_adasyn=use_adasyn, include_speciation=False)

        x_test = create_input_classification(df_class=df_test_all, include_speciation=False)
        prediction = model.predict(x_test)
        df_test_all = df_test_all.copy()
        df_test_all["prediction"] = model.predict(x_test)
        df_test_all[["prediction_proba_nrb", "prediction_proba_rb"]] = model.predict_proba(x_test)
        df_test_all.to_csv(f"datasets/curated_data/class_curated_scs_multiple_test_set_{dataset_name}_predicted.csv")

        accu, f1_score, sens, speci = get_class_results(true=df_test_all["y_true"], pred=prediction)
        metrics_all = ["accuracy", "sensitivity", "specificity", "f1"]
        metrics_values_all = [accu, sens, speci, f1_score]
        for metric, metric_values in zip(metrics_all, metrics_values_all):
            log.info(f"{metric}: ", score="{:.1f}".format(np.mean(metric_values) * 100))

    for indx, (dataset_name, dataset) in enumerate(datasets.items()):
        print("") # TODO
        log.info(f"Entries in {dataset_name}", entries=len(dataset))

        # remove substances from test set from training set
        df_smiles_correct = remove_smiles_with_incorrect_format(df=dataset, col_name_smiles="smiles")
        dataset = openbabel_convert(
            df=df_smiles_correct,
            input_type="smiles",
            column_name_input="smiles",
            output_type="inchi",
        )
        lst_accu: List[float] = []
        lst_sensitivity: List[float] = []
        lst_specificity: List[float] = []
        lst_f1: List[float] = []

        n_splits = args.nsplits 

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_seed)
        for _, test_index in skf.split(test_set[["cas", "smiles", "inchi_from_smiles"]], test_set["y_true"]):
            df_test = test_set[test_set.index.isin(test_index)]
            dataset_train = dataset[~dataset["cas"].isin(df_test["cas"])]
            dataset_train = dataset_train[~dataset_train["inchi_from_smiles"].isin(df_test["inchi_from_smiles"])]
            if len(df_test) > len(dataset)-len(dataset_train):
                df_test = get_inchi_main_layer(df=df_test, inchi_col="inchi_from_smiles", layers=3) # df_test
                dataset_train = get_inchi_main_layer(df=dataset_train, inchi_col="inchi_from_smiles", layers=3)
                dataset_train = dataset_train[~dataset_train["inchi_from_smiles_main_layer"].isin(df_test["inchi_from_smiles_main_layer"])]
            log.info("Entries train set, test set, dataset before - after removing test set", df_train=len(dataset_train), df_test=len(df_test), before_minus_after=len(dataset)-len(dataset_train))

            assert len(df_test) <= len(dataset)-len(dataset_train)

            model = train_XGBClassifier_Huang_Zhang_on_all_data(df=dataset_train, random_seed=args.random_seed, use_adasyn=use_adasyn, include_speciation=False)

            x_test = create_input_classification(df_class=df_test, include_speciation=False)
            prediction = model.predict(x_test)
            df_test = df_test.copy()
            df_test["prediction"] = model.predict(x_test)

            accu, f1_score, sens, speci = get_class_results(true=df_test["y_true"], pred=prediction)
            
            lst_accu.append(round(accu, 4))
            lst_sensitivity.append(round(sens, 4))
            lst_specificity.append(round(speci, 4))
            lst_f1.append(round(f1_score, 4))

        average_test_size = round(len(test_set)/n_splits, 1)
        test_percent = round(((average_test_size) / len(dataset)) * 100, 1)
        log.info("Test set size", size=average_test_size, percent=f"{test_percent}%")

        metrics = ["accuracy", "sensitivity", "specificity", "f1"]
        metrics_values = [
            lst_accu,
            lst_sensitivity,
            lst_specificity,
            lst_f1,
        ]
        for metric, metric_values in zip(metrics, metrics_values):
            log.info(
                f"{metric}: ",
                score="{:.1f}".format(np.mean(metric_values) * 100) + " ± " + "{:.1f}".format(np.std(metric_values) * 100),
            )
        log.info("Accuracy: ", accuracy=lst_accu)
        log.info("Sensitivity: ", sensitivity=lst_sensitivity)
        log.info("Specificity: ", specificity=lst_specificity)
        log.info("F1: ", f1=lst_f1)
            
        accuracy[indx] = np.asarray(lst_accu)
        f1[indx] = np.asarray([lst_f1])
        sensitivity[indx] = np.asarray([lst_sensitivity])
        specificity[indx] = np.asarray([lst_specificity])
    return accuracy, f1, sensitivity, specificity


def get_labels_colors_progress() -> Tuple[List[str], List[str]]:
    labels = [
        "Huang-Dataset \n reported",
        "Huang-Dataset \n replicated",
        "$\mathregular{Curated_{SCS}}$",
        "$\mathregular{Curated_{BIOWIN}}$",
        "$\mathregular{Curated_{FINAL}}$",
    ]
    colors = [
        "white",
        "plum",
        "royalblue",
        "lightgreen",
        # "mediumseagreen",
        "seagreen",
    ]
    return labels, colors


def run_paper_progress(mode: str) -> None:
    if mode == "regression":
        rmse, mae, r2, _ = train_regression_models(
            comparison_or_progress="progress",
            column_for_grouping="inchi_from_smiles",
        )
        rmse = [np.array([0.25])] + rmse  # reproted RMSE from Huang and Zhang
        mae = [np.array([0.18])] + mae  # reproted MAE from Huang and Zhang
        r2 = [np.array([0.54])] + r2  # reproted R2 from Huang and Zhang
        title_to_data = {"RMSE": rmse[0:8], "MAE": mae[0:8], "$\mathregular{R^{2}}$": r2[0:8]}
    elif mode == "classification":
        # accuracy, f1, sensitivity, specificity = train_classification_models(
        #     comparison_or_progress="progress",
        #     with_lunghini=True,
        #     use_adasyn=True,
        # )
        accuracy, f1, sensitivity, specificity = train_classification_models_test_set_multiple(
            use_adasyn=True,
        )
    
        accuracy = [np.array([0.851])] + accuracy  # reported accuracy from Huang and Zhang
        # accuracy.insert(2, np.array([0.876]))  # reported accuracy with chemical speciation from Huang and Zhang
        f1 = [np.array([0.862])] + f1  # reported F1 from Huang and Zhang
        # f1.insert(2, np.array([0.878]))  # reported F1 with chemical speciation from Huang and Zhang
        sensitivity = [np.array([0.89])] + sensitivity  # reported sensitivity from Huang and Zhang
        # sensitivity.insert(2, np.array([0.874]))  # reported sensitivity with chemical speciation from Huang and Zhang
        specificity = [np.array([0.809])] + specificity  # reported specificity from Huang and Zhang
        # specificity.insert(2, np.array([0.879]))  # reported specificity with chemical speciation from Huang and Zhang
        title_to_data = {
            "Accuracy": accuracy[0:5],
            "$\mathregular{F_{1}}$": f1[0:5],
            "Sensitivity": sensitivity[0:5],
            "Specificity": specificity[0:5],
        }
    else:
        log.fatal("Wrong mode given")

    labels, colors = get_labels_colors_progress()
    for title, data in title_to_data.items():
        if (title != "Accuracy") and (title != "$\mathregular{R^{2}}$") and (title != "MAE"):
            continue
        if title == "Accuracy":
            data = [array * 100 for array in data]
        # plot_results( #plot_results_with_stanard_deviation(
        #     all_data=data,
        #     labels=labels,
        #     colors=colors,
        #     title=title,
        #     seed=args.random_seed,
        #     save_ending="progress",
        # )
        plot_results_with_stanard_deviation(
            all_data=data,
            labels=labels,
            colors=colors,
            nsplits=args.nsplits,
            title=title,
            mode=mode,
            seed=args.random_seed,
            plot_with_paper=True,
            save_ending="progress",
        )


if __name__ == "__main__":
    if (args.mode == "classification") or (args.mode == "both"):
        if (args.progress_or_comparison == "progress") or (args.progress_or_comparison == "both"):
            log.info(" \n Running classification progress")
            run_paper_progress(mode="classification")
        if (args.progress_or_comparison == "comparison") or (args.progress_or_comparison == "both"):
            log.info(" \n Running classification comparison")
    if (args.mode == "regression") or (args.mode == "both"):
        if (args.progress_or_comparison == "progress") or (args.progress_or_comparison == "both"):
            log.info(" \n Running regression progress")
            run_paper_progress(mode="regression")
        if (args.progress_or_comparison == "comparison") or (args.progress_or_comparison == "both"):
            log.info(" \n Running regression comparison")
