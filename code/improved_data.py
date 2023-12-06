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
from processing_functions import load_class_data_paper
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
from ml_functions import train_class_model_on_all_data

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
    with_lunghini: bool,
    use_adasyn: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

    datasets = get_class_datasets()
    df_smallest = datasets["df_curated_scs_biowin"].copy() # TODO df_curated_scs_biowin_readded, df_curated_scs, df_curated_scs_biowin
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
        accuracy, f1, sensitivity, specificity = train_classification_models(
            with_lunghini=True,
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
