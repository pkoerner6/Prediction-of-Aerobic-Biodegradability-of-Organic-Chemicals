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

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from biodegradation.processing_functions import get_class_datasets
from biodegradation.processing_functions import get_regression_datasets
from biodegradation.processing_functions import get_comparison_datasets_regression
from biodegradation.processing_functions import get_comparison_datasets_classification
from biodegradation.model_results import results_improved_data_regression_nsplits5_seed42_progress
from biodegradation.model_results import results_improved_data_regression_nsplits5_seed42_comparison
from biodegradation.model_results import results_improved_data_regression_nsplits5_seed42_fixed_test_progress
from biodegradation.model_results import results_improved_data_regression_nsplits5_seed42_fixed_test_comparison
from biodegradation.model_results import results_improved_data_classification_nsplits5_seed42_progress
from biodegradation.model_results import results_improved_data_classification_nsplits5_seed42_comparison
from biodegradation.model_results import results_improved_data_classification_nsplits5_seed42_fixed_testset_progress
from biodegradation.model_results import results_improved_data_classification_nsplits5_seed42_fixed_testset_comparison
from biodegradation.ml_functions import run_XGBClassifier_Huang_Zhang
from biodegradation.ml_functions import run_XGBRegressor_Huang_Zhang

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
parser.add_argument(
    "--with_further_tests",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Whether to reduce the biowin both dataset further: only reliability1, no DOC die away tests...",
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

    plt.figure(figsize=(15, 6))

    bplot = plt.boxplot(all_data, vert=True, patch_artist=True, labels=labels, meanline=True, showmeans=True)

    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)

    if plot_with_paper:
        plt.plot(1, np.mean(all_data[0]), marker="o", markersize=14)
        if mode == "classification":
            plt.plot(3, np.mean(all_data[2]), marker="o", markersize=14)

    plt.xlabel("Datasets", fontsize=16)
    ylabel = f"{title} (%)" if title == "Accuracy" else title
    plt.ylabel(ylabel, fontsize=16)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.grid(axis="y")

    fixed = "_fixed_testset" if args.fixed_testset else ""
    title = "R2" if title == "$\mathregular{R^{2}}$" else title
    plt.savefig(
        f"biodegradation/figures/{nsplits}fold_improved_{title}_seed{seed}_paper_hyperparameter_{save_ending}{fixed}.png"
    )
    plt.close()


def train_regression_models(
    comparison_or_progress: str,
    column_for_grouping: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

    if comparison_or_progress == "progress":
        datasets = get_regression_datasets(with_further_tests=args.with_further_tests)
        df_smallest = datasets["df_improved_env_biowin_both_readded"].copy()
    if comparison_or_progress == "comparison":
        datasets = get_comparison_datasets_regression(mode="regression", include_speciation=True)
        df_smallest = datasets["df_improved_env_biowin_both"].copy()
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
            or (dataset_name == "df_paper_biowin_both")
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
        datasets = get_class_datasets(
            with_further_tests=args.with_further_tests,
        )
        df_smallest = datasets["df_improved_env_biowin_both_readded"].copy()
    if comparison_or_progress == "comparison":
        datasets = get_comparison_datasets_classification(
            mode="classification",
            include_speciation=True,
            with_lunghini=with_lunghini,
            create_new=True,
        )
        df_smallest = datasets["df_improved_env_biowin_both"].copy()
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


def get_labels_colors_progress(mode: str) -> Tuple[List[str], List[str]]:
    labels = [
        "Huang-Dataset \n reported",
        "Huang-Dataset \n replicated",
        "Huang-Dataset \n reported \n with chemical \n speciation",
        "Huang-Dataset \n replicated \n with chemical \n speciation",
        "$\mathregular{Curated_{S}}$",
        "$\mathregular{Curated_{SCS}}$",
        "$\mathregular{Curated_{1BIOWIN}}$",
        "$\mathregular{Curated_{2BIOWIN}}$",
        "$\mathregular{Curated_{FINAL}}$",
    ]
    colors = [
        "white",
        "pink",
        "white",
        "plum",
        "cornflowerblue",
        "royalblue",
        "lightgreen",
        "mediumseagreen",
        "seagreen",
    ]

    if mode == "regression":
        del labels[2]
        del colors[2]

    return labels, colors


def run_paper_progress(mode: str) -> None:
    if mode == "regression":
        if args.train_new:
            rmse, mae, r2, _ = train_regression_models(
                comparison_or_progress="progress",
                column_for_grouping="inchi_from_smiles",
            )
        elif (args.nsplits == 5) and (args.random_seed == 42) and not args.fixed_testset:
            rmse, mae, r2, _ = results_improved_data_regression_nsplits5_seed42_progress()
        elif (args.nsplits == 5) and (args.random_seed == 42) and args.fixed_testset:
            rmse, mae, r2, _ = results_improved_data_regression_nsplits5_seed42_fixed_test_progress()
        else:
            log.fatal("No results for this split for regression")
        rmse = [np.array([0.25])] + rmse  # reproted RMSE from Huang and Zhang
        mae = [np.array([0.18])] + mae  # reproted MAE from Huang and Zhang
        r2 = [np.array([0.54])] + r2  # reproted R2 from Huang and Zhang
        title_to_data = {"RMSE": rmse[0:8], "MAE": mae[0:8], "$\mathregular{R^{2}}$": r2[0:8]}
    elif mode == "classification":
        if args.train_new:
            accuracy, f1, sensitivity, specificity = train_classification_models(
                comparison_or_progress="progress",
                with_lunghini=True,
                use_adasyn=True,
            )
        elif (args.nsplits == 5) and (args.random_seed == 42) and not args.fixed_testset:
            accuracy, f1, sensitivity, specificity = results_improved_data_classification_nsplits5_seed42_progress()
        elif (args.nsplits == 5) and (args.random_seed == 42) and args.fixed_testset:
            (
                accuracy,
                f1,
                sensitivity,
                specificity,
            ) = results_improved_data_classification_nsplits5_seed42_fixed_testset_progress()
        else:
            log.fatal("No results for this split for classification")
        accuracy = [np.array([0.851])] + accuracy  # reported accuracy from Huang and Zhang
        accuracy.insert(2, np.array([0.876]))  # reported accuracy with chemical speciation from Huang and Zhang
        f1 = [np.array([0.862])] + f1  # reported F1 from Huang and Zhang
        f1.insert(2, np.array([0.878]))  # reported F1 with chemical speciation from Huang and Zhang
        sensitivity = [np.array([0.89])] + sensitivity  # reported sensitivity from Huang and Zhang
        sensitivity.insert(2, np.array([0.874]))  # reported sensitivity with chemical speciation from Huang and Zhang
        specificity = [np.array([0.809])] + specificity  # reported specificity from Huang and Zhang
        specificity.insert(2, np.array([0.879]))  # reported specificity with chemical speciation from Huang and Zhang
        title_to_data = {
            "Accuracy": accuracy[0:9],
            "$\mathregular{F_{1}}$": f1[0:9],
            "Sensitivity": sensitivity[0:9],
            "Specificity": specificity[0:9],
        }
    else:
        log.fatal("Wrong mode given")

    labels, colors = get_labels_colors_progress(mode=mode)
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


def get_labels_colors_comparison() -> Tuple[List[str], List[str]]:
    labels = [
        "Huang-Dataset",
        "$\mathregular{Curated_{S}}$",
        "$\mathregular{Curated_{SCS}}$",
        "Huang-Dataset, \n checked with \n BIOWIN 5 or 6",
        "$\mathregular{Curated_{S}}$, \n checked with \n BIOWIN 5 or 6",
        "$\mathregular{Curated_{1BIOWIN}}$",
        "Huang-Dataset, \n checked with \n BIOWIN 5 and 6",
        "$\mathregular{Curated_{S}}$, \n checked with \n BIOWIN 5 and 6",
        "$\mathregular{Curated_{2BIOWIN}}$",
    ]
    colors = [
        "pink",
        "plum",
        "violet",
        "lightblue",
        "cornflowerblue",
        "royalblue",
        "lightgreen",
        "mediumseagreen",
        "seagreen",
    ]
    return labels, colors


def run_models_comparison(mode: str) -> None:
    if mode == "regression":
        if args.train_new:
            rmse, mae, r2, mse = train_regression_models(
                comparison_or_progress="comparison",
                column_for_grouping="inchi_from_smiles",
            )
        elif (args.nsplits == 5) and (args.random_seed == 42) and not args.fixed_testset:
            rmse, mae, r2, _ = results_improved_data_regression_nsplits5_seed42_comparison()
        elif (args.nsplits == 5) and (args.random_seed == 42) and args.fixed_testset:
            rmse, mae, r2, _ = results_improved_data_regression_nsplits5_seed42_fixed_test_comparison()
        else:
            log.fatal("No results for this split for regression")
        title_to_data = {"RMSE": rmse, "MAE": mae, "$\mathregular{R^{2}}$": r2}
    if mode == "classification":
        if args.train_new:
            accuracy, f1, sensitivity, specificity = train_classification_models(
                comparison_or_progress="comparison",
                with_lunghini=True,
                use_adasyn=True,
            )
        elif (args.nsplits == 5) and (args.random_seed == 42) and not args.fixed_testset:
            accuracy, f1, sensitivity, specificity = results_improved_data_classification_nsplits5_seed42_comparison()
        elif (args.nsplits == 5) and (args.random_seed == 42) and args.fixed_testset:
            (
                accuracy,
                f1,
                sensitivity,
                specificity,
            ) = results_improved_data_classification_nsplits5_seed42_fixed_testset_comparison()
        else:
            log.fatal("No results for this split for classification")
        title_to_data = {
            "Accuracy": accuracy,
            "$\mathregular{F_{1}}$": f1,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
        }
    labels, colors = get_labels_colors_comparison()
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
            plot_with_paper=False,
            save_ending="comparison",
        )


if __name__ == "__main__":
    if (args.mode == "classification") or (args.mode == "both"):
        if (args.progress_or_comparison == "progress") or (args.progress_or_comparison == "both"):
            log.info(" \n Running classification progress")
            run_paper_progress(mode="classification")
        if (args.progress_or_comparison == "comparison") or (args.progress_or_comparison == "both"):
            log.info(" \n Running classification comparison")
            run_models_comparison(mode="classification")
    if (args.mode == "regression") or (args.mode == "both"):
        if (args.progress_or_comparison == "progress") or (args.progress_or_comparison == "both"):
            log.info(" \n Running regression progress")
            run_paper_progress(mode="regression")
        if (args.progress_or_comparison == "comparison") or (args.progress_or_comparison == "both"):
            log.info(" \n Running regression comparison")
            run_models_comparison(mode="regression")
