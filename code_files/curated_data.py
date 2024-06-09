
import numpy as np
import structlog
from typing import List, Tuple
import sys
import os
import argparse
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from code_files.processing_functions import get_class_datasets
from code_files.processing_functions import get_labels_colors_progress
from code_files.processing_functions import plot_results_with_standard_deviation
from code_files.ml_functions import train_XGBClassifier
from code_files.ml_functions import train_XGBClassifier_on_all_data

log = structlog.get_logger()

parser = argparse.ArgumentParser()

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
    "--test_set",
    type=str,
    choices=["df_curated_final", "df_curated_scs", "df_curated_biowin"],
    default="df_curated_scs",
    help="Choose the fixed test set",
)
parser.add_argument(
    "--create_model_trained_on_all_data",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If this is true then at the end an XGBClassifier will be trained on each of the 3 datasets and saved. ",
)
args = parser.parse_args()


def train_classification_models(
    use_adasyn: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

    datasets = get_class_datasets()
    df_test = datasets[args.test_set].copy()
    accuracy: List[np.ndarray] = [np.asarray([])] * len(datasets)
    f1: List[np.ndarray] = [np.asarray([])] * len(datasets)
    sensitivity: List[np.ndarray] = [np.asarray([])] * len(datasets)
    specificity: List[np.ndarray] = [np.asarray([])] * len(datasets)

    for indx, (dataset_name, dataset) in enumerate(datasets.items()):
        log.info(f"Entries in {dataset_name}", entries=len(dataset))

        lst_accu_paper, lst_f1_paper, lst_sensitivity_paper, lst_specificity_paper = train_XGBClassifier(
            df=dataset,
            random_seed=args.random_seed,
            nsplits=args.nsplits,
            use_adasyn=use_adasyn,
            include_speciation=False,
            df_test=df_test,
            dataset_name=dataset_name,
            target_col="y_true",
        )
        accuracy[indx] = np.asarray(lst_accu_paper)
        f1[indx] = np.asarray(lst_f1_paper)
        sensitivity[indx] = np.asarray(lst_sensitivity_paper)
        specificity[indx] = np.asarray(lst_specificity_paper)
    
    return accuracy, f1, sensitivity, specificity


def run_with_improved_data() -> None:
    balanced_accuracy, f1, sensitivity, specificity = train_classification_models(
        use_adasyn=True,
    )

    balanced_accuracy = [np.array([0.876])] + balanced_accuracy  # reported balanced accuracy from Huang and Zhang with pKa and alpha values
    f1 = [np.array([0.879])] + f1  # reported F1 from Huang and Zhang
    sensitivity = [np.array([0.878])] + sensitivity  # reported sensitivity from Huang and Zhang
    specificity = [np.array([0.874])] + specificity  # reported specificity from Huang and Zhang
    title_to_data = {
        "Balanced accuracy": balanced_accuracy,
    }

    labels, colors = get_labels_colors_progress()
    for title, data in title_to_data.items():
        data = [array * 100 for array in data]

        plot_results_with_standard_deviation(
            all_data=data,
            labels=labels,
            colors=colors,
            title=title,
            seed=args.random_seed,
            plot_with_paper=True,
            save_ending="progress",
            test_set_name=args.test_set,
        )


def create_model_trained_on_all_data():
    datasets = get_class_datasets()
    for df_name, df in datasets.items():
        if df_name == "df_paper":
            continue
        model = train_XGBClassifier_on_all_data(
            df=df,
            random_seed=args.random_seed,
            include_speciation=False,
        )
        pickle.dump(model, open(f"models/xgbc_{df_name}.pkl", 'wb'))

if __name__ == "__main__":
    run_with_improved_data()
    if args.create_model_trained_on_all_data:
        create_model_trained_on_all_data()

