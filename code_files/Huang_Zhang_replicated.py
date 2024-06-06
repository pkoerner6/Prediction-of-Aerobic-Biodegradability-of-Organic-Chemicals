
import argparse
from typing import List
import sys
import os
import pandas as pd
import structlog
from rdkit import RDLogger

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from code_files.processing_functions import openbabel_convert
from code_files.processing_functions import remove_smiles_with_incorrect_format
from code_files.processing_functions import load_class_data_paper
from code_files.ml_functions import train_XGBClassifier


RDLogger.DisableLog("rdApp.*")  # Disable warnings from rdkit
log = structlog.get_logger()

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_info",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Whether to print data information",
)
parser.add_argument("--random_state", type=int, default=42)
parser.add_argument("--nsplits", type=int, default=5)
parser.add_argument(
    "--chemical_speciation_included",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Whether to include chemical speciation",
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
    log.info("Unique CAS RN in dataset", unique_cas=df["cas"].nunique())
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


def create_paper_classification() -> pd.DataFrame:
    _, _, df_class = load_class_data_paper()
    if not args.chemical_speciation_included:
        df_class = df_class[["name", "name_type", "cas", "smiles", "y_true"]]
    return df_class


def run_classification() -> None:
    df_paper_class = create_paper_classification()

    log.info(
        f" \n Newly trained classification model trained on paper data, {args.nsplits}-fold, tested on test set"
    )
    train_XGBClassifier(
        df=df_paper_class,
        random_seed=args.random_state,
        nsplits=args.nsplits,
        use_adasyn=True,
        include_speciation=args.chemical_speciation_included,
        df_test=df_paper_class,
        dataset_name="df_classification_paper",
        target_col="y_true",
    )
    if args.data_info:
        log.info(" \n Analysing paper classification data")
        info_dataset(
            df_paper_class,
            dataset_type="classification",
            inchi_col_name="inchi_from_smiles",
        )


if __name__ == "__main__":
    run_classification()
