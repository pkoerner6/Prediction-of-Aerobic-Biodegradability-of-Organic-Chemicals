"""File to create all dataframes that will be used in the improved_data.py and improved_data_analysis.py file"""
"""Prior to this file, the data_processing.py file needs to be run"""

import pandas as pd
import numpy as np
import structlog

log = structlog.get_logger()
from typing import List, Dict
import matplotlib.pyplot as plt
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
import sys
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--with_lunghini",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="If data from Lunghini should be added to classification datasets",
)
parser.add_argument(
    "--prnt",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If dataprocessing information should be added",
)
parser.add_argument(
    "--random_seed",
    type=int,
    default=42,
    help="Choose the random seed",
)
args = parser.parse_args()

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from biodegradation.processing_functions import load_regression_df
from biodegradation.processing_functions import load_regression_df_improved_no_metal
from biodegradation.processing_functions import load_regression_df_improved_no_metal_env_smiles
from biodegradation.processing_functions import convert_to_maccs_fingerprints
from biodegradation.processing_functions import create_classification_data_based_on_regression_data
from biodegradation.processing_functions import create_classification_biowin
from biodegradation.processing_functions import process_df_biowin
from biodegradation.processing_functions import create_input_classification
from biodegradation.ml_functions import train_XGBClassifier_Huang_Zhang_on_all_data


def create_regression_datasets() -> None:
    reg = load_regression_df()
    improved = load_regression_df_improved_no_metal()
    improved_env = load_regression_df_improved_no_metal_env_smiles()
    paper_biowin, _ = process_df_biowin(
        df=reg,
        mode="reg",
        match_both=False,
        remove_doc=False,
        only_reliability_1=False,
        only_ready=False,
        only_28=False,
        biowin56=True,
    )
    paper_biowin_both, _ = process_df_biowin(
        df=reg,
        mode="reg",
        match_both=True,
        remove_doc=False,
        only_reliability_1=False,
        only_ready=False,
        only_28=False,
        biowin56=True,
    )
    improved_biowin, _ = process_df_biowin(
        df=improved,
        mode="reg",
        match_both=False,
        remove_doc=False,
        only_reliability_1=False,
        only_ready=False,
        only_28=False,
        biowin56=True,
    )
    improved_biowin_both, improved_biowin_both_removed = process_df_biowin(
        df=improved,
        mode="reg",
        match_both=True,
        remove_doc=False,
        only_reliability_1=False,
        only_ready=False,
        only_28=False,
        biowin56=True,
    )
    biowin_env, _ = process_df_biowin(
        df=improved_env,
        mode="reg",
        match_both=False,
        remove_doc=False,
        only_reliability_1=False,
        only_ready=False,
        only_28=False,
        biowin56=True,
    )
    biowin_env_both, biowin_env_both_removed = process_df_biowin(
        df=improved_env,
        mode="reg",
        match_both=True,
        only_reliability_1=False,
        remove_doc=False,
        only_ready=False,
        only_28=False,
        biowin56=True,
    )
    biowin_env_both_no_doc, _ = process_df_biowin(
        df=improved_env,
        mode="reg",
        match_both=True,
        remove_doc=True,
        only_reliability_1=False,
        only_ready=False,
        only_28=False,
        biowin56=True,
    )
    biowin_env_both_reliability1, _ = process_df_biowin(
        df=improved_env,
        mode="reg",
        match_both=True,
        remove_doc=False,
        only_reliability_1=True,
        only_ready=False,
        only_28=False,
        biowin56=True,
    )
    biowin_env_both_ready, _ = process_df_biowin(
        df=improved_env,
        mode="reg",
        match_both=True,
        remove_doc=False,
        only_reliability_1=False,
        only_ready=True,
        only_28=False,
        biowin56=True,
    )
    biowin_env_both_28days, _ = process_df_biowin(
        df=improved_env,
        mode="reg",
        match_both=True,
        remove_doc=False,
        only_reliability_1=False,
        only_ready=False,
        only_28=True,
        biowin56=True,
    )
    paper_biowin.to_csv("biodegradation/dataframes/improved_data/reg_paper_biowin.csv")
    paper_biowin_both.to_csv("biodegradation/dataframes/improved_data/reg_paper_biowin_both.csv")
    improved_biowin.to_csv("biodegradation/dataframes/improved_data/reg_improved_biowin.csv")
    improved_biowin_both.to_csv("biodegradation/dataframes/improved_data/reg_improved_biowin_both.csv")
    improved_biowin_both_removed.to_csv("biodegradation/dataframes/improved_data/reg_improved_biowin_both_removed.csv")
    biowin_env.to_csv("biodegradation/dataframes/improved_data/reg_improved_env_biowin.csv")
    biowin_env_both.to_csv("biodegradation/dataframes/improved_data/reg_improved_env_biowin_both.csv")
    biowin_env_both_removed.to_csv("biodegradation/dataframes/improved_data/reg_improved_env_biowin_both_removed.csv")
    biowin_env_both_no_doc.to_csv("biodegradation/dataframes/improved_data/reg_improved_env_biowin_both_no_doc.csv")
    biowin_env_both_reliability1.to_csv(
        "biodegradation/dataframes/improved_data/reg_improved_env_biowin_both_reliability1.csv"
    )
    biowin_env_both_ready.to_csv("biodegradation/dataframes/improved_data/reg_improved_env_biowin_both_ready.csv")
    biowin_env_both_28days.to_csv("biodegradation/dataframes/improved_data/reg_improved_env_biowin_both_28days.csv")


def create_class_datasets(with_lunghini: bool, include_speciation: bool, prnt: bool) -> None:
    reg = load_regression_df()
    df_reg_improved = load_regression_df_improved_no_metal()
    df_reg_improved_env = load_regression_df_improved_no_metal_env_smiles()
    log.info("Creating improved")
    improved = create_classification_data_based_on_regression_data(
        df_reg_improved.copy(),
        with_lunghini=with_lunghini,
        include_speciation=include_speciation,
        env_smiles=False,
        prnt=prnt,
    )
    improved.to_csv("biodegradation/dataframes/improved_data/class_improved.csv")
    log.info("Creating improved_env")
    improved_env = create_classification_data_based_on_regression_data(
        df_reg_improved.copy(),
        with_lunghini=with_lunghini,
        include_speciation=include_speciation,
        env_smiles=True,
        prnt=prnt,
    )
    improved_env.to_csv("biodegradation/dataframes/improved_data/class_improved_env.csv")
    log.info("Creating paper_biowin")
    paper_biowin, _ = create_classification_biowin(
        reg_df=reg,
        with_lunghini=with_lunghini,
        include_speciation=include_speciation,
        match_both=False,
        remove_doc=False,
        only_reliability_1=False,
        only_ready=False,
        only_28=False,
        env_smiles=False,
        biowin56=True,
        prnt=False,
    )
    paper_biowin.to_csv("biodegradation/dataframes/improved_data/class_paper_biowin.csv")
    log.info("Creating paper_biowin_both")
    paper_biowin_both, _ = create_classification_biowin(
        reg_df=reg,
        with_lunghini=with_lunghini,
        include_speciation=include_speciation,
        match_both=True,
        remove_doc=False,
        only_reliability_1=False,
        only_ready=False,
        only_28=False,
        env_smiles=False,
        biowin56=True,
        prnt=False,
    )
    paper_biowin_both.to_csv("biodegradation/dataframes/improved_data/class_paper_biowin_both.csv")
    log.info("Creating improved_biowin")
    improved_biowin, _ = create_classification_biowin(
        reg_df=df_reg_improved,
        with_lunghini=with_lunghini,
        include_speciation=include_speciation,
        match_both=False,
        remove_doc=False,
        only_reliability_1=False,
        only_ready=False,
        only_28=False,
        env_smiles=False,
        biowin56=True,
        prnt=False,
    )
    improved_biowin.to_csv("biodegradation/dataframes/improved_data/class_improved_biowin.csv")
    log.info("Creating improved_biowin_both")
    improved_biowin_both, improved_biowin_both_removed = create_classification_biowin(
        reg_df=df_reg_improved,
        with_lunghini=with_lunghini,
        include_speciation=include_speciation,
        match_both=True,
        remove_doc=False,
        only_reliability_1=False,
        only_ready=False,
        only_28=False,
        env_smiles=False,
        biowin56=True,
        prnt=False,
    )
    improved_biowin_both.to_csv("biodegradation/dataframes/improved_data/class_improved_biowin_both.csv")
    improved_biowin_both_removed.to_csv(
        "biodegradation/dataframes/improved_data/class_improved_biowin_both_removed.csv"
    )
    log.info("Creating biowin_env")
    biowin_env, _ = create_classification_biowin(
        reg_df=df_reg_improved_env,
        with_lunghini=with_lunghini,
        include_speciation=include_speciation,
        match_both=False,
        remove_doc=False,
        only_reliability_1=False,
        only_ready=False,
        only_28=False,
        env_smiles=True,
        biowin56=True,
        prnt=prnt,
    )
    biowin_env.to_csv("biodegradation/dataframes/improved_data/class_improved_env_biowin.csv")
    log.info("Creating biowin_env_both")
    biowin_env_both, biowin_env_both_removed = create_classification_biowin(
        reg_df=df_reg_improved_env,
        with_lunghini=with_lunghini,
        include_speciation=include_speciation,
        match_both=True,
        remove_doc=False,
        only_reliability_1=False,
        only_ready=False,
        only_28=False,
        env_smiles=True,
        biowin56=True,
        prnt=prnt,
    )
    biowin_env_both.to_csv("biodegradation/dataframes/improved_data/class_improved_env_biowin_both.csv")
    biowin_env_both_removed.to_csv("biodegradation/dataframes/improved_data/class_improved_env_biowin_both_removed.csv")
    log.info("Creating biowin_env_both_speciation")
    biowin_env_both_speciation, _ = create_classification_biowin(
        reg_df=df_reg_improved_env,
        with_lunghini=with_lunghini,
        include_speciation=True,
        match_both=True,
        remove_doc=False,
        only_reliability_1=False,
        only_ready=False,
        only_28=False,
        env_smiles=True,
        biowin56=True,
        prnt=prnt,
    )
    biowin_env_both_speciation.to_csv(
        "biodegradation/dataframes/improved_data/class_improved_env_biowin_both_speciation.csv"
    )
    log.info("Creating biowin_env_both_no_doc")
    biowin_env_both_no_doc, _ = create_classification_biowin(
        reg_df=df_reg_improved_env,
        with_lunghini=with_lunghini,
        include_speciation=include_speciation,
        match_both=True,
        remove_doc=True,
        only_reliability_1=False,
        only_ready=False,
        only_28=False,
        env_smiles=True,
        biowin56=True,
        prnt=prnt,
    )
    biowin_env_both_no_doc.to_csv("biodegradation/dataframes/improved_data/class_improved_env_biowin_both_no_doc.csv")
    log.info("Creating biowin_env_both_reliability1")
    biowin_env_both_reliability1, _ = create_classification_biowin(
        reg_df=df_reg_improved_env,
        with_lunghini=with_lunghini,
        include_speciation=include_speciation,
        match_both=True,
        remove_doc=False,
        only_reliability_1=True,
        only_ready=False,
        only_28=False,
        env_smiles=True,
        biowin56=True,
        prnt=prnt,
    )
    biowin_env_both_reliability1.to_csv(
        "biodegradation/dataframes/improved_data/class_improved_env_biowin_both_reliability1.csv"
    )


def create_readded() -> None:

    class_biowin_env_both = pd.read_csv(
        "biodegradation/dataframes/improved_data/class_improved_env_biowin_both.csv", index_col=0
    )
    class_biowin_env_both_removed = pd.read_csv(
        "biodegradation/dataframes/improved_data/class_improved_env_biowin_both_removed.csv", index_col=0
    )
    reg_biowin_env_both_removed = pd.read_csv(
        "biodegradation/dataframes/improved_data/reg_improved_env_biowin_both_removed.csv", index_col=0
    )

    df_class = class_biowin_env_both.copy()
    df_class_removed = class_biowin_env_both_removed.copy()
    df_reg_removed = reg_biowin_env_both_removed.copy()

    model_class = train_XGBClassifier_Huang_Zhang_on_all_data(
        df=df_class, random_seed=args.random_seed, use_adasyn=True, include_speciation=False
    )

    x_removed = create_input_classification(df_class_removed, include_speciation=False)
    df_class_removed["prediction_class"] = model_class.predict(x_removed)
    df_class_removed.to_csv(
        "biodegradation/dataframes/improved_data/class_improved_env_biowin_both_removed_predicted.csv"
    )

    df_both_removed_fp = convert_to_maccs_fingerprints(df_reg_removed)
    x_test = create_input_classification(df_both_removed_fp, include_speciation=False)
    df_reg_removed["prediction_class"] = model_class.predict(x_test)
    df_reg_removed.to_csv("biodegradation/dataframes/improved_data/reg_improved_env_biowin_both_removed_predicted.csv")

    class_biowin_env_both = pd.read_csv(
        "biodegradation/dataframes/improved_data/class_improved_env_biowin_both.csv", index_col=0
    )
    reg_biowin_env_both = pd.read_csv(
        "biodegradation/dataframes/improved_data/reg_improved_env_biowin_both.csv", index_col=0
    )

    dfs_biowin = {
        "reg_improved_env_biowin_both": reg_biowin_env_both,
        "class_improved_env_biowin_both": class_biowin_env_both,
    }
    for df_name, df in dfs_biowin.items():
        log.info(f"Readding entries to {df_name}")
        df_removed = pd.read_csv(
            f"biodegradation/dataframes/improved_data/{df_name}_removed_predicted.csv", index_col=0
        )
        df_removed.astype({"miti_linear_label": "int32", "miti_non_linear_label": "int32"})

        df_test_model_match = df_removed[df_removed["label"] == df_removed["prediction_class"]]
        log.info("Entries where our model matches test label", entries=len(df_test_model_match))
        log.info(
            "Entries where our model matches test label and the label is 0",
            nrb=len(df_test_model_match[df_test_model_match["label"] == 0]),
        )
        log.info(
            "Entries where our model matches test label and the label is 1",
            rb=len(df_test_model_match[df_test_model_match["label"] == 1]),
        )

        # Add data for which our prediction matched test label and train again
        df_readded = pd.concat([df, df_test_model_match], ignore_index=True)
        df_readded.to_csv(f"biodegradation/dataframes/improved_data/{df_name}_readded.csv")


if __name__ == "__main__":
    create_regression_datasets()
    create_class_datasets(with_lunghini=args.with_lunghini, include_speciation=False, prnt=args.prnt)
    create_readded()
