
import pandas as pd
import structlog

log = structlog.get_logger()
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
    default=True,
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
from code_files.processing_functions import load_regression_df_curated_scs_no_metal
from code_files.processing_functions import create_classification_data_based_on_regression_data
from code_files.processing_functions import create_classification_biowin
from code_files.processing_functions import create_input_classification
from code_files.ml_functions import train_XGBClassifier_on_all_data


def create_class_datasets(with_lunghini: bool, include_speciation: bool) -> None:
    df_reg_curated_scs = load_regression_df_curated_scs_no_metal()

    log.info("\n Creating curated_scs")
    curated_scs, _ = create_classification_data_based_on_regression_data(
        df_reg_curated_scs.copy(),
        with_lunghini=with_lunghini,
        include_speciation_lunghini=True,
        include_speciation=False,
        prnt=args.prnt,
    )
    log.info("Entries in curated_scs", entries=len(curated_scs))
    log.info("Entries in curated_scs labeled as RB", entries=len(curated_scs[curated_scs["y_true"]==1]))
    log.info("Entries in curated_scs labeled as NRB", entries=len(curated_scs[curated_scs["y_true"]==0]))


    log.info("\n Creating curated_biowin")
    curated_scs_biowin, curated_scs_biowin_problematic = create_classification_biowin(
        reg_df=df_reg_curated_scs.copy(),
        with_lunghini=with_lunghini,
        include_speciation_lunghini=True,
        prnt=args.prnt,
    )
    curated_scs_biowin.reset_index(inplace=True, drop=True)
    curated_scs_biowin_problematic.reset_index(inplace=True, drop=True)
    log.info("Entries in curated_biowin", entries=len(curated_scs_biowin))
    log.info("Entries in curated_biowin labeled as RB", entries=len(curated_scs_biowin[curated_scs_biowin["y_true"]==1]))
    log.info("Entries in curated_biowin labeled as NRB", entries=len(curated_scs_biowin[curated_scs_biowin["y_true"]==0]))

    curated_scs.to_csv("datasets/curated_data/class_curated_scs.csv")

    curated_scs_biowin.to_csv("datasets/curated_data/class_curated_biowin.csv")
    curated_scs_biowin_problematic.to_csv("datasets/curated_data/class_curated_biowin_problematic.csv")



def create_curated_final() -> None:
    log.info("\n Creating curated_final")

    class_biowin = pd.read_csv("datasets/curated_data/class_curated_biowin.csv", index_col=0)
    class_biowin_problematic = pd.read_csv("datasets/curated_data/class_curated_biowin_problematic.csv", index_col=0)

    df_class = class_biowin.copy()
    df_problematic = class_biowin_problematic.copy()

    model_class = train_XGBClassifier_on_all_data(df=df_class, random_seed=args.random_seed, include_speciation=False)

    x_removed, _ = create_input_classification(df_problematic, include_speciation=False, target_col="y_true")
    df_problematic["prediction_class"] = model_class.predict(x_removed)
    df_problematic.to_csv("datasets/curated_data/class_curated_scs_biowin_problematic_predicted.csv")

    df_problematic.astype({"miti_linear_label": "int32", "miti_non_linear_label": "int32"})

    df_label_and_model_match = df_problematic[df_problematic["label"] == df_problematic["prediction_class"]]
    df_no_match = df_problematic[df_problematic["label"] != df_problematic["prediction_class"]]
    log.info("Entries where our model matches test label", entries=len(df_label_and_model_match))
    log.info(
        "Entries where our model matches test label and the label is 0",
        nrb=len(df_label_and_model_match[df_label_and_model_match["label"] == 0]),
    )
    log.info(
        "Entries where our model matches test label and the label is 1",
        rb=len(df_label_and_model_match[df_label_and_model_match["label"] == 1]),
    )

    # Add data for which our prediction matched test label and train again
    df_curated_final = pd.concat([class_biowin, df_label_and_model_match], ignore_index=True)
    log.info("Entries in df_curated_final: ", df_curated_final=len(df_curated_final))
    log.info("Entries in df_curated_final labeled as RB", entries=len(df_curated_final[df_curated_final["y_true"]==1]))
    log.info("Entries in df_curated_final labeled as NRB", entries=len(df_curated_final[df_curated_final["y_true"]==0]))

    df_curated_final.reset_index(inplace=True, drop=True) 
    df_curated_final.to_csv(f"datasets/curated_data/class_curated_final.csv")
    df_no_match.reset_index(inplace=True, drop=True) 
    df_no_match.to_csv(f"datasets/curated_data/class_curated_final_removed.csv")



if __name__ == "__main__":
    create_class_datasets(with_lunghini=args.with_lunghini, include_speciation=False)
    create_curated_final()
