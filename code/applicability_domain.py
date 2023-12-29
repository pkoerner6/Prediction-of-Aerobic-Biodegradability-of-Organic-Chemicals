import argparse
import pandas as pd
import structlog
import sys
import os
import statistics
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from pyADA import ApplicabilityDomain

log = structlog.get_logger()
from typing import List, Dict, Tuple


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from processing_functions import load_class_data_paper
from processing_functions import load_regression_df
from processing_functions import create_fingerprint_df
from ml_functions import split_regression_df_with_grouping


def get_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_curated_scs = pd.read_csv(
        "datasets/curated_data/class_curated_scs.csv", index_col=0
    )
    df_curated_biowin = pd.read_csv(
        "datasets/curated_data/class_curated_biowin.csv", index_col=0
    )
    df_curated_final = pd.read_csv(
        "datasets/curated_data/class_curated_final.csv", index_col=0
    )
    df_curated_scs = df_curated_scs[:100] # TODO
    df_curated_biowin = df_curated_biowin[:100] # TODO
    df_curated_final = df_curated_final[:100] # TODO
    return df_curated_scs, df_curated_biowin, df_curated_final


def get_dsstox(new=False) -> pd.DataFrame:
    if new:
        df_dsstox_huang = pd.read_excel("external_data/Huang_Zhang_DSStox.xlsx", index_col=0)
        df_dsstox_huang.rename(columns={"Smiles": "smiles", "CASRN": "cas"}, inplace=True)

        df_dsstox = df_dsstox_huang[["cas", "smiles"]].copy()
        df_dsstox.to_csv("datasets/external_data/DSStox.csv")
    df_dsstox = pd.read_csv("datasets/external_data/DSStox.csv", index_col=0)
    return df_dsstox


def calculate_tanimoto_similarity_class(df: pd.DataFrame, model_with_best_params, nsplits=5, random_state=42):
    x = create_fingerprint_df(df=df)
    x = x.values
    y = df["y_true"]

    threshold_to_data_below: Dict[float, List[int]] = defaultdict(list)
    threshold_to_max: Dict[float, List[float]] = defaultdict(list)
    skf = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=random_state)
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train = y_train.values
        y_test = y_test.values

        model_with_best_params.fit(x_train, y_train)
        ad = ApplicabilityDomain(verbose=True)
        threshold_to_value = ad.fit(
            model=model_with_best_params,
            base_test=x_test,
            base_train=x_train,
            y_true=y_test,
            threshold_reference="max",
            threshold_step=(0, 1.01, 0.1),
            similarity_metric="tanimoto",
            metric_avaliation="acc",
        )
        for threshold, value in threshold_to_value.items():
            threshold_to_data_below[threshold].append(len(value[1]))
            threshold_to_max[threshold].append(value[0][0])

    for threshold, _ in threshold_to_max.items():
        max_threshold = threshold_to_max[threshold]
        data_below = threshold_to_data_below[threshold]
        log.info(
            f"Data points below {threshold}",
            max_accuracy=f"{'%.1f' % (statistics.mean(max_threshold)*100)}"
            + " ± "
            + f"{'%.1f' % (statistics.stdev(max_threshold)*100)} %",
            datapoints_below_threshold=f"{'%.1f' % statistics.mean(data_below)}",
            percentage_below_threshold=f"{'%.1f' % ((statistics.mean(data_below) / len(x_test)) * 100)} %",
        )
    return


def calculate_tanimoto_similarity_reg(df: pd.DataFrame, model_with_best_params, nsplits=5, random_state=42):
    threshold_to_data_below: Dict[float, List[int]] = defaultdict(list)
    threshold_to_max: Dict[float, List[float]] = defaultdict(list)

    train_dfs, test_dfs = split_regression_df_with_grouping(
        df=df, nsplits=nsplits, column_for_grouping="cas", random_seed=random_state
    )

    for train_df, test_df in zip(train_dfs, test_dfs):
        x_train = create_fingerprint_df(df=train_df)
        x_train = x_train.values
        y_train = train_df["biodegradation_percent"]
        y_train = y_train.values
        x_test = create_fingerprint_df(df=test_df)
        x_test = x_test.values
        y_test = test_df["biodegradation_percent"]
        y_test = y_test.values

        model_with_best_params.fit(x_train, y_train)
        ad = ApplicabilityDomain(verbose=True)
        threshold_to_value = ad.fit(
            model=model_with_best_params,
            base_test=x_test,
            base_train=x_train,
            y_true=y_test,
            threshold_reference="max",
            threshold_step=(0, 1.01, 0.1),
            similarity_metric="tanimoto",
            metric_avaliation="rmse",
        )
        for threshold, value in threshold_to_value.items():
            threshold_to_data_below[threshold].append(len(value[1]))
            threshold_to_max[threshold].append(value[0][0])

    for threshold, _ in threshold_to_max.items():
        max_threshold = threshold_to_max[threshold]
        data_below = threshold_to_data_below[threshold]
        log.info(
            f"Data points below {threshold}",
            max_rmse=f"{'%.2f' % statistics.mean(max_threshold)}"
            + " ± "
            + f"{'%.2f' % statistics.stdev(max_threshold)}",
            datapoints_below_threshold=f"{'%.1f' % statistics.mean(data_below)}",
            percentage_below_threshold=f"{'%.1f' % ((statistics.mean(data_below) / len(x_test)) * 100)} %",
        )
    return


def check_external_test_in_ad(df_train: pd.DataFrame, df_test: pd.DataFrame):
    x_train = create_fingerprint_df(df=df_train)
    x_train = x_train.values
    x_test = create_fingerprint_df(df=df_test)
    x_test = x_test.values

    AD = ApplicabilityDomain(verbose=True)
    df_similarities = AD.analyze_similarity(base_test=x_test, base_train=x_train, similarity_metric="tanimoto")
    threshold_below_05 = len(df_similarities[df_similarities["Max"] < 0.5])
    threshold_below06 = len(df_similarities[(df_similarities["Max"] >= 0.5) & (df_similarities["Max"] < 0.6)])
    threshold_below07 = len(df_similarities[(df_similarities["Max"] >= 0.6) & (df_similarities["Max"] < 0.7)])
    threshold_below08 = len(df_similarities[(df_similarities["Max"] >= 0.7) & (df_similarities["Max"] < 0.8)])
    threshold_below09 = len(df_similarities[(df_similarities["Max"] >= 0.8) & (df_similarities["Max"] < 0.9)])
    threshold_below1 = len(df_similarities[(df_similarities["Max"] >= 0.9) & (df_similarities["Max"] < 1.0)])
    threshold_equal1 = len(df_similarities[(df_similarities["Max"] == 1.0)])
    assert (
        len(df_test)
        == threshold_below_05
        + threshold_below06
        + threshold_below07
        + threshold_below08
        + threshold_below09
        + threshold_below1
        + threshold_equal1
    )
    log.info(
        "Datapoints in each threshold",
        threshold_below_05=threshold_below_05,
        threshold_below06=threshold_below06,
        threshold_below07=threshold_below07,
        threshold_below08=threshold_below08,
        threshold_below09=threshold_below09,
        threshold_below1=threshold_below1,
        threshold_equal1=threshold_equal1,
    )
    log.info(
        "Percentage in each threshold",
        threshold_below_05=f"{'%.1f' % ((threshold_below_05 / len(df_test))*100)}",
        threshold_below06=f"{'%.1f' % ((threshold_below06 / len(df_test))*100)}%",
        threshold_below07=f"{'%.1f' % ((threshold_below07 / len(df_test))*100)}%",
        threshold_below08=f"{'%.1f' % ((threshold_below08 / len(df_test))*100)}%",
        threshold_below09=f"{'%.1f' % ((threshold_below09 / len(df_test))*100)}%",
        threshold_below1=f"{'%.1f' % ((threshold_below1 / len(df_test))*100)}%",
        threshold_equal1=f"{'%.1f' % ((threshold_equal1 / len(df_test))*100)}%",
    )


def calculate_tanimoto_similarity_class_huang():
    log.info("\n Define AD of classification data Huang and Zhang")
    _, _, df_class_huang = load_class_data_paper()
    df_class_huang = df_class_huang[:100] # TODO
    model = XGBClassifier()
    calculate_tanimoto_similarity_class(df=df_class_huang, model_with_best_params=model)


def calculate_tanimoto_similarity_curated_scs():
    log.info("\n Define AD of df_curated_scs")
    df_curated_scs, _, _ = get_datasets()
    # best_params = get_lazy_xgbc_parameters()
    model = XGBClassifier() # TODO
    calculate_tanimoto_similarity_class(df=df_curated_scs, model_with_best_params=model)

def calculate_tanimoto_similarity_curated_biowin():
    log.info("\n Define AD of df_curated_biowin")
    _, df_curated_biowin, df_curated_final = get_datasets()
    # best_params = get_lazy_xgbc_parameters()
    model = XGBClassifier() # TODO
    calculate_tanimoto_similarity_class(df=df_curated_biowin, model_with_best_params=model)

def calculate_tanimoto_similarity_curated_final():
    log.info("\n Define AD of df_curated_final")
    _, _, df_curated_final = get_datasets()
    # best_params = get_lazy_xgbc_parameters()
    model = XGBClassifier() # TODO
    calculate_tanimoto_similarity_class(df=df_curated_final, model_with_best_params=model)


def check_how_much_of_dsstox_in_ad_class():
    df_dsstox = get_dsstox()
    log.info("\n Check if DSStox sets in AD of Readded classification")
    df_curated_scs, df_curated_biowin, df_curated_final = get_datasets()
    log.info(f"\n                 Checking if entries of DSStox in AD of df_curated_scs")
    check_external_test_in_ad(df_train=df_curated_scs, df_test=df_dsstox)
    log.info(f"\n                 Checking if entries of DSStox in AD of df_curated_biowin")
    check_external_test_in_ad(df_train=df_curated_biowin, df_test=df_dsstox)
    log.info(f"\n                 Checking if entries of DSStox in AD of df_curated_final")
    check_external_test_in_ad(df_train=df_curated_final, df_test=df_dsstox)


if __name__ == "__main__":
    calculate_tanimoto_similarity_class_huang()
    calculate_tanimoto_similarity_curated_scs()
    calculate_tanimoto_similarity_curated_biowin()
    calculate_tanimoto_similarity_curated_final()
    check_how_much_of_dsstox_in_ad_class()
