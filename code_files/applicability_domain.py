
import pandas as pd
import structlog
import sys
import os
import statistics
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

log = structlog.get_logger()
from typing import List, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from code_files.processing_functions import load_class_data_paper
from code_files.processing_functions import get_datasets_for_ad
from code_files.processing_functions import create_fingerprint_df
from code_files.processing_functions import get_datasets_for_ad
from code_files.processing_functions import ApplicabilityDomain


def get_dsstox(new=True) -> pd.DataFrame:
    if new:
        df_dsstox_huang = pd.read_excel("datasets/external_data/Huang_Zhang_DSStox.xlsx", index_col=0)
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
    threshold_to_data_between: Dict[float, List[int]] = defaultdict(list)
    threshold_to_acc_between_threshold: Dict[float, List[float]] = defaultdict(list)
    skf = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=random_state)
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train = y_train.values
        y_test = y_test.values

        model_with_best_params.fit(x_train, y_train)
        AD = ApplicabilityDomain(verbose=True)

        sims = AD.analyze_similarity(base_test=x_test, base_train=x_train,
                             similarity_metric='tanimoto')
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for index, threshold in enumerate(thresholds):
            threshold_to_data_below[threshold].append(len(sims[sims["Max"] <= threshold]))
            if index > 0:
                threshold_to_data_between[threshold].append(len(sims[(sims["Max"] <= threshold) & (sims["Max"] > thresholds[index-1])]))
            else:
                threshold_to_data_between[threshold].append(len(sims[sims["Max"] <= threshold]))

        threshold_to_value = AD.fit_ad(
            model=model_with_best_params,
            base_test=x_test,
            base_train=x_train,
            y_true=y_test,
            threshold_reference="max",
            threshold_step=(0, 1.1, 0.1),
            similarity_metric="tanimoto",
            metric_evaliation="acc",
        )
        for threshold, value in zip(thresholds[3:], threshold_to_value.values()):
            if value != None:
                threshold_to_acc_between_threshold[threshold].append(value)

    for index, threshold in enumerate(thresholds):
        if threshold < 0.4:
            continue
        if threshold in threshold_to_acc_between_threshold:
            acc_between_threshold = threshold_to_acc_between_threshold[threshold]
            if len(acc_between_threshold) == 1:
                expected_accuracy_between_thresholds = f"{acc_between_threshold[0]} %"
            else:
                mean_acc = statistics.mean(acc_between_threshold) * 100
                stdev_acc = statistics.stdev(acc_between_threshold) * 100
                expected_accuracy_between_thresholds = f"{mean_acc:.1f} ± {stdev_acc:.1f} %"
        else:
            expected_accuracy_between_thresholds = "None"
        data_below = threshold_to_data_below[threshold]
        data_between = threshold_to_data_between[threshold]
        
        if threshold==0.4:
            text = f"Data points below {threshold}"
        else: 
            text = f"Data points between {thresholds[index-1]} and <{threshold}"
        log.info(
            text,
            expected_accuracy_between_thresholds=expected_accuracy_between_thresholds,
            datapoints_below_t=f"{'%.1f' % sum(data_below)}",
            perc_below_t=f"{'%.1f' % ((sum(data_below) / len(df)) * 100)} %",
            perc_between_t=f"{'%.1f' % ((sum(data_between) / len(df)) * 100)} %",

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
    model = XGBClassifier()
    calculate_tanimoto_similarity_class(df=df_class_huang, model_with_best_params=model)


def calculate_tanimoto_similarity_curated_scs():
    log.info("\n Define AD of df_curated_scs")
    df_curated_scs, _, _ = get_datasets_for_ad()
    model = XGBClassifier()
    calculate_tanimoto_similarity_class(df=df_curated_scs, model_with_best_params=model)

def calculate_tanimoto_similarity_curated_biowin():
    log.info("\n Define AD of df_curated_biowin")
    _, df_curated_biowin, _ = get_datasets_for_ad()
    model = XGBClassifier()
    calculate_tanimoto_similarity_class(df=df_curated_biowin, model_with_best_params=model)

def calculate_tanimoto_similarity_curated_final():
    log.info("\n Define AD of df_curated_final")
    _, _, df_curated_final = get_datasets_for_ad()
    model = XGBClassifier()
    calculate_tanimoto_similarity_class(df=df_curated_final, model_with_best_params=model)


def check_how_much_of_dsstox_in_ad_class():
    df_dsstox = get_dsstox()
    log.info("\n Check if DSStox sets in AD of Readded classification")
    df_curated_scs, df_curated_biowin, df_curated_final = get_datasets_for_ad()
    df_dsstox1 = df_dsstox[:200000]
    df_dsstox2 = df_dsstox[200000:400000]
    df_dsstox3 = df_dsstox[400000:600000]
    df_dsstox4 = df_dsstox[600000:]
    
    log.info(f"\n                 Checking if entries of DSStox in AD of df_curated_scs")
    log.info(f"\n Part 1")
    check_external_test_in_ad(df_train=df_curated_scs, df_test=df_dsstox1)
    log.info(f"\n Part 2")
    check_external_test_in_ad(df_train=df_curated_scs, df_test=df_dsstox2)
    log.info(f"\n Part 3")
    check_external_test_in_ad(df_train=df_curated_scs, df_test=df_dsstox3)
    log.info(f"\n Part 4")
    check_external_test_in_ad(df_train=df_curated_scs, df_test=df_dsstox4)

    log.info(f"\n                 Checking if entries of DSStox in AD of df_curated_biowin")
    log.info(f"\n Part 1")
    check_external_test_in_ad(df_train=df_curated_biowin, df_test=df_dsstox1)
    log.info(f"\n Part 2")
    check_external_test_in_ad(df_train=df_curated_biowin, df_test=df_dsstox2)
    log.info(f"\n Part 3")
    check_external_test_in_ad(df_train=df_curated_biowin, df_test=df_dsstox3)
    log.info(f"\n Part 4")
    check_external_test_in_ad(df_train=df_curated_biowin, df_test=df_dsstox4)

    log.info(f"\n                 Checking if entries of DSStox in AD of df_curated_final")
    log.info(f"\n Part 1")
    check_external_test_in_ad(df_train=df_curated_final, df_test=df_dsstox1)
    log.info(f"\n Part 2")
    check_external_test_in_ad(df_train=df_curated_final, df_test=df_dsstox2)
    log.info(f"\n Part 3")
    check_external_test_in_ad(df_train=df_curated_final, df_test=df_dsstox3)
    log.info(f"\n Part 4")
    check_external_test_in_ad(df_train=df_curated_final, df_test=df_dsstox4)


if __name__ == "__main__":
    calculate_tanimoto_similarity_class_huang()
    calculate_tanimoto_similarity_curated_scs()
    calculate_tanimoto_similarity_curated_biowin()
    calculate_tanimoto_similarity_curated_final()
    check_how_much_of_dsstox_in_ad_class()
