import pandas as pd
import numpy as np
import structlog
import sys
import os
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from time import time
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.class_weight import compute_sample_weight

log = structlog.get_logger()


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from code_files.processing_functions import create_input_classification
from code_files.processing_functions import create_input_regression
from code_files.processing_functions import get_speciation_col_names
from code_files.processing_functions import openbabel_convert
from code_files.processing_functions import remove_smiles_with_incorrect_format



def get_class_results(true: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float, float]:
    accuracy = balanced_accuracy_score(true, pred)
    f1 = f1_score(true, pred)
    sensitivity = recall_score(true, pred)
    specificity = recall_score(true, pred, pos_label=0)
    return accuracy, f1, sensitivity, specificity


def print_class_results(accuracy: float, sensitivity: float, specificity: float, f1: float) -> float:
    log.info("Balanced accuracy", accuracy="{:.1f}".format(accuracy * 100))
    log.info("Sensitivity", sensitivity="{:.1f}".format(sensitivity * 100))
    log.info("Specificity", specificity="{:.1f}".format(specificity * 100))
    log.info("F1", f1="{:.2f}".format(f1))
    return accuracy



def split_classification_df_with_fixed_test_set(
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    nsplits: int,
    random_seed: int,
    cols: List[str],
    paper: bool,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:

    train_sets: List[pd.DataFrame] = []
    test_sets: List[pd.DataFrame] = []

    dfs = [df, df_test]
    for i, df in enumerate(dfs):
        if "inchi_from_smiles" not in df.columns:
            df_smiles_correct = remove_smiles_with_incorrect_format(df=df, col_name_smiles="smiles")
            dfs[i] = openbabel_convert(
                df=df_smiles_correct,
                input_type="smiles",
                column_name_input="smiles",
                output_type="inchi",
            )
    df = dfs[0][cols + ["inchi_from_smiles"]]
    df_test = dfs[1][cols + ["inchi_from_smiles"]]

    skf = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=random_seed)
    for _, test_index in skf.split(df_test[cols + ["inchi_from_smiles"]], df_test["y_true"]):
        df_test_set = df_test[df_test.index.isin(test_index)]
        train_set = df[~df["inchi_from_smiles"].isin(df_test_set["inchi_from_smiles"])]
        if paper:
            df_checked = pd.read_excel("datasets/chemical_speciation.xlsx", index_col=0)
            test_checked = df_checked[
                df_checked["env_smiles"].isin(df_test_set["smiles"]) | 
                df_checked["inchi_from_smiles"].isin(df_test_set["inchi_from_smiles"])]
            train_set = train_set[~(train_set["inchi_from_smiles"].isin(test_checked["inchi_from_smiles"]))]
            train_set = train_set[~(train_set["smiles"].isin(test_checked["env_smiles"]))]
            train_set = train_set.loc[~((train_set["cas"].isin(test_checked["cas"])) & (test_checked["cas"].notna())), :]
            train_set = train_set.loc[~((train_set["cas"].isin(df_test_set["cas"])) & (df_test_set["cas"].notna())), :]
        train_sets.append(train_set)
        test_sets.append(df_test_set)

    return train_sets, test_sets


def skf_class_fixed_testset(
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    nsplits: int,
    random_seed: int,
    include_speciation: bool,
    cols: List[str],
    paper: bool,
    target_col: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[pd.DataFrame], List[int]]:
    train_sets, test_sets = split_classification_df_with_fixed_test_set(
        df=df,
        df_test=df_test,
        nsplits=nsplits,
        random_seed=random_seed,
        cols=cols,
        paper=paper,
    )
    x_train_fold_lst: List[np.ndarray] = []
    y_train_fold_lst: List[np.ndarray] = []
    x_test_fold_lst: List[np.ndarray] = []
    y_test_fold_lst: List[np.ndarray] = []
    df_test_lst: List[pd.DataFrame] = []
    test_set_sizes: List[int] = []

    for split in range(nsplits):
        x_train_fold = train_sets[split][cols]
        x, y = create_input_classification(x_train_fold, include_speciation=include_speciation, target_col=target_col)
        x_train_fold_lst.append(x)
        y_train_fold_lst.append(y)
        x_test_fold = test_sets[split][cols]
        x, y = create_input_classification(x_test_fold, include_speciation=include_speciation, target_col=target_col)
        x_test_fold_lst.append(x)
        y_test_fold_lst.append(y)
        df_test = test_sets[split].copy()
        df_test_lst.append(df_test)
        test_set_sizes.append(len(df_test))
    return x_train_fold_lst, y_train_fold_lst, x_test_fold_lst, y_test_fold_lst, df_test_lst, test_set_sizes


def get_balanced_data_adasyn(random_seed: int, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    entries_class_1 = len(y[y == 1])
    entries_class_0 = len(y[y == 0])
    ratio = entries_class_0 / entries_class_1
    if (ratio > 1.17) or (ratio < 0.83):
        ada = ADASYN(random_state=random_seed, sampling_strategy="minority")
        x, y = ada.fit_resample(x, y)
    else:
        log.warn(
            "Dataset too balanced to use ADASYN",
            ratio=ratio,
            entries_class_1=entries_class_1,
            entries_class_0=entries_class_0,
        )
    return x, y



def run_balancing_and_training(
    df: pd.DataFrame,
    x_train_fold_lst: List[np.ndarray], 
    y_train_fold_lst: List[np.ndarray], 
    x_test_fold_lst: List[np.ndarray], 
    y_test_fold_lst: List[np.ndarray],
    test_set_sizes: List[int],
    use_adasyn: bool,
    random_seed: int,
    model,
):  
    lst_accu: List[float] = []
    lst_sensitivity: List[float] = []
    lst_specificity: List[float] = []
    lst_f1: List[float] = []

    for x_train, y_train, x_test, y_test in zip(x_train_fold_lst, y_train_fold_lst, x_test_fold_lst, y_test_fold_lst):  

        if use_adasyn:
            x_train_fold, y_train_fold = get_balanced_data_adasyn(random_seed=random_seed, x=x_train, y=y_train)
            model.fit(x_train_fold, y_train_fold)
        else:
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_fold)
            model.fit(x_train, y_train, sample_weight=sample_weights)

        prediction = model.predict(x_test)
        accuracy, f1, sensitivity, specificity = get_class_results(true=y_test, pred=prediction)

        lst_accu.append(round(accuracy, 4))
        lst_sensitivity.append(round(sensitivity, 4))
        lst_specificity.append(round(specificity, 4))
        lst_f1.append(round(f1, 4))


    average_test_size = round(sum(test_set_sizes) / len(test_set_sizes), 1)
    test_percent = round(((average_test_size) / len(df)) * 100, 1)
    log.info("Test set size", size=average_test_size, percent=f"{test_percent}%")

    return lst_accu, lst_sensitivity, lst_specificity, lst_f1


def skf_classification(
    df: pd.DataFrame,
    random_seed: int,
    nsplits: int,
    use_adasyn: bool,
    include_speciation: bool,
    df_test: pd.DataFrame,
    dataset_name: str,
    target_col: str,
    model,
) -> Tuple[List[float], List[float], List[float], List[float]]:

    df.reset_index(inplace=True, drop=True)

    cols = ["cas", "smiles", "y_true"]
    if include_speciation:
        cols += get_speciation_col_names()

    paper = True if dataset_name == "df_paper" else False
    (
        x_train_fold_lst,
        y_train_fold_lst,
        x_test_fold_lst,
        y_test_fold_lst,
        df_test_lst,
        test_set_sizes,
    ) = skf_class_fixed_testset(
        df=df,
        df_test=df_test,
        nsplits=nsplits,
        random_seed=random_seed,
        include_speciation=include_speciation,
        cols=cols,
        paper=paper,
        target_col=target_col,
    )

    lst_accu, lst_sensitivity, lst_specificity, lst_f1 = run_balancing_and_training(
        df=df,
        x_train_fold_lst=x_train_fold_lst, 
        y_train_fold_lst=y_train_fold_lst, 
        x_test_fold_lst=x_test_fold_lst, 
        y_test_fold_lst=y_test_fold_lst,
        test_set_sizes=test_set_sizes,
        use_adasyn=use_adasyn,
        random_seed=random_seed,
        model=model,
    )

    metrics = ["Balanced accuracy", "Sensitivity", "Specificity"]
    metrics_values = [
        lst_accu,
        lst_sensitivity,
        lst_specificity,
    ]
    for metric, metric_values in zip(metrics, metrics_values):
        log.info(
            f"{metric}: ",
            score="{:.1f}".format(np.mean(metric_values) * 100) + " ± " + "{:.1f}".format(np.std(metric_values) * 100),
        )
    log.info(
        f"F1: ",
        score="{:.2f}".format(np.mean(lst_f1)) + " ± " + "{:.2f}".format(np.std(lst_f1)),
    )
    return (
        lst_accu,
        lst_sensitivity,
        lst_specificity,
        lst_f1,
    )
    


def split_regression_df_with_grouping(
    df: pd.DataFrame,
    nsplits: int,
    column_for_grouping: str,
    random_seed: int,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    df_unique_id = df.drop_duplicates(subset=[column_for_grouping], keep="first").reset_index(drop=True)
    x = df_unique_id[[column_for_grouping]].values

    train_dfs: List[pd.DataFrame] = []
    test_dfs: List[pd.DataFrame] = []

    kf = KFold(n_splits=nsplits, shuffle=True, random_state=random_seed)

    for train_index, test_index in kf.split(x):
        id_to_split: Dict[str, str] = {}
        for id in x[train_index]:
            id_to_split[id[0]] = "train"
        for id in x[test_index]:
            id_to_split[id[0]] = "test"
        df["split"] = df[column_for_grouping].apply(lambda id: id_to_split[id])
        train_df = df.loc[df["split"] == "train"].drop("split", axis=1)
        test_df = df[df["split"] == "test"].drop("split", axis=1)
        train_dfs.append(train_df)
        test_dfs.append(test_df)
    return train_dfs, test_dfs


def split_regression_df_with_grouping_and_fixed_test_set(
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    nsplits: int,
    column_for_grouping: str,
    random_seed: int,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:

    train_dfs: List[pd.DataFrame] = []
    test_dfs: List[pd.DataFrame] = []

    dfs = [df, df_test]
    for i, df in enumerate(dfs):
        if "inchi_from_smiles" not in list(df.columns):
            df_smiles_correct = remove_smiles_with_incorrect_format(df=df, col_name_smiles="smiles")
            dfs[i] = openbabel_convert(
                df=df_smiles_correct,
                input_type="smiles",
                column_name_input="smiles",
                output_type="inchi",
            )
    df = dfs[0]
    df_test = dfs[1]

    _, test_dfs_smallest = split_regression_df_with_grouping(
        df=df_test,
        nsplits=nsplits,
        column_for_grouping=column_for_grouping,
        random_seed=random_seed,
    )

    for df_test in test_dfs_smallest:
        inchi_for_test = df_test["inchi_from_smiles"].tolist()

        train_df = df[~df["inchi_from_smiles"].isin(inchi_for_test)]
        train_dfs.append(train_df)
        test_dfs.append(df_test)

    return train_dfs, test_dfs


def create_train_test_sets_regression(
    train_df: pd.DataFrame, test_df: pd.DataFrame, include_speciation: bool
) -> Tuple[np.ndarray, pd.Series, np.ndarray, pd.Series]:
    x_train = create_input_regression(df=train_df, include_speciation=include_speciation)
    y_train = train_df["biodegradation_percent"]
    x_test = create_input_regression(df=test_df, include_speciation=include_speciation)
    y_test = test_df["biodegradation_percent"]
    return x_train, y_train, x_test, y_test


def kf_regression(
    df: pd.DataFrame,
    column_for_grouping: str,
    random_seed: int,
    nsplits: int,
    include_speciation: bool,
    fixed_testset: bool,
    df_test: pd.DataFrame,
    dataset_name: str,
    model,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    lst_rmse_stratified: List[float] = []
    lst_mae_stratified: List[float] = []
    lst_r2_stratified: List[float] = []
    lst_mse_stratified: List[float] = []
    if fixed_testset:
        train_dfs, test_dfs = split_regression_df_with_grouping_and_fixed_test_set(
            df=df,
            df_test=df_test,
            nsplits=nsplits,
            column_for_grouping=column_for_grouping,
            random_seed=random_seed,
        )
    else:
        train_dfs, test_dfs = split_regression_df_with_grouping(
            df=df,
            nsplits=nsplits,
            column_for_grouping=column_for_grouping,
            random_seed=random_seed,
        )

    test_set_sizes: List[int] = []
    for i, (df_train, df_test) in enumerate(zip(train_dfs, test_dfs)):
        x_train, y_train, x_test, y_test = create_train_test_sets_regression(
            train_df=df_train, test_df=df_test, include_speciation=include_speciation
        )
        test_set_sizes.append(len(df_test))

        model.fit(x_train, y_train)
        prediction = model.predict(x_test)

        rmse = mean_squared_error(y_test, prediction, squared=False)
        mae = mean_absolute_error(y_test, prediction)
        r2 = r2_score(y_test, prediction)
        mse = mean_squared_error(y_test, prediction)
        lst_rmse_stratified.append(round(rmse, 4))
        lst_mae_stratified.append(round(mae, 4))
        lst_r2_stratified.append(round(r2, 4))
        lst_mse_stratified.append(round(mse, 4))

        if i == 1:
            df_test["prediction"] = model.predict(x_test)
            test_set = ""
            if fixed_testset:
                test_set = "_fixed_testset"
            plot_regression_error(
                df=df_test,
                figure_name=f"XGBR_trained_on_{dataset_name}{test_set}",
                figure_title=f"XGBR trained on {dataset_name}",
            )

    average_test_size = round(sum(test_set_sizes) / len(test_set_sizes), 1)
    test_percent = round((average_test_size / len(df)) * 100, 1)
    log.info("Test set size", size=average_test_size, percent=f"{test_percent}%")
    metrics = ["RMSE", "MAE", "R2", "MSE"]
    metrics_values = [
        lst_rmse_stratified,
        lst_mae_stratified,
        lst_r2_stratified,
        lst_mse_stratified,
    ]
    for metric, metric_values in zip(metrics, metrics_values):
        log.info(
            f"{metric}: ",
            score="{:.2f}".format(np.mean(metric_values)) + " ± " + "{:.2f}".format(np.std(metric_values)),
        )
    return (
        lst_rmse_stratified,
        lst_mae_stratified,
        lst_r2_stratified,
        lst_mse_stratified,
    )


def report_perf_hyperparameter_tuning(optimizer, x, y, title="Model", callbacks=None):
    start = time()
    optimizer.fit(x, y, callback=callbacks)

    df_cv_results = pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = df_cv_results.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_
    log.info(
        (title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f " + "\u00B1" + " %.3f")
        % (
            time() - start,
            len(optimizer.cv_results_["params"]),
            best_score,
            best_score_std,
        )
    )
    log.info(
        "Mean train score",
        mean_train_score=round(np.mean(optimizer.cv_results_["mean_train_score"]), 3),
        std_train_score=round(np.mean(optimizer.cv_results_["std_train_score"]), 3),
    )
    log.info(
        "Mean test score",
        mean_test_score=round(np.mean(optimizer.cv_results_["mean_test_score"]), 3),
        std_test_score=round(np.mean(optimizer.cv_results_["std_test_score"]), 3),
    )
    return best_params


def get_Huang_Zhang_regression_parameters() -> Dict[str, object]:
    best_params = {
        "alpha": 2.41,
        "base_score": 0.5,
        "booster": "gbtree",
        "colsample_bylevel": 0.81,
        "colsample_bynode": 0.67,
        "colsample_bytree": 0.58,
        "gamma": 0.02,
        "importance_type": "gain",
        "learning_rate": 0.50,
        "max_delta_step": 3.41,
        "max_depth": 24,
        "min_child_weight": 2.12,
        "n_estimators": 305,
        "num_parallel_tree": 8,
        "reg_alpha": 2.41,
        "reg_lambda": 2.61,
        "scale_pos_weight": 1,
        "subsample": 0.58,
        "tree_method": "exact",
        "validate_parameters": 1,
    }
    return best_params


def train_XGBRegressor_Huang_Zhang(
    df: pd.DataFrame,
    column_for_grouping: str,
    random_seed: int,
    nsplits: int,
    include_speciation: bool,
    fixed_testset: bool,
    df_test: pd.DataFrame,
    dataset_name: str,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    model = XGBRegressor
    best_params = get_Huang_Zhang_regression_parameters()

    rmse, mae, r2, mse = kf_regression(
        df=df,
        column_for_grouping=column_for_grouping,
        random_seed=random_seed,
        nsplits=nsplits,
        include_speciation=include_speciation,
        fixed_testset=fixed_testset,
        df_test=df_test,
        dataset_name=dataset_name,
        model=model(**best_params),
    )
    log.info("RMSE: ", rmse=rmse)
    log.info("MAE: ", mae=mae)
    log.info("r2: ", r2=r2)
    log.info("MSE: ", mse=mse)
    return rmse, mae, r2, mse


def train_XGBClassifier(
    df: pd.DataFrame,
    random_seed: int,
    nsplits: int,
    use_adasyn: bool,
    include_speciation: bool,
    df_test: pd.DataFrame,
    dataset_name: str,
    target_col: str,
) -> Tuple[List[float], List[float], List[float], List[float]]:

    accu, sensitivity, specificity, f1 = skf_classification(
        df=df,
        random_seed=random_seed,
        nsplits=nsplits,
        use_adasyn=use_adasyn,
        include_speciation=include_speciation,
        df_test=df_test,
        dataset_name=dataset_name,
        target_col=target_col,
        model=XGBClassifier(),  # Default parameters as in paper
    )
    log.info("Balanced accuracy", accuracy=accu)
    log.info("Sensitivity", sensitivity=sensitivity)
    log.info("Specificity", specificity=specificity)
    log.info("F1", f1=f1)
    return accu, f1, sensitivity, specificity


def train_XGBClassifier_on_all_data( 
    df: pd.DataFrame,
    random_seed: int,
    include_speciation: bool,
) -> XGBClassifier:

    x, y = create_input_classification(df, include_speciation=include_speciation, target_col="y_true")
    x, y = get_balanced_data_adasyn(random_seed=random_seed, x=x, y=y)

    model_class = XGBClassifier()
    model_class.fit(x, y)

    return model_class


def plot_regression_error(
    df: pd.DataFrame,
    figure_name: str,
    figure_title: str,
    with_title=False,
) -> None:
    true = df["biodegradation_percent"]
    pred = df["prediction"]
    r2 = r2_score(true, pred)

    plt.scatter(true, pred, c="cornflowerblue")
    p1 = max(max(pred), max(true))
    p2 = min(min(pred), min(true))
    plt.plot([p1, p2], [p1, p2], "g-", linewidth=3)  # plotting identity line
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Experimental values", fontsize=18)
    plt.ylabel("Predicted values", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if with_title:
        plt.title(f"{figure_title}", fontsize=18)
    plt.legend([(r"R$^2$" + f"= {'%.2f' % r2}"), "identity"], prop={"size": 16}, loc="upper left")
    plt.tight_layout()
    plt.savefig(f"figures/regression_prediction_error_{figure_name}.png")
    plt.close()


def analyze_regression_results_and_plot(df: pd.DataFrame, figure_name: str, figure_title: str) -> None:
    rmse = mean_squared_error(
        df["biodegradation_percent"],
        df["prediction"],
        squared=False,
    )
    mae = mean_absolute_error(df["biodegradation_percent"], df["prediction"])
    r2 = r2_score(df["biodegradation_percent"], df["prediction"])
    mse = mean_squared_error(df["biodegradation_percent"], df["prediction"])
    log.info("RMSE", rmse="{:.2f}".format(rmse))
    log.info("MAE", mae="{:.2f}".format(mae))
    log.info(f"R2", r2="{:.2f}".format(r2))
    log.info("MSE", mse="{:.2f}".format(mse))
    plot_regression_error(df, figure_name, figure_title)


