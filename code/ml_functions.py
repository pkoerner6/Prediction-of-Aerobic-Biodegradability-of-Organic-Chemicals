import pandas as pd
import numpy as np
import structlog
import sys
import os
import pickle
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from time import time
from collections import OrderedDict
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgbm
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.class_weight import compute_sample_weight

np.int = int  # Because of scikit-optimize
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper

log = structlog.get_logger()


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from processing_functions import create_input_classification
from processing_functions import create_input_regression
from processing_functions import get_speciation_col_names
from processing_functions import openbabel_convert
from processing_functions import remove_smiles_with_incorrect_format
from processing_functions import get_inchi_main_layer


def get_class_results(true: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float, float]:
    accuracy = balanced_accuracy_score(true, pred)
    f1 = f1_score(true, pred)
    sensitivity = recall_score(true, pred)
    specificity = recall_score(true, pred, pos_label=0)
    return accuracy, f1, sensitivity, specificity


def print_class_results(accuracy: float, sensitivity: float, specificity: float, f1: float) -> float:
    log.info("Accuracy score", accuracy="{:.1f}".format(accuracy * 100))
    log.info("Sensitivity", sensitivity="{:.1f}".format(sensitivity * 100))
    log.info("Specificity", specificity="{:.1f}".format(specificity * 100))
    log.info("F1", f1="{:.1f}".format(f1 * 100))
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
    df = dfs[0][cols + ["inchi_from_smiles", "y_true"]]
    df_test = dfs[1][cols + ["inchi_from_smiles", "y_true"]]

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
        # log.info("Test sizes: ", test_set=len(df_test_set), deleted=len(df)-len(train_set)) # TODO
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
        x_train_fold_lst.append(create_input_classification(x_train_fold, include_speciation=include_speciation))
        y_train_fold_lst.append(train_sets[split]["y_true"])
        x_test_fold = test_sets[split][cols]
        x_test_fold_lst.append(create_input_classification(x_test_fold, include_speciation=include_speciation))
        y_test_fold_lst.append(test_sets[split]["y_true"])
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
    model,
) -> Tuple[List[float], List[float], List[float], List[float]]:

    df.reset_index(inplace=True, drop=True)

    cols = ["cas", "smiles"]
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


def tune_regressor(
    df: pd.DataFrame,
    random_seed: int,
    nsplits: int,
    search_spaces: Dict,
    include_speciation: bool,
    fixed_testset: bool,
    n_jobs: int,
    model,
) -> Dict:
    scoring = make_scorer(
        r2_score, greater_is_better=True
    )  # if want rmse, use: make_scorer(mean_squared_error, greater_is_better=False)

    train_dfs, test_dfs = split_regression_df_with_grouping(
        df=df,
        nsplits=nsplits,
        column_for_grouping="cas",
        random_seed=random_seed,
    )

    cv_strategy: List = []
    for train_df, test_df in zip(train_dfs, test_dfs):
        cv_strategy.append((train_df.index, test_df.index))

    x = create_input_regression(df=df, include_speciation=include_speciation)
    y = df["biodegradation_percent"]

    opt = BayesSearchCV(
        estimator=model(),
        search_spaces=search_spaces,
        scoring=scoring,
        cv=cv_strategy,
        n_iter=200,
        n_points=5,
        n_jobs=n_jobs,
        return_train_score=True,
        refit=False,
        optimizer_kwargs={"base_estimator": "GP"},
        random_state=random_seed,
        verbose=0,
    )
    overdone_control = DeltaYStopper(delta=0.0001)  # Stop if the gain of the optimization becomes too small
    time_limit_control = DeadlineStopper(total_time=60 * 60 * 4)
    opt.fit(x, y, callback=[overdone_control, time_limit_control])
    best_params = report_perf_hyperparameter_tuning(opt, x, y, callbacks=[overdone_control, time_limit_control])

    log.info("Best hyperparameters", best_params=best_params)
    return best_params


def train_regressor_with_best_hyperparamters(
    df: pd.DataFrame,
    random_seed: int,
    nsplits: int,
    include_speciation: bool,
    fixed_testset: bool,
    df_test: pd.DataFrame,
    dataset_name: str,
    best_params: Dict,
    model,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    rmse, mae, r2, mse = kf_regression(
        df=df,
        column_for_grouping="inchi_from_smiles",
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


def tune_and_train_regressor(
    df: pd.DataFrame,
    random_seed: int,
    nsplits: int,
    search_spaces: Dict,
    include_speciation: bool,
    fixed_testset: bool,
    df_test: pd.DataFrame,
    dataset_name: str,
    n_jobs: int,
    model,
) -> Tuple[List[float], List[float], List[float], List[float]]:

    best_params = tune_regressor(
        df=df,
        random_seed=random_seed,
        nsplits=nsplits,
        search_spaces=search_spaces,
        include_speciation=include_speciation,
        fixed_testset=fixed_testset,
        n_jobs=n_jobs,
        model=model,
    )

    rmse, mae, r2, mse = train_regressor_with_best_hyperparamters(
        df=df,
        random_seed=random_seed,
        nsplits=nsplits,
        include_speciation=include_speciation,
        fixed_testset=fixed_testset,
        df_test=df_test,
        dataset_name=dataset_name,
        best_params=best_params,
        model=model,
    )
    return rmse, mae, r2, mse


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


def get_lazy_xgbr_parameters() -> Dict[str, object]:
    best_params = {
        "alpha": 4.7780585065361,
        "base_score": 0.8055328348792045,
        "booster": "gbtree",
        "colsample_bylevel": 0.7399509073594608,
        "colsample_bynode": 0.17398551573835808,
        "colsample_bytree": 0.6995726624815766,
        "gamma": 0.03235404457038847,
        "importance_type": "gain",
        "lambda": 0.7084783537630203,
        "learning_rate": 0.16214176723141335,
        "max_delta_step": 0.09230985280173427,
        "max_depth": 40,
        "min_child_weight": 1.8340273097140976,
        "n_estimators": 971,
        "num_parallel_tree": 16,
        "reg_alpha": 2.4987822315018455,
        "reg_lambda": 3.4693236850371694,
        "scale_pos_weight": 0.49009783649381233,
        "subsample": 0.9420718385084459,
        "tree_method": "exact",
        "validate_parameters": 1,
    }
    return best_params


def get_lazy_xgbc_parameters() -> Dict[str, object]:
    best_params = {
        "alpha": 2.7736319580937456,
        "base_score": 0.4808439580915848,
        "booster": "gbtree",
        "colsample_bylevel": 0.9153371495471292,
        "colsample_bynode": 0.3042219673148875,
        "colsample_bytree": 0.9604420158222629,
        "gamma": 0.009848233225016628,
        "lambda": 0.403432692925663,
        "learning_rate": 0.2311373411076377,
        "max_delta_step": 3.061110290373153,
        "max_depth": 206,
        "min_child_weight": 1.3585293908683982,
        "n_estimators": 5155,
        "num_parallel_tree": 20,
        "scale_pos_weight": 1.8583803082857837,
        "subsample": 0.7622612926036432,
        "tree_method": "exact",
        "validate_parameters": True,
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


def tune_classifiers(
    df: pd.DataFrame,
    random_seed: int,
    nsplits: int,
    search_spaces: Dict,
    include_speciation: bool,
    n_jobs: int,
    model,
):
    x = create_input_classification(df, include_speciation)
    y = df["y_true"]

    x, y = get_balanced_data_adasyn(random_seed=random_seed, x=x, y=y)

    scoring = make_scorer(balanced_accuracy_score, greater_is_better=True)
    skf = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=random_seed)
    cv_strategy = list(skf.split(x, y))

    opt = BayesSearchCV(
        estimator=model(),
        search_spaces=search_spaces,
        scoring=scoring,
        cv=cv_strategy,
        n_iter=120,
        n_points=5,  # number of hyperparameter sets evaluated at the same time
        n_jobs=n_jobs,
        return_train_score=True,
        refit=False,
        optimizer_kwargs={"base_estimator": "GP"},  # optmizer parameters: use Gaussian Process (GP)
        random_state=random_seed,
        verbose=0,
    )
    overdone_control = DeltaYStopper(delta=0.0001)  # Stop if the gain of the optimization becomes too small
    time_limit_control = DeadlineStopper(total_time=60 * 60 * 4)
    best_params = report_perf_hyperparameter_tuning(
        opt,
        x,
        y,
        callbacks=[overdone_control, time_limit_control],
    )

    log.info("Best hyperparameters", best_params=best_params)
    return best_params


def train_classifier_with_best_hyperparamters(
    df: pd.DataFrame,
    random_seed: int,
    nsplits: int,
    include_speciation: bool,
    df_test: pd.DataFrame,
    dataset_name: str,
    best_params: Dict,
    paper: bool, 
    model,
):
    accu, sensitivity, specificity, f1 = skf_classification(
        df=df,
        random_seed=random_seed,
        nsplits=nsplits,
        use_adasyn=True,
        include_speciation=include_speciation,
        df_test=df_test,
        dataset_name=dataset_name,
        model=model(**best_params),
    )
    log.info("Accuracy: ", accuracy=accu)
    log.info("Sensitivity: ", sensitivity=sensitivity)
    log.info("Specificity: ", specificity=specificity)
    log.info("F1: ", f1=f1)
    return accu, f1, sensitivity, specificity


def tune_and_train_classifiers(
    df: pd.DataFrame,
    random_seed: int,
    nsplits: int,
    search_spaces: Dict,
    include_speciation: bool,
    df_test: pd.DataFrame,
    dataset_name: str,
    n_jobs: int,
    model,
) -> Tuple[List[float], List[float], List[float], List[float]]:

    best_params = tune_classifiers(
        df=df,
        random_seed=random_seed,
        nsplits=nsplits,
        search_spaces=search_spaces,
        include_speciation=include_speciation,
        n_jobs=n_jobs,
        model=model,
    )
    paper = True if dataset_name == "df_paper" else False
    accu, f1, sensitivity, specificity = train_classifier_with_best_hyperparamters(
        df=df,
        random_seed=random_seed,
        nsplits=nsplits,
        include_speciation=include_speciation,
        df_test=df_test,
        dataset_name=dataset_name,
        best_params=best_params,
        paper=paper,
        model=model,
    )
    return accu, f1, sensitivity, specificity


def train_XGBClassifier( # run_XGBClassifier_Huang_Zhang
    df: pd.DataFrame,
    random_seed: int,
    nsplits: int,
    use_adasyn: bool,
    include_speciation: bool,
    df_test: pd.DataFrame,
    dataset_name: str,
) -> Tuple[List[float], List[float], List[float], List[float]]:

    accu, sensitivity, specificity, f1 = skf_classification(
        df=df,
        random_seed=random_seed,
        nsplits=nsplits,
        use_adasyn=use_adasyn,
        include_speciation=include_speciation,
        df_test=df_test,
        dataset_name=dataset_name,
        model=XGBClassifier(),  # Default parameters as in paper
    )
    log.info("Accuracy: ", accuracy=accu)
    log.info("Sensitivity: ", sensitivity=sensitivity)
    log.info("Specificity: ", specificity=specificity)
    log.info("F1: ", f1=f1)
    return accu, f1, sensitivity, specificity


def train_XGBClassifier_on_all_data( # train_XGBClassifier_Huang_Zhang_on_all_data
    df: pd.DataFrame,
    random_seed: int,
    use_adasyn: bool,
    include_speciation: bool,
) -> XGBClassifier:

    x = create_input_classification(df, include_speciation=include_speciation)
    y = df["y_true"]
    x, y = get_balanced_data_adasyn(random_seed=random_seed, x=x, y=y)

    model_class = XGBClassifier()
    model_class.fit(x, y)

    return model_class


def train_class_model_on_all_data(
    df: pd.DataFrame,
    random_seed: int,
    use_adasyn: bool,
    include_speciation: bool,
    model_with_best_hyperparams,
):
    x = create_input_classification(df, include_speciation=include_speciation)
    y = df["y_true"]
    x, y = get_balanced_data_adasyn(random_seed=random_seed, x=x, y=y)
    model_class = model_with_best_hyperparams
    model_class.fit(x, y)
    return model_class


def train_reg_model_on_all_data(
    df: pd.DataFrame,
    random_seed: int,
    include_speciation: bool,
    model_with_best_hyperparams,
):
    x = create_input_regression(df, include_speciation=include_speciation)
    y = df["biodegradation_percent"]
    model_reg = model_with_best_hyperparams
    model_reg.fit(x, y)
    return model_reg


def tune_and_train_XGBRegressor(
    df: pd.DataFrame,
    random_seed: int,
    nsplits: int,
    include_speciation: bool,
    fixed_testset: bool,
    df_test: pd.DataFrame,
    dataset_name: str,
    n_jobs: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    model = XGBRegressor
    search_spaces = {
        "alpha": Real(3.5, 5.5, "uniform"),
        "base_score": Real(0.7, 0.99, "uniform"),
        "booster": Categorical(["gbtree"]),
        "colsample_bylevel": Real(0.65, 0.9, "uniform"),
        "colsample_bynode": Real(0.1, 0.3, "uniform"),
        "colsample_bytree": Real(0.5, 0.8, "uniform"),
        "gamma": Real(0.001, 0.1, "uniform"),
        "importance_type": Categorical(["gain"]),
        "lambda": Real(0.6, 0.8, "uniform"),
        "learning_rate": Real(0.01, 0.3, "uniform"),
        "max_delta_step": Real(0.01, 0.2, "uniform"),
        "max_depth": Integer(30, 500),
        "min_child_weight": Real(1.0, 2.5, "uniform"),
        "n_estimators": Integer(800, 1500),
        "num_parallel_tree": Integer(1, 50),
        "reg_alpha": Real(1.8, 3.0, "uniform"),
        "reg_lambda": Real(2.5, 4.5, "uniform"),
        "scale_pos_weight": Real(0.3, 0.7, "uniform"),
        "subsample": Real(0.8, 0.99, "uniform"),
        "tree_method": Categorical(["auto", "exact", "hist"]),
        "validate_parameters": Categorical([1]),
    }

    rmse, mae, r2, mse = tune_and_train_regressor(
        df=df,
        random_seed=random_seed,
        nsplits=nsplits,
        search_spaces=search_spaces,
        include_speciation=include_speciation,
        fixed_testset=fixed_testset,
        df_test=df_test,
        dataset_name=dataset_name,
        n_jobs=n_jobs,
        model=model,
    )
    return rmse, mae, r2, mse


def tune_and_train_HistGradientBoostingRegressor(
    df: pd.DataFrame,
    random_seed: int,
    nsplits: int,
    include_speciation: bool,
    fixed_testset: bool,
    df_test: pd.DataFrame,
    dataset_name: str,
    n_jobs: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    model = HistGradientBoostingRegressor
    search_spaces = {
        "learning_rate": Real(0.01, 0.1, "uniform"),
        "loss": Categorical(["squared_error", "absolute_error", "poisson", "quantile"]),
        "max_depth": Categorical([None]),
        "max_iter": Integer(200, 1400),
        "max_leaf_nodes": Categorical([None]),
        "min_samples_leaf": Integer(10, 300),
        "quantile": Real(0.1, 0.99, "uniform"),
        "random_state": Categorical([random_seed]),
        "tol": Real(1e-11, 1e-2, "uniform"),
    }

    rmse, mae, r2, mse = tune_and_train_regressor(
        df=df,
        random_seed=random_seed,
        nsplits=nsplits,
        search_spaces=search_spaces,
        include_speciation=include_speciation,
        fixed_testset=fixed_testset,
        df_test=df_test,
        dataset_name=dataset_name,
        n_jobs=n_jobs,
        model=model,
    )
    return rmse, mae, r2, mse


def tune_and_train_RandomForestRegressor(
    df: pd.DataFrame,
    random_seed: int,
    nsplits: int,
    include_speciation: bool,
    fixed_testset: bool,
    df_test: pd.DataFrame,
    dataset_name: str,
    n_jobs: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    model = RandomForestRegressor
    search_spaces = {
        "criterion": Categorical(["squared_error", "absolute_error", "friedman_mse", "poisson"]),
        "max_features": Categorical(["sqrt", "log2", None]),
        "min_samples_leaf": Integer(1, 10),
        "min_samples_split": Integer(2, 10),
        "n_estimators": Integer(500, 1000),
        "random_state": Categorical([random_seed]),
    }

    rmse, mae, r2, mse = tune_and_train_regressor(
        df=df,
        random_seed=random_seed,
        nsplits=nsplits,
        search_spaces=search_spaces,
        include_speciation=include_speciation,
        fixed_testset=fixed_testset,
        df_test=df_test,
        dataset_name=dataset_name,
        n_jobs=n_jobs,
        model=model,
    )
    return rmse, mae, r2, mse


def tune_and_train_LGBMRegressor(
    df: pd.DataFrame,
    random_seed: int,
    nsplits: int,
    include_speciation: bool,
    fixed_testset: bool,
    df_test: pd.DataFrame,
    dataset_name: str,
    n_jobs: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    model = lgbm.LGBMRegressor
    search_spaces: Dict = {
        "boosting_type": Categorical(["gbdt", "dart"]),
        "learning_rate": Real(0.01, 0.5, "uniform"),
        "max_depth": Categorical([-1]),
        "min_child_samples": Integer(1, 20),
        "n_estimators": Integer(200, 1800),
        "num_leaves": Integer(5, 100),
        "random_state": Categorical([random_seed]),
    }

    rmse, mae, r2, mse = tune_and_train_regressor(
        df=df,
        random_seed=random_seed,
        nsplits=nsplits,
        search_spaces=search_spaces,
        include_speciation=include_speciation,
        fixed_testset=fixed_testset,
        df_test=df_test,
        dataset_name=dataset_name,
        n_jobs=n_jobs,
        model=model,
    )
    return rmse, mae, r2, mse


def tune_and_train_SVR(
    df: pd.DataFrame,
    random_seed: int,
    nsplits: int,
    include_speciation: bool,
    fixed_testset: bool,
    df_test: pd.DataFrame,
    dataset_name: str,
    n_jobs: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    model = SVR
    search_spaces: Dict = {
        "C": Real(0.3, 30.0, "uniform"),
        "coef0": Real(5.0, 30.0, "uniform"),
        "degree": Integer(2, 3),
        "gamma": Categorical(["scale", "auto"]),
        "kernel": Categorical(["poly", "rbf", "sigmoid"]),
        "max_iter": Categorical([-1]),
        "tol": Real(1e-4, 1e-2, "uniform"),
    }

    rmse, mae, r2, mse = tune_and_train_regressor(
        df=df,
        random_seed=random_seed,
        nsplits=nsplits,
        search_spaces=search_spaces,
        include_speciation=include_speciation,
        fixed_testset=fixed_testset,
        df_test=df_test,
        dataset_name=dataset_name,
        n_jobs=n_jobs,
        model=model,
    )
    return rmse, mae, r2, mse


def tune_and_train_LGBMClassifier(
    df: pd.DataFrame,
    random_seed: int,
    nsplits: int,
    include_speciation: bool,
    df_test: pd.DataFrame,
    dataset_name: str,
    n_jobs: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    model = lgbm.LGBMClassifier
    search_spaces = {
        "boosting_type": Categorical(["gbdt"]),
        "learning_rate": Real(0.001, 0.1, "uniform"),
        "max_depth": Categorical([-1]),
        "n_estimators": Integer(100, 1600),
        "num_leaves": Integer(100, 1000),
        "objective": Categorical(["binary"]),
    }

    accu, f1, sensitivity, specificity = tune_and_train_classifiers(
        df=df,
        random_seed=random_seed,
        nsplits=nsplits,
        search_spaces=search_spaces,
        include_speciation=include_speciation,
        df_test=df_test,
        dataset_name=dataset_name,
        n_jobs=n_jobs,
        model=model,
    )
    return accu, f1, sensitivity, specificity


def tune_and_train_XGBClassifier(
    df: pd.DataFrame,
    random_seed: int,
    nsplits: int,
    include_speciation: bool,
    df_test: pd.DataFrame,
    dataset_name: str,
    n_jobs: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    model = XGBClassifier
    search_spaces = {
        "alpha": Real(1.0, 4.0, "uniform"),
        "base_score": Real(0.4, 0.60, "uniform"),
        "booster": Categorical(["gbtree"]),
        "colsample_bylevel": Real(0.7, 1.0, "uniform"),
        "colsample_bynode": Real(0.1, 0.5, "uniform"),
        "colsample_bytree": Real(0.7, 1.0, "uniform"),
        "gamma": Real(0.001, 0.1, "uniform"),
        "lambda": Real(0.1, 0.6, "uniform"),
        "learning_rate": Real(0.01, 0.8, "uniform"),
        "max_delta_step": Real(2.0, 4.0, "uniform"),
        "max_depth": Integer(100, 300),
        "min_child_weight": Real(1.0, 3.0, "uniform"),
        "n_estimators": Integer(4800, 5400),
        "num_parallel_tree": Integer(15, 25),
        "scale_pos_weight": Real(0.5, 3.0, "uniform"),
        "subsample": Real(0.6, 0.99, "uniform"),
        "tree_method": Categorical(["exact"]),
        "validate_parameters": Categorical([True]),
    }

    accu, f1, sensitivity, specificity = tune_and_train_classifiers(
        df=df,
        random_seed=random_seed,
        nsplits=nsplits,
        search_spaces=search_spaces,
        include_speciation=include_speciation,
        df_test=df_test,
        dataset_name=dataset_name,
        n_jobs=n_jobs,
        model=model,
    )
    return accu, f1, sensitivity, specificity


def tune_and_train_ExtraTreesClassifier(
    df: pd.DataFrame,
    random_seed: int,
    nsplits: int,
    include_speciation: bool,
    df_test: pd.DataFrame,
    dataset_name: str,
    n_jobs: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    model = ExtraTreesClassifier
    search_spaces = {
        "criterion": Categorical(["gini", "entropy", "log_loss"]),
        "max_depth": Categorical([None]),
        "max_features": Categorical(["sqrt", "log2", None]),
        "min_samples_leaf": Integer(1, 10),
        "min_samples_split": Integer(2, 10),
        "n_estimators": Integer(1000, 3000),
        "random_state": Categorical([random_seed]),
    }

    accu, f1, sensitivity, specificity = tune_and_train_classifiers(
        df=df,
        random_seed=random_seed,
        nsplits=nsplits,
        search_spaces=search_spaces,
        include_speciation=include_speciation,
        df_test=df_test,
        dataset_name=dataset_name,
        n_jobs=n_jobs,
        model=model,
    )
    return accu, f1, sensitivity, specificity


def tune_and_train_RandomForestClassifier(
    df: pd.DataFrame,
    random_seed: int,
    nsplits: int,
    include_speciation: bool,
    df_test: pd.DataFrame,
    dataset_name: str,
    n_jobs: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    model = RandomForestClassifier
    search_spaces = {
        "criterion": Categorical(["gini", "entropy", "log_loss"]),
        "max_features": Categorical(["sqrt", "log2", None]),
        "min_samples_leaf": Integer(1, 5),
        "min_samples_split": Integer(2, 5),
        "n_estimators": Integer(1000, 2500),
        "random_state": Categorical([random_seed]),
    }

    accu, f1, sensitivity, specificity = tune_and_train_classifiers(
        df=df,
        random_seed=random_seed,
        nsplits=nsplits,
        search_spaces=search_spaces,
        include_speciation=include_speciation,
        df_test=df_test,
        dataset_name=dataset_name,
        n_jobs=n_jobs,
        model=model,
    )
    return accu, f1, sensitivity, specificity


def tune_and_train_BaggingClassifier(
    df: pd.DataFrame,
    random_seed: int,
    nsplits: int,
    include_speciation: bool,
    df_test: pd.DataFrame,
    dataset_name: str,
    n_jobs: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    model = BaggingClassifier
    search_spaces = {
        "max_features": Integer(1, 164),
        "max_samples": Integer(5, 3000),
        "n_estimators": Integer(200, 3000),
        "n_jobs": Categorical([-1]),
        "random_state": Categorical([random_seed]),
    }

    accu, f1, sensitivity, specificity = tune_and_train_classifiers(
        df=df,
        random_seed=random_seed,
        nsplits=nsplits,
        search_spaces=search_spaces,
        include_speciation=include_speciation,
        df_test=df_test,
        dataset_name=dataset_name,
        n_jobs=n_jobs,
        model=model,
    )
    return accu, f1, sensitivity, specificity


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



def get_best_classifier_readded(train_new: bool, random_seed: int) -> Tuple[XGBClassifier, np.ndarray]:
    file_name = "models/xgbc.pkl"

    df_class_readded = pd.read_csv(
        "datasets/curated_data/class_curated_scs_biowin_readded.csv", index_col=0
    )
    best_params = get_lazy_xgbc_parameters()
    classifier = XGBClassifier(**best_params)

    x = create_input_classification(df_class_readded, include_speciation=False)
    y = df_class_readded["y_true"]
    x, y = get_balanced_data_adasyn(random_seed=random_seed, x=x, y=y)

    if train_new: 
        log.info("Fitting model")
        classifier.fit(x, y)
        log.info("Finished fitting model")
        pickle.dump(classifier, open(file_name, "wb"))

    else: 
        classifier = pickle.load(open(file_name, "rb"))

    return classifier, x
