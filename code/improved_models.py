import argparse
import pandas as pd
import numpy as np
import structlog
import sys
import os

log = structlog.get_logger()
from typing import List, Tuple
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin
import lightgbm as lgbm
from xgboost import XGBRegressor

parser = argparse.ArgumentParser()

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from biodegradation.processing_functions import create_input_regression
from biodegradation.processing_functions import create_input_classification
from biodegradation.ml_functions import tune_and_train_XGBRegressor
from biodegradation.ml_functions import tune_and_train_XGBClassifier
from biodegradation.ml_functions import split_regression_df_with_grouping
from biodegradation.ml_functions import get_balanced_data_adasyn
from biodegradation.ml_functions import tune_and_train_HistGradientBoostingRegressor
from biodegradation.ml_functions import tune_and_train_RandomForestRegressor
from biodegradation.ml_functions import tune_and_train_SVR
from biodegradation.ml_functions import tune_and_train_LGBMRegressor
from biodegradation.ml_functions import tune_and_train_LGBMClassifier
from biodegradation.ml_functions import tune_and_train_ExtraTreesClassifier
from biodegradation.ml_functions import tune_and_train_RandomForestClassifier
from biodegradation.ml_functions import tune_and_train_BaggingClassifier

from biodegradation.model_results import results_lazy_classifiers_nsplit5_seed42
from biodegradation.model_results import results_lazy_regressors_nsplit5_seed42


parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    type=str,
    choices=["classification", "regression", "both"],
    default="both",
    help="Training modus: either classification or regression",
)
parser.add_argument(
    "--run_lazy",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Whether to run lazy regressor and classifier",
)
parser.add_argument(
    "--random_seed",
    type=int,
    default=42,
    help="Select the random seed",
)
parser.add_argument(
    "--nsplits",
    type=int,
    default=5,
    help="Select the number of splits for cross validation",
)
parser.add_argument(
    "--plot",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Whether plot the results",
)
parser.add_argument(
    "--train_new",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Whether run the hyperparamter tuning again and train the models",
)
parser.add_argument(
    "--njobs",
    default=1,
    type=int,
    help="Define n_jobs: 1 when running locally or 12 when running on Euler",
)

args = parser.parse_args()


def get_improved_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_reg_improved = pd.read_csv(
        "biodegradation/dataframes/improved_data/reg_improved_env_biowin_both_readded.csv", index_col=0
    )
    df_class_improved = pd.read_csv(
        "biodegradation/dataframes/improved_data/class_improved_env_biowin_both_readded.csv", index_col=0
    )
    return df_class_improved, df_reg_improved


def get_classification_model_input(df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series, np.ndarray, pd.Series]:
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=args.random_seed)
    train_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)

    x_train = create_input_classification(train_df, include_speciation=False)
    y_train = train_df["y_true"]
    x_test = create_input_classification(test_df, include_speciation=False)
    y_test = test_df["y_true"]

    x_balanced, y_balanced = get_balanced_data_adasyn(random_seed=args.random_seed, x=x_train, y=y_train)

    return x_balanced, y_balanced, x_test, y_test


def get_regression_model_input(df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series, np.ndarray, pd.Series]:
    train_dfs, test_dfs = split_regression_df_with_grouping(
        df=df, nsplits=args.nsplits, column_for_grouping="cas", random_seed=args.random_seed
    )
    train_df = train_dfs[0].reset_index(drop=True)
    test_df = test_dfs[0].reset_index(drop=True)
    x_train = create_input_regression(df=train_df, include_speciation=False)
    y_train = train_df["biodegradation_percent"]
    x_test = create_input_regression(df=test_df, include_speciation=False)
    y_test = test_df["biodegradation_percent"]
    return x_train, y_train, x_test, y_test


def run_lazy_classifier(df_class: pd.DataFrame) -> None:
    (
        class_x_balanced,
        class_y_balanced,
        class_x_test,
        class_y_test,
    ) = get_classification_model_input(df_class)
    clf = LazyClassifier(predictions=True)
    models, _ = clf.fit(class_x_balanced, class_x_test, class_y_balanced, class_y_test)
    log.info(models)


def run_lazy_regressor(df_reg: pd.DataFrame) -> None:
    x_train, y_train, x_test, y_test = get_regression_model_input(df=df_reg)
    chosen_regressors = [
        "AdaBoostRegressor",
        "BaggingRegressor",
        "BayesianRidge",
        "DecisionTreeRegressor",
        "DummyRegressor",
        "ElasticNet",
        "ElasticNetCV",
        "ExtraTreeRegressor",
        "ExtraTreesRegressor",
        # 'GammaRegressor', # Did not work
        "GaussianProcessRegressor",
        "GradientBoostingRegressor",
        "HistGradientBoostingRegressor",
        "HuberRegressor",
        "KNeighborsRegressor",
        "KernelRidge",
        "Lars",
        "LarsCV",
        "Lasso",
        "LassoCV",
        "LassoLars",
        "LassoLarsCV",
        "LassoLarsIC",
        "LinearRegression",
        "LinearSVR",
        "MLPRegressor",
        "NuSVR",
        "OrthogonalMatchingPursuit",
        "OrthogonalMatchingPursuitCV",
        "PassiveAggressiveRegressor",
        "PoissonRegressor",
        # 'QuantileRegressor', # Did not work
        "RANSACRegressor",
        "RandomForestRegressor",
        "Ridge",
        "RidgeCV",
        "SGDRegressor",
        "SVR",
        "TransformedTargetRegressor",
        "TweedieRegressor",
    ]

    regressors = [
        est[1] for est in all_estimators() if (issubclass(est[1], RegressorMixin) and (est[0] in chosen_regressors))
    ]
    regressors.append(XGBRegressor)
    regressors.append(lgbm.LGBMRegressor)

    regressor = LazyRegressor(
        verbose=0,
        ignore_warnings=False,
        custom_metric=None,
        predictions=True,
        regressors=regressors,
    )
    models, _ = regressor.fit(x_train, x_test, y_train, y_test)
    log.info(models)


def plot_results_classification(all_data: List[np.ndarray], title: str) -> None:
    all_data = [array * 100 for array in all_data]

    plt.figure(figsize=(10, 5))
    labels = [
        "XGBoost",
        "LGBM",
        "ExtraTrees",
        "RandomForest",
        "Bagging",
    ]
    bplot = plt.boxplot(all_data, vert=True, patch_artist=True, labels=labels, meanline=True, showmeans=True)
    colors = ["pink", "lightblue", "mediumpurple", "lightgreen", "orange"]
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)

    # plt.title(f"{title} data", fontsize=15)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Classifiers", fontsize=18)
    plt.ylabel("Accuracy (%)", fontsize=18)
    plt.tight_layout()
    plt.grid(axis="y")
    plt.savefig(f"biodegradation/figures/lazy_predict_results_accuracy_classification.png")
    plt.close()


def plot_results_regression(all_data: List[np.ndarray], title: str) -> None:
    plt.figure(figsize=(10, 5))
    labels = [
        "LGBM",
        "HistGradient-\nBoosting",
        "RandomForest",
        "XGBoost",
        "SVR",
    ]
    colors = ["pink", "lightblue", "mediumpurple", "lightgreen", "orange"]
    bplot = plt.boxplot(all_data, vert=True, patch_artist=True, labels=labels, meanline=True, showmeans=True)
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)

    # plt.title(f"{title} data", fontsize=15)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Regressors", fontsize=18)
    plt.ylabel("$\mathregular{R^{2}}$", fontsize=18)
    plt.tight_layout()
    plt.grid(axis="y")
    plt.savefig(f"biodegradation/figures/lazy_predict_results_R2_regression.png")
    plt.close()


def run_and_plot_classifiers(df: pd.DataFrame) -> None:

    classifiers = {
        "XGBClassifier": tune_and_train_XGBClassifier,
        "LGBMClassifier": tune_and_train_LGBMClassifier,
        "ExtraTreesClassifier": tune_and_train_ExtraTreesClassifier,
        "RandomForestClassifier": tune_and_train_RandomForestClassifier,
        "BaggingClassifier": tune_and_train_BaggingClassifier,
    }

    accuracy_all = []
    f1_all = []
    sensitivity_all = []
    specificity_all = []

    if args.train_new:
        for classifier in classifiers:
            log.info(f" \n Currently running the classifier {classifier}")
            accu, f1, sensitivity, specificity = classifiers[classifier](
                df=df,
                random_seed=args.random_seed,
                nsplits=args.nsplits,
                include_speciation=False,
                fixed_testset=False,
                df_smallest=df,
                dataset_name="improved_final",
                n_jobs=args.njobs,
            )
            accuracy_all.append(np.asarray(accu))
            f1_all.append(np.asarray(f1))
            sensitivity_all.append(np.asarray(sensitivity))
            specificity_all.append(np.asarray(specificity))
    else:
        if (args.nsplits == 5) and (args.random_seed == 42):
            (
                accuracy_all_saved,
                f1_all_saved,
                sensitivity_all_saved,
                specificity_all_saved,
            ) = results_lazy_classifiers_nsplit5_seed42()
            accuracy_all += accuracy_all_saved
            f1_all += f1_all_saved
            sensitivity_all += sensitivity_all_saved
            specificity_all += specificity_all_saved
    if args.plot:
        plot_results_classification(accuracy_all, "Accuracy")
    log.info("Finished plotting")


def run_and_plot_regressors(
    df: pd.DataFrame,
) -> None:
    regressors = {
        "LGBMRegressor": tune_and_train_LGBMRegressor,
        "HistGradientBoostingRegressor": tune_and_train_HistGradientBoostingRegressor,
        "RandomForestRegressor": tune_and_train_RandomForestRegressor,
        "XGBRegressor": tune_and_train_XGBRegressor,
        "SVR": tune_and_train_SVR,
    }

    rmse_all = []
    mae_all = []
    r2_all = []
    mse_all = []

    if args.train_new:
        for regressor in regressors:
            log.info(f" \n Currently running the regressor {regressor}")
            rmse, mae, r2, mse = regressors[regressor](
                df=df.copy(),
                random_seed=args.random_seed,
                nsplits=args.nsplits,
                include_speciation=False,
                fixed_testset=False,
                df_smallest=df,
                dataset_name="improved_final",
                n_jobs=args.njobs,
            )
            rmse_all.append(np.asarray(rmse))
            mae_all.append(np.asarray(mae))
            r2_all.append(np.asarray(r2))
            mse_all.append(np.asarray(mse))
    else:
        (
            rmse_all_saved,
            mae_all_saved,
            r2_all_saved,
            mse_all_saved,
        ) = results_lazy_regressors_nsplit5_seed42()
        rmse_all += rmse_all_saved
        mae_all += mae_all_saved
        r2_all += r2_all_saved
        mse_all += mse_all_saved
    if args.plot:
        plot_results_regression(r2_all, "R2")
    log.info("Finished plotting")


if __name__ == "__main__":

    df_class, df_reg = get_improved_datasets()

    if args.run_lazy:
        if args.mode == "both" or args.mode == "classification":
            run_lazy_classifier(df_class=df_class)
        if args.mode == "both" or args.mode == "regression":
            run_lazy_regressor(df_reg=df_reg)

    if args.mode == "both" or args.mode == "classification":
        run_and_plot_classifiers(df=df_class)
    if args.mode == "both" or args.mode == "regression":
        run_and_plot_regressors(df=df_reg)
