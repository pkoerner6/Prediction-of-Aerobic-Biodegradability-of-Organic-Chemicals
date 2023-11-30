import argparse
import pandas as pd
import numpy as np
import structlog
import sys
import os

log = structlog.get_logger()
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin
import lightgbm as lgbm
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
np.int = int  # Because of scikit-optimize
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    f1_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

parser = argparse.ArgumentParser()

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from processing_functions import create_input_regression
from processing_functions import create_input_classification
from processing_functions import bit_vec_to_lst_of_lst
# from ml_functions import tune_and_train_XGBRegressor
# from ml_functions import tune_and_train_XGBClassifier
from ml_functions import split_regression_df_with_grouping
from ml_functions import get_balanced_data_adasyn
from ml_functions import report_perf_hyperparameter_tuning
from ml_functions import get_class_results
# from ml_functions import tune_and_train_HistGradientBoostingRegressor
# from ml_functions import tune_and_train_RandomForestRegressor
# from ml_functions import tune_and_train_SVR
# from ml_functions import tune_and_train_LGBMRegressor
# from ml_functions import tune_and_train_LGBMClassifier
# from ml_functions import tune_and_train_ExtraTreesClassifier
# from ml_functions import tune_and_train_RandomForestClassifier
# from ml_functions import tune_and_train_BaggingClassifier

from model_results import results_lazy_classifiers_nsplit5_seed42
from model_results import results_lazy_regressors_nsplit5_seed42


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
        "datasets/curated_data/reg_curated_scs_biowin_readded.csv", index_col=0
    )
    df_class_improved = pd.read_csv(
        "datasets/curated_data/class_curated_scs_biowin_readded.csv", index_col=0
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
    # (
    #     class_x_balanced,
    #     class_y_balanced,
    #     class_x_test,
    #     class_y_test,
    # ) = get_classification_model_input(df_class)

    train_df = pd.read_csv("datasets/curated_data/class_curated_scs.csv", index_col=0)
    # train_df = pd.read_csv("datasets/curated_data/class_curated_scs_biowin_readded.csv", index_col=0)
    test_df = pd.read_csv("datasets/curated_data/class_curated_scs_multiple.csv", index_col=0)
    train_df = train_df[~train_df["cas"].isin(test_df["cas"])]
    test_df.reset_index(inplace=True, drop=True)
    train_df.reset_index(inplace=True, drop=True)

    x_train = create_input_classification(train_df, include_speciation=False)
    y_train = train_df["y_true"]

    x_test = create_input_classification(test_df, include_speciation=False)
    y_test = test_df["y_true"]

    x_balanced, y_balanced = get_balanced_data_adasyn(random_seed=args.random_seed, x=x_train, y=y_train)

    clf = LazyClassifier(predictions=True, verbose=0)
    models, _ = clf.fit(x_balanced, x_test, y_balanced, y_test)
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


# def plot_results_classification(all_data: List[np.ndarray], title: str) -> None:
#     all_data = [array * 100 for array in all_data]

#     plt.figure(figsize=(10, 5))
#     labels = [
#         "XGBoost",
#         "LGBM",
#         "ExtraTrees",
#         "RandomForest",
#         "Bagging",
#     ]
#     bplot = plt.boxplot(all_data, vert=True, patch_artist=True, labels=labels, meanline=True, showmeans=True)
#     colors = ["pink", "lightblue", "mediumpurple", "lightgreen", "orange"]
#     for patch, color in zip(bplot["boxes"], colors):
#         patch.set_facecolor(color)

#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.xlabel("Classifiers", fontsize=18)
#     plt.ylabel("Accuracy (%)", fontsize=18)
#     plt.tight_layout()
#     plt.grid(axis="y")
#     plt.savefig(f"figures/lazy_predict_results_accuracy_classification.png")
#     plt.close()


# def plot_results_regression(all_data: List[np.ndarray], title: str) -> None:
#     plt.figure(figsize=(10, 5))
#     labels = [
#         "LGBM",
#         "HistGradient-\nBoosting",
#         "RandomForest",
#         "XGBoost",
#         "SVR",
#     ]
#     colors = ["pink", "lightblue", "mediumpurple", "lightgreen", "orange"]
#     bplot = plt.boxplot(all_data, vert=True, patch_artist=True, labels=labels, meanline=True, showmeans=True)
#     for patch, color in zip(bplot["boxes"], colors):
#         patch.set_facecolor(color)

#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.xlabel("Regressors", fontsize=18)
#     plt.ylabel("$\mathregular{R^{2}}$", fontsize=18)
#     plt.tight_layout()
#     plt.grid(axis="y")
#     plt.savefig(f"figures/lazy_predict_results_R2_regression.png")
#     plt.close()











def tune_classifiers(
    df: pd.DataFrame, 
    nsplits: int, 
    df_test: pd.DataFrame, 
    search_spaces: Dict, 
    n_jobs: int,
    model,
):
    df_tune = df[~df["inchi_from_smiles"].isin(df_test["inchi_from_smiles"])]
    df_tune = df_tune[~df_tune["cas"].isin(df_test["cas"])]
    assert len(df_tune) + len(df_test) == len(df)

    x = create_input_classification(df_tune, include_speciation=False)
    y = df_tune["y_true"]
    x, y, = get_balanced_data_adasyn(random_seed=args.random_seed, x=x, y=y)

    scoring = make_scorer(accuracy_score, greater_is_better=True)
    skf = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=args.random_seed)
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
        random_state=args.random_seed,
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
    df_test: pd.DataFrame, 
    model,
):  
    df_train = df[~df["cas"].isin(df_test["cas"])]
    df_train = df_train[~df_train["inchi_from_smiles"].isin(df_test["inchi_from_smiles"])]

    assert len(df_train) + len(df_test) == len(df)
    x = create_input_classification(df_train, include_speciation=False)
    y = df_train["y_true"]
    x, y = get_balanced_data_adasyn(random_seed=args.random_seed, x=x, y=y)

    model.fit(x, y)

    x_test = create_input_classification(df_class=df_test, include_speciation=False)
    prediction = model.predict(x_test)

    accu, f1, sensitivity, specificity = get_class_results(true=df_test["y_true"], pred=prediction)
    metrics_all = ["accuracy", "sensitivity", "specificity", "f1"]
    metrics_values_all = [accu, sensitivity, specificity, f1]
    for metric, metric_values in zip(metrics_all, metrics_values_all):
        log.info(f"{metric}: ", score="{:.1f}".format(metric_values * 100))
    
    return accu, f1, sensitivity, specificity


def tune_and_train_classifiers(
    df: pd.DataFrame, 
    nsplits: int, 
    df_test: pd.DataFrame,
    n_jobs: int,
    search_spaces: Dict,
    model,
):
    best_params = tune_classifiers(
        df=df,
        nsplits=nsplits,
        df_test=df_test,
        search_spaces=search_spaces,
        n_jobs=n_jobs,
        model=model,
    )

    accu, f1, sensitivity, specificity = train_classifier_with_best_hyperparamters(
        df=df,
        df_test=df_test,
        # dataset_name=dataset_name,
        model=model(**best_params),
    )
    return accu, f1, sensitivity, specificity


def tune_and_train_XGBClassifier(df: pd.DataFrame, nsplits: int, df_test: pd.DataFrame):
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
    # Best results:
    # OrderedDict([
    # ('alpha', 2.2303118765599415), 
    # ('base_score', 0.545545148635465), 
    # ('booster', 'gbtree'), 
    # ('colsample_bylevel', 0.7947398780461146), 
    # ('colsample_bynode', 0.3680591793075739), 
    # ('colsample_bytree', 0.8242355897456615), 
    # ('gamma', 0.03574220215501526), 
    # ('lambda', 0.46975211709521025), 
    # ('learning_rate', 0.2505260157188399), 
    # ('max_delta_step', 3.2955831766937553), 
    # ('max_depth', 210), 
    # ('min_child_weight', 1.2536676367917012), 
    # ('n_estimators', 4907), 
    # ('num_parallel_tree', 21), 
    # ('scale_pos_weight', 0.6947280728431414), 
    # ('subsample', 0.8600632698356062), 
    # ('tree_method', 'exact'), 
    # ('validate_parameters', True)])
    accu, f1, sensitivity, specificity = tune_and_train_classifiers(
        df=df,
        nsplits=nsplits,
        df_test=df_test,
        n_jobs=1,
        search_spaces=search_spaces,
        model=model,
    )
    return accu, f1, sensitivity, specificity

def tune_and_train_ExtraTreesClassifier(df: pd.DataFrame, nsplits: int, df_test: pd.DataFrame):
    model = ExtraTreesClassifier
    search_spaces = {
        "criterion": Categorical(["gini", "entropy", "log_loss"]),
        "max_depth": Categorical([None]),
        "max_features": Categorical(["sqrt", "log2", None]),
        "min_samples_leaf": Integer(1, 10),
        "min_samples_split": Integer(2, 10),
        "n_estimators": Integer(50, 1000),
        "random_state": Categorical([args.random_seed]),
    }
    # Best results: 
    # OrderedDict([
    # ('criterion', 'log_loss'), 
    # ('max_depth', None), 
    # ('max_features', 'sqrt'), 
    # ('min_samples_leaf', 1), 
    # ('min_samples_split', 2), 
    # ('n_estimators', 1000), 
    # ('random_state', 42)])
    accu, f1, sensitivity, specificity = tune_and_train_classifiers(
        df=df,
        nsplits=nsplits,
        df_test=df_test,
        n_jobs=1,
        search_spaces=search_spaces,
        model=model,
    )
    return accu, f1, sensitivity, specificity

def tune_and_train_RandomForestClassifier(df: pd.DataFrame, nsplits: int, df_test: pd.DataFrame):
    model = RandomForestClassifier
    search_spaces = {
        "criterion": Categorical(["gini", "entropy", "log_loss"]),
        "max_features": Categorical(["sqrt", "log2", None]),
        "min_samples_leaf": Integer(1, 5),
        "min_samples_split": Integer(2, 5),
        "n_estimators": Integer(1000, 2500),
        "random_state": Categorical([args.random_seed]),
    }
    # Best results: 
    accu, f1, sensitivity, specificity = tune_and_train_classifiers(
        df=df,
        nsplits=nsplits,
        df_test=df_test,
        n_jobs=1,
        search_spaces=search_spaces,
        model=model,
    )
    return accu, f1, sensitivity, specificity


def run_and_plot_classifiers() -> None:
    # df = pd.read_csv("datasets/curated_data/class_curated_scs.csv", index_col=0)
    df = pd.read_csv("datasets/curated_data/class_curated_scs_biowin_readded.csv", index_col=0)
    df_test = pd.read_csv("datasets/curated_data/class_curated_scs_multiple.csv", index_col=0)

    classifiers = {
        # "XGBClassifier": tune_and_train_XGBClassifier,
        # "LGBMClassifier": tune_and_train_LGBMClassifier,
        "ExtraTreesClassifier": tune_and_train_ExtraTreesClassifier,
        "RandomForestClassifier": tune_and_train_RandomForestClassifier,
        # "BaggingClassifier": tune_and_train_BaggingClassifier,
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
                nsplits=args.nsplits,
                df_test=df_test,
            )
            accuracy_all.append(np.asarray(accu))
            f1_all.append(np.asarray(f1))
            sensitivity_all.append(np.asarray(sensitivity))
            specificity_all.append(np.asarray(specificity))
    # if args.plot:
    #     plot_results_classification(accuracy_all, "Accuracy")
    # log.info("Finished plotting")


# def run_and_plot_regressors(
#     df: pd.DataFrame,
# ) -> None:
#     regressors = {
#         "LGBMRegressor": tune_and_train_LGBMRegressor,
#         "HistGradientBoostingRegressor": tune_and_train_HistGradientBoostingRegressor,
#         "RandomForestRegressor": tune_and_train_RandomForestRegressor,
#         "XGBRegressor": tune_and_train_XGBRegressor,
#         "SVR": tune_and_train_SVR,
#     }

#     rmse_all = []
#     mae_all = []
#     r2_all = []
#     mse_all = []

#     if args.train_new:
#         for regressor in regressors:
#             log.info(f" \n Currently running the regressor {regressor}")
#             rmse, mae, r2, mse = regressors[regressor](
#                 df=df.copy(),
#                 random_seed=args.random_seed,
#                 nsplits=args.nsplits,
#                 include_speciation=False,
#                 fixed_testset=False,
#                 df_smallest=df,
#                 dataset_name="improved_final",
#                 n_jobs=args.njobs,
#             )
#             rmse_all.append(np.asarray(rmse))
#             mae_all.append(np.asarray(mae))
#             r2_all.append(np.asarray(r2))
#             mse_all.append(np.asarray(mse))
#     else:
#         (
#             rmse_all_saved,
#             mae_all_saved,
#             r2_all_saved,
#             mse_all_saved,
#         ) = results_lazy_regressors_nsplit5_seed42()
#         rmse_all += rmse_all_saved
#         mae_all += mae_all_saved
#         r2_all += r2_all_saved
#         mse_all += mse_all_saved
#     if args.plot:
#         plot_results_regression(r2_all, "R2")
#     log.info("Finished plotting")


if __name__ == "__main__":

    df_class, df_reg = get_improved_datasets()

    if args.run_lazy:
        if args.mode == "both" or args.mode == "classification":
            run_lazy_classifier(df_class=df_class)
        # if args.mode == "both" or args.mode == "regression":
        #     run_lazy_regressor(df_reg=df_reg)
    if args.mode == "both" or args.mode == "classification":
        run_and_plot_classifiers()
    # if args.mode == "both" or args.mode == "regression":
    #     run_and_plot_regressors(df=df_reg)
