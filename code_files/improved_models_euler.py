
import argparse
import pandas as pd
import numpy as np
import structlog
import sys
import os

from tqdm import tqdm

log = structlog.get_logger()
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier

np.int = int  # Because of scikit-optimize
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from code_files.processing_functions import get_class_datasets
from code_files.processing_functions import bit_vec_to_lst_of_lst
from code_files.processing_functions import convert_to_morgan_fingerprints
from code_files.processing_functions import convert_to_maccs_fingerprints
from code_files.processing_functions import convert_to_rdk_fingerprints
from code_files.ml_functions import get_balanced_data_adasyn
from code_files.ml_functions import report_perf_hyperparameter_tuning
from code_files.ml_functions import get_class_results
from code_files.ml_functions import run_balancing_and_training
from code_files.ml_functions import split_classification_df_with_fixed_test_set



parser = argparse.ArgumentParser()

parser.add_argument(
    "--run_lazy",
    default=True,
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
    "--njobs",
    default=1,
    type=int,
    help="Define n_jobs: 1 when running locally or 12 when running on Euler",
)
parser.add_argument(
    "--train_set",
    type=str,
    default="df_curated_final",
    choices=["df_curated_scs", "df_curated_biowin", "df_curated_final"],
    help="How to generate the features",
)
parser.add_argument(
    "--test_set",
    type=str,
    default="df_curated_biowin",
    choices=["df_curated_scs", "df_curated_biowin"],
    help="How to generate the features",
)
parser.add_argument(
    "--feature_type",
    type=str,
    default="MACCS",
    choices=["MACCS", "RDK", "Morgan"],
    help="How to generate the features",
)
args = parser.parse_args()


def create_input_classification_other_features(df: pd.DataFrame, feature_type="MACCS") -> np.ndarray:
    """Function to create fingerprints and put fps into one array that can than be used as one feature for model training."""
    if feature_type=="MACCS":
        df = convert_to_maccs_fingerprints(df)
        x_class = bit_vec_to_lst_of_lst(df, include_speciation=False)
    elif feature_type=="RDK":
        df = convert_to_rdk_fingerprints(df)
        x_class = bit_vec_to_lst_of_lst(df, include_speciation=False)
    elif feature_type=="Morgan":
        df = convert_to_morgan_fingerprints(df)
        x_class = bit_vec_to_lst_of_lst(df, include_speciation=False)
    x_array = np.array(x_class, dtype=object)
    return x_array


def get_classification_model_input(df: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[np.ndarray, pd.Series, np.ndarray, pd.Series]:
    _, test_df = train_test_split(df_test, test_size=0.2, random_state=args.random_seed)
    train_df = df[~df["inchi_from_smiles"].isin(test_df["inchi_from_smiles"])]
    train_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)

    x_train = create_input_classification_other_features(train_df, feature_type=args.feature_type)
    y_train = train_df["y_true"]
    x_test = create_input_classification_other_features(test_df, feature_type=args.feature_type)
    y_test = test_df["y_true"]

    x_balanced, y_balanced = get_balanced_data_adasyn(random_seed=args.random_seed, x=x_train, y=y_train)

    return x_balanced, y_balanced, x_test, y_test


def run_lazy_classifier(df: pd.DataFrame, df_test: pd.DataFrame) -> None:
    x_balanced, y_balanced, x_test, y_test = get_classification_model_input(df, df_test)

    removed_classifiers = [
        "ClassifierChain",
        "ComplementNB",
        "MultiOutputClassifier", 
        "MultinomialNB", 
        "OneVsOneClassifier",
        "OneVsRestClassifier",
        "OutputCodeClassifier",
        "RadiusNeighborsClassifier",
        "VotingClassifier",
        "CategoricalNB",
        "StackingClassifier",
    ]

    classifiers = [
        est[1]
        for est in all_estimators()
        if (issubclass(est[1], ClassifierMixin) and (est[0] not in removed_classifiers))
    ]

    classifiers.append(XGBClassifier)

    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=False,
        custom_metric=None,
        predictions=True,
        classifiers=classifiers,
    )

    models, _ = clf.fit(x_balanced, x_test, y_balanced, y_test)
    log.info(models)



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

    x = create_input_classification_other_features(df_tune, feature_type=args.feature_type)
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


def skf_class_fixed_testset_other_features(
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_type: str,
    paper: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[pd.DataFrame], List[int]]:

    cols=["cas", "smiles", "y_true"]

    train_sets, test_sets = split_classification_df_with_fixed_test_set(
        df=df,
        df_test=df_test,
        nsplits=args.nsplits,
        random_seed=args.random_seed,
        cols=cols,
        paper=paper,
    )
    x_train_fold_lst: List[np.ndarray] = []
    y_train_fold_lst: List[np.ndarray] = []
    x_test_fold_lst: List[np.ndarray] = []
    y_test_fold_lst: List[np.ndarray] = []
    df_test_lst: List[pd.DataFrame] = []
    test_set_sizes: List[int] = []

    for split in range(args.nsplits):
        x_train_fold = train_sets[split][cols]
        x_train_fold_lst.append(create_input_classification_other_features(x_train_fold, feature_type))
        y_train_fold_lst.append(train_sets[split]["y_true"])
        x_test_fold = test_sets[split][cols]
        x_test_fold_lst.append(create_input_classification_other_features(x_test_fold, feature_type))
        y_test_fold_lst.append(test_sets[split]["y_true"])
        df_test_set = test_sets[split].copy()
        df_test_lst.append(df_test_set)
        test_set_sizes.append(len(df_test_set))
    return x_train_fold_lst, y_train_fold_lst, x_test_fold_lst, y_test_fold_lst, df_test_lst, test_set_sizes


def train_classifier_with_best_hyperparamters(
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    model,
):  
    df.reset_index(inplace=True, drop=True)

    (
        x_train_fold_lst,
        y_train_fold_lst,
        x_test_fold_lst,
        y_test_fold_lst,
        df_test_lst,
        test_set_sizes,
    ) = skf_class_fixed_testset_other_features(
        df=df,
        df_test=df_test,
        feature_type=args.feature_type,
        paper=False,
    )

    lst_accu, lst_sensitivity, lst_specificity, lst_f1 = run_balancing_and_training(
        df=df,
        x_train_fold_lst=x_train_fold_lst, 
        y_train_fold_lst=y_train_fold_lst, 
        x_test_fold_lst=x_test_fold_lst, 
        y_test_fold_lst=y_test_fold_lst,
        test_set_sizes=test_set_sizes,
        use_adasyn=True,
        random_seed=args.random_seed,
        model=XGBClassifier(),
    )


    metrics = ["balanced accuracy", "sensitivity", "specificity"]
    metrics_values = [
        lst_accu,
        lst_sensitivity,
        lst_specificity,
    ]
    for metric, metric_values in zip(metrics, metrics_values):
        log.info(
            f"{metric} for {args.feature_type}: ",
            score="{:.1f}".format(np.mean(metric_values) * 100) + " ± " + "{:.1f}".format(np.std(metric_values) * 100),
        )
    log.info(
        f"F1 for {args.feature_type}: ",
        score="{:.2f}".format(np.mean(lst_f1)) + " ± " + "{:.2f}".format(np.std(lst_f1)),
    )
    return (
        lst_accu,
        lst_sensitivity,
        lst_specificity,
        lst_f1,
    )


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

    lst_accu, lst_sensitivity, lst_specificity, lst_f1 = train_classifier_with_best_hyperparamters(
        df=df,
        df_test=df_test,
        model=model(**best_params),
    )
    return lst_accu, lst_sensitivity, lst_specificity, lst_f1


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
    log.info("Started tuning XGBClassifier")
    lst_accu, lst_sensitivity, lst_specificity, lst_f1 = tune_and_train_classifiers(
        df=df,
        nsplits=nsplits,
        df_test=df_test,
        n_jobs=1,
        search_spaces=search_spaces,
        model=model,
    )
    log.info("Finished tuning XGBClassifier")
    return lst_accu, lst_sensitivity, lst_specificity, lst_f1

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
    log.info("Started tuning ExtraTreesClassifier")
    lst_accu, lst_sensitivity, lst_specificity, lst_f1 = tune_and_train_classifiers(
        df=df,
        nsplits=nsplits,
        df_test=df_test,
        n_jobs=1,
        search_spaces=search_spaces,
        model=model,
    )
    log.info("Finished tuning ExtraTreesClassifier")
    return lst_accu, lst_sensitivity, lst_specificity, lst_f1


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
    log.info("Started tuning RandomForestClassifier")
    lst_accu, lst_sensitivity, lst_specificity, lst_f1 = tune_and_train_classifiers(
        df=df,
        nsplits=nsplits,
        df_test=df_test,
        n_jobs=1,
        search_spaces=search_spaces,
        model=model,
    )
    log.info("Finished tuning RandomForestClassifier")
    return lst_accu, lst_sensitivity, lst_specificity, lst_f1


def tune_and_train_MLPClassifier(df: pd.DataFrame, nsplits: int, df_test: pd.DataFrame):
    model = MLPClassifier
    search_spaces = {
        "random_state": Categorical([args.random_seed]),
        "activation": Categorical(["identity", "logistic", "tanh", "relu"]),
        "solver": Categorical(["lbfgs", "sgd", "adam"]),
        "alpha": Real(0.00001, 0.01, "uniform"),
        "learning_rate_init": Real(0.0001, 0.01, "uniform"),
        "max_iter": Integer(150, 250),
        "early_stopping": Categorical([True]),
        "hidden_layer_sizes": Integer(50, 150),
    }
    log.info("Started tuning MLPClassifier")
    lst_accu, lst_sensitivity, lst_specificity, lst_f1 = tune_and_train_classifiers(
        df=df,
        nsplits=nsplits,
        df_test=df_test,
        n_jobs=1,
        search_spaces=search_spaces,
        model=model,
    )
    log.info("Finished tuning MLPClassifier")
    return lst_accu, lst_sensitivity, lst_specificity, lst_f1


def tune_and_train_HistGradientBoostingClassifier(df: pd.DataFrame, nsplits: int, df_test: pd.DataFrame):
    model = HistGradientBoostingClassifier
    search_spaces = {
        "learning_rate": Real(0.01, 0.4, "uniform"),
        "max_iter": Integer(80, 200),
        "max_leaf_nodes": Integer(25, 50),
        "min_samples_leaf": Integer(10, 25),
        "random_state": Categorical([args.random_seed]),
    }
    log.info("Started tuning HistGradientBoostingClassifier")
    lst_accu, lst_sensitivity, lst_specificity, lst_f1 = tune_and_train_classifiers(
        df=df,
        nsplits=nsplits,
        df_test=df_test,
        n_jobs=1,
        search_spaces=search_spaces,
        model=model,
    )
    log.info("Finished tuning HistGradientBoostingClassifier")
    return lst_accu, lst_sensitivity, lst_specificity, lst_f1


def tune_and_train_LogisticRegressionCV(df: pd.DataFrame, nsplits: int, df_test: pd.DataFrame):
    model = LogisticRegressionCV
    search_spaces = {
        "random_state": Categorical([args.random_seed]),
        # "cv": Integer(5),
        "penalty": Categorical(["l1", "l2", "elasticnet"]),
        "solver": Categorical(["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]),
        "max_iter": Integer(80, 200),
    }
    log.info("Started tuning LogisticRegressionCV")
    lst_accu, lst_sensitivity, lst_specificity, lst_f1 = tune_and_train_classifiers(
        df=df,
        nsplits=nsplits,
        df_test=df_test,
        n_jobs=1,
        search_spaces=search_spaces,
        model=model,
    )
    log.info("Finished tuning LogisticRegressionCV")
    return lst_accu, lst_sensitivity, lst_specificity, lst_f1


def tune_and_train_GaussianProcessClassifier(df: pd.DataFrame, nsplits: int, df_test: pd.DataFrame):
    model = GaussianProcessClassifier
    search_spaces = {
        "max_iter_predict": Integer(80, 200),
        "random_state": Categorical([args.random_seed]),
    }
    log.info("Started tuning GaussianProcessClassifier")
    lst_accu, lst_sensitivity, lst_specificity, lst_f1 = tune_and_train_classifiers(
        df=df,
        nsplits=nsplits,
        df_test=df_test,
        n_jobs=1,
        search_spaces=search_spaces,
        model=model,
    )
    log.info("Finished tuning GaussianProcessClassifier")
    return lst_accu, lst_sensitivity, lst_specificity, lst_f1


def tune_and_train_LinearSVC(df: pd.DataFrame, nsplits: int, df_test: pd.DataFrame):
    model = LinearSVC
    search_spaces = {
        "penalty": Categorical(["l1", "l2"]),
        "loss": Categorical(["hinge", "squared_hinge"]),
        "multi_class": Categorical(["ovr", "crammer_singer"]),
        "max_iter": Integer(800, 1200),
        "random_state": Categorical([args.random_seed]),
    }
    log.info("Started tuning LinearSVC")
    lst_accu, lst_sensitivity, lst_specificity, lst_f1 = tune_and_train_classifiers(
        df=df,
        nsplits=nsplits,
        df_test=df_test,
        n_jobs=1,
        search_spaces=search_spaces,
        model=model,
    )
    log.info("Finished tuning LinearSVC")
    return lst_accu, lst_sensitivity, lst_specificity, lst_f1


def tune_and_train_PassiveAggressiveClassifier(df: pd.DataFrame, nsplits: int, df_test: pd.DataFrame):
    model = PassiveAggressiveClassifier
    search_spaces = {
        "max_iter": Integer(800, 1200),
        "early_stopping": Categorical([True]),
        "random_state": Categorical([args.random_seed]),
    }
    log.info("Started tuning PassiveAggressiveClassifier")
    lst_accu, lst_sensitivity, lst_specificity, lst_f1 = tune_and_train_classifiers(
        df=df,
        nsplits=nsplits,
        df_test=df_test,
        n_jobs=1,
        search_spaces=search_spaces,
        model=model,
    )
    log.info("Finished tuning PassiveAggressiveClassifier")
    return lst_accu, lst_sensitivity, lst_specificity, lst_f1


def tune_and_train_Perceptron(df: pd.DataFrame, nsplits: int, df_test: pd.DataFrame):
    model = Perceptron
    search_spaces = {
        "penalty": Categorical(["l1", "l2", "elasticnet"]),
        "alpha": Real(0.00001, 0.001, "uniform"),
        "max_iter": Integer(800, 1200),
        "random_state": Categorical([args.random_seed]),
        "early_stopping": Categorical([True]),
    }
    log.info("Started tuning Perceptron")
    lst_accu, lst_sensitivity, lst_specificity, lst_f1 = tune_and_train_classifiers(
        df=df,
        nsplits=nsplits,
        df_test=df_test,
        n_jobs=1,
        search_spaces=search_spaces,
        model=model,
    )
    log.info("Finished tuning Perceptron")
    return lst_accu, lst_sensitivity, lst_specificity, lst_f1


def tune_and_train_LogisticRegression(df: pd.DataFrame, nsplits: int, df_test: pd.DataFrame):
    model = LogisticRegression
    search_spaces = {
        "penalty": Categorical(["l1", "l2", "elasticnet"]),
        "random_state": Categorical([args.random_seed]),
        "solver": Categorical(["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]),
        "max_iter": Integer(80, 120),
        "multi_class": Categorical(["auto", "ovr", "multinomial"]),
    }
    log.info("Started tuning LogisticRegression")
    lst_accu, lst_sensitivity, lst_specificity, lst_f1 = tune_and_train_classifiers(
        df=df,
        nsplits=nsplits,
        df_test=df_test,
        n_jobs=1,
        search_spaces=search_spaces,
        model=model,
    )
    log.info("Finished tuning LogisticRegression")
    return lst_accu, lst_sensitivity, lst_specificity, lst_f1


def tune_and_train_SVC(df: pd.DataFrame, nsplits: int, df_test: pd.DataFrame):
    model = SVC
    search_spaces = {
        "kernel": Categorical(["linear", "poly", "rbf", "sigmoid", "precomputed"]),
        "degree": Integer(2, 3),
        "gamma": Categorical(["auto", "scale"]),
        "random_state": Categorical([args.random_seed]),
    }
    log.info("Started tuning SVC")
    lst_accu, lst_sensitivity, lst_specificity, lst_f1 = tune_and_train_classifiers(
        df=df,
        nsplits=nsplits,
        df_test=df_test,
        n_jobs=1,
        search_spaces=search_spaces,
        model=model,
    )
    log.info("Finished tuning SVC")
    return lst_accu, lst_sensitivity, lst_specificity, lst_f1


def tune_and_train_RidgeClassifierCV(df: pd.DataFrame, nsplits: int, df_test: pd.DataFrame):
    model = RidgeClassifierCV
    search_spaces = {
        "cv": Categorical([None]),
    }
    log.info("Started tuning RidgeClassifierCV")
    lst_accu, lst_sensitivity, lst_specificity, lst_f1 = tune_and_train_classifiers(
        df=df,
        nsplits=nsplits,
        df_test=df_test,
        n_jobs=1,
        search_spaces=search_spaces,
        model=model,
    )
    log.info("Finished tuning RidgeClassifierCV")
    return lst_accu, lst_sensitivity, lst_specificity, lst_f1


def tune_and_train_RidgeClassifier(df: pd.DataFrame, nsplits: int, df_test: pd.DataFrame):
    model = RidgeClassifier
    search_spaces = {
        "solver": Categorical(["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"]),
        "random_state": Categorical([args.random_seed]),
    }
    log.info("Started tuning RidgeClassifier")
    lst_accu, lst_sensitivity, lst_specificity, lst_f1 = tune_and_train_classifiers(
        df=df,
        nsplits=nsplits,
        df_test=df_test,
        n_jobs=1,
        search_spaces=search_spaces,
        model=model,
    )
    log.info("Finished tuning RidgeClassifier")
    return lst_accu, lst_sensitivity, lst_specificity, lst_f1


def tune_and_train_GradientBoostingClassifier(df: pd.DataFrame, nsplits: int, df_test: pd.DataFrame):
    model = GradientBoostingClassifier
    search_spaces = {
        "loss": Categorical(["log_loss", "exponential"]),
        "learning_rate": Real(0.01, 0.4, "uniform"),
        "n_estimators": Integer(80, 120),
        "criterion": Categorical(["friedman_mse", "squared_error"]),
        "max_depth": Integer(2, 5),
        "random_state": Categorical([args.random_seed]),
        "max_features": Categorical(["sqrt", "log2"]),
    }
    log.info("Started tuning GradientBoostingClassifier")
    lst_accu, lst_sensitivity, lst_specificity, lst_f1 = tune_and_train_classifiers(
        df=df,
        nsplits=nsplits,
        df_test=df_test,
        n_jobs=1,
        search_spaces=search_spaces,
        model=model,
    )
    log.info("Finished tuning GradientBoostingClassifier")
    return lst_accu, lst_sensitivity, lst_specificity, lst_f1


def run_classifiers_MACCS(datasets: Dict[str, pd.DataFrame], df_test: str) -> None:
    train_data = datasets[args.train_set]
    test_data = datasets[df_test].sample(frac=0.2, random_state=args.random_seed)

    if args.test_set == "df_curated_scs":
        classifiers = {
            "MLPClassifier": tune_and_train_MLPClassifier,
            "HistGradientBoostingClassifier": tune_and_train_HistGradientBoostingClassifier,
            "LogisticRegressionCV": tune_and_train_LogisticRegressionCV,
            "GaussianProcessClassifier": tune_and_train_GaussianProcessClassifier,
            "LinearSVC": tune_and_train_LinearSVC,
        }
    elif args.test_set == "df_curated_biowin":
        classifiers = {
            "XGBClassifier": tune_and_train_XGBClassifier,
            "RandomForestClassifier": tune_and_train_RandomForestClassifier,
            "ExtraTreesClassifier": tune_and_train_ExtraTreesClassifier,
            "HistGradientBoostingClassifier": tune_and_train_HistGradientBoostingClassifier,
            "MLPClassifier": tune_and_train_MLPClassifier,
        }

    for classifier in classifiers:
        log.info(f" \n Currently running the classifier {classifier}")
        _, _, _, _ = classifiers[classifier](
            df=train_data,
            nsplits=args.nsplits,
            df_test=test_data,
        )


def run_classifiers_RDK(datasets: Dict[str, pd.DataFrame], df_test: str) -> None:
    train_data = datasets[args.train_set]
    test_data = datasets[df_test]

    if args.test_set == "df_curated_scs":
        classifiers = {
            "MLPClassifier": tune_and_train_MLPClassifier,
            "LogisticRegressionCV": tune_and_train_LogisticRegressionCV,
            "PassiveAggressiveClassifier": tune_and_train_PassiveAggressiveClassifier,
            "Perceptron": tune_and_train_Perceptron,
            "LogisticRegression": tune_and_train_LogisticRegression,
        }
    elif args.test_set == "df_curated_biowin":
        classifiers = {
            "MLPClassifier": tune_and_train_MLPClassifier,
            "GradientBoostingClassifier": tune_and_train_GradientBoostingClassifier,
            "ExtraTreesClassifier": tune_and_train_ExtraTreesClassifier,
            "XGBClassifier": tune_and_train_XGBClassifier,
            "HistGradientBoostingClassifier": tune_and_train_HistGradientBoostingClassifier,
        }

    for classifier in classifiers:
        log.info(f" \n Currently running the classifier {classifier}")
        _, _, _, _ = classifiers[classifier](
            df=train_data,
            nsplits=args.nsplits,
            df_test=test_data,
        )


def run_classifiers_Morgan(datasets: Dict[str, pd.DataFrame], df_test: str) -> None:
    train_data = datasets[args.train_set]
    test_data = datasets[df_test]

    if args.test_set == "df_curated_scs":
        classifiers = {
            "ExtraTreesClassifier": tune_and_train_ExtraTreesClassifier,
            "HistGradientBoostingClassifier": tune_and_train_HistGradientBoostingClassifier,
            "XGBClassifier": tune_and_train_XGBClassifier,
            "RandomForestClassifier": tune_and_train_RandomForestClassifier,
            "LogisticRegressionCV": tune_and_train_LogisticRegressionCV,
        }
    elif args.test_set == "df_curated_biowin":
        classifiers = {
            "ExtraTreesClassifier": tune_and_train_ExtraTreesClassifier,
            "RandomForestClassifier": tune_and_train_RandomForestClassifier,
            "LogisticRegressionCV": tune_and_train_LogisticRegressionCV,
            "XGBClassifier": tune_and_train_XGBClassifier,
            "HistGradientBoostingClassifier": tune_and_train_HistGradientBoostingClassifier,
        }

    for classifier in classifiers:
        log.info(f" \n Currently running the classifier {classifier}")
        _, _, _, _ = classifiers[classifier](
            df=train_data,
            nsplits=args.nsplits,
            df_test=test_data,
        )

if __name__ == "__main__":

    datasets = get_class_datasets()

    if args.run_lazy:
        run_lazy_classifier(df=datasets[args.train_set], df_test=datasets[args.test_set])

    if args.feature_type == "MACCS":
        run_classifiers_MACCS(datasets=datasets, df_test=args.test_set)
    if args.feature_type == "RDK":
        run_classifiers_RDK(datasets=datasets, df_test=args.test_set)
    if args.feature_type == "Morgan":
        run_classifiers_Morgan(datasets=datasets, df_test=args.test_set)



