import argparse
import pandas as pd
import numpy as np
import structlog
import sys
import os

import torch
import yaml
from tqdm import tqdm
from argparse import Namespace
from fast_transformers.masking import LengthMask as LM

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

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from processing_functions import get_class_datasets
from processing_functions import bit_vec_to_lst_of_lst
from processing_functions import convert_to_morgan_fingerprints
from processing_functions import convert_to_maccs_fingerprints
from processing_functions import convert_to_rdk_fingerprints
from ml_functions import split_regression_df_with_grouping
from ml_functions import get_balanced_data_adasyn
from ml_functions import report_perf_hyperparameter_tuning
from ml_functions import get_class_results
from MolFormer.finetune.tokenizer.tokenizer import MolTranBertTokenizer
from MolFormer.training.train_pubchem_light import LightningModule



parser = argparse.ArgumentParser()

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
    choices=["MACCS", "RDK", "Morgan", "Molformer"],
    help="How to generate the features",
)
args = parser.parse_args()


def load_checkpoint():
    with open("MoLFormer/data/Pretrained MoLFormer/hparams.yaml", "r") as f:
        config = Namespace(**yaml.safe_load(f))

    tokenizer = MolTranBertTokenizer("MoLFormer/training/bert_vocab.txt")

    ckpt = "MoLFormer/data/Pretrained MoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt"
    lm = LightningModule(config, tokenizer.vocab).load_from_checkpoint(ckpt, config=config, vocab=tokenizer.vocab)

    return tokenizer, lm


def batch_split(data, batch_size=64):
    i = 0
    while i < len(data):
        yield data[i : min(i + batch_size, len(data))]
        i += batch_size


def get_embeddings(model, smiles, tokenizer, batch_size=64):
    model.eval()
    embeddings = []
    with tqdm(total=len(smiles), desc="Getting embeddings from MolFormer", ncols=100) as pbar:
        for batch in batch_split(smiles, batch_size=batch_size):
            batch_enc = tokenizer.batch_encode_plus(batch, padding=True, add_special_tokens=True)
            idx, mask = torch.tensor(batch_enc["input_ids"]), torch.tensor(batch_enc["attention_mask"])
            with torch.no_grad():
                token_embeddings = model.blocks(model.tok_emb(idx), length_mask=LM(mask.sum(-1)))
            # average pooling over tokens
            input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            embeddings.append(embedding.detach().cpu())
            pbar.update(len(batch))
    return torch.cat(embeddings)


def create_features_molformer(df: pd.DataFrame, tokenizer, lm) -> pd.DataFrame:
    # Pretrained features
    def canonicalize(s):
        return Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False)

    smiles = df.smiles.apply(canonicalize)
    embeddings = get_embeddings(lm, smiles, tokenizer).numpy()
    df = df.copy()
    df["fingerprint"] = [embedding.tolist() for embedding in embeddings]
    return df


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
    elif feature_type=="Molformer":
        tokenizer, lm = load_checkpoint()
        df = create_features_molformer(df, tokenizer, lm)
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

    clf = LazyClassifier(predictions=True, verbose=0)
    models, _ = clf.fit(x_balanced, x_test, y_balanced, y_test)
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


def train_classifier_with_best_hyperparamters(
    df: pd.DataFrame, 
    df_test: pd.DataFrame, 
    model,
):  
    df_train = df[~df["cas"].isin(df_test["cas"])]
    df_train = df_train[~df_train["inchi_from_smiles"].isin(df_test["inchi_from_smiles"])]

    assert len(df_train) + len(df_test) == len(df)
    x = create_input_classification_other_features(df=df_train, feature_type=args.feature_type)
    y = df_train["y_true"]
    x, y = get_balanced_data_adasyn(random_seed=args.random_seed, x=x, y=y)

    model.fit(x, y)

    x_test = create_input_classification_other_features(df=df_test, feature_type=args.feature_type)
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
    accu, f1, sensitivity, specificity = tune_and_train_classifiers(
        df=df,
        nsplits=nsplits,
        df_test=df_test,
        n_jobs=1,
        search_spaces=search_spaces,
        model=model,
    )
    return accu, f1, sensitivity, specificity


def run_and_plot_classifiers(df_class: pd.DataFrame) -> None:

    train_data, test_data = train_test_split(df_class, test_size=0.2, random_state=args.random_seed)

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
                df=train_data,
                nsplits=args.nsplits,
                df_test=test_data,
            )
            accuracy_all.append(np.asarray(accu))
            f1_all.append(np.asarray(f1))
            sensitivity_all.append(np.asarray(sensitivity))
            specificity_all.append(np.asarray(specificity))


if __name__ == "__main__":

    datasets = get_class_datasets()

    if args.run_lazy:
        run_lazy_classifier(df=datasets[args.train_set], df_test=datasets[args.test_set])
    # run_and_plot_classifiers(df_class=df_class)
