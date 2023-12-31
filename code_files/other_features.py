"""Needs to be run with venv 'molformer_venv'"""

import numpy as np
import pandas as pd
import argparse
import torch
import yaml
from tqdm import tqdm
from argparse import Namespace
from typing import List, Dict, Tuple
from rdkit import Chem
import structlog
log = structlog.get_logger()
import sys
import os

from fast_transformers.masking import LengthMask as LM
from xgboost import XGBClassifier

tqdm.pandas()

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from code_files.processing_functions import get_class_datasets
from code_files.processing_functions import get_labels_colors_progress
from code_files.processing_functions import plot_results_with_standard_deviation
from code_files.processing_functions import bit_vec_to_lst_of_lst
from code_files.processing_functions import convert_to_morgan_fingerprints
from code_files.processing_functions import convert_to_maccs_fingerprints
from code_files.processing_functions import convert_to_rdk_fingerprints
from code_files.ml_functions import split_classification_df_with_fixed_test_set
from code_files.ml_functions import run_balancing_and_training
from MolFormer.finetune.tokenizer.tokenizer import MolTranBertTokenizer
from MolFormer.training.train_pubchem_light import LightningModule


parser = argparse.ArgumentParser()
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
    help="Number of KFold splits",
)
parser.add_argument(
    "--feature_type",
    type=str,
    default="Molformer",
    choices=["MACCS", "RDK", "Morgan", "Molformer"],
    help="How to generate the features",
)
parser.add_argument(
    "--test_set",
    type=str,
    default="df_curated_scs",
    choices=["df_curated_scs", "df_curated_biowin"],
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


def skf_class_fixed_testset_other_features(
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_type: str,
    paper: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[pd.DataFrame], List[int]]:

    cols=["cas", "smiles"]

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


def train_XGBClassifier_other_features(
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_type: str,
    paper: bool,
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
        feature_type=feature_type,
        paper=paper,
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


    metrics = ["balanced accuracy", "sensitivity", "specificity", "f1"]
    metrics_values = [
        lst_accu,
        lst_sensitivity,
        lst_specificity,
        lst_f1,
    ]
    for metric, metric_values in zip(metrics, metrics_values):
        log.info(
            f"{metric} for {feature_type}: ",
            score="{:.1f}".format(np.mean(metric_values) * 100) + " ± " + "{:.1f}".format(np.std(metric_values) * 100),
        )
    return (
        lst_accu,
        lst_sensitivity,
        lst_specificity,
        lst_f1,
    )


def run_for_different_dfs():

    datasets = get_class_datasets()
    df_test = datasets[args.test_set].copy()

    balanced_accuracy: List[np.ndarray] = [np.asarray([])] * len(datasets)
    f1: List[np.ndarray] = [np.asarray([])] * len(datasets)
    sensitivity: List[np.ndarray] = [np.asarray([])] * len(datasets)
    specificity: List[np.ndarray] = [np.asarray([])] * len(datasets)

    for indx, (dataset_name, dataset) in enumerate(datasets.items()):
        log.info(f"Entries in {dataset_name}", entries=len(dataset))

        paper = True if dataset_name=="df_paper" else False
        lst_accu_paper, _, _, _ = train_XGBClassifier_other_features(
            df=dataset,
            df_test=df_test,
            feature_type=args.feature_type,
            paper=paper,
        )
        balanced_accuracy[indx] = np.asarray(lst_accu_paper)

    balanced_accuracy = [np.array([0.876])] + balanced_accuracy  # reported balanced accuracy from Huang and Zhang with pKa and alpha values
    title_to_data = {
        "Balanced_accuracy": balanced_accuracy,
    }

    labels, colors = get_labels_colors_progress()
    for title, data in title_to_data.items():
        data = [array * 100 for array in data]

        plot_results_with_standard_deviation(
            all_data=data,
            labels=labels,
            colors=colors,
            title=title,
            seed=args.random_seed,
            plot_with_paper=True,
            save_ending=f"{args.feature_type}_features",
            test_set_name=args.test_set,
        )

if __name__ == "__main__":
    run_for_different_dfs()
