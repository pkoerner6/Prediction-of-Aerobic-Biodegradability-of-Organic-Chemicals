import pandas as pd
import argparse
import torch
import yaml
from argparse import Namespace
from typing import List, Dict, Tuple
from rdkit import Chem
import structlog
import sys
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from fast_transformers.masking import LengthMask as LM
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from biodegradation.ml_functions import get_balanced_data_adasyn
from biodegradation.ml_functions import get_class_results
from biodegradation.ml_functions import print_class_results
from biodegradation.MolFormer.finetune.tokenizer.tokenizer import MolTranBertTokenizer
from biodegradation.MolFormer.training.train_pubchem_light import LightningModule


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
args = parser.parse_args()


def load_checkpoint():
    with open("biodegradation/MoLFormer/data/Pretrained MoLFormer/hparams.yaml", "r") as f:
        config = Namespace(**yaml.safe_load(f))

    tokenizer = MolTranBertTokenizer("biodegradation/MoLFormer/training/bert_vocab.txt")

    ckpt = "biodegradation/MoLFormer/data/Pretrained MoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt"
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
        print(len(embedding))
    print(len(embeddings))
    return torch.cat(embeddings)


def get_improved_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_reg_improved = pd.read_csv(
        "biodegradation/dataframes/improved_data/reg_improved_env_biowin_both_readded.csv", index_col=0
    )
    df_class_improved = pd.read_csv(
        "biodegradation/dataframes/improved_data/class_improved_env_biowin_both_readded.csv", index_col=0
    )
    return df_class_improved, df_reg_improved


def run_class_model_with_new_features(df: pd.DataFrame, tokenizer, lm):
    def canonicalize(s):
        return Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False)

    skf = StratifiedKFold(n_splits=args.nsplits, shuffle=True, random_state=args.random_seed)
    x = df.drop("y_true", axis=1)
    y = df["y_true"]
    for train_index, test_index in skf.split(X=x, y=y):
        df_train = df.loc[train_index]
        df_test = df.loc[test_index]

        smiles = df_train.smiles.apply(canonicalize)
        x_train = get_embeddings(lm, smiles, tokenizer).numpy()
        y_train = df_train["y_true"]

        x_train, y_train = get_balanced_data_adasyn(random_seed=args.random_seed, x=x_train, y=y_train)
        model = XGBClassifier()
        head = model.fit(x_train, y_train)

        x_test = get_embeddings(lm, df_test.smiles.apply(canonicalize), tokenizer).numpy()
        prediction = model.predict(x_test)
        accuracy, f1, sensitivity, specificity = get_class_results(true=df_test["y_true"], pred=prediction)
        accuracy = print_class_results(accuracy, sensitivity, specificity, f1)


if __name__ == "__main__":
    df_class, df_reg = get_improved_datasets()
    tokenizer, lm = load_checkpoint()
    run_class_model_with_new_features(df_class, tokenizer, lm)
