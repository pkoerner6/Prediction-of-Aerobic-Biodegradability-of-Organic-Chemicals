

import pandas as pd
import argparse
import torch
import yaml
from tqdm import tqdm
from argparse import Namespace
from typing import List, Dict, Tuple
from rdkit import Chem
import structlog
import sys
import os

from sklearn.model_selection import StratifiedKFold

from fast_transformers.masking import LengthMask as LM
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
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


def create_features_molformer(df: pd.DataFrame, df_name: str, tokenizer, lm) -> pd.DataFrame:
    def canonicalize(s):
        return Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False)

    smiles = df.smiles.apply(canonicalize)
    embeddings = get_embeddings(lm, smiles, tokenizer).numpy()
    df_with_fp = df.copy()
    df_with_fp["fingerprint"] = [embedding.tolist() for embedding in embeddings]
    df_with_fp.to_csv(f"datasets/different_features/{df_name}_molformer_embeddings.csv")
    return df_with_fp


def load_dfs() -> Dict[str, pd.DataFrame]:
    df_class = pd.read_csv("datasets/external_data/class_all.csv", index_col=0)
    curated_scs = pd.read_csv("datasets/curated_data/class_curated_scs.csv", index_col=0)
    biowin = pd.read_csv("datasets/curated_data/class_curated_biowin.csv", index_col=0)

    curated_final = pd.read_csv(
        "datasets/curated_data/class_curated_final.csv", index_col=0
    )
    df_removed = pd.read_csv("datasets/curated_data/class_curated_final_removed.csv", index_col=0)
    df_removed = df_removed[df_removed["principle"].isnull()]
    df_removed.reset_index(inplace=True, drop=True)
    datasets = {
        "class_curated_scs": curated_scs,
        "class_curated_final": curated_final,
    }
    for df_name, df in datasets.items():
        df.reset_index(inplace=True, drop=True)
        datasets[df_name] = df
    return datasets

if __name__ == "__main__":
    tokenizer, lm = load_checkpoint()

    datasets = load_dfs()

    create_features_molformer(datasets, tokenizer, lm)


