import argparse
import pandas as pd
import numpy as np
import structlog
import sys
import os

log = structlog.get_logger()
from typing import List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
# from tensorflow import keras
# from tensorflow.keras import layers
from rdkit import Chem
from rdkit.Chem import AllChem
import deepchem as dc
from deepchem.feat import RdkitGridFeaturizer
from deepchem.models.torch_models import AtomConvModel
import tempfile

parser = argparse.ArgumentParser()
tqdm.pandas(desc='Description')

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from processing_functions import get_class_datasets

parser.add_argument(
    "--random_seed",
    type=int,
    default=42,
    help="Select the random seed",
)
args = parser.parse_args()


def run_lazy():
  train_df = pd.read_csv("datasets/different_features/class_curated_scs_biowin_readded_molformer_embeddings.csv", index_col=0)
  train_df['fingerprint'] = train_df['fingerprint'].apply(eval)
  test_df = pd.read_csv("datasets/different_features/class_curated_scs_biowin_readded_multiple_molformer_embeddings.csv", index_col=0)
  test_df['fingerprint'] = test_df['fingerprint'].apply(eval)
  train_df = train_df[~train_df["cas"].isin(test_df["cas"])]
  test_df.reset_index(inplace=True, drop=True)
  train_df.reset_index(inplace=True, drop=True)
  print(len(train_df[train_df["y_true"]==1]))
  print(len(train_df[train_df["y_true"]==0]))

  print(len(test_df[test_df["y_true"]==1]))
  print(len(test_df[test_df["y_true"]==0]))

  size = 768
  cols = [*range(0, size, 1)] 
  x_train = pd.DataFrame(train_df.fingerprint.tolist(), columns=cols)
  y_train = train_df["y_true"]
  x_test = pd.DataFrame(test_df.fingerprint.tolist(), columns=cols)
  y_test = test_df["y_true"]

  clf = LazyClassifier(predictions=True, verbose=2)
  models, _ = clf.fit(x_train, x_test, y_train, y_test)
  log.info(models)


def test_GraphConvModel():
  df = pd.read_csv("datasets/curated_data/class_curated_final.csv", index_col=0)
  train_df, test_df = train_test_split(df, test_size=0.2, random_state=args.random_seed)

  # Function to generate 3D coordinates
  with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
    train_df.to_csv(tmpfile.name)
    featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
    loader = dc.data.CSVLoader(["y_true"], feature_field="smiles", featurizer=featurizer)
    train_dataset = loader.create_dataset(tmpfile.name)
    print(len(train_dataset))
  
  with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
    test_df.to_csv(tmpfile.name)
    featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
    loader = dc.data.CSVLoader(["y_true"], feature_field="smiles", featurizer=featurizer)
    test_dataset = loader.create_dataset(tmpfile.name)
    print(len(test_dataset))

  model = dc.models.GraphConvModel(
    n_tasks=1, 
    mode='classification', 
    graph_conv_layers=[256, 128, 64, 64],
    dense_layer_size = 128,
    dropout=0.1,
    number_atom_features=75,
    n_classes=2,
    batch_normalize=True,
    uncertainty=False,
  )
  model.fit(train_dataset, nb_epoch=50)

  metric = dc.metrics.Metric(dc.metrics.balanced_accuracy_score)
  print('Training set score:', model.evaluate(train_dataset, [metric], transformers=[]))
  print('Test set score:', model.evaluate(test_dataset, [metric], transformers=[]))


if __name__ == "__main__":
  # run_lazy()
  test_GraphConvModel()










