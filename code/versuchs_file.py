import numpy as np
import pandas as pd
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.Chem import AllChem
import sys
import os
from typing import List, Tuple
import pandas as pd
import structlog
import tqdm
from tqdm.auto import tqdm
import sys

log = structlog.get_logger()
tqdm.pandas()


import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from biodegradation.processing_functions import remove_smiles_with_incorrect_format
from biodegradation.processing_functions import openbabel_convert
from biodegradation.processing_functions import load_class_data_paper

df_regression = pd.read_excel("biodegradation/datasets/Huang_Zhang_RegressionDataset.xlsx", index_col=0)
df_regression.rename(
    columns={
        "Substance Name": "name",
        "Name type": "name_type",
        "CAS Number": "cas",
        "Smiles": "smiles",
        "Reliability": "reliability",
        "Endpoint": "endpoint",
        "Guideline": "guideline",
        "Principle": "principle",
        "Time (day)": "time_day",
        "Biodegradation (0.1 means 10%)": "biodegradation_percent",
    },
    inplace=True,
)

df_regression = remove_smiles_with_incorrect_format(df=df_regression, col_name_smiles="smiles")
df_regression = openbabel_convert(
    df=df_regression,
    input_type="smiles",
    column_name_input="smiles",
    output_type="inchi",
)

print("Unique cas in huang_reg: ", df_regression["cas"].nunique())
print("Unique smiles in huang_reg: ", df_regression["smiles"].nunique())
print("Unique inchi in huang_reg: ", df_regression["inchi_from_smiles"].nunique())



class_original, class_external, class_all = load_class_data_paper()

class_all = remove_smiles_with_incorrect_format(df=class_all, col_name_smiles="smiles")
class_all = openbabel_convert(
    df=class_all,
    input_type="smiles",
    column_name_input="smiles",
    output_type="inchi",
)

print("Unique cas in huang_class: ", class_all["cas"].nunique())
print("Unique smiles in huang_class: ", class_all["smiles"].nunique())
print("Unique inchi in huang_class: ", class_all["inchi_from_smiles"].nunique())

# counted_duplicates = (
#     df_regression.groupby(df_regression["inchi_from_smiles"].tolist(), as_index=False).size().sort_values(by="size", ascending=False)
# )
# replicates_df = counted_duplicates[counted_duplicates["size"] > 1]
# replicates = replicates_df["index"].tolist()

# inchi_with_more_than_one_smiles: List[str] = []
# for inchi in replicates:
#     df_inchi = df_regression[df_regression["inchi_from_smiles"] == inchi]
#     unique_smiles = df_inchi["smiles"].nunique()
#     if unique_smiles > 1:
#         inchi_with_more_than_one_smiles.append(inchi)

# print("InChI with more than one SMILES: ", len(inchi_with_more_than_one_smiles))


# inchi_with_more_than_one_cas: List[str] = []
# for inchi in replicates:
#     df_inchi = df_regression[df_regression["inchi_from_smiles"] == inchi]
#     unique_cas = df_inchi["cas"].nunique()
#     if unique_cas > 1:
#         inchi_with_more_than_one_cas.append(inchi)

# print("InChI with more than one CAS in Huang reg data: ", len(inchi_with_more_than_one_cas))




# class_all = pd.read_csv("biodegradation/dataframes/class_all.csv", index_col=0)
# class_all = remove_smiles_with_incorrect_format(df=class_all, col_name_smiles="smiles")
# class_all = openbabel_convert(
#     df=class_all,
#     input_type="smiles",
#     column_name_input="smiles",
#     output_type="inchi",
# )
# unique_inchi_in_class_data = class_all["inchi_from_smiles"].nunique()
# entries_class_data = len(class_all)
# print("Data points that appeared more than once in the class dataset: ", entries_class_data-unique_inchi_in_class_data)


# inchi_with_more_than_one_cas: List[str] = []
# for inchi in replicates:
#     df_inchi = df_regression[df_regression["inchi_from_smiles"] == inchi]
#     unique_cas = df_inchi["cas"].nunique()
#     if unique_cas > 1:
#         inchi_with_more_than_one_cas.append(inchi)

# print("InChI with more than one CAS in Reg data: ", len(inchi_with_more_than_one_cas))




# counted_duplicates_class = (
#     class_all.groupby(class_all["inchi_from_smiles"].tolist(), as_index=False).size().sort_values(by="size", ascending=False)
# )
# replicates_df = counted_duplicates_class[counted_duplicates_class["size"] > 1]
# replicates_class = replicates_df["index"].tolist()

# inchi_with_more_than_one_cas_class: List[str] = []
# for inchi in replicates_class:
#     df_inchi = df_regression[df_regression["inchi_from_smiles"] == inchi]
#     unique_cas = df_inchi["cas"].nunique()
#     if unique_cas > 1:
#         inchi_with_more_than_one_cas_class.append(inchi)

# print("InChI with more than one CAS in Class data: ", len(inchi_with_more_than_one_cas_class))
