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
import statistics

log = structlog.get_logger()
tqdm.pandas()


import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from processing_functions import remove_smiles_with_incorrect_format
from processing_functions import openbabel_convert
from processing_functions import load_class_data_paper
from processing_functions import load_checked_organics6
from processing_functions import get_inchi_main_layer
from processing_functions import load_regression_df

df_regression = pd.read_excel("datasets/Huang_Zhang_RegressionDataset.xlsx", index_col=0)
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

gluege_data = load_checked_organics6()

substances_from_reg_all_in_gluege_data = df_regression[df_regression["cas"].isin(gluege_data["cas"])]
print("Unique CAS from regression data that are in Gluege data: ", substances_from_reg_all_in_gluege_data["cas"].nunique())
substances_from_class_all_in_gluege_data = class_all[class_all["cas"].isin(gluege_data["cas"])]
print("Unique CAS from classification data that are in Gluege data: ", len(substances_from_class_all_in_gluege_data))

# def add_inchi_gluege(row):
#     cas = row['cas']
#     gluege_subset = gluege_data[gluege_data["cas"] == cas]
#     inchi_gluege = gluege_subset["inchi_from_smiles"].values[0]
#     return inchi_gluege

# substances_from_class_all_in_gluege_data["inchi_gluege"] = substances_from_class_all_in_gluege_data.apply(add_inchi_gluege, axis=1)
# substances_in_gluege_with_matching_inchi = substances_from_class_all_in_gluege_data[substances_from_class_all_in_gluege_data['inchi_from_smiles'] == substances_from_class_all_in_gluege_data['inchi_gluege']]
# substances_in_gluege_without_matching_inchi = substances_from_class_all_in_gluege_data[substances_from_class_all_in_gluege_data['inchi_from_smiles'] != substances_from_class_all_in_gluege_data['inchi_gluege']]
# print("Unique CAS from classification data that are in Gluege data and have matching CAS: ", len(substances_in_gluege_with_matching_inchi))
# print("Unique CAS from classification data that are in Gluege data and have no matching CAS: ", len(substances_in_gluege_without_matching_inchi))
# print("Percentage without macthcing CAS: ", (len(substances_in_gluege_without_matching_inchi))/ len(substances_from_class_all_in_gluege_data)* 100)

# substances_in_gluege_without_matching_inchi = get_inchi_main_layer(df=substances_in_gluege_without_matching_inchi, inchi_col='inchi_from_smiles', layers=2)
# substances_in_gluege_without_matching_inchi = get_inchi_main_layer(df=substances_in_gluege_without_matching_inchi, inchi_col='inchi_gluege', layers=2)
# matching_mol_formula = substances_in_gluege_without_matching_inchi[substances_in_gluege_without_matching_inchi['inchi_from_smiles_main_layer']==substances_in_gluege_without_matching_inchi['inchi_gluege_main_layer']]
# no_matching_mol_formula = substances_in_gluege_without_matching_inchi[substances_in_gluege_without_matching_inchi['inchi_from_smiles_main_layer']!=substances_in_gluege_without_matching_inchi['inchi_gluege_main_layer']]
# print("No matching molecular formula: ", len(no_matching_mol_formula))
# print("No matching molecular formula in percent of all in Gluege: ", len(no_matching_mol_formula)/ len(substances_from_class_all_in_gluege_data)* 100)
# print("Matching molecular formula: ", len(matching_mol_formula))

# substances_in_gluege_without_matching_inchi = get_inchi_main_layer(df=substances_in_gluege_without_matching_inchi, inchi_col='inchi_from_smiles', layers=4)
# substances_in_gluege_without_matching_inchi = get_inchi_main_layer(df=substances_in_gluege_without_matching_inchi, inchi_col='inchi_gluege', layers=4)
# substances_in_gluege_without_matching_inchi.to_csv("datasets/substances_from_class_Huang_in_gluege_without_matching_inchi.csv")
# matching_main_layer = substances_in_gluege_without_matching_inchi[substances_in_gluege_without_matching_inchi['inchi_from_smiles_main_layer']==substances_in_gluege_without_matching_inchi['inchi_gluege_main_layer']]
# no_matching_main_layer = substances_in_gluege_without_matching_inchi[substances_in_gluege_without_matching_inchi['inchi_from_smiles_main_layer']!=substances_in_gluege_without_matching_inchi['inchi_gluege_main_layer']]
# print("No matching main layer: ", len(no_matching_main_layer))
# print("No matching main layer in percent of all in Gluege: ", len(no_matching_main_layer)/ len(substances_from_class_all_in_gluege_data)* 100)
# print("Matching main layer: ", len(matching_main_layer))



# counted_duplicates = (
#     df_regression.groupby(df_regression["inchi_from_smiles"].tolist(), as_index=False).size().sort_values(by="size", ascending=False)
# )
# replicates_df = counted_duplicates[counted_duplicates["size"] > 1]
# replicates = replicates_df["index"].tolist()

# # inchi_with_more_than_one_smiles: List[str] = []
# # for inchi in replicates:
# #     df_inchi = df_regression[df_regression["inchi_from_smiles"] == inchi]
# #     unique_smiles = df_inchi["smiles"].nunique()
# #     if unique_smiles > 1:
# #         inchi_with_more_than_one_smiles.append(inchi)

# # print("InChI with more than one SMILES: ", len(inchi_with_more_than_one_smiles))


# # inchi_with_more_than_one_cas: List[str] = []
# # for inchi in replicates:
# #     df_inchi = df_regression[df_regression["inchi_from_smiles"] == inchi]
# #     unique_cas = df_inchi["cas"].nunique()
# #     if unique_cas > 1:
# #         inchi_with_more_than_one_cas.append(inchi)

# # print("InChI with more than one CAS in Huang reg data: ", len(inchi_with_more_than_one_cas))




# class_all = pd.read_csv("datasets/class_all.csv", index_col=0)
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


# # inchi_with_more_than_one_cas: List[str] = []
# # for inchi in replicates:
# #     df_inchi = df_regression[df_regression["inchi_from_smiles"] == inchi]
# #     unique_cas = df_inchi["cas"].nunique()
# #     if unique_cas > 1:
# #         inchi_with_more_than_one_cas.append(inchi)

# # print("InChI with more than one CAS in Reg data: ", len(inchi_with_more_than_one_cas))




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



##### Analyse variance among study results 
# df = load_regression_df()
df = pd.read_csv("datasets/data_processing/reg_improved_no_metal_env_smiles.csv", index_col=0)
df = remove_smiles_with_incorrect_format(df=df, col_name_smiles="smiles")
df = openbabel_convert(
    df=df,
    input_type="smiles",
    column_name_input="smiles",
    output_type="inchi",
)

print("Entries in reg df: ", len(df))
df = df[df["time_day"] == 28.0]
print("Entries in reg df with studies for 28 days: ", len(df))
print("Unique inchi in reg df: ", df["inchi_from_smiles"].nunique())

counted_duplicates = df.groupby(df["inchi_from_smiles"].tolist(), as_index=False).size().sort_values(by="size", ascending=False)
df_more_than_1_replicates = counted_duplicates[counted_duplicates["size"] > 1]
df_1_replicate = counted_duplicates[counted_duplicates["size"] == 1]

print("Entries with more than 1 replicate: ", len(df_more_than_1_replicates))
print("Entries with 1 replicate: ", len(df_1_replicate))

df_multi = df_more_than_1_replicates

all_std = []
std_over = []
std_under = []
for inchi in df_multi['index']:
    df_curr = df[df["inchi_from_smiles"]==inchi]
    std = statistics.stdev(df_curr["biodegradation_percent"])
    if std > 0.3:
        std_over.append(inchi)
    if std < 0.15:
        std_under.append(inchi)
    all_std.append(std)
print("Average variance: ", sum(all_std)/len(df_multi))
print("Number of substances with variance over 0.3: ", len(std_over))
print("Percentage of substances over 0.3 (relative to all substances with more than 1 study): ", len(std_over)/len(df_more_than_1_replicates)*100)
print("Number of substances with variance under 0.15: ", len(std_under))
print("Percentage of substances under 0.15 (relative to all substances with more than 1 study): ", len(std_under)/len(df_more_than_1_replicates)*100)
df_over = df[df["inchi_from_smiles"].isin(std_over)]
print("Number of studies associated to the substances with variance over 0.5: ", len(df_over))









