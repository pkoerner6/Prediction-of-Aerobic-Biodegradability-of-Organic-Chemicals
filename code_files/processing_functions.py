import json
import math
import re
import ssl
import subprocess
import time
import tqdm
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import structlog
import glob
import requests
import os
import cirpy
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from pubchempy import get_compounds
import urllib.request
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import matplotlib.pyplot as plt
import statistics
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


RDLogger.DisableLog("rdApp.*")  # Disable warnings from rdkit
log = structlog.get_logger()
tqdm.pandas()


def openbabel_convert(df: pd.DataFrame, input_type: str, column_name_input: str, output_type: str) -> pd.DataFrame:
    """Turn pandas df column into text file and then use openbabel to convert SMILES, InChIKey or InChI into each other.
    Add column with result to df."""
    assert input_type == "inchi" or input_type == "smiles" or input_type == "inchikey"
    assert output_type == "inchi" or output_type == "smiles" or output_type == "inchikey"
    assert input_type != output_type

    input = list(df[column_name_input])
    input_len = len(input)
    with open("input.txt", "w") as f:
        for item in input:
            f.write(item + "\n")
    with open("input.txt") as f:
        lines = [line.rstrip() for line in f]
    input_df = pd.DataFrame(np.array(lines), columns=[input_type])
    len_input_df = len(input_df)
    assert input_len == len_input_df

    process = subprocess.run(
        ["obabel", f"-i{input_type}", "input.txt", f"-o{output_type}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    df[f"{output_type}_from_{column_name_input}"] = process.stdout.decode("utf-8").split("\n")[
        : len(df)
    ]  # Don't take last empty line
    return df


def remove_smiles_with_incorrect_format(df: pd.DataFrame, col_name_smiles: str, prnt=False) -> pd.DataFrame:
    """SMILES can be in a format that cannot be converted by openbabel"""
    df_clean = df.copy()
    df_clean[col_name_smiles] = df_clean[col_name_smiles].apply(lambda x: "nan" if "*" in x or "|" in x else x)
    df_clean = df_clean[df_clean[col_name_smiles] != "nan"]
    invalid_smiles = ["c1cccc1"] # Invalid SMILES string: not convertable to mol
    len_df_clean = len(df_clean)
    df_clean = df_clean[~df_clean[col_name_smiles].isin(invalid_smiles)]
    len_df_clean_after = len(df_clean)
    if len_df_clean_after < len_df_clean:
        log.warn("Removed this many SMILES with incorrect format", removed=len_df_clean-len_df_clean_after)
    df_clean.reset_index(inplace=True, drop=True)
    if prnt: 
        log.warn("Removed this many data points because SMILES had incorrect format", removed=len(df)-len(df_clean))
    df_clean.reset_index(inplace=True, drop=True)
    return df_clean


def label_data_based_on_percentage(df: pd.DataFrame) -> pd.DataFrame:
    """Add column with label based on percentage -> 1 means degradable, 0 means not degradable."""
    df = df.copy()
    df["y_true"] = np.where(
        ((df["principle"] == "DOC Die Away") & (df["biodegradation_percent"] >= 0.7))
        | ((df["principle"] != "DOC Die Away") & (df["biodegradation_percent"] >= 0.6)),
        1,
        0,
    )
    return df


def group_and_label_chemicals(df: pd.DataFrame, col_to_group_by: str) -> pd.DataFrame:
    df_no_duplicates = df.drop_duplicates(subset=col_to_group_by, keep="first", ignore_index=True)
    for name, group in df.groupby(col_to_group_by):
        mean_label = group["y_true"].mean()
        index = df_no_duplicates[df_no_duplicates[col_to_group_by] == name].index[0]
        if mean_label > 0.5:
            df_no_duplicates.at[index, "y_true"] = 1
        elif mean_label < 0.5:
            df_no_duplicates.at[index, "y_true"] = 0
        elif mean_label == 0.5:
            df_no_duplicates.at[index, "y_true"] = -1
    df_no_duplicates = df_no_duplicates[df_no_duplicates["y_true"] != -1]
    return df_no_duplicates


def get_cid_from_inchi_pubchempy(df: pd.DataFrame) -> str:
    inchi_for_which_no_cid: List[str] = []

    def get_cid(row):
        inchi = row["inchi_from_smiles"]
        ssl._create_default_https_context = ssl._create_unverified_context
        time.sleep(0.2)
        comps = get_compounds(inchi, "inchi")
        if len(comps) >= 1:  # Even if muliple results for a CAS number, we take the first ("best match" on website)
            cid = comps[0].cid
        else:
            cid = None
            inchi_for_which_no_cid.append(inchi)
        return cid

    df["cid"] = df.progress_apply(get_cid, axis=1)
    log.warn(
        "InChI for which no CID was found on PubChem",
        inchi_no_cid=inchi_for_which_no_cid,
    )
    return df


def go_to_values(data: Dict) -> pd.DataFrame:
    """Takes the response from pubchem using pug-view and return df that contains the information"""
    df_layer1 = pd.json_normalize(data)
    df_layer2 = pd.json_normalize(df_layer1["Record.Section"].iloc[0])
    df_layer3 = pd.json_normalize(df_layer2["Section"].iloc[0])
    df_layer4 = pd.json_normalize(df_layer3["Section"].iloc[0])
    df_info = pd.json_normalize(df_layer4["Information"].iloc[0])
    return df_info


def go_to_references(data: dict) -> pd.DataFrame:
    """Takes the response from pubchem using pug-view and return df that contains the references of the information"""
    df = pd.json_normalize(data)
    df_ref = pd.json_normalize(df["Record.Reference"].iloc[0])
    return df_ref


def get_all_cas_pubchem(cid: int) -> Tuple[str, str]:
    """Takes a CID as input and returns the CAS number(s) and the references as strings"""
    num_to_ref: Dict[str, str] = {}
    ref_str, unique_cas = "", ""
    if math.isnan(cid):  # if the given cid is NaN, then return empty strings
        return unique_cas, ref_str
    cid = int(cid)
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        time.sleep(0.2)
        req = urllib.request.urlopen(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=CAS"
        )
        elevations = req.read()
        data = json.loads(elevations)
    except:
        return unique_cas, ref_str
    df_values = go_to_values(data)
    df_references = go_to_references(data)
    df_values.rename(
        columns={"ReferenceNumber": "reference", "Value.StringWithMarkup": "cas"},
        inplace=True,
    )
    df_values["cas"] = df_values["cas"].apply(lambda x: x[0]["String"])
    for index in range(len(df_references)):
        num = df_references.at[index, "ReferenceNumber"]
        ref = df_references.at[index, "SourceName"]
        num_to_ref[num] = ref
    df_values["reference"] = df_values["reference"].map(num_to_ref)
    unique_cas_lst = df_values["cas"].unique()
    for indx, cas in enumerate(unique_cas_lst):
        refs = df_values.loc[df_values["cas"] == cas, "reference"]
        refs_str = ", ".join([str(elem) for elem in refs])
        ref_str += cas + ": " + refs_str
        if indx < len(unique_cas_lst) - 1:
            ref_str += "; "
    unique_cas = ", ".join([str(elem) for elem in unique_cas_lst])
    return unique_cas, ref_str


def get_deprecated_cas(cid: int) -> str:
    """Takes a CID as input and returns the deprecated CAS number(s) as string"""
    deprecated_cas = ""
    if math.isnan(cid):
        return deprecated_cas
    cid = int(cid)
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        time.sleep(0.2)
        req = urllib.request.urlopen(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=Deprecated+CAS"
        )
        elevations = req.read()
        data = json.loads(elevations)
    except:
        return deprecated_cas
    df_values = go_to_values(data)
    df_values.rename(
        columns={"ReferenceNumber": "reference", "Value.StringWithMarkup": "cas"},
        inplace=True,
    )
    df_values["cas"] = df_values["cas"].apply(lambda x: x[0]["String"])
    deprecated_cas_lst = df_values["cas"].tolist()
    deprecated_cas = ", ".join([str(elem) for elem in deprecated_cas_lst])
    return deprecated_cas


def is_cas_right_format(cas_number: str) -> bool:
    """Checks if CAS has the right format"""
    regex = re.compile(r"\d+\-\d\d\-\d")
    return bool(regex.fullmatch(cas_number))


def get_inchi_layers(row, col_name: str, layers: int):
    inchi = str(row[col_name])
    inchi_new = ""
    if (inchi != np.nan) and (inchi != "nan") and (inchi != ""):
        if layers == 4:
            try:
                inchi_split = inchi.split("/")
                inchi_new = inchi_split[0] + "/" + inchi_split[1] + "/" + inchi_split[2] + "/" + inchi_split[3]
            except:
                pass
                # log.warn("No main layer for inchi", inchi=inchi)
        elif layers == 3:
            try:
                inchi_split = inchi.split("/")
                inchi_new = inchi_split[0] + "/" + inchi_split[1] + "/" + inchi_split[2]
            except:
                log.warn("No main layer for inchi", inchi=inchi)
        elif layers == 2:
            try:
                inchi_split = inchi.split("/")
                inchi_new = inchi_split[0] + "/" + inchi_split[1]
            except:
                log.warn("No main layer for inchi", inchi=inchi)
        else:
            log.error("This number of layers cannot be returned!")
    return inchi_new


def add_cas_from_pubchem(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a dataframe that has CIDs in column 'cid' and adds columns containing the CAS from pubchem,
    CAS from pubchem with references, and deprecated cas"""

    def get_cas(row):
        cid = row["cid"]
        unique_cas, ref_str = get_all_cas_pubchem(cid=cid)
        deprecated_cas = get_deprecated_cas(cid)
        return pd.Series([unique_cas, ref_str, deprecated_cas])

    df[["cas_pubchem", "cas_ref_pubchem", "deprecated_cas_pubchem"]] = df.progress_apply(get_cas, axis=1)
    return df


def pubchem_cas_to_ref_dict(cas_ref: str) -> dict:
    cas_ref = str(cas_ref)
    ref_lst: List[str] = []
    num_to_ref: Dict[str, str] = {}
    first_split = cas_ref.split(": ")
    for item in first_split:
        second_split = item.split("; ")
        for item in second_split:
            ref_lst.append(item)
    for i in range(len(ref_lst)):
        if i % 2 == 0 and i < len(ref_lst) - 1:
            num_to_ref[ref_lst[i]] = ref_lst[i + 1]
    return num_to_ref


def find_best_cas_pubchem_based_on_ref(cas_to_ref_pubchem: Dict[str, str]) -> str:
    for cas in cas_to_ref_pubchem.keys():
        if "CAS Common Chemistry" in cas_to_ref_pubchem[cas]:
            return cas
        elif "European Chemicals Agency (ECHA)" in cas_to_ref_pubchem[cas]:
            return cas
    return min(cas_to_ref_pubchem.keys(), key=len)


def remove_organo_metals_function(df: pd.DataFrame, smiles_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if smiles_column not in list(df.columns):
        log.fatal("smiles_column not in columns")
    non_metals = [
        "Ar",
        "At",
        "Br",
        "C",
        "Cl",
        "F",
        "H",
        "He",
        "I",
        "Kr",
        "N",
        "Ne",
        "O",
        "Og",
        "P",
        "Rn",
        "S",
        "Se",
        "Ts",
        "Xe",
        "K",
        "Li",
        "Na",
        "Si",
    ]

    def remove_organo_metals_by_row(row):
        smiles = row[smiles_column]
        components: set[str] = set()
        if smiles != np.nan:
            regex = re.compile("[^a-zA-Z]")
            smiles = regex.sub("", smiles)
            elements = re.findall("[A-Z][^A-Z]*", smiles)
            for item in set(elements):
                components.add(item)
        components = {i for i in components if i not in non_metals}
        components = {
            x.replace("n", "") if (("n" in x) and ("Mn" not in x) and ("In" not in x) and ("Rn" not in x)) else x
            for x in components
        }
        components = {
            x.replace("o", "")
            if (
                ("o" in x)
                and ("Co" not in x)
                and ("Mo" not in x)
                and ("Ho" not in x)
                and ("Po" not in x)
                and ("No" not in x)
            )
            else x
            for x in components
        }
        components = {
            x.replace("s", "")
            if (
                ("s" in x)
                and ("As" not in x)
                and ("Cs" not in x)
                and ("Os" not in x)
                and ("Es" not in x)
                and ("Hs" not in x)
                and ("Ds" not in x)
            )
            else x
            for x in components
        }
        components = {
            x.replace("c", "") if (("c" in x) and ("Mc" not in x) and ("Ac" not in x) and ("Tc" not in x)) else x
            for x in components
        }
        components = {
            x.replace("i", "") if (("i" in x) and ("Ti" not in x) and ("Ni" not in x) and ("Bi" not in x)) else x
            for x in components
        }
        components = {
            x.replace("f", "") if (("f" in x) and ("Hf" not in x) and ("Cf" not in x) and ("Rf" not in x)) else x
            for x in components
        }
        components = {x.replace("k", "") if (("k" in x) and ("Bk" not in x)) else x for x in components}
        components = {i for i in components if i not in non_metals}
        return len(components) > 0

    df["organo_metal"] = df.apply(remove_organo_metals_by_row, axis=1)
    organo_metals = df[df["organo_metal"] == True]
    none_organo_metals = df[df["organo_metal"] == False]
    organo_metals = remove_smiles_with_incorrect_format(df=organo_metals, col_name_smiles="smiles")
    organo_metals = openbabel_convert(
        df=organo_metals,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )
    none_organo_metals = remove_smiles_with_incorrect_format(df=none_organo_metals, col_name_smiles="smiles")
    none_organo_metals = openbabel_convert(
        df=none_organo_metals,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )
    log.info("Number of organo-metals found", organo_metals_substances=organo_metals["inchi_from_smiles"].nunique())
    return none_organo_metals, organo_metals


def get_smiles_from_cas_pubchempy(df: pd.DataFrame) -> pd.DataFrame:
    def get_smiles_pubchem(row):
        isomeric_smiles = ""
        canonical_smiles = ""
        inchi = ""
        cas = row["cas"]
        ssl._create_default_https_context = ssl._create_unverified_context
        time.sleep(0.2)
        comps = get_compounds(cas, "name")
        if len(comps) >= 1:  # Even if muliple results for a CAS number, we take the first ("best match" on website)
            isomeric_smiles = comps[0].isomeric_smiles
            canonical_smiles = comps[0].canonical_smiles
            inchi = comps[0].inchi
        return pd.Series([isomeric_smiles, canonical_smiles, inchi])

    df[["isomeric_smiles_pubchem", "canonical_smiles_pubchem", "inchi_pubchem"]] = df.progress_apply(
        func=get_smiles_pubchem, axis=1
    )
    return df


def turn_cas_column_into_string(df: pd.DataFrame, cas_column: str) -> str:
    return "\n".join(df[cas_column])


def get_comptox(cas_string: str) -> None:
    driver = webdriver.Chrome()
    driver.get("https://comptox.epa.gov/dashboard/batch-search")
    time.sleep(3)  # Wait until page opens
    checkboxe_cas = driver.find_elements(By.XPATH, "//label[contains(@for,'CASRN')]")[0]
    checkboxe_cas.click()
    time.sleep(0.5)
    textarea = driver.find_elements(By.ID, "identifiers")[0]
    action = ActionChains(driver)
    action.click(on_element=textarea)
    action.send_keys(cas_string)
    action.perform()
    button = driver.find_elements(By.CLASS_NAME, "button.btn.btn-sm.btn-default")[1]  # Choose export options
    button.click()
    dropdown = driver.find_elements(By.ID, "export-detail-btn")[0]
    dropdown.click()
    dropdown_menu = driver.find_elements(By.CLASS_NAME, "dropdown-item")[15]  # Select CSV as export format
    dropdown_menu.click()
    cid_box = driver.find_elements(By.XPATH, "//label[contains(@for,'DTXCID-output')]")[0]
    cas_box = driver.find_elements(By.XPATH, "//label[contains(@for,'CASRN-output')]")[0]
    smiles_box = driver.find_elements(By.XPATH, "//label[contains(@for,'SMILES-output')]")[0]
    inchi_box = driver.find_elements(By.XPATH, "//label[contains(@for,'INCHI_STRING-output')]")[0]
    mf_box = driver.find_elements(By.XPATH, "//label[contains(@for,'MOLECULAR_FORMULA-output')]")[0]
    am_box = driver.find_elements(By.XPATH, "//label[contains(@for,'AVERAGE_MASS-output')]")[0]
    boxes = [
        cid_box,
        cas_box,
        smiles_box,
        inchi_box,
        mf_box,
        am_box,
    ]
    for box in boxes:
        time.sleep(0.2)
        box.click()
    time.sleep(0.2)
    download_button = driver.find_elements(By.CLASS_NAME, "fa.fa-download")[0]  # Click download export file
    download_button.click()
    time.sleep(8)  # Wait until popup opens
    download_button2 = driver.find_elements(By.CLASS_NAME, "btn.btn-default")[2]  # Download
    download_button2.click()
    time.sleep(8)
    driver.close()


def load_comptox_file_and_save() -> pd.DataFrame:
    downloads = glob.glob("/Users/paulina_koerner/Downloads/*")
    latest_download = max(downloads, key=os.path.getctime)
    df = pd.read_csv(latest_download)
    return df


def get_smiles_from_cas_comptox(df: pd.DataFrame) -> pd.DataFrame:
    """Need to do batchsearch on comptox and download the results as csv"""
    cas_string = turn_cas_column_into_string(df=df, cas_column="cas")
    get_comptox(cas_string)
    df_comptox = load_comptox_file_and_save()
    df_comptox.drop(
        columns=[
            "FOUND_BY",
            "DTXSID",
            "PREFERRED_NAME",
            "AVERAGE_MASS",
        ],
        inplace=True,
    )
    df_comptox.rename(
        columns={
            "INPUT": "cas",
            "CASRN": "cas_found_comptox",
            "DTXCID": "dtxcid_comptox",
            "INCHIKEY": "inchikey_comptox",
            "IUPAC_NAME": "iupac_name_comptox",
            "SMILES": "smiles_comptox",
            "INCHI_STRING": "inchi_comptox",
            "MOLECULAR_FORMULA": "molecular_formula_comptox",
        },
        inplace=True,
    )
    df_comptox = df_comptox[
        df_comptox["cas"] == df_comptox["cas_found_comptox"]
    ]  # Because sometimes comptox gives an result for a different CAS RN
    df_comptox = df_comptox[(df_comptox["smiles_comptox"].notna()) & (df_comptox["smiles_comptox"] != "N/A")]
    log.info("Found SMILES for this many CAS on Comptox", entries_comptox=len(df_comptox))
    df_comp_merged = df.merge(df_comptox, on="cas", how="left").drop(
        columns=["dtxcid_comptox", "cas_found_comptox", "molecular_formula_comptox"]
    )
    return df_comp_merged


def get_info_cas_common_chemistry(cas: str) -> Tuple[str, str]:
    smiles = ""
    inchi = ""
    url = "https://commonchemistry.cas.org/api/detail?cas_rn=" + cas
    resp = requests.get(url=url)
    data = resp.json()
    if "smile" in data.keys():
        smiles = data["smile"]
    if "inchi" in data.keys():
        inchi = data["inchi"]
    return smiles, inchi


def add_biowin_label(df: pd.DataFrame, mode: str, create_biowin_batch_all=False) -> pd.DataFrame:
    if create_biowin_batch_all:
        biowin_paper = pd.read_csv("datasets/biowin_batch/df_reg_paper.csv", index_col=0)
        biowin_curated_s = pd.read_csv("datasets/biowin_batch/df_reg_curated_s.csv", index_col=0) # df_reg_improved
        biowin_curated_scs = pd.read_csv("datasets/biowin_batch/df_reg_curated_scs.csv", index_col=0) # df_reg_improved_env
        biowin_lunghini = pd.read_csv("datasets/biowin_batch/lunghini_added_cas.csv", index_col=0) # lunghini_added_cas
        biowin_lunghini["7"] = biowin_lunghini["7"].str.strip()
        biowin_additional = pd.read_csv(
            "datasets/biowin_batch/df_regression_additional.csv", index_col=0
        )
        df_biowin = pd.concat(
            [
                biowin_paper,
                biowin_curated_s,
                biowin_curated_scs,
                biowin_lunghini,
                biowin_additional,
            ],
            ignore_index=True,
        )
        df_biowin.rename(
            columns={
                "0": "linear",
                "1": "non-linear",
                "2": "ultimate",
                "3": "primary",
                "4": "MITI-lin",
                "5": "MITI-non",
                "6": "anaerobic",
                "7": "smiles",
            },
            inplace=True,
        )
        df_biowin.drop_duplicates(["smiles"], keep="first", inplace=True, ignore_index=True)
        df_biowin = df_biowin[~df_biowin.linear.str.contains("INCOMPATIBLE")]
        df_biowin = df_biowin[~df_biowin.linear.str.contains("SMILES NOTATION PROBLEM")]
        df_biowin = df_biowin.astype(
            {
                "linear": float,
                "non-linear": float,
                "ultimate": float,
                "primary": float,
                "MITI-lin": float,
                "MITI-non": float,
                "anaerobic": float,
            }
        )
        df_biowin.to_csv("datasets/biowin_batch/biowin_batch.csv")

    df_biowin = pd.read_csv("datasets/biowin_batch/biowin_batch.csv", index_col=0)

    def probability_of_rapid_biodegradation_and_miti1_test_labeling(prob: float) -> int:
        if prob > 0.5:
            label = 1  # means likely to biodegrade fast for BIOWIN 1 & 2 and means ready biodegradable for BIOWIN5 & 6, source: https://www.epa.gov/sites/default/files/2015-05/documents/05-iad_discretes_june2013.pdf
        else:
            label = 0  # means NOT likely to biodegrade fast
        return label

    def expert_survey_biodegradation_labeling(rating: float) -> float:
        # aerobic biodegradation half-life (days), source: https://www.ipcp.ethz.ch/Armenia/Using_EpiSuite_and_Box_Models.pdf
        # according to Boethling et al., for BioWIN3 weeks or faster are RB and all predictions slower are NRB
        # for BioWin4, days or faster = RB; days to weeks or slower = NRB
        if rating > 4.75:
            label = 0.17
        elif (rating > 4.25) and (rating <= 4.75):
            label = 1.25
        elif (rating > 3.75) and (rating <= 4.25):
            label = 2.33

        elif (rating > 3.25) and (rating <= 3.75):
            label = 8.67
        elif (rating > 2.75) and (rating <= 3.25):
            label = 15.0
        elif (rating > 2.25) and (rating <= 2.75):
            label = 37.5
        elif (rating > 1.75) and (rating <= 2.25):
            label = 120
        elif (rating > 1.25) and (rating <= 1.75):
            label = 240
        elif rating < 1.25:
            label = 720
        return label

    def give_biowin_meaning(row):
        linear = row["linear"]
        linear_label = probability_of_rapid_biodegradation_and_miti1_test_labeling(linear)
        non_linear = row["non-linear"]
        non_linear_label = probability_of_rapid_biodegradation_and_miti1_test_labeling(non_linear)
        ultimate = row["ultimate"]
        ultimate_label = expert_survey_biodegradation_labeling(ultimate)
        primary = row["primary"]
        primary_label = expert_survey_biodegradation_labeling(primary)
        miti_linear = row["MITI-lin"]
        miti_linear_label = probability_of_rapid_biodegradation_and_miti1_test_labeling(miti_linear)
        miti_non_linear = row["MITI-non"]
        miti_non_linear_label = probability_of_rapid_biodegradation_and_miti1_test_labeling(miti_non_linear)
        total_class_label = 0
        if (linear_label + non_linear_label + miti_linear_label + miti_non_linear_label) >= 3.0:
            total_class_label = 1
        return pd.Series(
            [
                linear_label,
                non_linear_label,
                ultimate_label,
                primary_label,
                miti_linear_label,
                miti_non_linear_label,
                total_class_label,
            ]
        )

    df_biowin[
        [
            "linear_label",
            "non_linear_label",
            "ultimate_label",
            "primary_label",
            "miti_linear_label",
            "miti_non_linear_label",
            "total_class_label",
        ]
    ] = df_biowin.apply(give_biowin_meaning, axis=1)

    def label_data(row):
        biodeg = row["biodegradation_percent"]
        principle = row["principle"]
        if (principle != "DOC Die Away") and (biodeg >= 0.6):
            label = 1
        elif (principle == "DOC Die Away") and (biodeg >= 0.7):
            label = 1
        else:
            label = 0
        return label

    def add_biowin_info(row):
        smiles = row["smiles"]
        biowin_match = df_biowin[df_biowin["smiles"] == smiles]
        linear_labels = biowin_match["linear_label"].values
        non_linear_labels = biowin_match["non_linear_label"].values
        miti_linear_label = biowin_match["miti_linear_label"].values
        miti_non_linear_label = biowin_match["miti_non_linear_label"].values
        if len(linear_labels) > 0:
            return pd.Series([linear_labels[0], non_linear_labels[0], miti_linear_label[0], miti_non_linear_label[0]])
        return pd.Series(["None", "None", "None", "None"])

    df = df.copy()
    if mode == "class":
        df[["label"]] = df[["y_true"]]
    else:
        df["label"] = df.apply(label_data, axis=1)

    df[["linear_label", "non_linear_label", "miti_linear_label", "miti_non_linear_label"]] = df.apply(
        add_biowin_info, axis=1
    )
    return df


def remove_selected_biowin(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    linear_label = "miti_linear_label"
    non_linear_label = "miti_non_linear_label"

    df_correct = df[
        (df["label"] == df[linear_label])
        | (df["label"] == df[non_linear_label])
        | (df[linear_label].isna())
        | (df[linear_label] == "None")
    ]
    df_correct = df[
        ((df["label"] == df[linear_label]) & (df["label"] == df[non_linear_label]))
        | (df[linear_label].isna())
        | (df[linear_label] == "None")
    ]
    df_false = df[~df.index.isin(df_correct.index)]
    # assert len(df_correct) + len(df_false) == len(df)
    return df_correct, df_false


def process_df_biowin(
    df: pd.DataFrame,
    mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = add_biowin_label(df=df, mode=mode)
    df, df_false = remove_selected_biowin(
        df=df,
    )
    return df, df_false


def replace_smiles_with_smiles_with_chemical_speciation(df_without_env_smiles: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_checked = pd.read_excel("datasets/chemical_speciation.xlsx", index_col=0)

    df_correct_smiles = remove_smiles_with_incorrect_format(df=df_without_env_smiles, col_name_smiles="smiles")
    df_without_env_smiles = openbabel_convert(
        df=df_correct_smiles,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )

    def get_env_smiles(row):
        smiles = row["smiles"]
        cas = row["cas"]
        inchi = row["inchi_from_smiles"]
        match = df_checked[df_checked["inchi_from_smiles"] == inchi]

        if len(match) < 1:
            log.warn("InChI not found, deleted this datapoint!! ", cas=cas, smiles=smiles, inchi=inchi)
            smiles = "delete"
            reason = ""
        else:
            smiles_checked = str(match["smiles"].values[0])
            env_smiles = str(match["env_smiles"].values[0])
            reason = str(match["comment_smiles_at_ph_7_4"].values[0])
            if (smiles_checked != env_smiles) and (env_smiles != "-"):
                smiles = env_smiles
        return pd.Series([smiles, reason])
    df_without_env_smiles = df_without_env_smiles.copy()
    df_without_env_smiles[["smiles", "reason"]] = df_without_env_smiles.apply(get_env_smiles, axis=1)

    df_removed = df_without_env_smiles[
        df_without_env_smiles["smiles"] == "delete"
    ]
    log.info("Deleted data points", data_points=len(df_removed), unique_cas=df_removed.cas.nunique(), unique_inchi=df_removed.inchi_from_smiles.nunique())
    df_removed_no_main_component = df_removed[df_removed["reason"]=="no main component at pH 7.4"]
    df_removed_mixture = df_removed[(df_removed["reason"]=="mixture")]
    log.info("Deleted because no main component at pH 7.4: ", no_main_component_cas=df_removed_no_main_component.cas.nunique(), no_main_component_inchi=df_removed_no_main_component.inchi_from_smiles.nunique())
    log.info("Deleted because mixture: ", mixture_cas=df_removed_mixture.cas.nunique(), mixture_inchi=df_removed_mixture.inchi_from_smiles.nunique())
    df_with_env_smiles = df_without_env_smiles[df_without_env_smiles["smiles"] != "delete"].copy()
    
    df_correct_smiles = remove_smiles_with_incorrect_format(df=df_with_env_smiles, col_name_smiles="smiles")
    df_with_env_smiles = openbabel_convert(
        df=df_correct_smiles,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )

    assert len(df_with_env_smiles[df_with_env_smiles["smiles"] == "delete"]) == 0
    df_with_env_smiles.drop(columns=["reason"], inplace=True)
    return df_with_env_smiles, df_removed


def remove_cas_connected_to_more_than_one_inchi(df: pd.DataFrame, prnt: bool) -> pd.DataFrame:
    df = df.astype({"smiles": str, "cas": str})
    df = openbabel_convert(df=df, input_type="smiles", column_name_input="smiles", output_type="inchi")
    if prnt:
        log.info(
            "Unique identifiers old",
            unique_cas=df["cas"].nunique(),
            unique_inchi=df["inchi_from_smiles"].nunique(),
            unique_smiles=df["smiles"].nunique(),
        )
    
    cas_connected_to_more_than_one_inchi: List[str] = []
    for cas, group in df.groupby("cas"):
        if group["inchi_from_smiles"].nunique() > 1:
            cas_connected_to_more_than_one_inchi.append(cas)
    if len(cas_connected_to_more_than_one_inchi) == 0:
        if prnt:
            log.info("No CAS RN found that are connected to more than one InChI")
    else:
        if prnt:
            studies_connected_to_more_than_one_inchi = df[df["cas"].isin(cas_connected_to_more_than_one_inchi)]
            substances_connected_to_more_than_one_inchi = studies_connected_to_more_than_one_inchi.inchi_from_smiles.nunique()
            log.warn(
                "Removing this many CAS RN because they are connected to more than one InChI",
                cas=len(cas_connected_to_more_than_one_inchi), substances=substances_connected_to_more_than_one_inchi
            )
    df = df[~df["cas"].isin(cas_connected_to_more_than_one_inchi)].copy()
    return df


def replace_multiple_cas_for_one_inchi(df: pd.DataFrame, prnt: bool) -> pd.DataFrame:

    df = remove_cas_connected_to_more_than_one_inchi(df=df, prnt=prnt)

    old_cas_to_new_cas: Dict[str, str] = {}
    for _, group in df.groupby("inchi_from_smiles"):
        if group["cas"].nunique() > 1:
            cass = group["cas"].unique()
            new_cas = min(cass, key=len)  # take shortest CAS RN
            for cas in cass:
                if cas != new_cas:
                    old_cas_to_new_cas[cas] = new_cas
    df["cas"] = df["cas"].replace(old_cas_to_new_cas)
    if prnt:
        log.info(
            "Unique identifiers",
            unique_cas=df["cas"].nunique(),
            unique_inchi=df["inchi_from_smiles"].nunique(),
            unique_smiles=df["smiles"].nunique(),
        )
    return df


def load_regression_df() -> pd.DataFrame:
    df_regression = pd.read_excel("datasets/external_data/Huang_Zhang_RegressionDataset.xlsx", index_col=0)
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
    return df_regression



def load_regression_df_curated_scs_no_metal() -> pd.DataFrame:
    df_regression = pd.read_csv(
        "datasets/data_processing/reg_curated_scs_no_metal.csv", index_col=0
    )
    return df_regression


def format_class_data_paper(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(
        columns={
            "Substance Name": "name",
            "Name type": "name_type",
            "CAS Number": "cas",
            "Source": "source",
            "Smiles": "smiles",
            "Class": "y_true",
        },
        inplace=True,
    )
    df.reset_index(inplace=True, drop=True)
    return df


def load_class_data_paper(
    load_new=False, # Set to True when running for the first time
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if load_new:
        xlsx = pd.ExcelFile("datasets/external_data/Data_huang_zhang.xlsx")
        tab_names = [
            "ClassDataset_original",
            "ClassDataset_external",
            "ClassDataset_all",
        ]
        for name in tab_names:
            class_file = pd.read_excel(xlsx, name)
            class_file = format_class_data_paper(class_file)
            class_file.to_csv(f"datasets/{name}.csv")
        class_original = pd.read_csv(f"datasets/{tab_names[0]}.csv", index_col=0)
        class_original.drop(labels="Index", inplace=True, axis=1)
        class_external = pd.read_csv(f"datasets/{tab_names[1]}.csv", index_col=0)
        class_external.drop(labels="Index", inplace=True, axis=1)
        class_all = pd.read_csv(f"datasets/{tab_names[2]}.csv", index_col=0)
        class_all.drop(labels="Index", inplace=True, axis=1)
        class_original.to_csv("datasets/external_data/class_original.csv")
        class_external.to_csv("datasets/external_data/class_external.csv")
        class_all.to_csv("datasets/external_data/class_all.csv")
    class_original = pd.read_csv("datasets/external_data/class_original.csv", index_col=0)
    class_external = pd.read_csv("datasets/external_data/class_external.csv", index_col=0)
    class_all = pd.read_csv("datasets/external_data/class_all.csv", index_col=0)
    return class_original, class_external, class_all


def reg_df_remove_inherent_only_28_for_classification(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["endpoint"] != "inherent") & (df["time_day"] == 28.0)]
    df.reset_index(inplace=True, drop=True)
    return df


def check_if_cas_match_pubchem(df: pd.DataFrame) -> pd.DataFrame:
    """Check if the CAS given in the dataframe match the added CAS from pubchem or if they are different"""

    def format_existing_cas_string(row):
        cas_string = str(row["cas_external"])
        split = cas_string.split(", ")
        existing_cas_lst = [cas.lstrip("0") for cas in split]
        existing_cas_set = {x for x in existing_cas_lst if is_cas_right_format(x)}
        return list(existing_cas_set)

    df["existing_cas"] = df.apply(format_existing_cas_string, axis=1)

    def find_best_cas(row):
        existing_cas_lst = row["existing_cas"]
        number_of_existing_cas = len(existing_cas_lst)
        if number_of_existing_cas == 0:
            cas_pubchem_str = str(row["cas_pubchem"]).replace(";", ",")
            cas_pubchem_lst = cas_pubchem_str.split(", ")
            number_cas_pubchem = len(cas_pubchem_lst)
            if (cas_pubchem_str == "nan") | (number_cas_pubchem == 0):
                return pd.Series(["No", ""])
            elif number_cas_pubchem == 1:
                return pd.Series(["Yes", cas_pubchem_lst[0]])
            elif number_cas_pubchem > 1:
                cas_to_ref_pubchem = pubchem_cas_to_ref_dict(row["cas_ref_pubchem"])
                best_cas = find_best_cas_pubchem_based_on_ref(cas_to_ref_pubchem)
                return pd.Series(["Yes", best_cas])
        elif number_of_existing_cas == 1:
            existing_cas = existing_cas_lst[0]
            if existing_cas in str(row["cas_pubchem"]):
                return pd.Series(["Yes", existing_cas])
            else:
                return pd.Series(["No", ""])
        elif number_of_existing_cas > 1:
            match = "No"
            best_cas = ""
            for cas in existing_cas_lst:
                if cas in str(row["cas_pubchem"]):
                    match = "Yes"
                    best_cas = cas
        return pd.Series([match, best_cas])

    df[["match", "best_cas"]] = df.apply(find_best_cas, axis=1)
    return df


def drop_rows_without_matching_cas(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all rows that have a No in the match column and return df"""
    df = df[df.match != "No"]
    df = df.drop(["match"], axis=1).reset_index(drop=True)
    return df


def process_external_dataset_lunghini(df: pd.DataFrame, class_df: pd.DataFrame, include_speciation_lunghini: bool) -> pd.DataFrame:
    df = check_if_cas_match_pubchem(df)
    df = drop_rows_without_matching_cas(df)
    class_df = remove_smiles_with_incorrect_format(df=class_df, col_name_smiles="smiles")
    class_df = openbabel_convert(
        df=class_df,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )
    df.rename(columns={"best_cas": "cas"}, inplace=True)
    if include_speciation_lunghini:
        df, _ = replace_smiles_with_smiles_with_chemical_speciation(df_without_env_smiles=df)

    # Find data points for substances not already in class_df
    df_new = df[~df["inchi_from_smiles"].isin(class_df["inchi_from_smiles"])].copy()
    df_new.drop_duplicates(subset=["inchi_from_smiles"], keep="first", inplace=True)
    return df_new


def get_external_dataset_lunghini(
    run_from_start: bool,
    class_df: pd.DataFrame,
    include_speciation_lunghini: bool,
) -> pd.DataFrame:
    if run_from_start:
        df_lunghini = pd.read_csv("datasets/lunghini.csv", sep=";", index_col=0)
        df_lunghini.rename(
            columns={
                "SMILES": "smiles",
                "CASRN": "cas_external",
                "ReadyBiodegradability": "y_true",
                "Dataset": "dataset",
            },
            inplace=True,
        )
        df_output = openbabel_convert(
            df=df_lunghini,
            input_type="smiles",
            column_name_input="smiles",
            output_type="inchi",
        )
        df = group_and_label_chemicals(df=df_output, col_to_group_by="inchi_from_smiles")
        df = get_cid_from_inchi_pubchempy(df)
        df.to_csv("datasets/lunghini_added_cids.csv")
        df = add_cas_from_pubchem(df)
        df.to_csv("datasets/external_data/lunghini_added_cas.csv")
        log.info("Finished getting cas from pubchem!")
        log.warn("Finished creating dataset external. But pKa and alpha values still need to be added!!!")
    
    df_lunghini_with_cas = pd.read_csv("datasets/external_data/lunghini_added_cas.csv", index_col=0)

    df_new = process_external_dataset_lunghini(df=df_lunghini_with_cas, class_df=class_df, include_speciation_lunghini=include_speciation_lunghini)
    df_new = df_new[["cas", "smiles", "y_true"]].copy()

    df_smiles_no_star = remove_smiles_with_incorrect_format(df=df_new, col_name_smiles="smiles")
    df_new = openbabel_convert(
        df=df_smiles_no_star,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )

    return df_new


def is_unique(s):
    a = s.to_numpy()
    return (a[0] == a).all()


def assign_group_label_and_drop_replicates(df: pd.DataFrame, by_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counted_duplicates = (
        df.groupby(df[by_column].tolist(), as_index=False).size().sort_values(by="size", ascending=False)
    )

    multiples_lst: List[str] = []
    removed_lst: List[str] = []
    m_df = counted_duplicates[counted_duplicates["size"] > 1]  
    multiples = m_df["index"].tolist()
    for substance in multiples:
        current_group = df[df[by_column] == substance] 
        if is_unique(current_group["y_true"]):
            multiples_lst.append(substance)
        else: 
            removed_lst.append(substance)

    s_df = counted_duplicates[counted_duplicates["size"] == 1] 
    singles_df = df[df[by_column].isin(s_df["index"])]

    df_removed_due_to_variance = df[df[by_column].isin(removed_lst)]
    df_multiple = df[df[by_column].isin(multiples_lst)].copy()
    df_multiple.drop_duplicates(subset=[by_column], inplace=True, ignore_index=True)
    df_multiple.reset_index(inplace=True, drop=True)

    return df_multiple, singles_df, df_removed_due_to_variance


def reg_df_remove_studies_not_to_consider(df: pd.DataFrame) -> pd.DataFrame:
    assert "ready" in df["endpoint"].unique()
    df = df[(df['time_day']==28.0) & (df["endpoint"]=="ready")] 
    df.reset_index(inplace=True, drop=True)
    return df


def create_classification_data_based_on_regression_data(
    reg_df: pd.DataFrame, 
    with_lunghini: bool, 
    include_speciation_lunghini: bool, 
    include_speciation: bool,
    run_from_start: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:  # reg_df MUST be without speciation for it to work propperly
    df_included = reg_df_remove_studies_not_to_consider(reg_df)

    log.info("Data points to consider for classification", data_points=len(df_included), unique_cas=df_included.cas.nunique())
    inchi_counts = df_included.groupby('inchi_from_smiles').size()
    num_inchi_multiple = inchi_counts[inchi_counts > 1].count()
    log.info("Substances with more than one study result to consider", substances=num_inchi_multiple)
    
    grouped = df_included.groupby('inchi_from_smiles')
    filtered_groups = grouped.filter(lambda x: len(x) > 1)
    std_devs = filtered_groups.groupby('inchi_from_smiles')['biodegradation_percent'].std()
    average_std_dev = round(std_devs.mean()*100, 1)
    log.info("Average standard deviation in biodegradation for substances with more than one study result to consider", average_std_dev=average_std_dev)
    
    group_stats = filtered_groups.groupby('inchi_from_smiles')['biodegradation_percent'].agg(['std'])
    over_30_percent_std = group_stats[group_stats['std'] > 0.3]
    num_entries_over_30_percent_std = len(over_30_percent_std)
    log.info("Substances with std over 30%", num_entries_over_30_percent_std=num_entries_over_30_percent_std, percent_of_num_inchi_multiple=round(num_entries_over_30_percent_std/num_inchi_multiple*100, 1))

    df_labelled = label_data_based_on_percentage(df_included)
    log.info("Substances remaining in dataset after removing studies not to consider and labelling the data by percentage", substances=df_labelled.inchi_from_smiles.nunique())

    columns = ["cas", "smiles", "principle", "biodegradation_percent", "y_true", "inchi_from_smiles"]
    if include_speciation:
        columns = columns + get_speciation_col_names()
    df_class = pd.DataFrame(data=df_labelled.copy(), columns=columns)

    df_multiples, df_singles, df_removed_due_to_variance = assign_group_label_and_drop_replicates(
        df=df_class, by_column="inchi_from_smiles"
    )
    log.info("Substances removed due to variance", num_substances=df_removed_due_to_variance.inchi_from_smiles.nunique(), percentage_of_num_inchi_multiple=round(df_removed_due_to_variance.inchi_from_smiles.nunique()/num_inchi_multiple*100, 1))
    log.info("Substances remaining in dataset after assigning group label and removing replicates", substances=len(df_singles)+len(df_multiples))
    assert len(df_multiples) + len(df_singles) + df_removed_due_to_variance["inchi_from_smiles"].nunique() == df_class["inchi_from_smiles"].nunique()

    if with_lunghini:
        log.info("Adding data from Lunghini et al.")
        df_lunghini_additional = get_external_dataset_lunghini(
            run_from_start=run_from_start,
            class_df=df_class,
            include_speciation_lunghini=include_speciation_lunghini,
        )
        log.info("Adding this many data points from Lunghini et al.", entries=len(df_lunghini_additional), unique_cas=df_lunghini_additional.cas.nunique(), unique_inchi=df_lunghini_additional.inchi_from_smiles.nunique())

        df_singles = pd.concat([df_singles, df_lunghini_additional], axis=0)
        df_singles.reset_index(inplace=True, drop=True)

    df_class = pd.concat([df_multiples, df_singles], axis=0)
    df_class.drop(["principle", "biodegradation_percent"], axis=1, inplace=True)
    df_class.reset_index(inplace=True, drop=True)

    return df_class, df_removed_due_to_variance



def create_classification_biowin(
    reg_df: pd.DataFrame,
    with_lunghini: bool,
    include_speciation_lunghini: bool,
    run_from_start: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df_class, _ = create_classification_data_based_on_regression_data(
        reg_df,
        with_lunghini=with_lunghini,
        include_speciation_lunghini=include_speciation_lunghini,
        include_speciation=False,
        run_from_start=run_from_start,
    )
    df_class_biowin, df_class_biowin_problematic = process_df_biowin(
        df=df_class,
        mode="class",
    )
    return df_class_biowin, df_class_biowin_problematic


def create_input_regression(
    df: pd.DataFrame, include_speciation: bool
) -> np.ndarray:  # function name used to be obtain_final_input
    """Function to put all features into the required shape for model training."""

    df = convert_regression_df_to_input(df)

    include_others = False
    if "guideline" in list(df.columns):
        include_others = True

    def create_x_reg(row) -> np.ndarray:
        other = []
        if include_others:
            other = row[["time_day", "endpoint", "guideline", "principle", "reliability"]].values.tolist()
        if include_speciation:
            speciation = row[get_speciation_col_names()].values.tolist()
            other = other + speciation
        record_fp = np.array(row["fingerprint"]).tolist()
        return record_fp + other

    x_reg = (df.apply(create_x_reg, axis=1)).to_list()
    x_array = np.array(x_reg, dtype=float)  # was object
    return x_array


def encode_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical data using principles of the OrdinalEncoder()"""
    cat_dict_guideline = {
        "EU Method C.4-A": 0,
        "EU Method C.4-C": 1,
        "EU Method C.4-D": 2,
        "EU Method C.4-E": 3,
        "OECD Guideline 301 A": 4,
        "OECD Guideline 301 B": 5,
        "OECD Guideline 301 C": 6,
        "OECD Guideline 301 D": 7,
        "OECD Guideline 301 E": 8,
        "OECD Guideline 301 F": 9,
        "OECD Guideline 302 B": 10,
        "OECD Guideline 302 C": 11,
        "OECD Guideline 310": 12,
    }
    cat_dict_principle = {
        "DOC Die Away": 0,
        "CO2 Evolution": 1,
        "Closed Respirometer": 2,
        "Closed Bottle Test": 3,
    }
    cat_dict_endpoint = {"Ready": 0, "ready": 0, "Inherent": 1, "inherent": 1}
    df = df.replace(
        {
            "guideline": cat_dict_guideline,
            "principle": cat_dict_principle,
            "endpoint": cat_dict_endpoint,
        }
    )
    return df

def bit_vec_to_lst_of_lst(df: pd.DataFrame, include_speciation: bool):
    def create_x_class(row) -> np.ndarray:
        speciation = []
        if include_speciation:
            speciation = row[get_speciation_col_names()].values.tolist()
        record_fp = np.array(row["fingerprint"]).tolist()
        return record_fp + speciation

    x_class = df.apply(create_x_class, axis=1)
    x_class = x_class.to_list()
    return x_class


def convert_to_morgan_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    # 1024 bit fingerprint
    df = df.copy()
    mols = [AllChem.MolFromSmiles(smiles) for smiles in df["smiles"]]
    df["fingerprint"] = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols] # Possible nBits are 1024, 2048, 4096
    return df

def convert_to_maccs_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    # 166 bit fingerprint
    df = df.copy()
    df.reset_index(drop=True, inplace=True) # TODO
    mols = [AllChem.MolFromSmiles(smiles) for smiles in df["smiles"]]
    for index, value in enumerate(mols): 
        if value is None:
            log.warn("This SMILES could not be converted to Mol file, deleting this datapoint", problematic_smiles=df.loc[index, "smiles"])
            df.drop(index, inplace=True)
            df.reset_index(drop=True, inplace=True)
            del mols[index]
    df["fingerprint"] = [GetMACCSKeysFingerprint(mol) for mol in mols]
    return df

def convert_to_rdk_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    # calculate 2048 bit RDK fingerprint
    df = df.copy()
    mols = [AllChem.MolFromSmiles(smiles) for smiles in df["smiles"]]
    df["fingerprint"] = [Chem.RDKFingerprint(mol) for mol in mols]
    return df


def create_input_classification(df_class: pd.DataFrame, include_speciation: bool, target_col: str) -> Tuple[np.ndarray, pd.Series]:
    """Function to create fingerprints and put fps into one array that can than be used as one feature for model training."""
    df = convert_to_maccs_fingerprints(df_class)
    x_class = bit_vec_to_lst_of_lst(df, include_speciation)
    x_array = np.array(x_class, dtype=object)
    y = df[target_col]
    return x_array, y


def get_speciation_col_names() -> List[str]:
    return [
        "pka_acid_1",
        "pka_acid_2",
        "pka_acid_3",
        "pka_acid_4",
        "pka_base_1",
        "pka_base_2",
        "pka_base_3",
        "pka_base_4",
        "α_acid_0",
        "α_acid_1",
        "α_acid_2",
        "α_acid_3",
        "α_acid_4",
        "α_base_0",
        "α_base_1",
        "α_base_2",
        "α_base_3",
        "α_base_4",
    ]


def convert_regression_df_to_input(df: pd.DataFrame) -> pd.DataFrame:
    df = convert_to_maccs_fingerprints(df)
    df = encode_categorical_data(df)
    return df


def load_gluege_data() -> pd.DataFrame:
    df = pd.read_pickle("reach_study_results/RegisteredSubstances_organic6.pkl")
    new_col_names_to_col_names = {
        "CAS_RN": "cas",
        "SMILES corresponding to CAS RN": "smiles",
        "SMILES at pH 7.4": "smiles_ph",
    }
    df.rename(columns=new_col_names_to_col_names, inplace=True)
    df = df[["cas", "smiles", "smiles_ph"]].astype({"smiles": str, "cas": str, "smiles_ph": str})
    df = df[df["smiles"] != "nan"]
    df = df[~df.smiles.str.contains("*", regex=False)]
    df = remove_smiles_with_incorrect_format(df=df, col_name_smiles="smiles")
    df = openbabel_convert(
        df=df,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )
    df.reset_index(inplace=True, drop=True)
    return df


def check_number_of_components(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_one_component = df[~df["smiles_paper"].str.contains(".", regex=False)].copy()
    df_multiple_components = df[df["smiles_paper"].str.contains(".", regex=False)].copy()
    return df_one_component, df_multiple_components


def openbabel_convert_smiles_to_inchi_with_nans(col_names_smiles_to_inchi: List[str], df: pd.DataFrame) -> pd.DataFrame:
    for col in col_names_smiles_to_inchi:
        df[col] = df[col].astype(str)
        df[col] = df[col].apply(lambda x: "nan" if "*" in x else x)  # openabbel cannot convert smiles containing *
        df[col] = df[col].apply(lambda x: "nan" if "|" in x else x)  # openabbel cannot convert smiles containing |
        df[col] = df[col].replace(
            to_replace=["nan", None, "", np.nan], value="c"
        )  # use "c" as placeholder because openbabel cannot handle nans
        df = openbabel_convert(
            df=df,
            input_type="smiles",
            column_name_input=col,
            output_type="inchi",
        )
        df[f"inchi_from_{col}"] = df[f"inchi_from_{col}"].replace(to_replace="InChI=1S/CH3/h1H3", value="")  # remove placeholder
        df[col] = df[col].replace(to_replace="c", value="")  # remove placeholder
    return df


def get_inchi_main_layer(df: pd.DataFrame, inchi_col: str, layers=4) -> pd.DataFrame:
    df = df.copy()

    def get_inchi_main_layer_inner_function(row):
        inchi_main_layer_smiles = get_inchi_layers(row, col_name=inchi_col, layers=layers)
        return inchi_main_layer_smiles

    df[f"{inchi_col}_main_layer"] = df.apply(get_inchi_main_layer_inner_function, axis=1)
    return df


def get_molecular_formula_from_inchi(df: pd.DataFrame, inchi_col: str) -> pd.DataFrame:
    df = df.copy()

    def get_molecular_formula_from_inchi_inner_function(row):
        inchi_main_layer_smiles = get_inchi_layers(row, col_name=inchi_col, layers=2)
        return inchi_main_layer_smiles

    df[f"{inchi_col}_molecular_formula"] = df.apply(get_molecular_formula_from_inchi_inner_function, axis=1)
    return df


def get_smiles_inchi_cirpy(cas: str) -> Tuple[str, str]:
    smile = ""
    inchi = ""
    ssl._create_default_https_context = ssl._create_unverified_context
    time.sleep(0.2)
    try:
        smile = cirpy.resolve(cas, "smiles")
        inchi = cirpy.resolve(cas, "stdinchi")
    except:
        log.fatal("Could not get info from CIRpy")
    return smile, inchi


def get_guidelines_and_principles_to_keep() -> Tuple[List[str], List[str]]:
    guidelines_to_keep = [
        "OECD Guideline 301 A",
        "OECD Guideline 301 B",
        "OECD Guideline 301 C",
        "OECD Guideline 301 D",
        "OECD Guideline 301 E",
        "OECD Guideline 301 F",
        "OECD Guideline 302 B",
        "OECD Guideline 302 C",
        "OECD Guideline 310",
        "EU Method C.4-A",
        "EU Method C.4-C",
        "EU Method C.4-D",
        "EU Method C.4-E",
    ]
    principles_to_keep = [
        "CO2 Evolution",
        "Closed Bottle Test",
        "Closed Respirometer",
        "DOC Die Away",
    ]
    return guidelines_to_keep, principles_to_keep


def further_processing_of_echa_data(df: pd.DataFrame) -> pd.DataFrame:
    def remove_unwanted(row):
        guideline = row["guideline"]
        principle = row["principle"]
        guidelines_to_keep, principles_to_keep = get_guidelines_and_principles_to_keep()
        if guideline not in guidelines_to_keep:
            guideline = ""
        if principle not in principles_to_keep:
            principle = ""
        return pd.Series([guideline, principle])

    df[["guideline", "principle"]] = df.apply(remove_unwanted, axis=1)
    df = df[df["guideline"] != ""]
    df = df[df["principle"] != ""]
    df = df[df["oxygen_conditions"] == "aerobic"]
    df = df[(df["reliability"] == 1) | (df["reliability"] == 2)]
    return df


def load_and_process_echa_additional(include_speciation: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_reg = load_regression_df()
    df_reg = remove_smiles_with_incorrect_format(df=df_reg, col_name_smiles="smiles")
    df_reg = openbabel_convert(
        df=df_reg,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )
    _, _, df_class = load_class_data_paper()
    df_class_correct_format = remove_smiles_with_incorrect_format(df=df_class, col_name_smiles="smiles")
    df_class = openbabel_convert(
        df=df_class_correct_format,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )
    df_echa = pd.read_csv("datasets/iuclid_echa.csv", index_col=0)
    df_echa = further_processing_of_echa_data(df_echa)
    cols_to_keep = [
        "cas",
        "ref_cas",
        "smiles",
        "reliability",
        "biodegradation_percent",
        "biodegradation_samplingtime",
        "endpoint",
        "guideline",
        "principle",
    ]
    if include_speciation:
        cols_to_keep += get_speciation_col_names()
    df_echa = df_echa[cols_to_keep]
    df_echa.rename(columns={"biodegradation_samplingtime": "time_day"}, inplace=True)
    df_echa = df_echa[df_echa["cas"] == df_echa["ref_cas"]]
    df_echa, _ = remove_organo_metals_function(df=df_echa, smiles_column="smiles")
    df_echa.drop(columns=["organo_metal", "ref_cas"], inplace=True)
    df_echa = openbabel_convert(
        df=df_echa,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )
    echa_additional_reg = df_echa[~df_echa["inchi_from_smiles"].isin(df_reg["inchi_from_smiles"])]
    echa_additional_class = df_echa[~df_echa["inchi_from_smiles"].isin(df_class["inchi_from_smiles"])]
    echa_additional_class, _ = create_classification_data_based_on_regression_data(
        reg_df=echa_additional_class,
        with_lunghini=False,
        include_speciation_lunghini=False,
        include_speciation=include_speciation,
        run_from_start=False,
    )

    echa_additional_reg_chemical_speciation, _ = replace_smiles_with_smiles_with_chemical_speciation(echa_additional_reg.copy())
    echa_additional_reg = echa_additional_reg[echa_additional_reg.index.isin(echa_additional_reg_chemical_speciation.index)]
    echa_additional_class_chemical_speciation, _ = replace_smiles_with_smiles_with_chemical_speciation(echa_additional_class.copy())
    echa_additional_class = echa_additional_class[echa_additional_class.index.isin(echa_additional_class_chemical_speciation.index)]

    echa_additional_reg.reset_index(inplace=True, drop=True)
    echa_additional_reg_chemical_speciation.reset_index(inplace=True, drop=True)
    echa_additional_class.reset_index(inplace=True, drop=True)
    echa_additional_class_chemical_speciation.reset_index(inplace=True, drop=True)

    return echa_additional_reg, echa_additional_reg_chemical_speciation, echa_additional_class, echa_additional_class_chemical_speciation


def get_df_with_unique_cas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["cas"], keep="first", ignore_index=True)
    if "smiles" not in df.columns:
        return df
    df = df[["name", "cas", "smiles"]]
    df.rename(columns={"smiles": "smiles_paper"}, inplace=True)
    return df


def get_class_datasets() -> Dict[str, pd.DataFrame]:
    _, _, df_class = load_class_data_paper()
    curated_scs = pd.read_csv("datasets/curated_data/class_curated_scs.csv", index_col=0)
    curated_biowin = pd.read_csv("datasets/curated_data/class_curated_biowin.csv", index_col=0)
    curated_final = pd.read_csv(
        "datasets/curated_data/class_curated_final.csv", index_col=0
    )
    return {
        "df_paper": df_class,
        "df_curated_scs": curated_scs,
        "df_curated_biowin": curated_biowin,
        "df_curated_final": curated_final,
    }


def create_dfs_for_curated_data_analysis() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    class_curated_scs = pd.read_csv("datasets/curated_data/class_curated_scs.csv", index_col=0)
    class_curated_biowin = pd.read_csv("datasets/curated_data/class_curated_biowin.csv", index_col=0)
    class_curated_biowin_problematic = pd.read_csv("datasets/curated_data/class_curated_biowin_problematic.csv", index_col=0)
    class_curated_final = pd.read_csv("datasets/curated_data/class_curated_final.csv", index_col=0)
    class_curated_final_removed = pd.read_csv("datasets/curated_data/class_curated_final_removed.csv", index_col=0)

    return (
        class_curated_scs,
        class_curated_biowin,
        class_curated_biowin_problematic,
        class_curated_final,
        class_curated_final_removed,
    )


def get_labels_colors_progress() -> Tuple[List[str], List[str]]:
    labels = [
        "Huang-Dataset \n reported",
        "Huang-Dataset \n replicated",
        r"$\mathregular{Curated_{SCS}}$",
        r"$\mathregular{Curated_{BIOWIN}}$",
        r"$\mathregular{Curated_{FINAL}}$",
    ]
    colors = [
        "white",
        "plum",
        "royalblue",
        "lightgreen",
        "seagreen",
    ]
    return labels, colors


def plot_results_with_standard_deviation(
    all_data: List[np.ndarray],
    labels: List[str],
    colors: List[str],
    title: str,
    seed: int,
    plot_with_paper: bool,
    save_ending: str,
    test_set_name: str,
) -> None:

    plt.figure(figsize=(15, 5))

    bplot = plt.boxplot(all_data, vert=True, patch_artist=True, labels=labels, meanline=True, showmeans=True)

    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)

    if plot_with_paper:
        plt.plot(1, np.mean(all_data[0]), marker="o", markersize=14)

    plt.xlabel("Datasets", fontsize=22)
    ylabel = f"Balanced accuracy (%)" if title == "Balanced_accuracy" else title
    plt.ylabel(ylabel, fontsize=22)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.grid(axis="y")

    plt.savefig(
        f"figures/{title}_seed{seed}_paper_hyperparameter_{save_ending}_test_set_{test_set_name}.png"
    )
    plt.close()



def get_datasets_for_ad() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_curated_scs = pd.read_csv("datasets/curated_data/class_curated_scs.csv", index_col=0)
    df_curated_biowin = pd.read_csv("datasets/curated_data/class_curated_biowin.csv", index_col=0)
    df_curated_final = pd.read_csv("datasets/curated_data/class_curated_final.csv", index_col=0)
    return df_curated_scs, df_curated_biowin, df_curated_final


def create_fingerprint_df(df: pd.DataFrame) -> pd.DataFrame:
    mols = [AllChem.MolFromSmiles(smiles) for smiles in df["smiles"]]
    fps = [np.array(GetMACCSKeysFingerprint(mol)) for mol in mols]
    df_fp = pd.DataFrame(data=fps)
    return df_fp


# Adapted from pyADA https://github.com/jeffrichardchemistry/pyADA/blob/main/pyADA/pyADA.py
class Similarity:
    """
        All similarity calculations have a range of [0, 1]
    """
    def __init__(self):        
        pass    
    
    def __coefs(self, vector1, vector2):
        A = np.array(vector1).astype(int)
        B = np.array(vector2).astype(int)

        AnB = A & B #intersection
        onlyA = np.array(B) < np.array(A) #A is a subset of B
        onlyB = np.array(A) < np.array(B) #B is a subset of A
        return AnB,onlyA,onlyB
    

    def tanimoto_similarity(self, vector1, vector2):
        """
        Structural similarity calculation based on tanimoto index. T(A,B) = (A ^ B)/(A + B - A^B)
        """
        AnB, onlyA, onlyB = Similarity.__coefs(self, vector1=vector1, vector2=vector2)
        return AnB.sum() / (onlyA.sum() + onlyB.sum() + AnB.sum())


class ApplicabilityDomain:
    def __init__(self, verbose):
        self.__sims = Similarity()    
        self.__verbose = verbose
        self.similarities_table_ = None
                
    def analyze_similarity(self, base_test, base_train, similarity_metric='tanimoto') -> pd.DataFrame:

        similarities = {}

        # get dictionary of all data tests similarities
        def get_dict(base_train, i_test, similarities, n):
            get_tests_similarities = [0]*len(base_train)
            for i, i_train in enumerate(base_train):
                if similarity_metric == 'tanimoto':
                    get_tests_similarities[i] = (self.__sims.tanimoto_similarity(i_test, i_train))               
                else:
                    log.error("This similarity_metric does not exist")
            similarities['Sample_test_{}'.format(n)] = np.array(get_tests_similarities)
            return similarities
        
        if self.__verbose:
            with tqdm(total=len(base_test)) as progbar:
                for n,i_test in enumerate(base_test):
                    similarities = get_dict(base_train, i_test, similarities, n)
                    progbar.update(1)
        else:
            for n,i_test in enumerate(base_test):            
                similarities = get_dict(base_train, i_test, similarities, n)
                    
        self.similarities_table_ = pd.DataFrame(similarities)
        
        analyze = pd.concat([self.similarities_table_.mean(),
                             self.similarities_table_.median(),
                             self.similarities_table_.std(),
                             self.similarities_table_.max(),
                             self.similarities_table_.min()],
                             axis=1)        
        analyze.columns = ['Mean', 'Median', 'Std', 'Max', 'Min']
        
        return analyze
            
    
    def fit_ad(
            self, 
            model, 
            base_test, 
            base_train, 
            y_true, 
            threshold_reference, 
            threshold_step, 
            similarity_metric, 
            metric_evaliation
    ) -> Dict[str, float]:
        #reference parameters
        if threshold_reference.lower() == 'max':
            thref = 'Max'
        elif threshold_reference.lower() == 'average':
            thref = 'Mean'
        elif threshold_reference.lower() == 'std':
            thref = 'Std'
        elif threshold_reference.lower() == 'median':
            thref = 'Median'
        else:
            thref = 'Max'
        
        #Get analysis table
        table_analysis = ApplicabilityDomain.analyze_similarity(self, base_test=base_test, base_train=base_train,
                                                            similarity_metric=similarity_metric)
        table_analysis.index = np.arange(0, len(table_analysis), 1)
        
        results = {}
        total_thresholds = np.arange(threshold_step[0], threshold_step[1], threshold_step[2])
        
        def get_table(thresholds, samples_between_thresholds, base_test, model, y_true, metric_evaliation, results):
            new_xitest = base_test[samples_between_thresholds.index, :] 
            new_ypred = model.predict(new_xitest)
            new_ytrue = y_true[samples_between_thresholds.index]
            assert len(new_xitest) == len(new_ypred) == len(new_ytrue)
            
            if metric_evaliation == 'acc':
                performance_metric = accuracy_score(y_true=new_ytrue, y_pred=new_ypred)
            else:
                log.error("This metric_evaliation is not defined")
                
            results['Threshold {}'.format(thresholds.round(5))] = performance_metric 
            return results


        length = 0 
        for index, thresholds in enumerate(tqdm(total_thresholds)):
            if thresholds<0.4:
                continue
            elif thresholds==0.4:
                samples_between_thresholds = table_analysis.loc[(table_analysis[thref] < thresholds)]
            elif thresholds==1.0:
                samples_between_thresholds = table_analysis.loc[(table_analysis[thref] <= thresholds) & (table_analysis[thref] >= (total_thresholds[index-1]))]
            else:
                samples_between_thresholds = table_analysis.loc[(table_analysis[thref] < thresholds) & (table_analysis[thref] >= (total_thresholds[index-1]))] 
            length += len(samples_between_thresholds)
            if len(samples_between_thresholds) == 0:
                results[thresholds] = None
            else:
                results = get_table(thresholds, samples_between_thresholds, base_test, model, y_true, metric_evaliation, results)
        assert len(table_analysis) == length 
        return results


def check_substances_in_ad(df_train: pd.DataFrame, df_train_name: str, df_test: pd.DataFrame) -> pd.DataFrame:
    x_train = create_fingerprint_df(df=df_train)
    x_train = x_train.values
    x_test = create_fingerprint_df(df=df_test)
    x_test = x_test.values

    AD = ApplicabilityDomain(verbose=True)
    df_similarities = AD.analyze_similarity(base_test=x_test, base_train=x_train, similarity_metric="tanimoto")
    assert len(df_test) == len(df_similarities)
    df_test.reset_index(inplace=True, drop=True)
    df_similarities.reset_index(inplace=True, drop=True)
    df_test[f"in_ad_of_{df_train_name}"] = 1
    df_test.loc[df_similarities['Max'] < 0.5, f"in_ad_of_{df_train_name}"] = 0

    return df_test








