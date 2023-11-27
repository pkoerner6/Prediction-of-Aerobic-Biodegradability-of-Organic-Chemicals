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
from rdkit.Chem import PandasTools
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from pyADA import ApplicabilityDomain
from pubchempy import get_compounds
from padelpy import from_smiles
import urllib.request
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains


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


def remove_smiles_with_incorrect_format(df: pd.DataFrame, col_name_smiles: str) -> pd.DataFrame:
    """SMILES can be in a format that cannot be converted by openbabel"""
    df_clean = df.copy()
    df_clean[col_name_smiles] = df_clean[col_name_smiles].apply(lambda x: "nan" if "*" in x or "|" in x else x)
    df_clean = df_clean[df_clean[col_name_smiles] != "nan"]
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


def assign_group_label_and_drop_replicates_based_on_mean_biodegradation(
    df: pd.DataFrame, by_column: str
) -> pd.DataFrame:
    df["threshold"] = np.where(df["principle"] == "DOC Die Away", 0.7, 0.6)
    counted_duplicates = (
        df.groupby(df[by_column].tolist(), as_index=False).size().sort_values(by="size", ascending=False)
    )

    replicates_df = counted_duplicates[counted_duplicates["size"] > 1]
    replicates = replicates_df["index"].tolist()
    indices_to_drop: List[int] = []
    for item in replicates:
        current_group = df.loc[df[by_column] == item]
        current_group_agg = current_group[["biodegradation_percent", "threshold"]].aggregate(["mean"])
        mean_biodegradation = current_group_agg.at["mean", "biodegradation_percent"]
        mean_threshold = current_group_agg.at["mean", "threshold"]
        label = 1 if mean_biodegradation >= mean_threshold else 0
        for i, indx in enumerate(list(current_group.index)):
            if i == 0:
                df.loc[indx, "threshold"] = mean_threshold
                df.loc[indx, "biodegradation_percent"] = mean_biodegradation
                df.loc[indx, "y_true"] = label
            else:
                indices_to_drop.append(indx)
                df = df.drop(index=indx)  # Remove other rows in main df
    df.reset_index(inplace=True, drop=True)
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
                log.warn("No main layer for inchi", inchi=inchi)
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
    log.info("Number of organo-metals found", organo_metals=len(organo_metals))
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
    ms_smiles_box = driver.find_elements(By.XPATH, "//label[contains(@for,'MS_READY_SMILES-output')]")[0]
    qsar_smiles_box = driver.find_elements(By.XPATH, "//label[contains(@for,'QSAR_READY_SMILES-output')]")[0]
    mf_box = driver.find_elements(By.XPATH, "//label[contains(@for,'MOLECULAR_FORMULA-output')]")[0]
    am_box = driver.find_elements(By.XPATH, "//label[contains(@for,'AVERAGE_MASS-output')]")[0]
    boxes = [
        cid_box,
        cas_box,
        smiles_box,
        inchi_box,
        ms_smiles_box,
        qsar_smiles_box,
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
            "MS_READY_SMILES",
            "QSAR_READY_SMILES",
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


def replace_smiles_with_env_relevant_smiles(df_without_env_smiles: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_checked = pd.read_excel("datasets/substances_with_env_smiles.xlsx", index_col=0)

    df_without_env_smiles = openbabel_convert(
        df=df_without_env_smiles,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )

    def get_env_smiles(row):
        smiles = row["smiles"]
        cas = row["cas"]
        inchi = row["inchi_from_smiles"]
        match = df_checked[df_checked["cas"] == cas]

        if len(match) < 1:
            log.warn("CAS not found, deleted this datapoint!! ", cas=cas, smiles=smiles, inchi=inchi)
            smiles = "delete"
        else:
            smiles_checked = str(match["smiles"].values[0])
            env_smiles = str(match["env_smiles"].values[0])
            if (smiles_checked != env_smiles) and (env_smiles != "-"):
                smiles = env_smiles
        return smiles
    df_without_env_smiles = df_without_env_smiles.copy()
    df_without_env_smiles["smiles"] = df_without_env_smiles.apply(get_env_smiles, axis=1)

    df_removed_because_no_main_component_at_ph_or_other_issue = df_without_env_smiles[
        df_without_env_smiles["smiles"] == "delete"
    ]
    df_with_env_smiles = df_without_env_smiles[df_without_env_smiles["smiles"] != "delete"].copy()
    df_correct_smiles = remove_smiles_with_incorrect_format(df=df_with_env_smiles, col_name_smiles="smiles")
    df_with_env_smiles = openbabel_convert(
        df=df_correct_smiles,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )
    assert len(df_with_env_smiles[df_with_env_smiles["smiles"] == "delete"]) == 0
    return df_with_env_smiles, df_removed_because_no_main_component_at_ph_or_other_issue


def remove_cas_connected_to_more_than_one_inchi(df: pd.DataFrame, prnt: bool) -> pd.DataFrame:
    df = df.astype({"smiles": str, "cas": str})
    df_correct = remove_smiles_with_incorrect_format(df=df, col_name_smiles="smiles") # TODO should not be necessary
    if prnt:
        log.info("Entries with correct format: ", entries=len(df), correct=len(df_correct))
    df = openbabel_convert(df=df_correct, input_type="smiles", column_name_input="smiles", output_type="inchi")
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
            df = df[df["cas"] != cas].copy()
    if len(cas_connected_to_more_than_one_inchi) == 0:
        log.info("No CAS RN found that are connected to more than one InChI")
    else:
        if prnt:
            log.warn(
                "Removing this many CAS RN because they are connected to more than one InChI",
                cas=len(cas_connected_to_more_than_one_inchi),
            )
            log.warn(
                f"Removing the following CAS RN because they are connected to more than one InChI",
                cas=cas_connected_to_more_than_one_inchi,
            )
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


def load_regression_df_curated_s_no_metal() -> pd.DataFrame:
    df_regression = pd.read_csv("datasets/data_processing/reg_curated_s_no_metal.csv", index_col=0)
    df_regression = df_regression[
        df_regression["cas"] != "1803551-73-6"
    ]  # Remove becuase cannot be converted to fingerprint (Explicit valence for atom # 0 F, 2, is greater than permitted)
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
    load_new=False,
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


def dixons_q_outlier_detection_removal_huang_zhang(df: pd.DataFrame) -> pd.DataFrame:
    """Find and remove outliers using dixons q test according to this video: https://www.youtube.com/watch?v=K3zxIJm0v70"""
    """Note that outliers are only removed when only one outlier is present!"""
    num_samples_to_critical_val: Dict[int, float] = {
        3: 0.941,
        4: 0.765,
        5: 0.642,
        6: 0.560,
        7: 0.507,
        8: 0.468,
        9: 0.437,
        10: 0.412,
        11: 0.392,
        12: 0.376,
        13: 0.361,
        14: 0.349,
        15: 0.338,
        16: 0.329,
        17: 0.320,
        18: 0.313,
        19: 0.306,
        20: 0.300,
    }
    indices_of_outliers: List[int] = []
    counted_duplicates = df.groupby(df["cas"].tolist(), as_index=False).size().sort_values(by="size", ascending=False)
    df_more_than_3_replicates = counted_duplicates[counted_duplicates["size"] > 3]
    df_for_dixons = df_more_than_3_replicates[df_more_than_3_replicates["size"] < 27]
    cas_numbers = list(df_for_dixons["index"])
    for cas_number in cas_numbers:
        current_group = df.loc[df["cas"] == cas_number]
        current_group_sorted = current_group.sort_values(by="biodegradation_percent")
        current_deg_values = list(
            current_group_sorted["biodegradation_percent"]
        )  # get list of degradation values for given cas
        range = abs(current_deg_values[-1] - current_deg_values[0])
        if range == 0:
            continue
        n = len(current_deg_values)
        critical_val = num_samples_to_critical_val.get(n, -1.0)
        if critical_val == -1.0:
            log.info("No outliers detected!")
            continue
        # Check if lowest value is outlier
        gap_low = current_deg_values[1] - current_deg_values[0]
        test_stat = gap_low / range
        if test_stat > critical_val:
            indices_of_outliers.append(
                current_group.index[current_group["biodegradation_percent"] == current_deg_values[0]].tolist()[0]
            )
        # Check if highest value is outlier
        gap_high = current_deg_values[-1] - current_deg_values[-2]
        test_stat = gap_high / range
        if test_stat > critical_val:
            indices_of_outliers.append(
                current_group.index[current_group["biodegradation_percent"] == current_deg_values[-1]].tolist()[0]
            )
    df = df.drop(index=indices_of_outliers).reset_index(drop=True)  # Remove rows with outlier in main df
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


def get_df_additional_external(df: pd.DataFrame, class_df: pd.DataFrame) -> pd.DataFrame:
    df_new = df[~df["inchi_from_smiles"].isin(class_df["inchi_from_smiles"])].copy()
    df_new.rename(columns={"best_cas": "cas"}, inplace=True)
    return df_new


def process_external_dataset_lunghini(df: pd.DataFrame, class_df: pd.DataFrame) -> pd.DataFrame:
    df = check_if_cas_match_pubchem(df)
    df = drop_rows_without_matching_cas(df)
    class_df = remove_smiles_with_incorrect_format(df=class_df, col_name_smiles="smiles")
    class_df = openbabel_convert(
        df=class_df,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )
    df_new = get_df_additional_external(df=df, class_df=class_df).copy()
    # df_new.drop_duplicates(subset=["inchi_from_smiles"], keep="first", inplace=True) # TODO should not be necessary
    return df_new


def get_external_dataset_lunghini( # TODO
    run_from_start: bool,
    class_df: pd.DataFrame,
    include_speciation: bool,
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
    df_lunghini_with_cas = pd.read_csv("datasets/external_data/lunghini_added_cas.csv", index_col=0)  # includes pka and alpha values
    df_new = process_external_dataset_lunghini(df=df_lunghini_with_cas, class_df=class_df)
    cols = ["cas", "smiles", "y_true"]
    if include_speciation:
        cols += get_speciation_col_names()
    df_new = df_new[cols].copy()

    df_smiles_no_star = remove_smiles_with_incorrect_format(df=df_new, col_name_smiles="smiles")
    df_new = openbabel_convert(
        df=df_smiles_no_star,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )

    return df_new


def create_classification_data_based_on_regression_data(
    reg_df: pd.DataFrame, 
    with_lunghini: bool, 
    include_speciation: bool, 
    env_smiles_lunghini: bool, 
    prnt: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # reg_df MUST be without speciation for it to work propperly
    df_included = reg_df_remove_studies_not_to_consider(reg_df)
    df_labelled = label_data_based_on_percentage(df_included)
    # df_labelled = dixons_q_outlier_detection_removal_huang_zhang(df_labelled)

    columns = ["cas", "smiles", "principle", "biodegradation_percent", "y_true", "inchi_from_smiles"]
    if include_speciation:
        columns += get_speciation_col_names()
    df_class = pd.DataFrame(data=df_labelled.copy(), columns=columns)

    if with_lunghini:
        df_lunghini_additional = get_external_dataset_lunghini(
            run_from_start=False, # TODO
            class_df=df_class,
            include_speciation=include_speciation,
        )
        if env_smiles_lunghini:
            df_lunghini_additional, _ = replace_smiles_with_env_relevant_smiles(df_without_env_smiles=df_lunghini_additional)
        df_lunghini_additional = df_lunghini_additional[df_lunghini_additional["smiles"].notna()]
        df_class = pd.concat([df_class, df_lunghini_additional], axis=0)
        df_class.reset_index(inplace=True, drop=True)

    df_class = replace_multiple_cas_for_one_inchi(df=df_class, prnt=prnt)

    df_multiples, df_singles, df_removed_due_to_variance = assign_group_label_and_drop_replicates(
        df=df_class, by_column="inchi_from_smiles"
    )
    assert len(df_multiples) + len(df_singles) + df_removed_due_to_variance["inchi_from_smiles"].nunique() == df_class["inchi_from_smiles"].nunique()

    df_class = pd.concat([df_multiples, df_singles], axis=0) # Here we just take the labels for the substances with one study result
    df_class.reset_index(inplace=True, drop=True)
    df_class.drop(["principle", "biodegradation_percent"], axis=1, inplace=True)

    log.info("Entries in df_class_multiple: ", df_class_multiple=len(df_multiples))
    log.info("Entries in df_removed_due_to_variance: ", df_removed_due_to_variance=len(df_removed_due_to_variance))
    return df_class, df_multiples, df_removed_due_to_variance


def is_unique(s):
    a = s.to_numpy()
    return (a[0] == a).all()


def assign_group_label_and_drop_replicates(df: pd.DataFrame, by_column: str) -> pd.DataFrame:
    counted_duplicates = (
        df.groupby(df[by_column].tolist(), as_index=False).size().sort_values(by="size", ascending=False)
    )

    multiples_lst: List[str] = []
    # doubles_lst: List[str] = []
    removed_lst: List[str] = []
    m_df = counted_duplicates[counted_duplicates["size"] > 1]  #tested > 2, and > 1
    multiples = m_df["index"].tolist()
    for substance in multiples:
        current_group = df[df[by_column] == substance] 
        if is_unique(current_group["y_true"]):
            multiples_lst.append(substance)
        else: 
            removed_lst.append(substance)
    
    # d_df = counted_duplicates[counted_duplicates["size"] == 2] # TODO make final decision and then delete
    # doubles = d_df["index"].tolist()
    # for substance in doubles:
    #     current_group = df[df[by_column] == substance] 
    #     if is_unique(current_group["y_true"]):
    #         doubles_lst.append(substance)
    #     else: 
    #         removed_lst.append(substance)

    s_df = counted_duplicates[counted_duplicates["size"] == 1] #tested <= 2, and == 1
    singles_df = df[df[by_column].isin(s_df["index"])]

    df_removed_due_to_variance = df[df[by_column].isin(removed_lst)]
    df_multiple = df[df[by_column].isin(multiples_lst)].copy()
    df_multiple.drop_duplicates(subset=[by_column], inplace=True, ignore_index=True)
    df_multiple.reset_index(inplace=True, drop=True)
    # df_double = df[df[by_column].isin(doubles_lst)].copy()
    # df_double.drop_duplicates(subset=[by_column], inplace=True, ignore_index=True)
    # singles_df = pd.concat([df_double, singles_df], axis=0)

    return df_multiple, singles_df, df_removed_due_to_variance


def reg_df_remove_studies_not_to_consider(df: pd.DataFrame) -> pd.DataFrame:
    # df = df[ # TODO
    #     ((df['time_day']==28.0) & (df["endpoint"]=="ready")) |
    #     ((df['time_day']>28.0) & (df["biodegradation_percent"]<0.7) & (df["principle"]=="DOC Die Away")) | 
    #     ((df['time_day']>28.0) & (df["biodegradation_percent"]<0.6) & (df["principle"]!="DOC Die Away")) | 
    #     ((df['time_day']<28.0) & (df["endpoint"]=="ready") & (df["biodegradation_percent"]>0.7) & (df["principle"]=="DOC Die Away")) | 
    #     ((df['time_day']<28.0) & (df["endpoint"]=="ready") & (df["biodegradation_percent"]>0.6) & (df["principle"]!="DOC Die Away"))
    #     ]
    df = df[(df['time_day']==28.0) & (df["endpoint"]=="ready")]
    df.reset_index(inplace=True, drop=True)
    return df


def create_classification_data_based_on_regression_data_adapted(
    reg_df: pd.DataFrame, 
    with_lunghini: bool, 
    env_smiles: bool, 
    prnt: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # reg_df MUST be without speciation for it to work propperly
    df_to_consider = reg_df_remove_studies_not_to_consider(reg_df)
    df = label_data_based_on_percentage(df_to_consider)

    columns = ["cas", "smiles", "principle", "biodegradation_percent", "y_true", "inchi_from_smiles"]
    df_class = pd.DataFrame(data=df.copy(), columns=columns)

    if with_lunghini:
        df_lunghini_additional = get_external_dataset_lunghini(
            run_from_start=False, # TODO
            class_df=df_class,
            include_speciation=False,
        )
        if env_smiles:
            df_lunghini_additional, _ = replace_smiles_with_env_relevant_smiles(df_without_env_smiles=df_lunghini_additional)
        df_lunghini_additional = df_lunghini_additional[df_lunghini_additional["smiles"].notna()]
        df_class = pd.concat([df_class, df_lunghini_additional], axis=0)
        df_class.reset_index(inplace=True, drop=True)

    df_class = replace_multiple_cas_for_one_inchi(df=df_class, prnt=prnt)

    df_class_multiple, df_singles, df_removed_due_to_variance = assign_group_label_and_drop_replicates(
        df=df_class, by_column="inchi_from_smiles"
    )
    df_class_multiple.drop(["principle", "biodegradation_percent"], axis=1, inplace=True)

    df_singles_biowin, df_problematic = process_df_biowin(
        df=df_singles,
        mode="class",
    )

    df_class = pd.concat([df_class_multiple, df_singles_biowin], axis=0)

    log.info("Entries in df_class_multiple: ", df_class_multiple=len(df_class_multiple))
    log.info("Entries in df_problematic: ", df_problematic=len(df_problematic))
    log.info("Entries in df_removed_due_to_variance: ", df_removed_due_to_variance=len(df_removed_due_to_variance))
    return df_class, df_problematic, df_removed_due_to_variance, df_class_multiple


def create_classification_biowin(
    reg_df: pd.DataFrame,
    with_lunghini: bool,
    include_speciation: bool,
    env_smiles: bool,
    prnt: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df_class = create_classification_data_based_on_regression_data(
        reg_df,
        with_lunghini=with_lunghini,
        include_speciation=include_speciation,
        env_smiles_lunghini=env_smiles,
        prnt=prnt,
    )
    df_class_biowin, df_class_biowin_removed = process_df_biowin(
        df=df_class,
        mode="class",
    )
    return df_class_biowin, df_class_biowin_removed


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


def convert_to_maccs_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    # 166 bit fingerprint
    df = df.copy()
    mols = [AllChem.MolFromSmiles(smiles) for smiles in df["smiles"]]
    df["fingerprint"] = [GetMACCSKeysFingerprint(mol) for mol in mols]

    return df

def convert_to_rdk_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    # calculate 2048 bit RDK fingerprint
    df = df.copy()
    mols = [AllChem.MolFromSmiles(smiles) for smiles in df["smiles"]]
    df["fingerprint"] = [Chem.RDKFingerprint(mol) for mol in mols]
    return df

def convert_to_pubchem_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    # calculate 881 bit PubChem fingerprint
    def create_x_class(row) -> np.ndarray:
        smiles = row["smiles"]
        fingerprint_dict = from_smiles(smiles, fingerprints=True, descriptors=False)
        fp = [int(bit) for bit in fingerprint_dict.values()]
        return np.array(fp)

    x_class = df.apply(create_x_class, axis=1).to_list()

    return x_class


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

    x_class = df.apply(create_x_class, axis=1).to_list()
    return x_class

def create_input_classification(df_class: pd.DataFrame, include_speciation: bool, fingerprint_type="MACCS") -> np.ndarray: # TODO change to MACCS
    """Function to create fingerprints and put fps into one array that can than be used as one feature for model training."""
    if fingerprint_type=="MACCS":
        df = convert_to_maccs_fingerprints(df_class)
        x_class = bit_vec_to_lst_of_lst(df, include_speciation)
    elif fingerprint_type=="RDK":
        df = convert_to_rdk_fingerprints(df_class)
        x_class = bit_vec_to_lst_of_lst(df, include_speciation)
    elif fingerprint_type=="PubChem":
        x_class = convert_to_pubchem_fingerprints(df_class)
    x_array = np.array(x_class, dtype=object)
    return x_array


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


def load_checked_organics6() -> pd.DataFrame:
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
        df[col].replace(
            to_replace=["nan", None, "", np.nan], value="c", inplace=True
        )  # use "c" as placeholder because openbabel cannot handle nans
        df = openbabel_convert(
            df=df,
            input_type="smiles",
            column_name_input=col,
            output_type="inchi",
        )
        df[f"inchi_from_{col}"].replace(to_replace="InChI=1S/CH3/h1H3", value="", inplace=True)  # remove placeholder
        df[col].replace(to_replace="c", value="", inplace=True)  # remove placeholder
    return df


def get_inchi_main_layer(df: pd.DataFrame, inchi_col: str, layers=int) -> pd.DataFrame:
    df = df.copy()

    def get_inchi_main_layer_inner_function(row):
        inchi_main_layer_smiles = get_inchi_layers(row, col_name=inchi_col, layers=layers)
        return inchi_main_layer_smiles

    df[f"{inchi_col}_main_layer"] = df.apply(get_inchi_main_layer_inner_function, axis=1)
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


def load_and_process_echa_additional() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    echa_additional_class = create_classification_data_based_on_regression_data(
        reg_df=echa_additional_class,
        with_lunghini=False,
        include_speciation=True,
        env_smiles_lunghini=False,
        prnt=False,
    )

    echa_additional_reg_env, _ = replace_smiles_with_env_relevant_smiles(echa_additional_reg.copy())
    echa_additional_reg = echa_additional_reg[echa_additional_reg.index.isin(echa_additional_reg_env.index)]
    echa_additional_class_env, _ = replace_smiles_with_env_relevant_smiles(echa_additional_class.copy())
    echa_additional_class = echa_additional_class[echa_additional_class.index.isin(echa_additional_class_env.index)]

    echa_additional_reg.reset_index(inplace=True, drop=True)
    echa_additional_reg_env.reset_index(inplace=True, drop=True)
    echa_additional_class.reset_index(inplace=True, drop=True)
    echa_additional_class_env.reset_index(inplace=True, drop=True)

    return echa_additional_reg, echa_additional_reg_env, echa_additional_class, echa_additional_class_env


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
    biowin = pd.read_csv("datasets/curated_data/class_curated_scs_biowin.csv", index_col=0)

    biowin_readded = pd.read_csv(
        "datasets/curated_data/class_curated_scs_biowin_readded.csv", index_col=0
    )
    return {
        "df_paper": df_class,
        "df_curated_scs": curated_scs,
        "df_curated_scs_biowin": biowin,
        "df_curated_scs_biowin_readded": biowin_readded,
    }


def get_regression_datasets() -> Dict[str, pd.DataFrame]:
    reg = load_regression_df()
    curated_scs = load_regression_df_curated_scs_no_metal()
    biowin = pd.read_csv("datasets/curated_data/reg_curated_scs_biowin.csv", index_col=0)
    biowin_readded = pd.read_csv(
        "datasets/curated_data/reg_curated_scs_biowin_readded.csv", index_col=0
    )
    name_to_df = {
        "df_paper": reg,
        "df_curated_scs": curated_scs,
        "df_curated_scs_biowin": biowin,
        "df_curated_scs_biowin_readded": biowin_readded,
    }

    for name, df in name_to_df.items():
        name_to_df[name] = convert_regression_df_to_input(df=df)

    return name_to_df


def get_comparison_datasets_regression(mode: str, include_speciation: bool) -> Dict[str, pd.DataFrame]:
    datasets = get_regression_datasets()
    df_biowin = pd.read_csv("datasets/curated_data/reg_paper_biowin.csv", index_col=0)
    df_curated_s_biowin = pd.read_csv(
        "datasets/curated_data/reg_curated_s_biowin.csv", index_col=0
    )
    df_curated_scs_biowin = pd.read_csv(
        "datasets/curated_data/reg_curated_scs_biowin.csv", index_col=0
    )

    name_to_df = {
        "df_paper": datasets["df_paper"],
        "df_curated_s": datasets["df_curated_s"],
        "df_curated_scs": datasets["df_curated_scs"],
        "df_paper_biowin": df_biowin,
        "df_curated_s_biowin": df_curated_s_biowin,
        "df_curated_scs_biowin": df_curated_scs_biowin,
    }
    for name, df in name_to_df.items():
        name_to_df[name] = convert_regression_df_to_input(df=df)
    return name_to_df


def get_comparison_datasets_classification(
    mode: str, include_speciation: bool, with_lunghini: bool, create_new: bool
) -> Dict[str, pd.DataFrame]:
    datasets = get_class_datasets()
    df_biowin = pd.read_csv("datasets/curated_data/class_paper_biowin.csv", index_col=0)
    df_curated_s_biowin = pd.read_csv(
        "datasets/curated_data/class_curated_s_biowin.csv", index_col=0
    )
    df_curated_scs_biowin = pd.read_csv(
        "datasets/curated_data/class_curated_scs_biowin.csv", index_col=0
    )

    name_to_df = {
        "df_paper": datasets["df_paper"],
        "df_curated_s": datasets["df_curated_s"],
        "df_curated_scs": datasets["df_curated_scs"],
        "df_paper_biowin": df_biowin,
        "df_curated_s_biowin": df_curated_s_biowin,
        "df_curated_scs_biowin": df_curated_scs_biowin,
    }
    return name_to_df


def create_dfs_for_curated_data_analysis() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    reg_curated_scs = load_regression_df_curated_scs_no_metal()
    reg_curated_scs_biowin = pd.read_csv(
        "datasets/curated_data/reg_curated_scs_biowin.csv", index_col=0
    )
    reg_curated_scs_biowin_readded = pd.read_csv(
        "datasets/curated_data/reg_curated_scs_biowin_readded.csv", index_col=0
    )

    class_curated_scs = pd.read_csv("datasets/curated_data/class_curated_scs.csv", index_col=0)
    class_curated_scs_biowin = pd.read_csv(
        "datasets/curated_data/class_curated_scs_biowin.csv", index_col=0
    )
    class_curated_scs_biowin_readded = pd.read_csv(
        "datasets/curated_data/class_curated_scs_biowin_readded.csv", index_col=0
    )

    return (
        reg_curated_scs,
        reg_curated_scs_biowin,
        reg_curated_scs_biowin_readded,
        class_curated_scs,
        class_curated_scs_biowin,
        class_curated_scs_biowin_readded,
    )


def load_opera_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_opera = PandasTools.LoadSDF("datasets/OPERA_data.sdf")
    df_opera = df_opera[["CAS", "Canonical_QSARr", "preferred_name", "InChI_Code_QSARr", "Ready_Biodeg"]].rename(
        columns={
            "CAS": "cas",
            "Canonical_QSARr": "smiles",
            "preferred_name": "name",
            "InChI_Code_QSARr": "inchi_from_smiles",
            "Ready_Biodeg": "label",
        }
    )
    df_opera["label"] = df_opera["label"].astype(float)
    log.info("Entries in OPERA data: ", opera=len(df_opera))
    log.info("Number of entries that are NRB", NRB=len(df_opera[df_opera["label"] == 0.0]))
    log.info("Number of entries that are RB", RB=len(df_opera[df_opera["label"] == 1.0]))

    df_opera = replace_multiple_cas_for_one_inchi(df=df_opera, prnt=True)
    df_opera, _ = remove_organo_metals_function(df=df_opera, smiles_column="smiles")

    df_readded = pd.read_csv(
        "datasets/curated_data/class_curated_scs_biowin_readded.csv", index_col=0
    )
    _, _, df_class_huang = load_class_data_paper()
    df_class_huang = remove_smiles_with_incorrect_format(df=df_class_huang, col_name_smiles="smiles")
    df_class_huang = openbabel_convert(
        df=df_class_huang, input_type="smiles", column_name_input="smiles", output_type="inchi"
    )

    df_readded = remove_smiles_with_incorrect_format(df=df_readded, col_name_smiles="smiles")
    df_readded = openbabel_convert(df=df_readded, input_type="smiles", column_name_input="smiles", output_type="inchi")

    df_opera_in_huang = df_opera[df_opera["inchi_from_smiles"].isin(df_class_huang["inchi_from_smiles"])]
    df_opera_additional = df_opera[~(df_opera["inchi_from_smiles"].isin(df_class_huang["inchi_from_smiles"]))]

    log.info("OPERA data in Huang", df_opera_in_huang=len(df_opera_in_huang))
    log.info("OPERA data NOT in Huang classification dataset", new_data=len(df_opera_additional))

    df_opera_additional_env_smiles, _ = replace_smiles_with_env_relevant_smiles(df_opera_additional.copy())
    df_opera_additional_env_smiles = df_opera_additional_env_smiles[
        ~df_opera_additional_env_smiles["inchi_from_smiles"].isin(df_readded["inchi_from_smiles"])
    ]
    df_opera_additional = df_opera_additional[df_opera_additional["cas"].isin(df_opera_additional_env_smiles["cas"])]
    assert len(df_opera_additional) == len(df_opera_additional_env_smiles)

    log.info("OPERA data NOT in readded and Huang classification data", new_data=len(df_opera_additional))

    assert len(df_opera_additional[df_opera_additional["inchi_from_smiles"].isin(df_readded["inchi_from_smiles"])]) == 0
    assert (
        len(df_opera_additional[df_opera_additional["inchi_from_smiles"].isin(df_class_huang["inchi_from_smiles"])])
        == 0
    )
    assert (
        len(
            df_opera_additional_env_smiles[
                df_opera_additional_env_smiles["inchi_from_smiles"].isin(df_readded["inchi_from_smiles"])
            ]
        )
        == 0
    )
    assert (
        len(
            df_opera_additional_env_smiles[
                df_opera_additional_env_smiles["inchi_from_smiles"].isin(df_class_huang["inchi_from_smiles"])
            ]
        )
        == 0
    )

    log.info(
        "Number of entries in df_opera_additional that are NRB",
        NRB=len(df_opera_additional[df_opera_additional["label"] == 0.0]),
    )
    log.info(
        "Number of entries in df_opera_additional that are RB",
        RB=len(df_opera_additional[df_opera_additional["label"] == 1.0]),
    )

    df_opera_additional.reset_index(inplace=True, drop=True)
    df_opera_additional_env_smiles.reset_index(inplace=True, drop=True)
    return df_opera_additional, df_opera_additional_env_smiles


def create_biowin_data() -> pd.DataFrame:
    df_biowin = pd.read_excel("datasets/biowin_original_dataset.xlsx", names=["Col", "second"])
    df_biowin.drop(columns=["second"], inplace=True)
    df_biowin.dropna(axis=0, how="any", inplace=True)
    df_biowin.dropna(axis=1, how="all", inplace=True)
    df_biowin.drop(index=[0, 2, 8, 10], inplace=True)
    df_biowin.reset_index(inplace=True, drop=True)
    df_biowin = df_biowin["Col"].str.split(" ", expand=True)
    df_biowin = df_biowin.astype(str)
    df_biowin = df_biowin.replace("None", "")
    df_biowin = df_biowin.applymap(lambda x: x.strip())

    def format_cols(row):
        cas = row[2]
        name = row[3] + " " + row[4] + " " + row[5] + " " + row[6] + " " + row[7]
        rest = ""
        for i in range(8, 32):
            r = row[i]
            if len(r) > 0:
                rest += r + ", "
        if len(cas) == 0:
            cas = row[3]
            name = row[4] + " " + row[5] + " " + row[6] + " " + row[7]
        cas = cas.lstrip("0")
        return pd.Series([cas, name, rest])

    df_biowin[["cas", "name", "rest"]] = df_biowin.apply(format_cols, axis=1)

    df_formatted = df_biowin[["cas", "name", "rest"]].copy()
    df_formatted[["label", "biowin1", "biowin2", "rest"]] = df_formatted["rest"].str.split(", ", expand=True)
    df_formatted = df_formatted[["cas", "name", "label", "biowin1", "biowin2"]]

    df_wrong = df_formatted[df_formatted["biowin1"] == ""]
    df_formatted = df_formatted[df_formatted["biowin1"] != ""]
    df_wrong.at[263, "name"] = "N'-(3-CHLORO-4-METHYLPHENYL)-N,N-DIMETHYLUREA"
    df_wrong.at[263, "label"] = 0
    df_wrong.at[263, "biowin1"] = 0.5186
    df_wrong.at[263, "biowin2"] = 0.1900
    df_formatted = pd.concat([df_formatted, df_wrong], axis=0)
    return df_formatted


def remove_biowin_entries_that_are_in_huang_class_and_readded(df_biowin_env: pd.DataFrame) -> pd.DataFrame:
    _, _, df_huang_class = load_class_data_paper()
    df_readded = pd.read_csv(
        "datasets/curated_data/class_curated_scs_biowin_readded.csv", index_col=0
    )
    df_huang_class = remove_smiles_with_incorrect_format(df=df_huang_class, col_name_smiles="smiles")
    df_huang_class = openbabel_convert(
        df=df_huang_class,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )

    df_biowin_additional = df_biowin_env[~df_biowin_env["inchi_from_smiles"].isin(df_readded["inchi_from_smiles"])]

    log.info(
        "Data points in biowin data that are not in the Huang classification data and the readded classification dataset",
        new_data=len(df_biowin_additional),
    )
    assert (
        len(df_biowin_additional[df_biowin_additional["inchi_from_smiles"].isin(df_readded["inchi_from_smiles"])]) == 0
    )
    assert (
        len(df_biowin_additional[df_biowin_additional["inchi_from_smiles"].isin(df_huang_class["inchi_from_smiles"])])
        == 0
    )

    return df_biowin_additional


def add_smiles_ccc(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info("Entries in that were not checked by Gluege", entries=len(df))

    def get_smiles_inchi(row):
        cas = row["cas"]
        smiles, cas = get_info_cas_common_chemistry(cas)
        return pd.Series([smiles, cas])

    df[["smiles_from_ccc", "inchi_from_ccc"]] = df.progress_apply(func=get_smiles_inchi, axis=1)
    df_smiles_found = df[df["smiles_from_ccc"] != ""].copy()
    df_smiles_not_found = df[df["smiles_from_ccc"] == ""].copy()
    log.info("SMILES found on CCC", num=len(df_smiles_found))

    df_multiple_components = df_smiles_found[df_smiles_found["smiles_from_ccc"].str.contains(".", regex=False)]
    if len(df_multiple_components) > 0:
        log.warn("Found SMILES with multiple components in data", num=len(df_multiple_components))

    df_smiles_found = df_smiles_found[~df_smiles_found["smiles_from_ccc"].str.contains(".", regex=False)]
    df_smiles_found.rename(columns={"smiles_from_ccc": "smiles", "inchi_from_ccc": "inchi"}, inplace=True)
    return df_smiles_found, df_smiles_not_found, df_multiple_components


def process_multiple_comp_data(
    df_multiple_components: pd.DataFrame, df_smiles_not_found: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_for_multiple = pd.concat([df_multiple_components, df_smiles_not_found], ignore_index=True)
    log.info("Need to run process_multiple_component_data for this many datapoints: ", entries=len(df_for_multiple))

    df_pubchem = get_smiles_from_cas_pubchempy(df_for_multiple)
    df_multiple = get_smiles_from_cas_comptox(df=df_pubchem)
    col_names_smiles_to_inchi = [
        "isomeric_smiles_pubchem",
        "smiles_comptox",
    ]
    df_multiple = openbabel_convert_smiles_to_inchi_with_nans(
        col_names_smiles_to_inchi=col_names_smiles_to_inchi, df=df_multiple
    )

    inchi_pubchem = "inchi_from_isomeric_smiles_pubchem"
    inchi_comptox = "inchi_from_smiles_comptox"
    df_no_smiles_pubchem_comptox = df_multiple[(df_multiple[inchi_pubchem] == "") & (df_multiple[inchi_comptox] == "")]
    df_no_smiles_pubchem = df_multiple[(df_multiple[inchi_pubchem] == "") & (df_multiple[inchi_comptox] != "")]
    df_no_smiles_comptox = df_multiple[(df_multiple[inchi_pubchem] != "") & (df_multiple[inchi_comptox] == "")]
    df_no_match_smiles_pubchem_comptox = df_multiple[
        (df_multiple[inchi_pubchem] != df_multiple[inchi_comptox])
        & ((df_multiple[inchi_pubchem] != "") & (df_multiple[inchi_comptox] != ""))
    ]
    df_match_smiles_pubchem_comptox = df_multiple[
        (df_multiple[inchi_pubchem] == df_multiple[inchi_comptox])
        & ((df_multiple[inchi_pubchem] != "") & (df_multiple[inchi_comptox] != ""))
    ]
    df_multiple_components_smiles_found = df_match_smiles_pubchem_comptox[
        ["name", "cas", "label", "biowin1", "biowin2", "isomeric_smiles_pubchem", inchi_pubchem]
    ].rename(
        columns={
            "isomeric_smiles_pubchem": "smiles",
            inchi_pubchem: "inchi",
        }
    )
    df_multiple_components_smiles_not_found = df_multiple[
        ~df_multiple["cas"].isin(list(df_multiple_components_smiles_found["cas"]))
    ]
    text_to_df = {
        "NO SMILES on PubChem and Comptox": df_no_smiles_pubchem_comptox,
        "No SMILES on PubChem but on Comptox": df_no_smiles_pubchem,
        "SMILES on PubChem but NOT on Comptox": df_no_smiles_comptox,
        "InChI from SMILES from PubChem and Comptox do NOT match": df_no_match_smiles_pubchem_comptox,
        "InChI from SMILES from PubChem and Comptox match": df_match_smiles_pubchem_comptox,
        "NO SMILES found yet (after PubChem and Comptox)": df_multiple_components_smiles_not_found,
    }
    for text, df in text_to_df.items():
        log.info(f"CAS for which {text}", entries=len(df))
    return df_multiple_components_smiles_found, df_multiple_components_smiles_not_found


def process_multiple_comps_smiles_not_found(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Process datapoints for which no SMILES found yet")
    df = get_inchi_main_layer(df=df, inchi_col="inchi_from_isomeric_smiles_pubchem", layers=4)
    df = get_inchi_main_layer(df=df, inchi_col="inchi_from_smiles_comptox", layers=4)

    # Check if inchi main layers from PubChem and Comptox match; if yes keep the datapoint
    inchi_pubchem_ml = "inchi_from_isomeric_smiles_pubchem_main_layer"
    inchi_comptox_ml = "inchi_from_smiles_comptox_main_layer"
    df_no_match_inchi_main_layer_pubchem_comptox = df[
        (df[inchi_pubchem_ml] != df[inchi_comptox_ml]) & ((df[inchi_pubchem_ml] != "") & (df[inchi_comptox_ml] != ""))
    ]
    df_match_inchi_main_layer_pubchem_comptox = df[
        (df[inchi_pubchem_ml] == df[inchi_comptox_ml]) & ((df[inchi_pubchem_ml] != "") & (df[inchi_comptox_ml] != ""))
    ]
    df_smiles_found_main_layer = df_match_inchi_main_layer_pubchem_comptox[
        ["name", "cas", "isomeric_smiles_pubchem", "inchi_from_isomeric_smiles_pubchem"]
    ].rename(
        columns={"isomeric_smiles_pubchem": "smiles", "inchi_from_isomeric_smiles_pubchem": "inchi"}
    )  # if main layer matches, take smiles from pubchem

    df_smiles_not_found = df[~df["cas"].isin(list(df_smiles_found_main_layer["cas"]))]

    # For the substances that have only a SMILES from pubchem, check cirpy
    def add_cirpy_info(row):
        cas = row["cas"]
        smile_cirpy, inchi_cirpy = get_smiles_inchi_cirpy(cas)
        return pd.Series([smile_cirpy, inchi_cirpy])

    df_smiles_not_found[["smiles_from_cas_cirpy", "inchi_from_cas_cirpy"]] = df_smiles_not_found.progress_apply(
        func=add_cirpy_info, axis=1
    )

    df_smiles_not_found = openbabel_convert_smiles_to_inchi_with_nans(
        col_names_smiles_to_inchi=["smiles_from_cas_cirpy"],
        df=df_smiles_not_found,
    )
    # Check if smiles from cirpy and pubchem match
    df_pubchem_cirpy_match = df_smiles_not_found[
        (
            df_smiles_not_found["inchi_from_isomeric_smiles_pubchem"]
            == df_smiles_not_found["inchi_from_smiles_from_cas_cirpy"]
        )
        & (df_smiles_not_found["inchi_from_isomeric_smiles_pubchem"] != "")
        & (df_smiles_not_found["inchi_from_smiles_from_cas_cirpy"] != "")
    ]
    df_pubchem_cirpy_match = df_pubchem_cirpy_match[
        ["name", "cas", "label", "biowin1", "biowin2", "isomeric_smiles_pubchem", "inchi_from_isomeric_smiles_pubchem"]
    ].rename(
        columns={
            "isomeric_smiles_pubchem": "smiles",
            "inchi_from_isomeric_smiles_pubchem": "inchi",
        }
    )  # if inchi from pubchem and cirpy match, take smiles from pubchem
    df_smiles_found = df_pubchem_cirpy_match.copy()
    df_smiles_not_found = df_smiles_not_found[~(df_smiles_not_found["cas"].isin(df_smiles_found["cas"]))]

    text_to_df = {
        "CAS for which InChI main layer from PubChem and Comptox match": df_match_inchi_main_layer_pubchem_comptox,
        "CAS for which InChI main layer from PubChem and Comptox do Not match": df_no_match_inchi_main_layer_pubchem_comptox,
        "Entries for which SMILES only in pubchem found and this matches with cirpy": df_pubchem_cirpy_match,
        "CAS for which SMILES found": df_smiles_found,
        "CAS for which no SMILES found yet": df_smiles_not_found,
    }
    for text, df in text_to_df.items():
        log.info(f"{text}", entries=len(df))
    return df_smiles_found, df_smiles_not_found


def process_data_not_checked_by_gluege(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_smiles_found, df_smiles_not_found, df_multiple_comps = add_smiles_ccc(df=df)
    df_multiple_found1, df_multiple_not_found1 = process_multiple_comp_data(
        df_multiple_components=df_multiple_comps,
        df_smiles_not_found=df_smiles_not_found,
    )
    df_multiple_found2, df_multiple_not_found2 = process_multiple_comps_smiles_not_found(df_multiple_not_found1)

    df_multiple_comps_smiles_found = pd.concat([df_multiple_found1, df_multiple_found2])
    df_not_found = df_multiple_not_found2.copy()
    df_found = pd.concat([df_smiles_found, df_multiple_comps_smiles_found])
    assert (len(df_found) + len(df_not_found)) == len(df)

    return df_found, df_not_found


def aggregate_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.astype(
        {
            "cas": str,
            "name": str,
            "label": float,
            "biowin1": float,
            "biowin2": float,
            "smiles": str,
            "inchi_from_smiles": str,
        }
    )
    aggregation_functions = {
        "cas": "first",
        "name": "first",
        "label": "mean",
        "biowin1": "mean",
        "biowin2": "mean",
        "smiles": "first",
        "inchi_from_smiles": "first",
    }
    df = df.groupby(["cas"]).aggregate(aggregation_functions).reset_index(drop=True)
    return df


def add_smiles(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_checked = load_checked_organics6()
    cas_in_checked = [cas for cas in df.cas if cas in df_checked.cas]
    log.info("This many cas are checked by Gluege", cas=len(cas_in_checked))
    df_found, _ = process_data_not_checked_by_gluege(df=df)

    df_found = replace_multiple_cas_for_one_inchi(df=df_found, prnt=False)
    df_found_scs, _ = replace_smiles_with_env_relevant_smiles(df_found.copy())

    df_found, _ = remove_organo_metals_function(df=df_found, smiles_column="smiles")
    df_found_scs, _ = remove_organo_metals_function(df=df_found_scs, smiles_column="smiles")

    df_found = df_found[["cas", "name", "label", "biowin1", "biowin2", "smiles", "inchi_from_smiles"]]
    df_found_scs = df_found[["cas", "name", "label", "biowin1", "biowin2", "smiles", "inchi_from_smiles"]]

    df_found_agg = aggregate_duplicates(df=df_found)
    df_found_scs_agg = aggregate_duplicates(df=df_found_scs)

    return df_found_agg, df_found_scs_agg


def load_biowin_data(new: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if new:
        df_biowin = create_biowin_data()
        _, df_biowin_smiles = add_smiles(df=df_biowin)

        df_biowin = remove_biowin_entries_that_are_in_huang_class_and_readded(df_biowin_smiles)

        df_biowin_smiles.to_csv("datasets/biowin_original_data_smiles.csv")
    else:
        df_biowin = pd.read_csv("datasets/biowin_original_data_smiles.csv", index_col=0)
    return df_biowin


def create_fingerprint_df(df: pd.DataFrame) -> pd.DataFrame:
    mols = [AllChem.MolFromSmiles(smiles) for smiles in df["smiles"]]
    fps = [np.array(GetMACCSKeysFingerprint(mol)) for mol in mols]
    df_fp = pd.DataFrame(data=fps)
    return df_fp


def check_if_in_AD(df_train: pd.DataFrame, df_test: pd.DataFrame):
    x_train = create_fingerprint_df(df=df_train)
    x_train = x_train.values
    x_test = create_fingerprint_df(df=df_test)
    x_test = x_test.values

    AD = ApplicabilityDomain(verbose=True)
    df_similarities = AD.analyze_similarity(base_test=x_test, base_train=x_train, similarity_metric="tanimoto")
    return df_similarities


def get_only_data_inside_AD_of_Huang_and_readded(df_test: pd.DataFrame, df_name: str, mode: str) -> pd.DataFrame:
    huang_threshold = 0.5
    readded_threshold = 0.5

    if mode == "class":
        _, _, df_huang = load_class_data_paper()
        df_readded = pd.read_csv(
            "datasets/curated_data/class_curated_scs_biowin_readded.csv", index_col=0
        )
    elif mode == "reg":
        df_huang = load_regression_df()
        df_readded = pd.read_csv(
            "datasets/curated_data/reg_curated_scs_biowin_readded.csv", index_col=0
        )
    df_similarities = check_if_in_AD(df_train=df_huang, df_test=df_test)
    df_similarities.reset_index(inplace=True, drop=True)
    subset_in_ad = df_similarities[df_similarities["Max"] >= huang_threshold]
    df_test_in_ad = df_test[df_test.index.isin(subset_in_ad.index)]

    df_similarities = check_if_in_AD(df_train=df_readded, df_test=df_test_in_ad)
    df_similarities.reset_index(inplace=True, drop=True)
    subset_in_ad = df_similarities[df_similarities["Max"] >= huang_threshold]
    df_in_ad = df_test_in_ad[df_test_in_ad.index.isin(subset_in_ad.index)]
    log.info(
        f"Entries from {df_name} that are in the AD of the Huang and Zhang dataset and the readded dataset",
        df_in_ad=len(df_in_ad),
    )

    return df_in_ad


def find_echa_additional(
    df_echa_additional: pd.DataFrame, df_echa_additional_env: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_readded = pd.read_csv(
        "datasets/curated_data/class_curated_scs_biowin_readded.csv", index_col=0
    )
    df_echa_additional_env = df_echa_additional_env[
        ~df_echa_additional_env["inchi_from_smiles"].isin(df_readded["inchi_from_smiles"])
    ].copy()
    df_echa_additional = df_echa_additional[df_echa_additional["cas"].isin(df_echa_additional_env["cas"])].copy()
    assert len(df_echa_additional) == len(df_echa_additional_env)
    assert len(df_echa_additional[df_echa_additional["inchi_from_smiles"].isin(df_readded["inchi_from_smiles"])]) == 0
    assert (
        len(df_echa_additional_env[df_echa_additional_env["inchi_from_smiles"].isin(df_readded["inchi_from_smiles"])])
        == 0
    )

    df_echa_additional.rename(columns={"y_true": "label"}, inplace=True)
    df_echa_additional_env.rename(columns={"y_true": "label"}, inplace=True)
    log.info(
        "Entries in echa_additional that are not in Huang class dataset and the readded class dataset",
        entries_echa_additional=len(df_echa_additional),
        entries_echa_additional_env=len(df_echa_additional_env),
    )
    return df_echa_additional, df_echa_additional_env


def load_external_class_test_dfs_additional_to_Huang_readded() -> Tuple[
    Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]
]:
    df_opera_additional, df_opera_additional_env = load_opera_data()
    log.info(
        "Entries in df_opera_additional",
        entries_opera_additional=len(df_opera_additional),
        entries_opera_additional_env=len(df_opera_additional_env),
    )

    df_biowin = load_biowin_data(new=True)
    log.info("Entries in df_biowin", entries_biowin=len(df_biowin))

    _, _, df_echa_additional, df_echa_additional_env = load_and_process_echa_additional()
    df_echa_additional, df_echa_additional_env = find_echa_additional(df_echa_additional, df_echa_additional_env)

    test_datasets = {
        "OPERA_data": df_opera_additional,
        "biowin_original_data": df_biowin,
        "echa_additional": df_echa_additional,
    }

    log.info("Checking if datasets without SMILES with chemical speciation in AD")
    test_datasets_in_ad: Dict[str, pd.DataFrame] = {}
    for df_name, df in test_datasets.items():
        df = get_only_data_inside_AD_of_Huang_and_readded(df_test=df, df_name=df_name, mode="class")
        test_datasets_in_ad[f"{df_name}_in_ad"] = df
    log.info("Checking if datasets with SMILES with chemical speciation in AD")

    return test_datasets, test_datasets_in_ad


def load_external_reg_test_dfs_additional_to_Huang_readded() -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    echa_additional_reg, echa_additional_reg_env, _, _ = load_and_process_echa_additional()
    df_readded = pd.read_csv(
        "datasets/curated_data/reg_curated_scs_biowin_readded.csv", index_col=0
    )
    echa_additional_reg_env = echa_additional_reg_env[
        ~echa_additional_reg_env["inchi_from_smiles"].isin(df_readded["inchi_from_smiles"])
    ]
    echa_additional_reg = echa_additional_reg[echa_additional_reg["cas"].isin(echa_additional_reg_env["cas"])]
    assert len(echa_additional_reg) == len(echa_additional_reg_env)
    assert len(echa_additional_reg[echa_additional_reg["inchi_from_smiles"].isin(df_readded["inchi_from_smiles"])]) == 0
    assert (
        len(echa_additional_reg_env[echa_additional_reg_env["inchi_from_smiles"].isin(df_readded["inchi_from_smiles"])])
        == 0
    )
    log.info(
        "Entries in echa_additional that are not in Huang regression dataset and the readded regression dataset",
        entries_echa_additional=len(echa_additional_reg),
        entries_echa_additional_env=len(echa_additional_reg_env),
    )
    echa_additional_reg_in_ad = get_only_data_inside_AD_of_Huang_and_readded(
        df_test=echa_additional_reg, df_name="echa_additional", mode="reg"
    )
    echa_additional_reg_env_in_ad = get_only_data_inside_AD_of_Huang_and_readded(
        df_test=echa_additional_reg_env, df_name="echa_additional", mode="reg"
    )

    return echa_additional_reg, echa_additional_reg_env, echa_additional_reg_in_ad, echa_additional_reg_env_in_ad




def column_to_structure() -> Dict[int, str]: 
    # source: http://www.mayachemtools.org/docs/modules/html/MACCSKeys.html
    column_to_structure_maccs = {
        0: "Index",
        1: "Isotope",
        2: "Atomic No. between 103 and 256",
        3: "Group IVA & VA & VIA Rows 4-6",
        4: "Actinide",
        5: "Group IIIB & IVB",
        6: "Lanthanide",
        7: "Group VB & VIB & VIIB ",
        8: "QAAA@1",
        9: "Group VIII",
        10: "Group IIA (Alkaline Earth)",
        11: "4M Ring",
        12: "Group IB & IIB",
        13: "ON(C)C",
        14: "S-S",
        15: "OC(O)O",
        16: "QAA@1",
        17: "CTC",
        18: "Group IIIA (B...)",
        19: "7M Ring",
        20: "SI",
        21: "C=C(Q)Q",
        22: "3M Ring",
        23: "NC(O)O",
        24: "N-O",
        25: "NC(N)N",
        26: "C$=C($A)$A",
        27: "I",
        28: "QCH2Q",
        29: "P",
        30: "CQ(C)(C)A",
        31: "QX",
        32: "CSN",
        33: "NS",
        34: "CH2=A",
        35: "Group IA (Alkali Metal)",
        36: "S Heterocycle",
        37: "NC(O)N",
        38: "NC(C)N",
        39: "OS(O)O",
        40: "S-O",
        41: "CTN",
        42: "F",
        43: "QHAQH",
        44: "Other",
        45: "C=CN",
        46: "BR",
        47: "SAN",
        48: "OQ(O)O",
        49: "Charge",
        50: "C=C(C)C",
        51: "CSO",
        52: "NN",
        53: "QHAAAQH",
        54: "QHAAQH",
        55: "OSO",
        56: "ON(O)C",
        57: "O Heterocycle",
        58: "QSQ",
        59: "Snot%A%A",
        60: "S=O",
        61: "AS(A)A",
        62: "A$A!A$A",
        63: "N=O",
        64: "A$A!S",
        65: "C%N",
        66: "CC(C)(C)A",
        67: "QS",
        68: "QHQH (&...)",
        69: "QQH",
        70: "QNQ",
        71: "NO",
        72: "OAAO",
        73: "S=A",
        74: "CH3ACH3",
        75: "A!N$A",
        76: "C=C(A)A",
        77: "NAN",
        78: "C=N",
        79: "NAAN",
        80: "NAAAN",
        81: "SA(A)A",
        82: "ACH2QH",
        83: "QAAAA@1",
        84: "NH2",
        85: "CN(C)C",
        86: "CH2QCH2",
        87: "X!A$A",
        88: "S",
        89: "OAAAO",
        90: "QHAACH2A",
        91: "QHAAACH2A",
        92: "OC(N)C",
        93: "QCH3",
        94: "QN",
        95: "NAAO",
        96: "5M Ring",
        97: "NAAAO",
        98: "QAAAAA@1",
        99: "C=C",
        100: "ACH2N",
        101: "8M Ring",
        102: "QO",
        103: "CL",
        104: "QHACH2A",
        105: "A$A($A)$A",
        106: "QA(Q)Q",
        107: "XA(A)A",
        108: "CH3AAACH2A",
        109: "ACH2O",
        110: "NCO",
        111: "NACH2A",
        112: "AA(A)(A)A",
        113: "Onot%A%A",
        114: "CH3CH2A",
        115: "CH3ACH2A",
        116: "CH3AACH2A",
        117: "NAO",
        118: "ACH2CH2A (more than 1)",
        119: "N=A",
        120: "Heterocyclic atom (more than 1)",
        121: "N Heterocycle",
        122: "AN(A)A",
        123: "OCO",
        124: "QQ",
        125: "Aromatic Ring (more than 1)",
        126: "A!O!A",
        127: "A$A!O (more than 1)",
        128: "ACH2AAACH2A",
        129: "ACH2AACH2A",
        130: "QQ (more than 1)",
        131: "QH (more than 1)",
        132: "OACH2A",
        133: "A$A!N",
        134: "X (Halogen)",
        135: "Nnot%A%A",
        136: "O=A (more than 1)",
        137: "Heterocycle",
        138: "QCH2A (more than 1)",
        139: "OH",
        140: "O (more than 3)",
        141: "CH3 (more than 2)",
        142: "N (more than 1)",
        143: "A$A!O",
        144: "Anot%A%Anot%A",
        145: "6M Ring (more than 1)",
        146: "O (more than 2)",
        147: "ACH2CH2A",
        148: "AQ(A)A",
        149: "CH3 (more than 1)",
        150: "A!A$A!A",
        151: "NH",
        152: "OC(C)C",
        153: "QCH2A",
        154: "C=O",
        155: "A!CH2!A",
        156: "NA(A)A",
        157: "C-O",
        158: "C-N",
        159: "O (more than 1)",
        160: "CH3",
        161: "N",
        162: "Aromatic",
        163: "6M Ring",
        164: "O",
        165: "Ring",
        166: "Fragments",    
    }
    return column_to_structure_maccs

def get_maccs_names() -> List[str]:
    maccs_names = [
        "Index",
        "Isotope",
        "103 < Atomic No. < 256",
        "Group IVA, VA, VIA Rows 4-6",
        "Actinide",
        "Group IIIB, IVB",
        "Lanthanide",
        "Group VB, VIB, VIIB ",
        "QAAA@1",
        "Group VIII",
        "Group IIA (Alkaline Earth)",
        "4M Ring",
        "Group IB & IIB",
        "ON(C)C",
        "S-S",
        "OC(O)O",
        "QAA@1",
        "CTC",
        "Group IIIA (B...)",
        "7M Ring",
        "SI",
        "C=C(Q)Q",
        "3M Ring",
        "NC(O)O",
        "N-O",
        "NC(N)N",
        "C$=C($A)$A",
        "I",
        "QCH2Q",
        "P",
        "CQ(C)(C)A",
        "QX",
        "CSN",
        "NS",
        "CH2=A",
        "Group IA (Alkali Metal)",
        "S Heterocycle",
        "NC(O)N",
        "NC(C)N",
        "OS(O)O",
        "S-O",
        "CTN",
        "F",
        "QHAQH",
        "Other",
        "C=CN",
        "BR",
        "SAN",
        "OQ(O)O",
        "Charge",
        "C=C(C)C",
        "CSO",
        "NN",
        "QHAAAQH",
        "QHAAQH",
        "OSO",
        "ON(O)C",
        "O Heterocycle",
        "QSQ",
        "Snot%A%A",
        "S=O",
        "AS(A)A",
        "A$A!A$A",
        "N=O",
        "A$A!S",
        "C%N",
        "CC(C)(C)A",
        "QS",
        "QHQH (&...)",
        "QQH",
        "QNQ",
        "NO",
        "OAAO",
        "S=A",
        "CH3ACH3",
        "A!N$A",
        "C=C(A)A",
        "NAN",
        "C=N",
        "NAAN",
        "NAAAN",
        "SA(A)A",
        "ACH2QH",
        "QAAAA@1",
        "NH2",
        "CN(C)C",
        "CH2QCH2",
        "X!A$A",
        "S",
        "OAAAO",
        "QHAACH2A",
        "QHAAACH2A",
        "OC(N)C",
        "QCH3",
        "QN",
        "NAAO",
        "5M Ring",
        "NAAAO",
        "QAAAAA@1",
        "C=C",
        "ACH2N",
        "8M Ring",
        "QO",
        "CL",
        "QHACH2A",
        "A$A($A)$A",
        "QA(Q)Q",
        "XA(A)A",
        "CH3AAACH2A",
        "ACH2O",
        "NCO",
        "NACH2A",
        "AA(A)(A)A",
        "Onot%A%A",
        "CH3CH2A",
        "CH3ACH2A",
        "CH3AACH2A",
        "NAO",
        "ACH2CH2A (more than 1)",
        "N=A",
        "Heterocyclic atom (more than 1)",
        "N Heterocycle",
        "AN(A)A",
        "OCO",
        "QQ",
        "Aromatic Ring (more than 1)",
        "A!O!A",
        "A$A!O (more than 1)",
        "ACH2AAACH2A",
        "ACH2AACH2A",
        "QQ (more than 1)",
        "QH (more than 1)",
        "OACH2A",
        "A$A!N",
        "X (Halogen)",
        "Nnot%A%A",
        "O=A (more than 1)",
        "Heterocycle",
        "QCH2A (more than 1)",
        "OH",
        "O (more than 3)",
        "CH3 (more than 2)",
        "N (more than 1)",
        "A$A!O",
        "Anot%A%Anot%A",
        "6M Ring (more than 1)",
        "O (more than 2)",
        "ACH2CH2A",
        "AQ(A)A",
        "CH3 (more than 1)",
        "A!A$A!A",
        "NH",
        "OC(C)C",
        "QCH2A",
        "C=O",
        "A!CH2!A",
        "NA(A)A",
        "C-O",
        "C-N",
        "O (more than 1)",
        "CH3",
        "N",
        "Aromatic",
        "6M Ring",
        "O",
        "Ring",
        "Fragments",    
    ]
    return maccs_names








