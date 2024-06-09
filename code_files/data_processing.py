
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
from code_files.processing_functions import further_processing_of_echa_data
from code_files.processing_functions import get_df_with_unique_cas
from code_files.processing_functions import remove_organo_metals_function
from code_files.processing_functions import get_smiles_from_cas_pubchempy
from code_files.processing_functions import get_smiles_from_cas_comptox
from code_files.processing_functions import get_info_cas_common_chemistry
from code_files.processing_functions import replace_smiles_with_smiles_with_chemical_speciation
from code_files.processing_functions import replace_multiple_cas_for_one_inchi
from code_files.processing_functions import load_regression_df
from code_files.processing_functions import load_gluege_data
from code_files.processing_functions import check_number_of_components
from code_files.processing_functions import openbabel_convert_smiles_to_inchi_with_nans
from code_files.processing_functions import get_inchi_main_layer
from code_files.processing_functions import get_molecular_formula_from_inchi
from code_files.processing_functions import get_smiles_inchi_cirpy
from code_files.processing_functions import openbabel_convert
from code_files.processing_functions import remove_smiles_with_incorrect_format
from code_files.processing_functions import get_speciation_col_names


parser = argparse.ArgumentParser()

parser.add_argument(
    "--new_cas_common_chemistry",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Whether to add new smiles from CAS Common Chemistry",
)
parser.add_argument(
    "--new_pubchem",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Whether to add new smiles from pubchem",
)
parser.add_argument(
    "--new_comptox",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Whether to add new smiles from comptox",
)
parser.add_argument(
    "--new_cirpy",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Whether to add new smiles from cirpy",
)
args = parser.parse_args()


def replace_smiles_with_smiles_gluege(df: pd.DataFrame, df_checked: pd.DataFrame) -> pd.DataFrame:
    df = remove_smiles_with_incorrect_format(df=df, col_name_smiles="smiles")
    df = openbabel_convert(
        df=df,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )

    df = get_inchi_main_layer(df=df, inchi_col="inchi_from_smiles", layers=4)
    df = get_molecular_formula_from_inchi(df=df, inchi_col="inchi_from_smiles")
    df_checked = get_inchi_main_layer(df=df_checked, inchi_col="inchi_from_smiles", layers=4)
    df_checked = get_molecular_formula_from_inchi(df=df_checked, inchi_col="inchi_from_smiles")

    inchi_huang_gluege_dont_match: List[str] = []
    inchi_main_huang_gluege_dont_match: List[str] = []
    molecular_formular_huang_gluege_dont_match: List[str] = []
    def get_smiles_ec_num(row):
        cas = row["cas"]
        match = df_checked[df_checked["cas"] == cas]
        inchi = row["inchi_from_smiles"]
        inchi_main = row["inchi_from_smiles_main_layer"]
        mf = row["inchi_from_smiles_molecular_formula"]
        if len(match["smiles"].values) == 0:
            log.warn("This CAS RN was not checked by Gluege et al.", cas=cas)
            smiles = row["smiles"]
        else:
            smiles = str(match["smiles"].values[0])
            inchi_checked = str(match["inchi_from_smiles"].values[0])
            inchi_checked_main = str(match["inchi_from_smiles_main_layer"].values[0])
            inchi_checked_mf = str(match["inchi_from_smiles_molecular_formula"].values[0])
            if inchi != inchi_checked:
                inchi_huang_gluege_dont_match.append(inchi)
                inchi = inchi_checked
            if inchi_main != inchi_checked_main:
                inchi_main_huang_gluege_dont_match.append(inchi)
            if mf != inchi_checked_mf:
                molecular_formular_huang_gluege_dont_match.append(inchi)
        return pd.Series([smiles, inchi])

    df[["smiles", "inchi_from_smiles"]] = df.apply(get_smiles_ec_num, axis=1)
    log.info(
        "InChI from SMILES from Huang et al. did not match the InChI from SMILES from Gluege et al.",
        no_match=len(inchi_huang_gluege_dont_match), 
        no_match_inchi=len(set(inchi_huang_gluege_dont_match)), 
        no_match_inchi_percent="{:.1f}".format(len(set(inchi_huang_gluege_dont_match))/df.cas.nunique()*100),
        no_match_inchi_main=len(set(inchi_main_huang_gluege_dont_match)), 
        no_match_inchi_main_percent="{:.1f}".format(len(set(inchi_main_huang_gluege_dont_match))/df.cas.nunique()*100),
        no_match_inchi_molecular_formula=len(set(molecular_formular_huang_gluege_dont_match)), 
        no_match_inchi_molecular_formula_percent="{:.1f}".format(len(set(molecular_formular_huang_gluege_dont_match))/df.cas.nunique()*100),
    )
    return df


def get_df_a_and_b(
    df_unique_cas: pd.DataFrame, df_checked: pd.DataFrame, df_reg: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_a = df_unique_cas[df_unique_cas["cas"].isin(df_checked["cas"].tolist())].copy()
    df_b = df_unique_cas[~df_unique_cas["cas"].isin(df_checked["cas"].tolist())].copy()
    df_a_full_original = df_reg[df_reg["cas"].isin(df_a["cas"].tolist())].copy()
    df_b_full_original = df_reg[df_reg["cas"].isin(df_b["cas"].tolist())].copy()
    df_a_full_original = df_reg[df_reg["cas"].isin(list(df_a["cas"]))].copy()
    df_b_full_original = df_reg[df_reg["cas"].isin(list(df_b["cas"]))].copy()
    return (
        df_a,
        df_b,
        df_a_full_original,
        df_b_full_original,
    )


def get_problematic_studies_echa() -> pd.DataFrame:
    df_echa = pd.read_csv("datasets/iuclid_echa.csv", index_col=0)
    df_echa = further_processing_of_echa_data(df_echa)
    log.info("Studies retrieved from ECHA with iuclid after processing", studies=len(df_echa))
    df_echa = df_echa.astype({"inventory_num": str, "ref_inventory_num": str, "cas": str, "ref_cas": str})

    def check_if_problematic(row):
        inventory_num = row["inventory_num"]
        ref_inventory_num = row["ref_inventory_num"]
        cas = row["cas"]
        ref_cas = row["ref_cas"]
        if (inventory_num not in ref_inventory_num) and (inventory_num != "nan") and (ref_inventory_num != "nan"):
            return True
        elif (cas not in ref_cas) and (cas != "nan") and (ref_cas != "nan"):
            return True
        return False

    df_echa["is_ref_problematic"] = df_echa.apply(check_if_problematic, axis=1)
    df_problematic = df_echa[df_echa["is_ref_problematic"] == True]
    df_problematic = df_problematic[
        [
            "inventory_num",
            "ref_inventory_num",
            "cas",
            "ref_cas",
            "smiles",
            "smiles_ph",
            "reliability",
            "biodegradation_percent",
            "biodegradation_samplingtime",
            "endpoint",
            "guideline",
            "principle",
        ]
    ].rename(columns={"biodegradation_samplingtime": "time_day"})
    log.info(
        "Read-across: Total number of studies for which inventory number or CAS RN don't match reference substance (ECHA)",
        problematic_studies=len(df_problematic),
    )
    return df_problematic


def remove_studies_where_registered_substance_dont_match_reference_substance(
    df_a_full: pd.DataFrame
) -> pd.DataFrame:
    df_problematic_studies = get_problematic_studies_echa()
    df_problematic_studies = df_problematic_studies[df_problematic_studies["cas"].isin(df_a_full["cas"])].copy()

    def remove_problematic_studies(row):
        cas = row["cas"]
        if cas not in list(df_problematic_studies["cas"]):
            return False
        df_cas_problematic_studies = df_problematic_studies[df_problematic_studies["cas"] == cas]
        df_a_cas = [
            row["reliability"],
            row["endpoint"],
            row["time_day"],
            row["biodegradation_percent"],
        ]
        for indx in df_cas_problematic_studies.index:
            if (
                (df_a_cas[0] == df_cas_problematic_studies.at[indx, "reliability"])
                and (df_a_cas[1] == df_cas_problematic_studies.at[indx, "endpoint"])
                and (df_a_cas[2] == df_cas_problematic_studies.at[indx, "time_day"])
                and (df_a_cas[3] == df_cas_problematic_studies.at[indx, "biodegradation_percent"])
            ):
                return True
        return False
    df_a_full["to_remove"] = df_a_full.apply(remove_problematic_studies, axis=1)
    df_removed_studies = df_a_full[df_a_full["to_remove"] == True].copy()
    # log.info("Read-across: Removed studies in df_a_full because reference CAS RN didn't match CAS RN", removed=len(df_removed_studies), removed_belonged_to_unique_cas=df_removed_studies.cas.nunique())
    df_a_full_clean = df_a_full[df_a_full["to_remove"] == False].copy()
    log.info(
        "Entries in df_a_full_clean without problematic studies",
        df_a_full_clean=len(df_a_full_clean),
        unique_cas = df_a_full_clean.cas.nunique(),
        unique_inchi = df_a_full_clean.inchi_from_smiles.nunique()
    )
    assert len(df_a_full_clean) + len(df_removed_studies) == len(df_a_full)
    log.info("Read-across: Number of unique CAS RN removed due to read-across", unique_cas_removed=df_a_full.cas.nunique()-df_a_full_clean.cas.nunique())

    return df_a_full_clean


def process_one_component_data(
    df_b: pd.DataFrame, new_cas_common_chemistry: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info("Entries in df_b (Not checked by Gluege) with one component", entries=len(df_b))

    def get_smiles_inchi_from_ccc(row):
        cas = row["cas"]
        smiles, cas = get_info_cas_common_chemistry(cas)
        return pd.Series([smiles, cas])

    if new_cas_common_chemistry:
        df_b[["smiles_from_ccc", "inchi_from_ccc"]] = df_b.progress_apply(func=get_smiles_inchi_from_ccc, axis=1)
        df_b.to_csv(f"datasets/data_processing/df_one_component_ccc.csv") 
    df_b_ccc = pd.read_csv(f"datasets/data_processing/df_one_component_ccc.csv", index_col=0)

    df_smiles_found = df_b_ccc[df_b_ccc["smiles_from_ccc"].notnull()].copy()
    df_smiles_not_found = df_b_ccc[df_b_ccc["smiles_from_ccc"].isnull()].copy()
    assert len(df_smiles_found) + len(df_smiles_not_found) == len(df_b_ccc)
    log.info("SMILES found on CAS CC for one component substances", num=len(df_smiles_found))
    df_multiple_components = df_smiles_found[df_smiles_found["smiles_from_ccc"].str.contains(".", regex=False)]
    if len(df_multiple_components) > 0:
        log.warn(
            "Found SMILES with multiple components in originally one component data",
            num=len(df_multiple_components),
        )
    df_smiles_found = df_smiles_found[~df_smiles_found["smiles_from_ccc"].str.contains(".", regex=False)]
    df_smiles_found.rename(
        columns={"smiles_from_ccc": "smiles", "inchi_from_cas_cc": "inchi"},
        inplace=True,
    )
    df_smiles_found.drop(labels="smiles", axis=1, inplace=True)
    return df_smiles_found, df_smiles_not_found, df_multiple_components


def process_multiple_component_data(
    df_multiple_components: pd.DataFrame,
    new_pubchem: bool,
    new_comptox: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if new_pubchem:
        df_pubchem = get_smiles_from_cas_pubchempy(df_multiple_components)
        df_pubchem.to_csv("datasets/data_processing/df_smiles_multiple_component_pubchem_no_metals.csv") 
    df_pubchem = pd.read_csv(
        "datasets/data_processing/df_smiles_multiple_component_pubchem_no_metals.csv",
        index_col=0,
    )

    if new_comptox:
        df_comptox = get_smiles_from_cas_comptox(df=df_pubchem)
        df_comptox.to_csv("datasets/data_processing/comptox_no_metals.csv") 
    df_multiple = pd.read_csv("datasets/data_processing/comptox_no_metals.csv", index_col=0)

    assert len(df_multiple_components) == len(df_multiple)

    assert len(df_multiple) == len(df_pubchem)
    col_names_smiles_to_inchi = [
        "smiles_paper",
        "smiles_from_ccc",
        "isomeric_smiles_pubchem",
        "isomeric_smiles_pubchem",
        "smiles_comptox",
    ]
    df_multiple = openbabel_convert_smiles_to_inchi_with_nans(
        col_names_smiles_to_inchi=col_names_smiles_to_inchi, df=df_multiple
    )
    inchi_pubchem = "inchi_from_isomeric_smiles_pubchem"
    inchi_comptox = "inchi_from_smiles_comptox"
    df_smiles_pubchem_comptox = df_multiple[(df_multiple[inchi_pubchem] != "") & (df_multiple[inchi_comptox] != "")]
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
        ["name", "cas", "isomeric_smiles_pubchem", inchi_pubchem]
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
        "SMILES found on PubChem and Comptox": df_smiles_pubchem_comptox,
        "NO SMILES on PubChem and Comptox": df_no_smiles_pubchem_comptox,
        "NO SMILES on PubChem but on Comptox": df_no_smiles_pubchem,
        "SMILES on PubChem but NOT on Comptox": df_no_smiles_comptox,
        "SMILES on either PubChem or Comptox but NOT both": df_no_smiles_pubchem + df_no_smiles_comptox,
        "InChI from SMILES from PubChem and Comptox do NOT match": df_no_match_smiles_pubchem_comptox,
        "InChI from SMILES from PubChem and Comptox match": df_match_smiles_pubchem_comptox,
        "NO SMILES found yet (after PubChem and Comptox)": df_multiple_components_smiles_not_found,
    }
    for text, df in text_to_df.items():
        log.info(f"CAS RN for which {text}", entries=len(df))

    assert len(df_multiple) == len(df_multiple_components_smiles_found) + len(df_multiple_components_smiles_not_found)
    return df_multiple_components_smiles_found, df_multiple_components_smiles_not_found


def process_multiple_components_smiles_not_found(
    df: pd.DataFrame,
    new_cirpy: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Process datapoints for which no SMILES found yet")
    df = get_inchi_main_layer(df=df, inchi_col="inchi_from_isomeric_smiles_pubchem", layers=4)
    df = get_inchi_main_layer(df=df, inchi_col="inchi_from_smiles_comptox", layers=4)

    # Check if inchi main layers from PubChem and Comptox match; if yes keep the datapoint
    inchi_pubchem_ml = "inchi_from_isomeric_smiles_pubchem_main_layer"
    inchi_comptox_ml = "inchi_from_smiles_comptox_main_layer"
    df_match_inchi_main_layer_pubchem_comptox = df[
        (df[inchi_pubchem_ml] == df[inchi_comptox_ml]) & ((df[inchi_pubchem_ml] != "") & (df[inchi_comptox_ml] != ""))
    ]
    df_smiles_found_main_layer = df_match_inchi_main_layer_pubchem_comptox[
        ["name", "cas", "isomeric_smiles_pubchem", "inchi_from_isomeric_smiles_pubchem"]
    ].rename(
        columns={
            "isomeric_smiles_pubchem": "smiles",
            "inchi_from_isomeric_smiles_pubchem": "inchi",
        }
    )  # if main layer matches, take smiles from pubchem

    # Further processing of the substances where the inchi main layers don't match
    df_smiles_not_found = df[~df["cas"].isin(list(df_smiles_found_main_layer["cas"]))]
    # Check if the smiles from comptox matches with the smile from the paper (because comptox was not used in paper)
    df_smiles_not_found = get_inchi_main_layer(df=df_smiles_not_found, inchi_col="inchi_from_smiles_paper", layers=4)
    df_smiles_not_found = get_inchi_main_layer(df=df_smiles_not_found, inchi_col="inchi_from_smiles_comptox", layers=4)
    df_comptox_paper_match = df_smiles_not_found[
        (
            df_smiles_not_found["inchi_from_smiles_comptox_main_layer"]
            == df_smiles_not_found["inchi_from_smiles_paper_main_layer"]
        )
        & (df_smiles_not_found["inchi_from_smiles_comptox_main_layer"] != "")
        & (df_smiles_not_found["inchi_from_smiles_paper_main_layer"] != "")
    ]
    df_comptox_paper_match = df_comptox_paper_match[["name", "cas", "smiles_comptox", "inchi_comptox"]].rename(
        columns={"smiles_comptox": "smiles", "inchi_comptox": "inchi"}
    )  # if inchi from comptox and paper match, take smiles from comptox

    df_smiles_found = pd.concat([df_smiles_found_main_layer, df_comptox_paper_match])
    df_smiles_not_found = df_smiles_not_found[~(df_smiles_not_found["cas"].isin(df_smiles_found["cas"]))]

    # For the substances that have only a SMILES from pubchem, check cirpy
    len_old_df_smiles_not_found = len(df_smiles_not_found)
    if new_cirpy:

        def add_cirpy_info(row):
            cas = row["cas"]
            smile_cirpy, inchi_cirpy = get_smiles_inchi_cirpy(cas)
            return pd.Series([smile_cirpy, inchi_cirpy])

        df_smiles_not_found[["smiles_from_cas_cirpy", "inchi_from_cas_cirpy"]] = df_smiles_not_found.progress_apply(
            func=add_cirpy_info, axis=1
        )
        df_smiles_not_found.to_csv(
            f"datasets/data_processing/df_multiple_components_smiles_not_found_ccc_cirpy_no_metals.csv"
        ) 
    len_old_df_smiles_not_found = len(df_smiles_not_found)
    df_smiles_not_found = pd.read_csv(
            f"datasets/data_processing/df_multiple_components_smiles_not_found_ccc_cirpy_no_metals.csv", index_col=0
        )
    if len(df_smiles_not_found) != len_old_df_smiles_not_found:
        log.fatal("Need to run new cirpy!!")

    df_smiles_not_found = openbabel_convert_smiles_to_inchi_with_nans(
        col_names_smiles_to_inchi=["smiles_from_cas_cirpy"],
        df=df_smiles_not_found,
    )
    # Check if smiles from cirpy and pubchem match
    df_smiles_not_found = get_inchi_main_layer(
        df=df_smiles_not_found, inchi_col="inchi_from_isomeric_smiles_pubchem", layers=4
    )
    df_smiles_not_found = get_inchi_main_layer(
        df=df_smiles_not_found, inchi_col="inchi_from_smiles_from_cas_cirpy", layers=4
    )
    df_pubchem_cirpy_match = df_smiles_not_found[
        (
            df_smiles_not_found["inchi_from_isomeric_smiles_pubchem_main_layer"]
            == df_smiles_not_found["inchi_from_smiles_from_cas_cirpy_main_layer"]
        )
        & (df_smiles_not_found["inchi_from_isomeric_smiles_pubchem_main_layer"] != "")
        & (df_smiles_not_found["inchi_from_smiles_from_cas_cirpy_main_layer"] != "")
    ]
    df_pubchem_cirpy_match = df_pubchem_cirpy_match[
        ["name", "cas", "isomeric_smiles_pubchem", "inchi_from_isomeric_smiles_pubchem"]
    ].rename(
        columns={
            "isomeric_smiles_pubchem": "smiles",
            "inchi_from_isomeric_smiles_pubchem": "inchi",
        }
    )  # if inchi from pubchem and cirpy match, take smiles from pubchem
    df_smiles_found = pd.concat([df_smiles_found, df_pubchem_cirpy_match])
    df_smiles_not_found = df_smiles_not_found[~(df_smiles_not_found["cas"].isin(df_smiles_found["cas"]))]
    assert len(df_smiles_found) + len(df_smiles_not_found) == len(df)

    text_to_df = {
        "CAS RN for which InChI main layer from PubChem and Comptox match": df_match_inchi_main_layer_pubchem_comptox,
        "Entries for which SMILES only in comptox found and this matches with paper": df_comptox_paper_match,
        "Entries for which SMILES only in pubchem found and this matches with cirpy": df_pubchem_cirpy_match,
    }
    for text, df in text_to_df.items():
        log.info(f"{text}", entries=len(df))
    return df_smiles_found, df_smiles_not_found


def get_full_df_b(
    df_b_full_original: pd.DataFrame,
    df_one_component_smiles_found: pd.DataFrame,
    df_multiple_components_smiles_found: pd.DataFrame,
    df_multiple_components_smiles_not_found: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df_one_component_smiles_found.rename(
        columns={"smiles_paper": "smiles"},
        inplace=True,
    )
    found = pd.concat([df_one_component_smiles_found, df_multiple_components_smiles_found])
    assert len(found) == len(df_one_component_smiles_found) + len(df_multiple_components_smiles_found)
    df_smiles_found_full = df_b_full_original[df_b_full_original["cas"].isin(list(found["cas"]))].drop(
        labels=["smiles"], axis=1
    )

    def get_found_smiles(row):
        cas = row["cas"]
        match = found[found["cas"] == cas]
        new_smiles = match["smiles"].values[0]
        return new_smiles

    df_smiles_found_full["smiles"] = df_smiles_found_full.apply(get_found_smiles, axis=1)
    df_smiles_found_full = remove_smiles_with_incorrect_format(df=df_smiles_found_full, col_name_smiles="smiles")
    df_smiles_found_full = openbabel_convert(
        df=df_smiles_found_full,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )

    df_multiple_components_smiles_not_found = df_multiple_components_smiles_not_found.drop(
        ["inchi_from_ccc", "inchi_pubchem", "inchi_comptox", "inchi_from_cas_cirpy"], axis=1
    ).reset_index(drop=True)
    df_multiple_components_smiles_not_found.to_csv("datasets/data_processing/substances_with_no_confirmed_smiles.csv")
    df_smiles_not_found_full = df_b_full_original[
        df_b_full_original["cas"].isin(list(df_multiple_components_smiles_not_found["cas"]))
    ]

    assert (len(df_smiles_found_full) + len(df_smiles_not_found_full)) == len(df_b_full_original)

    return df_smiles_found_full, df_smiles_not_found_full


def get_dfs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_regression_df()
    cols = [
        "name",
        "name_type",
        "cas",
        "smiles",
        "reliability",
        "endpoint",
        "guideline",
        "principle",
        "time_day",
        "biodegradation_percent",
    ]
    cols += get_speciation_col_names()
    df = df[cols]
    df_unique_cas = get_df_with_unique_cas(df)
    df_checked = load_gluege_data()
    return df, df_unique_cas, df_checked


def load_datasets() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    df_reg, df_unique_cas, df_checked = get_dfs()
    df_reg = remove_smiles_with_incorrect_format(df=df_reg, col_name_smiles="smiles")
    df_reg = openbabel_convert(df=df_reg, input_type="smiles", column_name_input="smiles", output_type="inchi")
    log.info("Entries in regression df", entries=len(df_reg), unique_cas=len(df_unique_cas), unique_smiles=df_reg.smiles.nunique(), unique_inchi=df_reg.inchi_from_smiles.nunique())
    
    inchi_groups = df_reg.groupby('inchi_from_smiles')['cas'].nunique()
    num_inchi_with_multiple_cas = inchi_groups[inchi_groups > 1].count()
    log.info("Number of InChI that are associated with multiple CAS RN", num_inchi_with_multiple_cas=num_inchi_with_multiple_cas)

    cas_groups = df_reg.groupby('cas')['inchi_from_smiles'].nunique()
    num_cas_with_multiple_inchi = cas_groups[cas_groups > 1].count()
    log.info("Number of CAS RN that are associated with multiple InChI", num_cas_with_multiple_inchi=num_cas_with_multiple_inchi)


    df_a, df_b, df_a_full_original, df_b_full_original = get_df_a_and_b(
        df_unique_cas=df_unique_cas, df_checked=df_checked, df_reg=df_reg
    )
    df_name_to_df = {
        "df_a (unique CAS RN checked by Gluege)": df_a,
        "df_b (unique CAS RN NOT checked by Gluege)": df_b,
        "df_a_full (datapoints checked by Gluege)": df_a_full_original,
        "df_b_full (datapoints NOT checked by Gluege)": df_b_full_original,
    }
    for df_name, df in df_name_to_df.items():
        log.info(f"Entries in {df_name}", df_a=len(df))
    log.info(
        "Total number of datapoints: ",
        entries=len(df_a_full_original) + len(df_b_full_original),
    )
    return (
        df_checked,
        df_a,
        df_b,
        df_a_full_original,
        df_b_full_original,
    )


def process_data_checked_by_gluege(
    df_a_full_original: pd.DataFrame, df_checked: pd.DataFrame
) -> pd.DataFrame:
    df_a_full = replace_smiles_with_smiles_gluege(df_a_full_original, df_checked)
    df_a_full = remove_studies_where_registered_substance_dont_match_reference_substance(df_a_full)
    return df_a_full


def process_data_not_checked_by_gluege(
    df_b: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_one_component, df_multiple_components = check_number_of_components(df_b)
    (
        df_smiles_found,
        df_smiles_not_found,
        df_problematic_multiple_components,
    ) = process_one_component_data(df_b=df_one_component, new_cas_common_chemistry=args.new_cas_common_chemistry)
    df_multiple = pd.concat(
        [
            df_multiple_components,
            df_smiles_not_found,
            df_problematic_multiple_components,
        ],
        ignore_index=True,
    )
    log.info("Need to run process_multiple_component_data for this many datapoints: ", entries=len(df_multiple))
    log.info(
        "These datapoints come from",
        df_multiple_components=len(df_multiple_components),
        df_smiles_not_found_cascc=len(df_smiles_not_found),
        df_problematic_multiple_components=len(df_problematic_multiple_components),
    )

    (
        df_multiple_components_smiles_found_first,
        df_multiple_components_smiles_not_found_first,
    ) = process_multiple_component_data(
        df_multiple_components=df_multiple,
        new_pubchem=args.new_pubchem,
        new_comptox=args.new_comptox,
    )
    (
        df_multiple_components_smiles_found_second,
        df_multiple_components_smiles_not_found,
    ) = process_multiple_components_smiles_not_found(
        df_multiple_components_smiles_not_found_first, new_cirpy=args.new_cirpy
    )
    df_multiple_components_smiles_found = pd.concat(
        [df_multiple_components_smiles_found_first, df_multiple_components_smiles_found_second]
    )
    log.info("Multiple component data found", found=len(df_multiple_components_smiles_found))
    log.info("Multiple component data NOT found", found=len(df_multiple_components_smiles_not_found))
    return (
        df_smiles_found,
        df_multiple_components_smiles_found,
        df_multiple_components_smiles_not_found,
    )


def process_full_dataset(
    df_a_full: pd.DataFrame,
    df_b_full_original: pd.DataFrame,
    df_one_component_smiles_found: pd.DataFrame,
    df_multiple_components_smiles_found: pd.DataFrame,
    df_multiple_components_smiles_not_found: pd.DataFrame,
) -> pd.DataFrame:
    df_b_found_full, df_smiles_not_found_full = get_full_df_b(
        df_b_full_original,
        df_one_component_smiles_found,
        df_multiple_components_smiles_found,
        df_multiple_components_smiles_not_found,
    )
    log.info("Entries in df_a_full after processing", df_a_full=len(df_a_full), unique_cas=df_a_full.cas.nunique(), unique_inchi=df_a_full.inchi_from_smiles.nunique())
    log.info("Entries in df_b_full after processing", df_b_full=len(df_b_found_full), unique_cas=df_b_found_full.cas.nunique(), unique_inchi=df_b_found_full.inchi_from_smiles.nunique())
    log.info("Dataset for which SMILES not found", datapoints=len(df_smiles_not_found_full))
    df_full = pd.concat([df_a_full, df_b_found_full])
    log.info("Entries in df_full", df_full=len(df_full), unique_cas=df_full.cas.nunique(), unique_inchi=df_full.inchi_from_smiles.nunique())

    log.info("Replacing SMILES with SMILES with chemical speciation")
    df_full_with_chemical_speciation, _ = replace_smiles_with_smiles_with_chemical_speciation(df_full.copy())
    log.info("Dataset size after replacing SMILES with chemical speciation", entries=len(df_full_with_chemical_speciation), unique_cas=df_full_with_chemical_speciation.cas.nunique(), unique_inchi=df_full_with_chemical_speciation.inchi_from_smiles.nunique())

    log.info("Replacing multiple CAS RN for one InChI")
    df_full = replace_multiple_cas_for_one_inchi(df_full, prnt=False)
    log.info("Number of substances after replacing multiple CAS RN for one InChI for df_full", unique_cas=df_full.cas.nunique(), unique_inchi=df_full.inchi_from_smiles.nunique())
    df_full_with_chemical_speciation = replace_multiple_cas_for_one_inchi(df_full_with_chemical_speciation, prnt=False)
    log.info("Number of substances after replacing multiple CAS RN for one InChI for df_full_with_chemical_speciation", unique_cas=df_full_with_chemical_speciation.cas.nunique(), unique_inchi=df_full_with_chemical_speciation.inchi_from_smiles.nunique())

    df_full, _ = remove_organo_metals_function(df=df_full, smiles_column="smiles")
    log.info("Number of substances after removing organo metals for df_full", unique_cas=df_full.cas.nunique(), unique_inchi=df_full.inchi_from_smiles.nunique())
    df_full_with_chemical_speciation, _ = remove_organo_metals_function(df=df_full_with_chemical_speciation, smiles_column="smiles")
    log.info("Number of substances after after removing organo metals for df_full_with_chemical_speciation", unique_cas=df_full_with_chemical_speciation.cas.nunique(), unique_inchi=df_full_with_chemical_speciation.inchi_from_smiles.nunique())
    return df_full, df_full_with_chemical_speciation


def aggregate_duplicates(df: pd.DataFrame):
    aggregation_functions = {
        "smiles": "first",
        "biodegradation_percent": "mean",
        "inchi_from_smiles": "first",
    }
    if "pka_acid_1" in df.columns:
        aggregation_functions = {
            "smiles": "first",
            "biodegradation_percent": "mean",
            "inchi_from_smiles": "first",
            "pka_acid_1": "first",
            "pka_acid_2": "first",
            "pka_acid_3": "first",
            "pka_acid_4": "first",
            "pka_base_1": "first",
            "pka_base_2": "first",
            "pka_base_3": "first",
            "pka_base_4": "first",
            "α_acid_0": "first",
            "α_acid_1": "first",
            "α_acid_2": "first",
            "α_acid_3": "first",
            "α_acid_4": "first",
            "α_base_0": "first",
            "α_base_1": "first",
            "α_base_2": "first",
            "α_base_3": "first",
            "α_base_4": "first",
        }
    df = (
        df.groupby(["cas", "reliability", "endpoint", "guideline", "principle", "time_day"])
        .aggregate(aggregation_functions)
        .reset_index()
    )
    return df


if __name__ == "__main__":

    df_checked, _, df_b, df_a_full_original, df_b_full_original = load_datasets()

    log.info(" \n Processing data checked by Gluege et al.")
    df_a_full = process_data_checked_by_gluege(df_a_full_original=df_a_full_original, df_checked=df_checked)

    log.info(" \n Processing data NOT checked by Gluege et al.")
    (
        df_one_component_smiles_found,
        df_multiple_components_smiles_found,
        df_multiple_components_smiles_not_found,
    ) = process_data_not_checked_by_gluege(df_b=df_b)

    log.info(" \n Full dataset")
    df_full, df_full_with_chemical_speciation = process_full_dataset(
        df_a_full=df_a_full,
        df_b_full_original=df_b_full_original,
        df_one_component_smiles_found=df_one_component_smiles_found,
        df_multiple_components_smiles_found=df_multiple_components_smiles_found,
        df_multiple_components_smiles_not_found=df_multiple_components_smiles_not_found,
    )

    cols_to_keep = [
        "cas",
        "smiles",
        "reliability",
        "endpoint",
        "guideline",
        "principle",
        "time_day",
        "biodegradation_percent",
        "inchi_from_smiles",
    ]
    cols_to_keep += get_speciation_col_names()
    df_full = df_full[cols_to_keep]
    df_full_with_chemical_speciation = df_full_with_chemical_speciation[cols_to_keep]

    df_full_agg = aggregate_duplicates(df=df_full)
    log.info("Entries in df_full_agg after aggregating", entries=len(df_full_agg), unique_cas=df_full_agg.cas.nunique(), unique_inchi=df_full_agg.inchi_from_smiles.nunique())
    df_full_with_chemical_speciation_agg = aggregate_duplicates(df=df_full_with_chemical_speciation)
    log.info("Entries in df_full_with_chemical_speciation_agg after aggregating", entries=len(df_full_with_chemical_speciation_agg), unique_cas=df_full_with_chemical_speciation_agg.cas.nunique(), unique_inchi=df_full_with_chemical_speciation_agg.inchi_from_smiles.nunique())

    dfs = {
        "df_full": df_full,
        "df_full_agg": df_full_agg,
        "df_full_with_chemical_speciation": df_full_with_chemical_speciation,
        "df_full_with_chemical_speciation_agg": df_full_with_chemical_speciation_agg,
    }
    for df_name, df in dfs.items():
        log.info(f"Entries in {df_name}", entries=len(df), unique_cas=df.cas.nunique(), unique_inchi=df.inchi_from_smiles.nunique())

    df_full_agg.to_csv("datasets/data_processing/reg_curated_s_no_metal.csv")
    df_full_with_chemical_speciation_agg.to_csv("datasets/data_processing/reg_curated_scs_no_metal.csv")
