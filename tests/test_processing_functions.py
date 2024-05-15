import sys
import os
from typing import List
import numpy as np
import pandas as pd
import collections
import re
import pandas.api.types as ptypes
from pandas.api.types import is_numeric_dtype

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from code_files.processing_functions import openbabel_convert
from code_files.processing_functions import remove_smiles_with_incorrect_format
from code_files.processing_functions import label_data_based_on_percentage
from code_files.processing_functions import group_and_label_chemicals
from code_files.processing_functions import get_cid_from_inchi_pubchempy
from code_files.processing_functions import go_to_values
from code_files.processing_functions import go_to_references
from code_files.processing_functions import get_all_cas_pubchem
from code_files.processing_functions import get_deprecated_cas
from code_files.processing_functions import is_cas_right_format
from code_files.processing_functions import get_inchi_layers
from code_files.processing_functions import add_cas_from_pubchem
from code_files.processing_functions import pubchem_cas_to_ref_dict
from code_files.processing_functions import find_best_cas_pubchem_based_on_ref
from code_files.processing_functions import remove_organo_metals_function
from code_files.processing_functions import get_smiles_from_cas_pubchempy
from code_files.processing_functions import turn_cas_column_into_string
from code_files.processing_functions import get_comptox
from code_files.processing_functions import load_comptox_file_and_save
from code_files.processing_functions import get_smiles_from_cas_comptox
from code_files.processing_functions import get_info_cas_common_chemistry
from code_files.processing_functions import add_biowin_label
from code_files.processing_functions import remove_selected_biowin
from code_files.processing_functions import process_df_biowin
from code_files.processing_functions import replace_smiles_with_smiles_with_chemical_speciation
from code_files.processing_functions import replace_multiple_cas_for_one_inchi
from code_files.processing_functions import load_regression_df
from code_files.processing_functions import format_class_data_paper
from code_files.processing_functions import load_class_data_paper
from code_files.processing_functions import reg_df_remove_inherent_only_28_for_classification
from code_files.processing_functions import check_if_cas_match_pubchem
from code_files.processing_functions import drop_rows_without_matching_cas
from code_files.processing_functions import process_external_dataset_lunghini
from code_files.processing_functions import get_external_dataset_lunghini
from code_files.processing_functions import assign_group_label_and_drop_replicates 
from code_files.processing_functions import reg_df_remove_studies_not_to_consider
from code_files.processing_functions import create_classification_data_based_on_regression_data
from code_files.processing_functions import create_classification_biowin
from code_files.processing_functions import create_input_regression
from code_files.processing_functions import encode_categorical_data
from code_files.processing_functions import bit_vec_to_lst_of_lst
from code_files.processing_functions import create_input_classification
from code_files.processing_functions import convert_to_morgan_fingerprints
from code_files.processing_functions import convert_to_maccs_fingerprints 
from code_files.processing_functions import convert_to_rdk_fingerprints
from code_files.processing_functions import get_speciation_col_names
from code_files.processing_functions import convert_regression_df_to_input
from code_files.processing_functions import load_gluege_data
from code_files.processing_functions import check_number_of_components
from code_files.processing_functions import openbabel_convert_smiles_to_inchi_with_nans
from code_files.processing_functions import get_inchi_main_layer
from code_files.processing_functions import get_smiles_inchi_cirpy
from code_files.processing_functions import further_processing_of_echa_data
from code_files.processing_functions import load_and_process_echa_additional
from code_files.processing_functions import get_df_with_unique_cas



def test_openbabel_convert(regression_paper):
    df_output = openbabel_convert(
        df=regression_paper,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )
    assert "inchi_from_smiles" in df_output.columns
    assert list(df_output["inchi_from_smiles"]) == list(df_output["inchi_from_smiles"])
    df_output = openbabel_convert(
        df=regression_paper,
        input_type="inchi",
        column_name_input="inchi_from_smiles",
        output_type="smiles",
    )
    assert "smiles_from_inchi_from_smiles" in df_output.columns


def test_remove_smiles_with_incorrect_format(regression_paper_with_star):
    len_before = len(regression_paper_with_star)
    regression_paper_with_star = remove_smiles_with_incorrect_format(
        df=regression_paper_with_star, col_name_smiles="smiles"
    )
    len_after = len(regression_paper_with_star)
    assert len_before > len_after
    assert len_after == 7
    assert "8029-68-3" not in regression_paper_with_star["cas"]
    assert "1330-39-8" not in regression_paper_with_star["cas"]


def test_label_data_based_on_percentage(class_for_labelling):
    df_additional = label_data_based_on_percentage(class_for_labelling)
    assert "y_true" in list(df_additional.columns)
    assert list(df_additional["correct_label"]) == list(df_additional["y_true"])


def test_group_and_label_chemicals(group_remove_duplicates):
    df_output = openbabel_convert(
        df=group_remove_duplicates,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )
    df = group_and_label_chemicals(df=df_output, col_to_group_by="inchi_from_smiles")
    assert df["inchi_from_smiles"].nunique() == len(df)
    assert df["cas"].to_list() == [
        "79-41-4",
        "268567-32-4",
        "534-52-1",
        "94232-67-4",
        "137863-20-8",
    ]
    assert df["y_true"].to_list() == [1, 0, 0, 0, 0]


def test_get_cid_from_inchi_pubchempy():
    df_lunghini = pd.read_csv("datasets/external_data/lunghini.csv", sep=";", index_col=0)[:6]
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
    assert "cid" in list(df.columns)
    assert list(df["cid"]) == [
        5314126.0,
        643776.0,
        5369379.0,
        5558.0,
        6588.0,
        3035339.0,
    ]


def test_get_all_cas_pubchem():
    cids = [5314126.0, 643776.0, 5369379.0, 5558.0, 6588.0, 3035339.0]
    unique_cass = []
    ref_strs = []
    for cid in cids:
        unique_cas, ref_str = get_all_cas_pubchem(cid=cid)
        unique_cass.append(unique_cas)
        ref_strs.append(ref_str)
    assert unique_cass == [
        "588-72-7, 1335-06-4, 103-64-0",
        "590-11-4, 540-49-8",
        "588-73-8, 1335-06-4, 588-72-7",
        "75-25-2, 4471-18-5",
        "79-27-6",
        "30171-80-3",
    ]
    assert ref_strs == [
        "588-72-7: CAS Common Chemistry, ChemIDplus, EPA DSSTox, FDA Global Substance Registration System (GSRS); 103-64-0: ChemIDplus, DTP/NCI, EPA Chemicals under the TSCA, European Chemicals Agency (ECHA), Hazardous Substances Data Bank (HSDB); 1335-06-4: European Chemicals Agency (ECHA)",
        "590-11-4: ChemIDplus, EPA DSSTox, FDA Global Substance Registration System (GSRS); 540-49-8: DTP/NCI",
        "588-73-8: ChemIDplus, DTP/NCI, FDA Global Substance Registration System (GSRS); 1335-06-4: ChemIDplus, EPA Chemicals under the TSCA, European Chemicals Agency (ECHA); 588-72-7: DTP/NCI",
        "75-25-2: CAMEO Chemicals, CAS Common Chemistry, ChemIDplus, DrugBank, DTP/NCI, EPA Chemicals under the TSCA, EPA DSSTox, European Chemicals Agency (ECHA), FDA Global Substance Registration System (GSRS), Hazardous Substances Data Bank (HSDB), ILO-WHO International Chemical Safety Cards (ICSCs), Occupational Safety and Health Administration (OSHA), The National Institute for Occupational Safety and Health (NIOSH); 4471-18-5: ChemIDplus",
        "79-27-6: CAMEO Chemicals, CAS Common Chemistry, ChemIDplus, DTP/NCI, EPA Chemicals under the TSCA, EPA DSSTox, European Chemicals Agency (ECHA), FDA Global Substance Registration System (GSRS), Hazardous Substances Data Bank (HSDB), ILO-WHO International Chemical Safety Cards (ICSCs), Occupational Safety and Health Administration (OSHA), The National Institute for Occupational Safety and Health (NIOSH)",
        "30171-80-3: ChemIDplus, EPA DSSTox, European Chemicals Agency (ECHA)",
    ]


def test_get_deprecated_cas():
    cids = [5314126.0, 643776.0, 5369379.0, 5558.0, 6588.0, 3035339.0]
    deprecated_cass = []
    for cid in cids:
        deprecated_cas = get_deprecated_cas(cid)
        deprecated_cass.append(deprecated_cas)
    assert deprecated_cass == ["1340-14-3, 1340-14-3", "", "41380-64-7", "", "", ""]


def test_is_cas_right_format():
    cass = [
        "103-64-0",
        "000103-64-0",
        "540-49-8",
        "1335-06-4",
        "75-25-2",
        "000075-25-2",
        "79-27-6",
        "000079-27-6",
        "1335-06-4",
        "NOCAS_865529",
    ]
    right_format_cas = [x for x in cass if is_cas_right_format(x)]
    assert right_format_cas == [
        "103-64-0",
        "000103-64-0",
        "540-49-8",
        "1335-06-4",
        "75-25-2",
        "000075-25-2",
        "79-27-6",
        "000079-27-6",
        "1335-06-4",
    ]


def test_get_inchi_layers(df_lunghini_added_cas):
    def get_inchi_main_layer(row):
        inchi_main_layer_smiles = get_inchi_layers(row, col_name="inchi_from_smiles", layers=4)
        return inchi_main_layer_smiles

    df_lunghini_added_cas["inchi_main_layer"] = df_lunghini_added_cas.apply(get_inchi_main_layer, axis=1)
    assert list(df_lunghini_added_cas["inchi_main_layer"]) == [
        "InChI=1S/C8H7Br/c9-7-6-8-4-2-1-3-5-8/h1-7H",
        "InChI=1S/C2H2Br2/c3-1-2-4/h1-2H",
        "InChI=1S/C8H7Br/c9-7-6-8-4-2-1-3-5-8/h1-7H",
        "InChI=1S/CHBr3/c2-1(3)4/h1H",
        "InChI=1S/C2H2Br4/c3-1(4)2(5)6/h1-2H",
        "InChI=1S/C10H10Br2O2/c11-10(12)8-3-1-2-4-9(8)14-6-7-5-13-7/h1-4,7,10H,5-6H2",
    ]


def test_add_cas_from_pubchem():
    df = pd.read_csv("datasets/external_data/lunghini_added_cids.csv")[:6]
    df = add_cas_from_pubchem(df=df)
    for col in ["cas_pubchem", "cas_ref_pubchem", "deprecated_cas_pubchem"]:
        assert col in list(df.columns)
    assert list(df["cas_pubchem"]) == [
        "588-72-7, 1335-06-4, 103-64-0",
        "590-11-4, 540-49-8",
        "588-73-8, 1335-06-4, 588-72-7",
        "75-25-2, 4471-18-5",
        "79-27-6",
        "30171-80-3",
    ]
    assert list(df["cas_ref_pubchem"]) == [
        "588-72-7: CAS Common Chemistry, ChemIDplus, EPA DSSTox, FDA Global Substance Registration System (GSRS); 103-64-0: ChemIDplus, DTP/NCI, EPA Chemicals under the TSCA, European Chemicals Agency (ECHA), Hazardous Substances Data Bank (HSDB); 1335-06-4: European Chemicals Agency (ECHA)",
        "590-11-4: ChemIDplus, EPA DSSTox, FDA Global Substance Registration System (GSRS); 540-49-8: DTP/NCI",
        "588-73-8: ChemIDplus, DTP/NCI, FDA Global Substance Registration System (GSRS); 1335-06-4: ChemIDplus, EPA Chemicals under the TSCA, European Chemicals Agency (ECHA); 588-72-7: DTP/NCI",
        "75-25-2: CAMEO Chemicals, CAS Common Chemistry, ChemIDplus, DrugBank, DTP/NCI, EPA Chemicals under the TSCA, EPA DSSTox, European Chemicals Agency (ECHA), FDA Global Substance Registration System (GSRS), Hazardous Substances Data Bank (HSDB), ILO-WHO International Chemical Safety Cards (ICSCs), Occupational Safety and Health Administration (OSHA), The National Institute for Occupational Safety and Health (NIOSH); 4471-18-5: ChemIDplus",
        "79-27-6: CAMEO Chemicals, CAS Common Chemistry, ChemIDplus, DTP/NCI, EPA Chemicals under the TSCA, EPA DSSTox, European Chemicals Agency (ECHA), FDA Global Substance Registration System (GSRS), Hazardous Substances Data Bank (HSDB), ILO-WHO International Chemical Safety Cards (ICSCs), Occupational Safety and Health Administration (OSHA), The National Institute for Occupational Safety and Health (NIOSH)",
        "30171-80-3: ChemIDplus, EPA DSSTox, European Chemicals Agency (ECHA)",
    ]
    assert list(df["deprecated_cas_pubchem"]) == [
        "1340-14-3",
        "",
        "41380-64-7",
        "",
        "",
        "",
    ]


def test_pubchem_cas_to_ref_dict(df_lunghini_added_cas):
    def get_ref(row):
        cas_to_ref_pubchem = pubchem_cas_to_ref_dict(row["cas_ref_pubchem"])
        return cas_to_ref_pubchem

    df_lunghini_added_cas["cas_to_ref"] = df_lunghini_added_cas.apply(get_ref, axis=1)
    assert list(df_lunghini_added_cas["cas_to_ref"]) == [
        {
            "588-72-7": "CAS Common Chemistry, ChemIDplus, FDA Global Substance Registration System (GSRS)",
            "103-64-0": "ChemIDplus, DTP/NCI, EPA Chemicals under the TSCA, EPA DSSTox, Hazardous Substances Data Bank (HSDB)",
            "1335-06-4": "European Chemicals Agency (ECHA)",
        },
        {
            "590-11-4": "ChemIDplus, EPA DSSTox, FDA Global Substance Registration System (GSRS)",
            "540-49-8": "DTP/NCI",
        },
        {
            "588-73-8": "ChemIDplus, DTP/NCI, FDA Global Substance Registration System (GSRS)",
            "1335-06-4": "ChemIDplus, EPA Chemicals under the TSCA, EPA DSSTox, European Chemicals Agency (ECHA)",
            "588-72-7": "DTP/NCI",
        },
        {
            "75-25-2": "CAMEO Chemicals, CAS Common Chemistry, ChemIDplus, DrugBank, DTP/NCI, EPA Chemicals under the TSCA, EPA DSSTox, European Chemicals Agency (ECHA), FDA Global Substance Registration System (GSRS), Hazardous Substances Data Bank (HSDB), ILO International Chemical Safety Cards (ICSC), Occupational Safety and Health Administration (OSHA), The National Institute for Occupational Safety and Health (NIOSH)",
            "4471-18-5": "ChemIDplus",
        },
        {
            "79-27-6": "CAMEO Chemicals, CAS Common Chemistry, ChemIDplus, DTP/NCI, EPA Chemicals under the TSCA, EPA DSSTox, European Chemicals Agency (ECHA), FDA Global Substance Registration System (GSRS), Hazardous Substances Data Bank (HSDB), ILO International Chemical Safety Cards (ICSC), Occupational Safety and Health Administration (OSHA), The National Institute for Occupational Safety and Health (NIOSH)"
        },
        {"30171-80-3": "ChemIDplus, EPA DSSTox, European Chemicals Agency (ECHA)"},
    ]


def test_find_best_cas_pubchem_based_on_ref(df_lunghini_added_cas):
    def get_ref(row):
        cas_to_ref_pubchem = pubchem_cas_to_ref_dict(row["cas_ref_pubchem"])
        best_cas = find_best_cas_pubchem_based_on_ref(cas_to_ref_pubchem)
        return best_cas

    df_lunghini_added_cas["best_cas"] = df_lunghini_added_cas.apply(get_ref, axis=1)
    assert list(df_lunghini_added_cas["best_cas"]) == [
        "588-72-7",
        "590-11-4",
        "1335-06-4",
        "75-25-2",
        "79-27-6",
        "30171-80-3",
    ]


def test_remove_organo_metals_function(metal_smiles):
    none_organo_metals, organo_metals = remove_organo_metals_function(df=metal_smiles, smiles_column="smiles")
    assert list(none_organo_metals.smiles) == [
        "BrN1C(=O)CCC1=O",
        "C=CC(=O)OCCCCOC(=O)OC1=CC=C(C=C1)C(=O)C1=CC=CC=C1",
        "CCCCCCCC(=O)NO",
        "CCOc1ccccc1C(N)=N",
        "CCCCCCCCCCCCCCCCCCn1c(=O)c2ccc3sc4ccccc4c4ccc(c2c34)c1=O",
        "O=c1[nH]cc(F)c(=O)[nH]1",
    ]
    metals = [
        "Be",
        "B",
        "Mg",
        "Al",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
    ]

    def find_organo_metals_by_row(row):
        smiles = row["smiles"]
        components: set[str] = set()
        if smiles != np.nan:
            regex = re.compile("[^a-zA-Z]")
            smiles = regex.sub("", smiles)
            elements = re.findall("[A-Z][^A-Z]*", smiles)
            for item in set(elements):
                components.add(item)
        components = {i for i in components if i in metals}
        return len(components) > 0

    none_organo_metals = none_organo_metals.copy()
    none_organo_metals["organo_metal"] = none_organo_metals.apply(find_organo_metals_by_row, axis=1)
    organo_metals_in_none_organo_metals = none_organo_metals[none_organo_metals["organo_metal"] == True]
    assert len(organo_metals_in_none_organo_metals) == 0


def test_get_smiles_from_cas_pubchempy(df_b):
    _, df_multiple_components = check_number_of_components(df_b)
    df_pubchem = get_smiles_from_cas_pubchempy(df_multiple_components)
    assert [
        "name",
        "cas",
        "smiles_paper",
        "isomeric_smiles_pubchem",
        "canonical_smiles_pubchem",
        "inchi_pubchem",
    ] == list(df_pubchem.columns)
    assert len(df_pubchem) == len(df_multiple_components)
    assert list(df_pubchem["cas"]) == [
        "132-16-1",
        "7775-09-9",
        "94891-43-7",
        "8029-68-3",
    ]
    assert list(df_pubchem["isomeric_smiles_pubchem"]) == [
        "C1=CC=C2C(=C1)C3=NC4=NC(=NC5=C6C=CC=CC6=C([N-]5)N=C7C8=CC=CC=C8C(=N7)N=C2[N-]3)C9=CC=CC=C94.[Fe+2]",
        "[O-]Cl(=O)=O.[Na+]",
        "",
        "C(=O)(N)NO",
    ]
    assert list(df_pubchem["canonical_smiles_pubchem"]) == [
        "",
        "[O-]Cl(=O)=O.[Na+]",
        "",
        "C(=O)(N)NO",
    ]
    assert list(df_pubchem["inchi_pubchem"]) == [
        "",
        "InChI=1S/ClHO3.Na/c2-1(3)4;/h(H,2,3,4);/q;+1/p-1",
        "",
        "InChI=1S/CH4N2O2/c2-1(4)3-5/h5H,(H3,2,3,4)",
    ]


def test_turn_cas_column_into_string(df_pubchem):
    cas_string = turn_cas_column_into_string(df=df_pubchem, cas_column="cas")

    def check_cas_string(row):
        assert row["cas"] in cas_string

    df_pubchem.apply(check_cas_string, axis=1)


def test_get_smiles_from_cas_comptox(df_pubchem):
    df_comptox = get_smiles_from_cas_comptox(df=df_pubchem)
    assert list(df_comptox["smiles_comptox"]) == [
        np.nan,
        r"NC1CCCC2=C1C=CC=C2",
        r"[Co++].[N-]1\C2=N/C3=N/C(=N\C4=C5C=CC=CC5=C([N-]4)\N=C4/N=C(/N=C1/C1=CC=CC=C21)C1=CC=CC=C41)/C1=CC=CC=C31",
        r"[H+].[Cr+3].[O-]C1=CC=C(Cl)C=C1\N=N\C1=C2C=CC=CC2=CC=C1[O-].[O-]C1=CC=C(Cl)C=C1\N=N\C1=C2C=CC=CC2=CC=C1[O-]",
        r"[Cu].I.C1=CC=C(C=C1)P(C1=CC=CC=C1)C1=CC=CC=C1",
    ]


def test_get_info_cas_common_chemistry(df_b):
    df_one_component, _ = check_number_of_components(df_b)
    cass = list(df_one_component["cas"])
    smiless: List[str] = []
    inchis: List[str] = []
    for cas in cass:
        smiles, inchi = get_info_cas_common_chemistry(cas)
        smiless.append(smiles)
        inchis.append(inchi)
    assert cass == [
        "31221-06-4",
        "875-74-1",
        "2579-20-6",
        "112-24-3",
        "4605-14-5",
        "13463-41-7",
    ]
    assert smiless == [
        "[N+](=[N-])=C1C(=O)NC(=O)NC1=O",
        "[C@@H](C(O)=O)(N)C1=CC=CC=C1",
        "C(N)C1CC(CN)CCC1",
        "C(CNCCN)NCCN",
        "C(CNCCCN)CNCCCN",
        "[Zn+2]12([O-]N3C(=[S]1)C=CC=C3)[O-]N4C(=[S]2)C=CC=C4",
    ]
    assert inchis == [
        "InChI=1S/C4H2N4O3/c5-8-1-2(9)6-4(11)7-3(1)10/h(H2,6,7,9,10,11)",
        "InChI=1S/C8H9NO2/c9-7(8(10)11)6-4-2-1-3-5-6/h1-5,7H,9H2,(H,10,11)",
        "InChI=1S/C8H18N2/c9-5-7-2-1-3-8(4-7)6-10/h7-8H,1-6,9-10H2",
        "InChI=1S/C6H18N4/c7-1-3-9-5-6-10-4-2-8/h9-10H,1-8H2",
        "InChI=1S/C9H24N4/c10-4-1-6-12-8-3-9-13-7-2-5-11/h12-13H,1-11H2",
        "InChI=1S/2C5H4NOS.Zn/c2*7-6-4-2-1-3-5(6)8;/h2*1-4H;/q2*-1;+2",
    ]


def test_add_biowin_label(df_curated_class, df_curated_reg):
    df_curated_reg = add_biowin_label(df=df_curated_reg, mode="reg")
    assert df_curated_reg["linear_label"].to_list() == [0.0, 0.0, 0.0, 1.0, "None"]
    assert df_curated_reg["non_linear_label"].to_list() == [0.0, 0.0, 0.0, 1.0, "None"]
    assert df_curated_reg["miti_linear_label"].to_list() == [0.0, 0.0, 0.0, 1.0, "None"]
    assert df_curated_reg["miti_non_linear_label"].to_list() == [0.0, 0.0, 0.0, 1.0, "None"]
    assert df_curated_reg["label"].to_list() == [0, 0, 0, 1, 0]

    df_curated_class = add_biowin_label(df=df_curated_class, mode="class")
    assert df_curated_class["linear_label"].to_list() == [1.0, 1.0, 0.0, 1.0, "None", 1.0, 1.0]
    assert df_curated_class["non_linear_label"].to_list() == [1.0, 1.0, 0.0, 1.0, "None", 1.0, 1.0]
    assert df_curated_class["miti_linear_label"].to_list() == [1.0, 1.0, 0.0, 1.0, "None", 0.0, 0.0]
    assert df_curated_class["miti_non_linear_label"].to_list() == [1.0, 1.0, 0.0, 1.0, "None", 1.0, 0.0]


def test_remove_selected_biowin(df_curated_class, df_curated_reg):
    df_curated_class = add_biowin_label(df=df_curated_class, mode="class")
    df_class, df_false_class = remove_selected_biowin(
        df=df_curated_class,
    )
    assert len(df_false_class[df_false_class["miti_linear_label"] == "None"]) == 0
    assert len(df_false_class[df_false_class["miti_non_linear_label"] == "None"]) == 0
    assert (len(df_class) + len(df_false_class)) == len(df_curated_class)
    assert df_class.cas.to_list() == ["100-09-4", "100-14-1", "100-37-8", "100-41-4"]
    assert df_false_class.cas.to_list() == ["100-10-7", "100-26-5", "100-40-3"]


def test_replace_smiles_with_env_relevant_smiles(class_curated):
    df_env, df_removed = replace_smiles_with_smiles_with_chemical_speciation(df_without_env_smiles=class_curated)
    assert df_env.smiles.to_list() == [
        "COC1=CC=C(C=C1)C([O-])=O",
        "CN(C)C1=CC=C(C=O)C=C1",
        "N(=O)(=O)C1=CC=C(CCl)C=C1",
        "[O-]C(=O)C1=CC=C(N=C1)C([O-])=O",
        "CC[NH+](CC)CCO",
    ]
    assert df_env.cas.to_list() == ["100-09-4", "100-10-7", "100-14-1", "100-26-5", "100-37-8"]
    assert df_removed.smiles.to_list() == ["delete"]
    assert df_removed.cas.to_list() == ["100-36-7"]
    assert len(df_env) == 5


def test_replace_multiple_cas_for_one_inchi(df_a_full, df_b_found_full):
    df_full = pd.concat([df_a_full, df_b_found_full])
    df_full = replace_multiple_cas_for_one_inchi(df=df_full, prnt=False)
    assert len(df_full) == len(df_a_full) + len(df_b_found_full)
    assert df_full["cas"].nunique() == df_full["inchi_from_smiles"].nunique()


def test_load_regression_df(regression_paper_full):
    new_regression_df = load_regression_df()
    col_names = list(new_regression_df.columns)
    target_col_names = list(regression_paper_full.columns)
    assert target_col_names == col_names


def test_format_class_data_paper(class_all_no_processing):
    len_before = len(class_all_no_processing)
    df = format_class_data_paper(class_all_no_processing)
    len_after = len(df)
    assert len_before == len_after
    assert list(df.columns) == [
        "name",
        "name_type",
        "cas",
        "source",
        "smiles",
        "y_true",
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


def test_load_class_data_paper():
    class_original, class_external, df_class = load_class_data_paper()
    assert len(class_original) == 4593
    assert list(class_original.columns) == [
        "name",
        "name_type",
        "cas",
        "source",
        "smiles",
        "y_true",
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
    assert len(class_external) == 1546
    assert list(class_external.columns) == [
        "smiles",
        "y_true",
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
    assert len(df_class) == 6139
    assert list(df_class.columns) == [
        "name",
        "name_type",
        "cas",
        "source",
        "smiles",
        "y_true",
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


def test_reg_df_remove_inherent_only_28_for_classification(regression_paper):
    df = reg_df_remove_inherent_only_28_for_classification(regression_paper)
    assert len(df[df["endpoint"] == "Inherent"]) == 0
    assert len(df[df["time_day"] != 28.0]) == 0


def test_check_if_cas_match_pubchem(df_lunghini_added_cas):
    df = group_and_label_chemicals(df=df_lunghini_added_cas, col_to_group_by="inchi_from_smiles")
    df = check_if_cas_match_pubchem(df)

    def get_df_cas_match(row):
        existing_cas = row["existing_cas"]
        best_cas = row["best_cas"]
        if len(existing_cas) > 0:
            if existing_cas[0] == best_cas:
                return "Yes"
        return "No"

    df["cas_match"] = df.apply(get_df_cas_match, axis=1)
    df_cas_match = df[df["cas_match"] == "Yes"]
    assert df_cas_match["match"].unique() == "Yes"

    def get_df_no_existing_cas(row):
        existing_cas = row["existing_cas"]
        best_cas = row["best_cas"]
        if (best_cas != "") & (existing_cas == []):
            return "Yes"
        return "No"

    df["no_existing_cas"] = df.apply(get_df_no_existing_cas, axis=1)
    df_no_existing_cas = df[df["no_existing_cas"] == "Yes"]
    assert df_no_existing_cas["match"].unique() == "Yes"

    def get_mismtach(row):
        existing_cas = row["existing_cas"]
        best_cas = row["best_cas"]
        if (best_cas != "") & (existing_cas != "") & (best_cas != existing_cas):
            return "Yes"
        return "No"

    df["mismatch"] = df.apply(get_mismtach, axis=1)
    df_mismatch = df[df["mismatch"] == "Yes"]
    assert df_mismatch["match"].unique() == "Yes"


def test_drop_rows_without_matching_cas(df_lunghini_added_cas):
    df = group_and_label_chemicals(df=df_lunghini_added_cas, col_to_group_by="inchi_from_smiles")
    df = check_if_cas_match_pubchem(df)
    df = drop_rows_without_matching_cas(df)
    assert "match" not in list(df.columns)


def test_get_external_dataset_lunghini(class_df):
    df_external_additional = get_external_dataset_lunghini(
        run_from_start=False, class_df=class_df, include_speciation_lunghini=True
    )
    assert len(class_df[class_df["cas"].isin(df_external_additional["cas"])]) == 0
    assert list(df_external_additional.columns) == [
        "cas",
        "smiles",
        "y_true",
        'inchi_from_smiles',
    ]


def test_assign_group_label_and_drop_replicates():
    curated_scs = pd.read_csv("datasets/data_processing/reg_curated_scs_no_metal.csv", index_col=0)[:100]

    df_included = reg_df_remove_studies_not_to_consider(curated_scs)
    df_labelled = label_data_based_on_percentage(df_included)
    columns = ["cas", "smiles", "principle", "biodegradation_percent", "y_true", "inchi_from_smiles"]
    df_class = pd.DataFrame(data=df_labelled.copy(), columns=columns)

    df_multiples, df_singles, df_removed_due_to_variance = assign_group_label_and_drop_replicates(
        df=df_class, by_column="inchi_from_smiles"
    )

    assert len(df_multiples) + len(df_singles) + len(df_removed_due_to_variance) == df_included["inchi_from_smiles"].nunique()
    
    for inchi in df_singles["inchi_from_smiles"]:
        df_current = df_included[df_included["inchi_from_smiles"] == inchi]
        assert len(df_current) == 1
        assert df_current["inchi_from_smiles"].nunique() == 1

    for inchi in df_multiples["inchi_from_smiles"]:
        df_current = df_included[df_included["inchi_from_smiles"] == inchi]
        assert len(df_current) > 1
        assert df_current["inchi_from_smiles"].nunique() == 1

    for inchi in df_removed_due_to_variance["inchi_from_smiles"]:
        df_current = df_included[df_included["inchi_from_smiles"] == inchi]
        assert len(df_current) > 1
        assert df_current["inchi_from_smiles"].nunique() != 1


def test_reg_df_remove_studies_not_to_consider(df_curated_reg):
    df_included = reg_df_remove_studies_not_to_consider(df_curated_reg)
    assert len(df_curated_reg[df_curated_reg['time_day']==28.0]) == len(df_included)
    assert len(df_included[df_included['endpoint'] == "inherent"]) == 0


def test_create_classification_data_based_on_regression_data(regression_paper):
    df_class, _ = create_classification_data_based_on_regression_data(
        reg_df=regression_paper,
        with_lunghini=False,
        include_speciation_lunghini=False,
        include_speciation=False,
        prnt=False,
    )

    assert df_class.cas.to_list() == ["85-00-7", "12040-58-3", "24245-27-0"]
    assert df_class.smiles.to_list() == [
        "[Br-].[Br-].c1cc[n+]2c(c1)-c1cccc[n+]1CC2",
        "[Ca+2].[Ca+2].[Ca+2].[O-]B([O-])[O-].[O-]B([O-])[O-]",
        "[Cl-].[NH2+]=C(Nc1ccccc1)Nc1ccccc1",
    ]
    assert df_class.inchi_from_smiles.to_list() == [
        "InChI=1S/C12H12N2.2BrH/c1-3-7-13-9-10-14-8-4-2-6-12(14)11(13)5-1;;/h1-8H,9-10H2;2*1H/q+2;;/p-2",
        "InChI=1S/2BO3.3Ca/c2*2-1(3)4;;;/q2*-3;3*+2",
        "InChI=1S/C13H13N3.ClH/c14-13(15-11-7-3-1-4-8-11)16-12-9-5-2-6-10-12;/h1-10H,(H3,14,15,16);1H",
    ]
    assert df_class.y_true.to_list() == [0, 0, 1]
    assert df_class.cas.nunique() == df_class.inchi_from_smiles.nunique()
    assert df_class.cas.nunique() == df_class.smiles.nunique()
    assert df_class.columns.to_list() == ["cas", "smiles", "y_true", "inchi_from_smiles"]

    cols = ["cas", "smiles", "y_true", "inchi_from_smiles"] + get_speciation_col_names()
    df_class, _ = create_classification_data_based_on_regression_data(
        reg_df=regression_paper, 
        with_lunghini=True, 
        include_speciation_lunghini=True, 
        include_speciation=True, 
        prnt=False,
        run_from_start=False
    )
    assert df_class.columns.to_list() == cols


def test_create_input_regression(regression_paper):
    x = create_input_regression(df=regression_paper, include_speciation=False)
    assert len(x[0]) == 172
    assert type(x) == np.ndarray
    assert type(x[0]) == np.ndarray

    reg = load_regression_df()[:5]  # with speciation
    x = create_input_regression(df=reg, include_speciation=True)
    assert len(x[0]) == 190


def test_convert_to_maccs_fingerprints(df_curated_class):
    df = convert_to_maccs_fingerprints(df_curated_class)
    for fp in df["fingerprint"]:
        assert str(type(fp)) == "<class 'rdkit.DataStructs.cDataStructs.ExplicitBitVect'>"


def test_convert_to_rdk_fingerprints(df_curated_class):
    df = convert_to_rdk_fingerprints(df_curated_class)
    for fp in df["fingerprint"]:
        assert str(type(fp)) == "<class 'rdkit.DataStructs.cDataStructs.ExplicitBitVect'>"


def test_convert_to_morgan_fingerprints(df_curated_class):
    df = convert_to_morgan_fingerprints(df_curated_class)
    for fp in df["fingerprint"]:
        assert str(type(fp)) == "<class 'rdkit.DataStructs.cDataStructs.ExplicitBitVect'>"


def test_encode_categorical_data(regression_paper):
    df = encode_categorical_data(regression_paper)

    def check_encoding(row):
        guideline = row["guideline"]
        principle = row["principle"]
        endpoint = row["endpoint"]
        assert (guideline >= 0) & (guideline <= 12)
        assert (principle >= 0) & (principle <= 3)
        assert (endpoint == 0) | (endpoint == 3)

    df.apply(check_encoding, axis=1)


def test_bit_vec_to_lst_of_lst(df_curated_class): 
    df = convert_to_maccs_fingerprints(df_curated_class)
    x_class = bit_vec_to_lst_of_lst(df, False)
    assert len(x_class[0]) == 167

    df = convert_to_rdk_fingerprints(df_curated_class)
    x_class = bit_vec_to_lst_of_lst(df, False)
    assert len(x_class[0]) == 2048

    df = convert_to_morgan_fingerprints(df_curated_class)
    x_class = bit_vec_to_lst_of_lst(df, False)
    assert len(x_class[0]) == 1024


def test_create_input_classification(class_df):
    x, _ = create_input_classification(df_class=class_df, include_speciation=False, target_col="y_true")
    assert (list(np.unique(x[0]))) == [0, 1]
    assert type(x) == np.ndarray
    assert type(x[0]) == np.ndarray
    assert len(x[0]) == 167

    _, _, df_class = load_class_data_paper()
    x, _ = create_input_classification(df_class=df_class[:5], include_speciation=True, target_col="y_true")
    assert len(x[0]) == 185

def test_convert_regression_df_to_input(regression_paper):
    df = convert_regression_df_to_input(df=regression_paper)
    assert "fingerprint" in df.columns
    assert "EU Method" not in df["guideline"].unique()
    assert "OECD Guideline" not in df["guideline"].unique()
    assert all(ptypes.is_numeric_dtype(df[col]) for col in ["guideline", "principle", "endpoint"])


def test_load_gluege_data():
    df = load_gluege_data()
    assert collections.Counter(list(df.columns)) == collections.Counter(
        [
            "cas",
            "smiles",
            "smiles_ph",
            "inchi_from_smiles",
        ]
    )
    assert len(df[df.smiles.str.contains("nan", regex=False)]) == 0
    assert len(df[df.smiles.str.contains("*", regex=False)]) == 0
    assert ptypes.is_string_dtype(df.dtypes["smiles"])
    assert ptypes.is_string_dtype(df.dtypes["cas"])


def test_check_number_of_components(df_b):
    df_one_component, df_multiple_components = check_number_of_components(df_b)
    assert len(df_one_component[df_one_component["smiles_paper"].str.contains(".", regex=False)]) == 0
    assert len(df_multiple_components[df_multiple_components["smiles_paper"].str.contains(".", regex=False)]) == len(
        df_multiple_components
    )


def test_openbabel_convert_smiles_to_inchi_with_nans():
    df = pd.read_csv(
        "datasets/data_processing/df_multiple_components_smiles_not_found_ccc_cirpy_no_metals.csv",
        index_col=0,
    )
    df_after = openbabel_convert_smiles_to_inchi_with_nans(col_names_smiles_to_inchi=["smiles_from_cas_cirpy"], df=df)
    assert len(df) == len(df_after)
    assert "inchi_from_smiles_from_cas_cirpy" in df_after.columns


def test_get_inchi_main_layer(df_b):
    _, df_multiple_components = check_number_of_components(df_b)
    df_pubchem = get_smiles_from_cas_pubchempy(df_multiple_components)
    df = get_inchi_main_layer(df=df_pubchem, inchi_col="inchi_pubchem", layers=4)
    assert list(df["cas"]) == ["132-16-1", "7775-09-9", "94891-43-7", "8029-68-3"]
    assert list(df["inchi_pubchem"]) == [
        "InChI=1S/C32H16N8.Fe/c1-2-10-18-17(9-1)25-33-26(18)38-28-21-13-5-6-14-22(21)30(35-28)40-32-24-16-8-7-15-23(24)31(36-32)39-29-20-12-4-3-11-19(20)27(34-29)37-25;/h1-16H;/q-2;+2",
        "InChI=1S/ClHO3.Na/c2-1(3)4;/h(H,2,3,4);/q;+1/p-1",
        "",
        "InChI=1S/CH4N2O2/c2-1(4)3-5/h5H,(H3,2,3,4)",
    ]
    assert list(df["inchi_pubchem_main_layer"]) == [
        "InChI=1S/C32H16N8.Fe/c1-2-10-18-17(9-1)25-33-26(18)38-28-21-13-5-6-14-22(21)30(35-28)40-32-24-16-8-7-15-23(24)31(36-32)39-29-20-12-4-3-11-19(20)27(34-29)37-25;/h1-16H;",
        "InChI=1S/ClHO3.Na/c2-1(3)4;/h(H,2,3,4);",
        "",
        "InChI=1S/CH4N2O2/c2-1(4)3-5/h5H,(H3,2,3,4)",
    ]


def test_get_smiles_inchi_cirpy(df_b):
    _, df_multiple_components = check_number_of_components(df_b)
    cass = list(df_multiple_components["cas"])
    smiless: List[str] = []
    inchis: List[str] = []
    for cas in cass:
        smiles, inchi = get_smiles_inchi_cirpy(cas)
        smiless.append(smiles)
        inchis.append(inchi)
    assert cass == ["132-16-1", "7775-09-9", "94891-43-7", "8029-68-3"]
    assert smiless == [
        "[Fe++].[n-]1c2nc3nc(nc4[n-]c(nc5nc(nc1c6ccccc26)c7ccccc57)c8ccccc48)c9ccccc39",
        "[Na+].[O-][Cl](=O)=O",
        None,
        None,
    ]
    assert inchis == [
        "InChI=1S/C32H16N8.Fe/c1-2-10-18-17(9-1)25-33-26(18)38-28-21-13-5-6-14-22(21)30(35-28)40-32-24-16-8-7-15-23(24)31(36-32)39-29-20-12-4-3-11-19(20)27(34-29)37-25;/h1-16H;/q-2;+2",
        "InChI=1S/ClHO3.Na/c2-1(3)4;/h(H,2,3,4);/q;+1/p-1",
        None,
        None,
    ]


def test_further_processing_of_echa_data():
    df_echa = pd.read_csv("datasets/iuclid_echa.csv", index_col=0)
    df_echa = further_processing_of_echa_data(df=df_echa)
    assert len(df_echa[df_echa["guideline"] == ""]) == 0
    assert len(df_echa[df_echa["principle"] == ""]) == 0
    assert len(df_echa[df_echa["oxygen_conditions"] != "aerobic"]) == 0


def test_load_and_process_echa_additional():
    (
        echa_additional_reg,
        echa_additional_reg_env,
        echa_additional_class,
        echa_additional_class_env,
    ) = load_and_process_echa_additional(include_speciation=False)
    for cas in echa_additional_reg["cas"]:
        assert is_cas_right_format(cas)
    for cas in echa_additional_class["cas"]:
        assert is_cas_right_format(cas)
    assert (
        len(echa_additional_reg[(echa_additional_reg["reliability"] != 1) & (echa_additional_reg["reliability"] != 2)])
        == 0
    )
    guidelines = [
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
    assert len(echa_additional_reg[echa_additional_reg["guideline"].isin(guidelines)]) == len(echa_additional_reg)
    assert len(
        echa_additional_reg[
            (echa_additional_reg["endpoint"] == "ready") | (echa_additional_reg["endpoint"] == "inherent")
        ]
    ) == len(echa_additional_reg)
    principles = [
        "CO2 Evolution",
        "DOC Die Away",
        "Closed Bottle Test",
        "Closed Respirometer",
    ]
    assert len(echa_additional_reg[echa_additional_reg["principle"].isin(principles)]) == len(echa_additional_reg)
    assert is_numeric_dtype(echa_additional_reg["time_day"]) == True
    assert is_numeric_dtype(echa_additional_reg["biodegradation_percent"]) == True
    assert (
        len(
            echa_additional_reg[
                (echa_additional_reg["biodegradation_percent"] < 0)
                | (echa_additional_reg["biodegradation_percent"] > 1)
            ]
        )
        == 0
    )
    echa_additional_reg_no_organo_metals, organo_metals = remove_organo_metals_function(
        df=echa_additional_reg, smiles_column="smiles"
    )
    assert len(echa_additional_reg_no_organo_metals) == len(echa_additional_reg)
    assert len(organo_metals) == 0
    echa_additional_class_no_organo_metals, organo_metals = remove_organo_metals_function(
        df=echa_additional_class, smiles_column="smiles"
    )
    assert len(echa_additional_class_no_organo_metals) == len(echa_additional_class)
    assert len(organo_metals) == 0


def test_get_df_with_unique_cas(regression_paper):
    df_unique_cas = get_df_with_unique_cas(regression_paper)
    assert (regression_paper["cas"].nunique()) == len(df_unique_cas)
    assert (list(df_unique_cas.columns)) == ["name", "cas", "smiles_paper"]


