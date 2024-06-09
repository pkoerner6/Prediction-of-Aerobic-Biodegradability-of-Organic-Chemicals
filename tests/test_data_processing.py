import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from code_files.data_processing import replace_smiles_with_smiles_gluege
from code_files.data_processing import get_df_a_and_b
from code_files.data_processing import get_problematic_studies_echa
from code_files.data_processing import remove_studies_where_registered_substance_dont_match_reference_substance
from code_files.data_processing import process_one_component_data
from code_files.data_processing import process_multiple_component_data
from code_files.data_processing import process_multiple_components_smiles_not_found
from code_files.data_processing import get_full_df_b
from code_files.data_processing import get_dfs
from code_files.data_processing import load_datasets
from code_files.data_processing import process_data_checked_by_gluege
from code_files.data_processing import process_data_not_checked_by_gluege
from code_files.data_processing import process_full_dataset
from code_files.data_processing import aggregate_duplicates
from code_files.processing_functions import openbabel_convert
from code_files.processing_functions import check_number_of_components
from code_files.processing_functions import remove_organo_metals_function
from code_files.processing_functions import get_speciation_col_names


def test_replace_smiles_with_smiles_gluege(class_original_paper):
    df_checked, _, _, _, _ = load_datasets()
    df_class_gluege = replace_smiles_with_smiles_gluege(class_original_paper, df_checked)
    assert list(df_class_gluege.smiles) == [
        r"COC1=C(C)C2=NC(=CC(O[C@@H]3C[C@@H]4[C@@H](C3)C(=O)N(C)CCCC\C=C/[C@@H]3C[C@]3(NC4=O)C(O)=O)=C2C=C1)C1=NC(=CS1)C(C)C",
        r"CC(=C)C(O)=O",
        r"CC(C)COP(=S)(OCC(C)C)SCC(C)C(O)=O",
        r"Cc1cc([N+](=O)[O-])cc([N+](=O)[O-])c1O",
        r"CCC(C(O)=O)C1=CC=C(C=C1)N1C(=O)C2=C(C=CC=C2)C1=O",
        r"CCCCC(=O)N(CC1=CC=C(C=C1)C1=C(C=CC=C1)C1=NN=NN1)[C@@H](C(C)C)C(=O)OCC1=CC=CC=C1",
    ]
    assert len(class_original_paper) == len(df_class_gluege)


def test_get_df_a_and_b():
    df_reg, df_unique_cas, df_checked = get_dfs()
    assert df_reg["cas"].nunique() == len(df_unique_cas)
    df_a, df_b, df_a_full_original, df_b_full_original = get_df_a_and_b(
        df_unique_cas=df_unique_cas,
        df_checked=df_checked,
        df_reg=df_reg,
    )
    assert len(df_a) + len(df_b) == df_reg["cas"].nunique()
    assert len(df_a_full_original) + len(df_b_full_original) == len(df_reg)


def test_get_problematic_studies_echa():
    df_echa = get_problematic_studies_echa()

    def check_if_problematic(row):
        inventory_num = row["inventory_num"]
        ref_inventory_num = row["ref_inventory_num"]
        cas = row["cas"]
        ref_cas = row["ref_cas"]
        assert (inventory_num not in ref_inventory_num) | (cas not in ref_cas)

    df_echa.apply(check_if_problematic, axis=1)


def test_remove_studies_where_registered_substance_dont_match_reference_substance():
    df_reg, df_unique_cas, df_checked = get_dfs()
    df_a, _, df_a_full_original, _ = get_df_a_and_b(df_unique_cas=df_unique_cas, df_checked=df_checked, df_reg=df_reg)
    df_checked, _, _, _, _ = load_datasets()
    df_a_full = replace_smiles_with_smiles_gluege(df_a_full_original, df_checked)
    df_a_full = remove_studies_where_registered_substance_dont_match_reference_substance(df_a_full, df_a)
    df_echa_problematic_ref = get_problematic_studies_echa()

    def remove_problematic_studies(row):
        cas = row["cas"]
        if cas not in list(df_echa_problematic_ref["cas"]):
            return 0
        df_cas_problematic_studies = df_echa_problematic_ref[df_echa_problematic_ref["cas"] == cas]
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
    assert len(df_a_full[df_a_full["to_remove"] == True]) == 0
    assert len(df_a_full[df_a_full["to_remove"] == False]) == len(df_a_full)


def test_process_one_component_data(df_b):
    df_reg, df_unique_cas, df_checked = get_dfs()
    _, df_b, _, _ = get_df_a_and_b(df_unique_cas=df_unique_cas, df_checked=df_checked, df_reg=df_reg)
    df_one_component, _ = check_number_of_components(df_b)
    (
        df_smiles_found,
        df_smiles_not_found,
        df_problematic_multiple_components,
    ) = process_one_component_data(df_b=df_one_component, new_cas_common_chemistry=False)
    assert len(df_smiles_found[df_smiles_found["smiles_paper"].str.contains(".", regex=False)]) == 0
    assert len(df_smiles_found) + len(df_smiles_not_found) + len(df_problematic_multiple_components) == len(
        df_one_component
    )
    assert len(
        df_problematic_multiple_components[
            df_problematic_multiple_components["smiles_from_ccc"].str.contains(".", regex=False)
        ]
    ) == len(df_problematic_multiple_components)


def test_process_multiple_component_data():
    df_reg, df_unique_cas, df_checked = get_dfs()
    _, df_b, _, _ = get_df_a_and_b(df_unique_cas=df_unique_cas, df_checked=df_checked, df_reg=df_reg)
    df_one_component, df_multiple_components = check_number_of_components(df_b)
    (
        _,
        df_smiles_not_found,
        df_problematic_multiple_components,
    ) = process_one_component_data(df_b=df_one_component, new_cas_common_chemistry=False)

    df_multiple = pd.concat(
        [df_multiple_components, df_smiles_not_found, df_problematic_multiple_components],
        ignore_index=True,
    )

    df_multiple_components_smiles_found, df_multiple_components_smiles_not_found = process_multiple_component_data(
        df_multiple_components=df_multiple,
        new_pubchem=False,
        new_comptox=False,
    )
    assert len(df_multiple_components_smiles_found) + len(df_multiple_components_smiles_not_found) == len(
        df_multiple_components
    ) + len(df_smiles_not_found) + len(df_problematic_multiple_components)
    assert len(
        df_multiple_components_smiles_found[
            ~df_multiple_components_smiles_found["cas"].isin(list(df_multiple_components_smiles_not_found["cas"]))
        ]
    )


def test_process_multiple_components_smiles_not_found():
    df_reg, df_unique_cas, df_checked = get_dfs()
    _, df_b, _, _ = get_df_a_and_b(df_unique_cas=df_unique_cas, df_checked=df_checked, df_reg=df_reg)
    df_one_component, df_multiple_components = check_number_of_components(df_b)
    (
        _,
        df_smiles_not_found,
        df_problematic_multiple_components,
    ) = process_one_component_data(df_b=df_one_component, new_cas_common_chemistry=False)

    df_multiple = pd.concat(
        [df_multiple_components, df_smiles_not_found, df_problematic_multiple_components],
        ignore_index=True,
    )

    _, df_multiple_components_smiles_not_found = process_multiple_component_data(
        df_multiple_components=df_multiple,
        new_pubchem=False,
        new_comptox=False,
    )
    (
        df_multiple_components_smiles_found2,
        df_multiple_components_smiles_not_found2,
    ) = process_multiple_components_smiles_not_found(df_multiple_components_smiles_not_found, new_cirpy=False)
    assert len(df_multiple_components_smiles_found2) + len(df_multiple_components_smiles_not_found2) == len(
        df_multiple_components_smiles_not_found
    )


def test_get_full_df_b():
    df_reg, df_unique_cas, df_checked = get_dfs()
    _, df_b, _, df_b_full_original = get_df_a_and_b(df_unique_cas=df_unique_cas, df_checked=df_checked, df_reg=df_reg)

    df_one_component, df_multiple_components = check_number_of_components(df_b)
    df_smiles_found, df_smiles_not_found, df_problematic_multiple_components = process_one_component_data(
        df_b=df_one_component, new_cas_common_chemistry=False
    )

    df_multiple = pd.concat(
        [df_multiple_components, df_smiles_not_found, df_problematic_multiple_components], ignore_index=True
    )

    df_multiple_components_smiles_found1, df_multiple_components_smiles_not_found1 = process_multiple_component_data(
        df_multiple,
        new_pubchem=False,
        new_comptox=False,
    )
    (
        df_multiple_components_smiles_found2,
        df_multiple_components_smiles_not_found,
    ) = process_multiple_components_smiles_not_found(df=df_multiple_components_smiles_not_found1, new_cirpy=False)
    assert len(df_smiles_found) + len(df_multiple_components_smiles_found1) + len(
        df_multiple_components_smiles_found2
    ) == len(pd.concat([df_smiles_found, df_multiple_components_smiles_found1, df_multiple_components_smiles_found2]))

    df_multiple_components_smiles_found = pd.concat(
        [df_multiple_components_smiles_found1, df_multiple_components_smiles_found2]
    )
    assert len(df_multiple_components_smiles_found) == len(df_multiple_components_smiles_found1) + len(
        df_multiple_components_smiles_found2
    )

    df_b_found_full, df_smiles_not_found_full = get_full_df_b(
        df_b_full_original,
        df_smiles_found,
        df_multiple_components_smiles_found,
        df_multiple_components_smiles_not_found,
    )
    assert (len(df_b_found_full) + len(df_smiles_not_found_full)) == len(df_b_full_original)

    df_b_full = openbabel_convert(
        df=df_b_found_full,
        input_type="smiles",
        column_name_input="smiles",
        output_type="inchi",
    )
    for _, group in df_b_full.groupby("cas"):
        assert group["inchi_from_smiles"].nunique() == 1


def test_get_dfs():
    df_reg, df_unique_cas, _ = get_dfs()
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
    for col in df_reg.columns:
        assert col in cols
    assert len(df_unique_cas) == df_reg["cas"].nunique()


def test_process_data_not_checked_by_gluege():
    _, _, df_b, _, _ = load_datasets()
    df_one_comp_found, df_multiple_comps_found, df_multiple_comps_not_found = process_data_not_checked_by_gluege(
        df_b=df_b
    )
    assert len(df_b) == len(df_one_comp_found) + len(df_multiple_comps_found) + len(df_multiple_comps_not_found)


def test_aggregate_duplicates(df_for_aggregate):
    df_full_with_env_smiles_agg = aggregate_duplicates(df=df_for_aggregate)
    assert len(df_full_with_env_smiles_agg) == 4
    assert list(df_full_with_env_smiles_agg.cas) == ["535-87-5", "535-87-5", "85-00-7", "865-49-6"]
    assert list(df_full_with_env_smiles_agg.time_day) == [29.0, 29.0, 28.0, 14.0]
    assert list(df_full_with_env_smiles_agg.smiles) == [
        "[Cl-].[Cl-].[NH3+]CCCCCCCC[NH3+]",
        "[Cl-].[Cl-].[NH3+]CCCCCCCC",
        "[Br-].[Br-].c1cc[n+]2c(c1)-c1cccc[n+]1CC2",
        "[2H]C(Cl)(Cl)Cl",
    ]
    assert list(df_full_with_env_smiles_agg.biodegradation_percent) == [0.5, 0.5, 0.5, 0.5]
