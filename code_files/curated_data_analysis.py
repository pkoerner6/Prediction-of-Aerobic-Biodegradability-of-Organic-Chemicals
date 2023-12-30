"""Analysis of the datasets resulting models when datapoints are removed based on BIOWIN5 and BIOWIN6"""
"""Prior to this file, the files data_processing.py and creating_datasets.py need to be run"""

import pandas as pd
import numpy as np
import structlog

log = structlog.get_logger()
from typing import List, Dict
import matplotlib.pyplot as plt
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import AllChem
import rdkit
import sys
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--nsplits",
    type=int,
    default=5,
    help="Number of KFold splits",
)
parser.add_argument(
    "--random_seed",
    type=int,
    default=0,
)
args = parser.parse_args()

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from code_files.processing_functions import convert_regression_df_to_input
from code_files.processing_functions import create_dfs_for_curated_data_analysis


def plot_bar_dataset_analysis(
    dfname_to_groupamount: pd.DataFrame, bin_names: List[str], xlabel: str, ylabel: str, title: str, saving_name: str
):
    x = np.arange(len(dfname_to_groupamount["curated_scs"]))
    width = 0.2
    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, dfname_to_groupamount["curated_scs"], width, color="mediumturquoise")
    plt.bar(x + 0.0, dfname_to_groupamount["curated_biowin"], width, color="royalblue")
    plt.bar(x + 0.2, dfname_to_groupamount["curated_final"], width, color="mediumseagreen")
    plt.xticks(x, bin_names, fontsize=16)
    if "percent" in saving_name:
        plt.ylim(0, 145)
        plt.yticks(np.arange(0, 101, step=20), fontsize=16)
    else:
        plt.yticks(fontsize=16)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.legend(
        [
            "$\mathregular{Curated_{SCS}}$",
            "$\mathregular{Curated_{BIOWIN}}$",
            "$\mathregular{Curated_{FINAL}}$",
        ],
        fontsize=16,
    )
    plt.savefig(f"figures/analysis_{saving_name}.png")
    plt.close()


def analyse_distribution(
    dfnames_to_dfs: Dict[str, pd.DataFrame],
    bins: List,
    df_description: str,
    col_name: str,
    xlabel: str,
    ylabel: str,
    title: str,
    saving_name: str,
    plot_percent: bool,
    percentage_per_bin=False,
) -> None:
    bin_names: List[str] = []
    if not percentage_per_bin:
        log.info(df_description)

    dfname_to_groupamount: Dict[str, List[int]] = defaultdict(list)
    dfname_to_grouppercent: Dict[str, List[float]] = defaultdict(list)
    for i, (df_name, df) in enumerate(dfnames_to_dfs.items()):
        if col_name == "molecular_weight":
            df = df.copy()
            df[col_name] = [ExactMolWt(Chem.MolFromSmiles(smiles)) for smiles in df["smiles"]]
            dfnames_to_dfs[df_name] = df
        df = df.groupby(pd.cut(df[col_name], bins=bins, include_lowest=True))[col_name].count()
        for idx, index in enumerate(df.index):
            dfname_to_groupamount[df_name].append(df[index])
            label = f"{int(index.left)} to {int(index.right)}"
            if idx == 0:  # need to change label because the first bin includes the left
                label = f"0 to {int(index.right)}"
            if col_name == "y_true":
                label = "RB"
                if idx == 0:
                    label = "NRB"
            if label not in bin_names:
                bin_names.append(label)
    for i, label in enumerate(bin_names):
        if percentage_per_bin:
            dfname_to_grouppercent["curated_scs"].append(
                (dfname_to_groupamount["curated_scs"][i] / (dfname_to_groupamount["curated_scs"][i])) * 100
            )
            dfname_to_grouppercent["curated_biowin"].append(
                (dfname_to_groupamount["curated_biowin"][i] / (dfname_to_groupamount["curated_scs"][i]))
                * 100
            )
            dfname_to_grouppercent["curated_final"].append(
                (
                    dfname_to_groupamount["curated_final"][i]
                    / (dfname_to_groupamount["curated_scs"][i])
                )
                * 100
            )
        else:
            dfname_to_grouppercent["curated_scs"].append(
                (dfname_to_groupamount["curated_scs"][i] / len(dfnames_to_dfs["curated_scs"])) * 100
            )
            dfname_to_grouppercent["curated_biowin"].append(
                (dfname_to_groupamount["curated_biowin"][i] / len(dfnames_to_dfs["curated_scs"])) * 100
            )
            dfname_to_grouppercent["curated_final"].append(
                (dfname_to_groupamount["curated_final"][i] / len(dfnames_to_dfs["curated_scs"]))
                * 100
            )
            log.info(
                f"Entries in bin ({label}) relative to curated_scs",
                curated_scs="{:.1f}".format(
                    (dfname_to_groupamount["curated_scs"][i] / dfname_to_groupamount["curated_scs"][i]) * 100
                ),
                improved_biowin_both="{:.1f}".format(
                    (dfname_to_groupamount["curated_biowin"][i] / dfname_to_groupamount["curated_scs"][i])
                    * 100
                ),
                improved_biowin_both_readded="{:.1f}".format(
                    (
                        dfname_to_groupamount["curated_final"][i]
                        / dfname_to_groupamount["curated_scs"][i]
                    )
                    * 100
                ),
            )
    if plot_percent:
        plot_bar_dataset_analysis(
            dfname_to_groupamount=dfname_to_grouppercent,
            bin_names=bin_names,
            xlabel=xlabel,
            ylabel=f"{ylabel} (%)",
            title=title,
            saving_name=f"{saving_name}_percent",
        )
    else:
        plot_bar_dataset_analysis(
            dfname_to_groupamount=dfname_to_groupamount,
            bin_names=bin_names,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            saving_name=saving_name,
        )


def analyse_distribution_of_substances(dfnames_to_dfs: pd.DataFrame) -> None:
    dfnames_to_dfs_grouped = {
        "curated_scs": dfnames_to_dfs["curated_scs"],
        "curated_biowin": dfnames_to_dfs["curated_biowin"],
        "curated_final": dfnames_to_dfs["curated_final"],
    }
    for df_name, df in dfnames_to_dfs_grouped.items():
        log.info(f"Unique substances in {df_name}: ", substances=df.cas.nunique())
        cas_count = df.groupby(["cas"]).cas.count()
        df_grouped = pd.DataFrame({"cas": cas_count.index, "count": cas_count.values})
        count_count = df_grouped.groupby(["count"]).cas.count()
        df_grouped_count = pd.DataFrame({"count": count_count.index, df_name: count_count.values})
        dfnames_to_dfs_grouped[df_name] = df_grouped_count.set_index("count")

    max_entries = max(dfnames_to_dfs_grouped["curated_scs"].index)
    df_cas_count: Dict[str, pd.DataFrame] = {}
    for df_name, df in dfnames_to_dfs.items():
        cas_count = df.groupby(["cas"]).cas.count()
        df_grouped = pd.DataFrame({"cas": cas_count.index, "count": cas_count.values})
        df_cas_count[df_name] = df_grouped
    occurence_to_df_occurences: Dict[int, List[int]] = {}
    for num_entries in range(1, max_entries + 1):
        original_occurence = df_cas_count["curated_scs"][df_cas_count["curated_scs"]["count"] == num_entries]
        biowin_both_occurence = df_cas_count["curated_biowin"][
            df_cas_count["curated_biowin"]["count"] == num_entries
        ]
        biowin_both_occurence = biowin_both_occurence[biowin_both_occurence["cas"].isin(original_occurence["cas"])]
        biowin_both_occurence_readded = df_cas_count["curated_final"][
            df_cas_count["curated_final"]["count"] == num_entries
        ]
        biowin_both_occurence_readded = biowin_both_occurence_readded[
            biowin_both_occurence_readded["cas"].isin(original_occurence["cas"])
        ]
        df_occurences = [
            len(original_occurence),
            len(biowin_both_occurence),
            len(biowin_both_occurence_readded),
        ]
        if len(original_occurence) > 0:
            occurence_to_df_occurences[num_entries] = df_occurences

    df_occurences_all = pd.DataFrame(occurence_to_df_occurences).T
    df_occurences_all.rename(
        columns={
            0: "curated_scs",
            1: "curated_biowin",
            2: "curated_final",
        },
        inplace=True,
    )

    for c in df_occurences_all.index[:6]:
        log.info(
            f"Relative amount of entries with {c} entry per substance",
            curated_scs="{:.1f}".format(
                (df_occurences_all["curated_scs"][c] / df_occurences_all["curated_scs"][c]) * 100
            ),
            improved_biowin_both="{:.1f}".format(
                (df_occurences_all["curated_biowin"][c] / df_occurences_all["curated_scs"][c]) * 100
            ),
            improved_biowin_both_readded="{:.1f}".format(
                (df_occurences_all["curated_final"][c] / df_occurences_all["curated_scs"][c]) * 100
            ),
        )

    ax = df_occurences_all.plot(
        kind="bar",
        figsize=(13, 6),
        rot=0,
        color=[
            "mediumturquoise",
            "royalblue",
            "mediumseagreen",
        ],
    )
    plt.xlabel("entries per substance", fontsize=18)
    plt.ylabel("Occurence", fontsize=18)
    plt.legend(
        [
            "$\mathregular{Curated_{SCS}}$",
            "$\mathregular{Curated_{BIOWIN}}$",
            "$\mathregular{Curated_{FINAL}}$",
        ],
        fontsize=16,
    )
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f"figures/analysis_entries_per_substance.png")
    plt.close()


def analyse_distribution_of_days(dfnames_to_dfs: pd.DataFrame) -> None:
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 200]
    analyse_distribution(
        dfnames_to_dfs=dfnames_to_dfs,
        bins=bins,
        df_description="\n Analyse distribution of days",
        col_name="time_day",
        xlabel="Days",
        ylabel="Number of studies",
        title="Testing time",
        saving_name="distribution_of_time_day",
        plot_percent=False,
    )


def analyse_occurence_of_halogens(dfnames_to_dfs: pd.DataFrame) -> None:
    entries_with_f: List[int] = []
    entries_with_br: List[int] = []
    entries_with_cl: List[int] = []
    entries_with_all_halogens: List[int] = []
    for df_name, df in dfnames_to_dfs.items():
        entries_with_f.append(len(df[df["smiles"].str.contains("F")]))
        entries_with_br.append(len(df[df["smiles"].str.contains("Br")]))
        entries_with_cl.append(len(df[df["smiles"].str.contains("Cl")]))
        entries_with_all_halogens.append(
            len(
                df[
                    (df["smiles"].str.contains("F"))
                    | (df["smiles"].str.contains("Br"))
                    | (df["smiles"].str.contains("Cl"))
                ]
            )
        )
    entry_lists = [entries_with_f, entries_with_br, entries_with_cl]
    halogens = ["F", "Br", "Cl"]
    for i, (halogen, entry_list) in enumerate(zip(halogens, entry_lists)):
        log.info(
            f"Relative number of entries that contain {halogen}",
            curated_scs="{:.1f}".format((entry_list[0] / entry_list[0]) * 100),
            curated_biowin="{:.1f}".format((entry_list[1] / entry_list[0]) * 100),
            curated_final="{:.1f}".format((entry_list[2] / entry_list[0]) * 100),
        )
        log.info(
            f"Occurence of substances with {halogen} relative to all entries",
            curated_scs="{:.1f}".format((entry_list[i] / len(dfnames_to_dfs["curated_scs"])) * 100),
            curated_biowin="{:.1f}".format(
                (entry_list[i] / len(dfnames_to_dfs["curated_biowin"])) * 100
            ),
            curated_final="{:.1f}".format(
                (entry_list[i] / len(dfnames_to_dfs["curated_final"])) * 100
            ),
        )
    log.info(
        f"Occurence of halogens relative to all entries",
        curated_scs="{:.1f}".format((entries_with_all_halogens[0] / len(dfnames_to_dfs["curated_scs"])) * 100),

        curated_biowin="{:.1f}".format(
            (entries_with_all_halogens[2] / len(dfnames_to_dfs["curated_biowin"])) * 100
        ),
        curated_final="{:.1f}".format(
            (entries_with_all_halogens[2] / len(dfnames_to_dfs["curated_final"])) * 100
        ),
    )

    df_names = ["curated_scs", "curated_biowin", "curated_final"]

    df_halogens = pd.DataFrame({"F": entries_with_f, "Br": entries_with_br, "Cl": entries_with_cl}, index=df_names).T
    ax = df_halogens.plot(
        kind="bar",
        figsize=(8, 6),
        rot=0,
        color=[
            "mediumturquoise",
            "royalblue",
            "mediumseagreen",
        ],
        fontsize=18,
    )
    plt.ylabel("Occurence", fontsize=18)
    plt.legend(
        [
            "$\mathregular{Curated_{SCS}}$",
            "$\mathregular{Curated_{BIOWIN}}$",
            "$\mathregular{Curated_{FINAL}}$",
        ],
        fontsize=16,
    )
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f"figures/analysis_halogens.png")
    plt.close()

    df_halogens_all = pd.DataFrame(
        {"all": entries_with_all_halogens, "F": entries_with_f, "Br": entries_with_br, "Cl": entries_with_cl},
        index=df_names,
    ).T
    for df_name in df_names:
        df_halogens_all[df_name] = df_halogens_all[df_name].apply(lambda x: (x / len(dfnames_to_dfs[df_name])) * 100)
    ax = df_halogens_all.plot(
        kind="bar",
        figsize=(10, 6),
        rot=0,
        color=[
            "mediumturquoise",
            "royalblue",
            "mediumseagreen",
        ],
        fontsize=18,
    )
    plt.ylabel("Occurence (%)", fontsize=18)
    plt.legend(
        [
            "$\mathregular{Curated_{SCS}}$",
            "$\mathregular{Curated_{BIOWIN}}$",
            "$\mathregular{Curated_{FINAL}}$",
        ],
        fontsize=16,
    )
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f"figures/analysis_halogens_relative_occurence.png")
    plt.close()


def analyse_biodeg_distribution(
    dfnames_to_dfs: pd.DataFrame, bins: List[float], plot_percent: bool, percentage_per_bin: bool
) -> None:
    analyse_distribution(
        dfnames_to_dfs=dfnames_to_dfs,
        bins=bins,
        df_description="Original df",
        col_name="y_true",
        xlabel="Class",
        ylabel="Number of substances",
        title="Biodegradation classes",
        saving_name="distribution_of_biodegradation_classes",
        plot_percent=plot_percent,
        percentage_per_bin=percentage_per_bin,
    )


def count_hydrogen_bonds(smiles):
    mol = Chem.MolFromSmiles(smiles)

    AllChem.EmbedMolecule(mol, randomSeed=42) # Generate 3D coordinates

    num_hbond_acceptors = AllChem.CalcNumHBA(mol)
    num_hbond_donors = AllChem.CalcNumHBD(mol)

    return num_hbond_acceptors, num_hbond_donors


def analyse_num_of_h_bond_donors_acceptors(dfnames_to_dfs: pd.DataFrame) -> None:
    for df_name, df in dfnames_to_dfs.items():
        num_hbond_acceptors: List[int] = []
        num_hbond_donors: List[int] = []

        def get_hbond(row):
            smiles = row["smiles"]
            hbond_acceptors, hbond_donors = count_hydrogen_bonds(smiles)
            num_hbond_acceptors.append(hbond_acceptors)
            num_hbond_donors.append(hbond_donors)
        df.apply(get_hbond, axis=1)

        log.info(f"Mean number of H bond acceptors and donors in {df_name}", mean_hbond_acceptors="{:.1f}".format(sum(num_hbond_acceptors)/len(num_hbond_acceptors)), mean_hbond_donors="{:.1f}".format(sum(num_hbond_donors)/len(num_hbond_donors)))


def analyse_dataset_composition():
    (
        class_curated_scs,
        class_curated_biowin,
        class_curated_problematic,
        class_curated_final,
        class_curated_removed,
    ) = create_dfs_for_curated_data_analysis()

    dfnames_to_dfs = {
        "curated_scs": class_curated_scs,
        "curated_biowin": class_curated_biowin,
        "curated_final": class_curated_final,
    }

    bins = [0, 250, 500, 750, 1000, 2000]
    analyse_distribution(
        dfnames_to_dfs=dfnames_to_dfs,
        bins=bins,
        df_description=f"\n Analyse distribution of molecular weight",
        col_name="molecular_weight",
        xlabel="Molecular weight (Da)",
        ylabel="Number of substances",
        title="Molecular weight",
        saving_name=f"distribution_of_molecular_weight",
        plot_percent=False,
    )

    log.info(f"\n Analyse distribution of halogens")
    analyse_occurence_of_halogens(dfnames_to_dfs=dfnames_to_dfs)

    # Analyse distribution of degradation class
    log.info("\n Analyse distribution of degradation class")
    analyse_biodeg_distribution(
        dfnames_to_dfs=dfnames_to_dfs,
        bins=[0.0, 0.5, 1.0],
        plot_percent=True,
        percentage_per_bin=True,
    )
    analyse_biodeg_distribution(
        dfnames_to_dfs=dfnames_to_dfs,
        bins=[0.0, 0.5, 1.0],
        plot_percent=False,
        percentage_per_bin=False,
    )

    # Analyse number of H bond donors and acceptors
    log.info("\n Analyse number of H bond donors and acceptors")
    dfnames_to_dfs["class_curated_problematic"] = class_curated_problematic
    dfnames_to_dfs["class_curated_removed"] = class_curated_removed
    analyse_num_of_h_bond_donors_acceptors(dfnames_to_dfs)


if __name__ == "__main__":
    analyse_dataset_composition()
