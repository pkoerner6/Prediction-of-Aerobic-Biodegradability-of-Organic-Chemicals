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
from biodegradation.processing_functions import convert_regression_df_to_input
from biodegradation.processing_functions import create_dfs_for_improved_data_analysis
from biodegradation.ml_functions import run_XGBClassifier_Huang_Zhang
from biodegradation.ml_functions import run_XGBRegressor_Huang_Zhang


def plot_bar_dataset_analysis(
    dfname_to_groupamount: pd.DataFrame, bin_names: List[str], xlabel: str, ylabel: str, title: str, saving_name: str
):
    x = np.arange(len(dfname_to_groupamount["improved_env"]))
    width = 0.2
    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.3, dfname_to_groupamount["improved_env"], width, color="mediumturquoise")
    plt.bar(x - 0.1, dfname_to_groupamount["improved_env_biowin"], width, color="cornflowerblue")
    plt.bar(x + 0.1, dfname_to_groupamount["improved_env_biowin_both"], width, color="royalblue")
    plt.bar(x + 0.3, dfname_to_groupamount["improved_env_biowin_both_readded"], width, color="mediumseagreen")
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
            "$\mathregular{Curated_{1BIOWIN}}$",
            "$\mathregular{Curated_{2BIOWIN}}$",
            "$\mathregular{Curated_{FINAL}}$",
        ],
        fontsize=16,
    )
    plt.savefig(f"biodegradation/figures/analysis_{saving_name}.png")
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
            dfname_to_grouppercent["improved_env"].append(
                (dfname_to_groupamount["improved_env"][i] / (dfname_to_groupamount["improved_env"][i])) * 100
            )
            dfname_to_grouppercent["improved_env_biowin"].append(
                (dfname_to_groupamount["improved_env_biowin"][i] / (dfname_to_groupamount["improved_env"][i])) * 100
            )
            dfname_to_grouppercent["improved_env_biowin_both"].append(
                (dfname_to_groupamount["improved_env_biowin_both"][i] / (dfname_to_groupamount["improved_env"][i]))
                * 100
            )
            dfname_to_grouppercent["improved_env_biowin_both_readded"].append(
                (
                    dfname_to_groupamount["improved_env_biowin_both_readded"][i]
                    / (dfname_to_groupamount["improved_env"][i])
                )
                * 100
            )
        else:
            dfname_to_grouppercent["improved_env"].append(
                (dfname_to_groupamount["improved_env"][i] / len(dfnames_to_dfs["improved_env"])) * 100
            )
            dfname_to_grouppercent["improved_env_biowin"].append(
                (dfname_to_groupamount["improved_env_biowin"][i] / len(dfnames_to_dfs["improved_env"])) * 100
            )
            dfname_to_grouppercent["improved_env_biowin_both"].append(
                (dfname_to_groupamount["improved_env_biowin_both"][i] / len(dfnames_to_dfs["improved_env"])) * 100
            )
            dfname_to_grouppercent["improved_env_biowin_both_readded"].append(
                (dfname_to_groupamount["improved_env_biowin_both_readded"][i] / len(dfnames_to_dfs["improved_env"]))
                * 100
            )
            log.info(
                f"Entries in bin ({label}) relative to improved_env",
                improved_env="{:.1f}".format(
                    (dfname_to_groupamount["improved_env"][i] / dfname_to_groupamount["improved_env"][i]) * 100
                ),
                improved_biowin="{:.1f}".format(
                    (dfname_to_groupamount["improved_env_biowin"][i] / dfname_to_groupamount["improved_env"][i]) * 100
                ),
                improved_biowin_both="{:.1f}".format(
                    (dfname_to_groupamount["improved_env_biowin_both"][i] / dfname_to_groupamount["improved_env"][i])
                    * 100
                ),
                improved_biowin_both_readded="{:.1f}".format(
                    (
                        dfname_to_groupamount["improved_env_biowin_both_readded"][i]
                        / dfname_to_groupamount["improved_env"][i]
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
        "improved_env": dfnames_to_dfs["improved_env"],
        "improved_env_biowin": dfnames_to_dfs["improved_env_biowin"],
        "improved_env_biowin_both": dfnames_to_dfs["improved_env_biowin_both"],
        "improved_env_biowin_both_readded": dfnames_to_dfs["improved_env_biowin_both_readded"],
    }
    for df_name, df in dfnames_to_dfs_grouped.items():
        log.info(f"Unique substances in {df_name}: ", substances=df.cas.nunique())
        cas_count = df.groupby(["cas"]).cas.count()
        df_grouped = pd.DataFrame({"cas": cas_count.index, "count": cas_count.values})
        count_count = df_grouped.groupby(["count"]).cas.count()
        df_grouped_count = pd.DataFrame({"count": count_count.index, df_name: count_count.values})
        dfnames_to_dfs_grouped[df_name] = df_grouped_count.set_index("count")

    max_entries = max(dfnames_to_dfs_grouped["improved_env"].index)
    df_cas_count: Dict[str, pd.DataFrame] = {}
    for df_name, df in dfnames_to_dfs.items():
        cas_count = df.groupby(["cas"]).cas.count()
        df_grouped = pd.DataFrame({"cas": cas_count.index, "count": cas_count.values})
        df_cas_count[df_name] = df_grouped
    occurence_to_df_occurences: Dict[int, List[int]] = {}
    for num_entries in range(1, max_entries + 1):
        original_occurence = df_cas_count["improved_env"][df_cas_count["improved_env"]["count"] == num_entries]
        biowin_occurence = df_cas_count["improved_env_biowin"][
            df_cas_count["improved_env_biowin"]["count"] == num_entries
        ]
        biowin_occurence = biowin_occurence[biowin_occurence["cas"].isin(original_occurence["cas"])]
        biowin_both_occurence = df_cas_count["improved_env_biowin_both"][
            df_cas_count["improved_env_biowin_both"]["count"] == num_entries
        ]
        biowin_both_occurence = biowin_both_occurence[biowin_both_occurence["cas"].isin(original_occurence["cas"])]
        biowin_both_occurence_readded = df_cas_count["improved_env_biowin_both_readded"][
            df_cas_count["improved_env_biowin_both_readded"]["count"] == num_entries
        ]
        biowin_both_occurence_readded = biowin_both_occurence_readded[
            biowin_both_occurence_readded["cas"].isin(original_occurence["cas"])
        ]
        df_occurences = [
            len(original_occurence),
            len(biowin_occurence),
            len(biowin_both_occurence),
            len(biowin_both_occurence_readded),
        ]
        if len(original_occurence) > 0:
            occurence_to_df_occurences[num_entries] = df_occurences

    df_occurences_all = pd.DataFrame(occurence_to_df_occurences).T
    df_occurences_all.rename(
        columns={
            0: "improved_env",
            1: "improved_env_biowin",
            2: "improved_env_biowin_both",
            3: "improved_env_biowin_both_readded",
        },
        inplace=True,
    )

    for c in df_occurences_all.index[:6]:
        log.info(
            f"Relative amount of entries with {c} entry per substance",
            improved_env="{:.1f}".format(
                (df_occurences_all["improved_env"][c] / df_occurences_all["improved_env"][c]) * 100
            ),
            improved_biowin="{:.1f}".format(
                (df_occurences_all["improved_env_biowin"][c] / df_occurences_all["improved_env"][c]) * 100
            ),
            improved_biowin_both="{:.1f}".format(
                (df_occurences_all["improved_env_biowin_both"][c] / df_occurences_all["improved_env"][c]) * 100
            ),
            improved_biowin_both_readded="{:.1f}".format(
                (df_occurences_all["improved_env_biowin_both_readded"][c] / df_occurences_all["improved_env"][c]) * 100
            ),
        )

    ax = df_occurences_all.plot(
        kind="bar",
        figsize=(13, 6),
        rot=0,
        color=[
            "mediumturquoise",
            "cornflowerblue",
            "royalblue",
            "mediumseagreen",
        ],
    )
    plt.xlabel("entries per substance", fontsize=18)
    plt.ylabel("Occurence", fontsize=18)
    plt.legend(
        [
            "$\mathregular{Curated_{SCS}}$",
            "$\mathregular{Curated_{1BIOWIN}}$",
            "$\mathregular{Curated_{2BIOWIN}}$",
            "$\mathregular{Curated_{FINAL}}$",
        ],
        fontsize=16,
    )
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f"biodegradation/figures/analysis_entries_per_substance.png")
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


def analyse_occurence_of_halogens(dfnames_to_dfs: pd.DataFrame, mode: str) -> None:
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
            improved_env="{:.1f}".format((entry_list[0] / entry_list[0]) * 100),
            biowin_one="{:.1f}".format((entry_list[1] / entry_list[0]) * 100),
            biowin_both="{:.1f}".format((entry_list[2] / entry_list[0]) * 100),
            readded="{:.1f}".format((entry_list[3] / entry_list[0]) * 100),
        )
        log.info(
            f"Occurence of substances with {halogen} relative to all entries",
            improved_env="{:.1f}".format((entry_list[i] / len(dfnames_to_dfs["improved_env"])) * 100),
            improved_biowin="{:.1f}".format((entry_list[i] / len(dfnames_to_dfs["improved_env_biowin"])) * 100),
            improved_biowin_both="{:.1f}".format(
                (entry_list[i] / len(dfnames_to_dfs["improved_env_biowin_both"])) * 100
            ),
            improved_biowin_both_readded="{:.1f}".format(
                (entry_list[i] / len(dfnames_to_dfs["improved_env_biowin_both_readded"])) * 100
            ),
        )
    log.info(
        f"Occurence of halogens relative to all entries",
        improved_env="{:.1f}".format((entries_with_all_halogens[0] / len(dfnames_to_dfs["improved_env"])) * 100),
        improved_biowin="{:.1f}".format(
            (entries_with_all_halogens[1] / len(dfnames_to_dfs["improved_env_biowin"])) * 100
        ),
        improved_biowin_both="{:.1f}".format(
            (entries_with_all_halogens[2] / len(dfnames_to_dfs["improved_env_biowin_both"])) * 100
        ),
        improved_biowin_both_readded="{:.1f}".format(
            (entries_with_all_halogens[2] / len(dfnames_to_dfs["improved_env_biowin_both_readded"])) * 100
        ),
    )

    df_names = ["improved_env", "improved_env_biowin", "improved_env_biowin_both", "improved_env_biowin_both_readded"]

    df_halogens = pd.DataFrame({"F": entries_with_f, "Br": entries_with_br, "Cl": entries_with_cl}, index=df_names).T
    ax = df_halogens.plot(
        kind="bar",
        figsize=(8, 6),
        rot=0,
        color=[
            "mediumturquoise",
            "cornflowerblue",
            "royalblue",
            "mediumseagreen",
        ],
        fontsize=18,
    )
    plt.ylabel("Occurence", fontsize=18)
    plt.legend(
        [
            "$\mathregular{Curated_{SCS}}$",
            "$\mathregular{Curated_{1BIOWIN}}$",
            "$\mathregular{Curated_{2BIOWIN}}$",
            "$\mathregular{Curated_{FINAL}}$",
        ],
        fontsize=16,
    )
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f"biodegradation/figures/analysis_halogens_{mode}.png")
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
            "cornflowerblue",
            "royalblue",
            "mediumseagreen",
        ],
        fontsize=18,
    )
    plt.ylabel("Occurence (%)", fontsize=18)
    plt.legend(
        [
            "$\mathregular{Curated_{SCS}}$",
            "$\mathregular{Curated_{1BIOWIN}}$",
            "$\mathregular{Curated_{2BIOWIN}}$",
            "$\mathregular{Curated_{FINAL}}$",
        ],
        fontsize=16,
    )
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f"biodegradation/figures/analysis_halogens_relative_occurence_{mode}.png")
    plt.close()


def analyse_biowin_models(df_improved_biowin_both: pd.DataFrame, mode: str, df_both_removed: pd.DataFrame) -> None:
    df_both = df_improved_biowin_both.copy()
    df_both_all = pd.concat([df_both, df_both_removed], ignore_index=True)
    log.info("Entries in df_both: ", entries=len(df_both))
    log.info("Entries in df_both_removed: ", entries=len(df_both_removed))
    log.info("Entries in df_both_all: ", entries=len(df_both_all))
    log.info("Entries in df_both_all: ", entries=len(df_both_all))
    both_all_class_occurences = {
        "tests": [len(df_both_all[df_both_all["label"] == 0]), len(df_both_all[df_both_all["label"] == 1])],
        "BIOWIN5": [
            len(df_both_all[df_both_all["miti_linear_label"] == 0]),
            len(df_both_all[df_both_all["miti_linear_label"] == 1]),
        ],
        "BIOWIN6": [
            len(df_both_all[df_both_all["miti_non_linear_label"] == 0]),
            len(df_both_all[df_both_all["miti_non_linear_label"] == 1]),
        ],
    }
    for key, value in both_all_class_occurences.items():
        log.info(
            f"Entries in df_both_all with {key} labels: ",
            nrb=value[0],
            rb=value[1],
            nrb_percent="{:.1f}".format((value[0] / sum(value)) * 100),
            rb_percent="{:.1f}".format((value[1] / sum(value)) * 100),
        )
    if mode == "reg":
        df_both_all_28 = df_both_all[df_both_all["time_day"] == 28.0]
        log.info("Entries in df_both_all_28: ", entries=len(df_both_all_28))
        both_all_28_class_occurences = {
            "tests": [
                len(df_both_all_28[df_both_all_28["label"] == 0]),
                len(df_both_all_28[df_both_all_28["label"] == 1]),
            ],
            "BIOWIN5": [
                len(df_both_all_28[df_both_all_28["miti_linear_label"] == 0]),
                len(df_both_all_28[df_both_all_28["miti_linear_label"] == 1]),
            ],
            "BIOWIN6": [
                len(df_both_all_28[df_both_all_28["miti_non_linear_label"] == 0]),
                len(df_both_all_28[df_both_all_28["miti_non_linear_label"] == 1]),
            ],
        }
        for key, value in both_all_28_class_occurences.items():
            log.info(
                f"Entries in df_both_all_28 with {key} labels: ",
                nrb=value[0],
                rb=value[1],
                nrb_percent="{:.1f}".format((value[0] / sum(value)) * 100),
                rb_percent="{:.1f}".format((value[1] / sum(value)) * 100),
            )

    df_both_all_class_occurences = pd.DataFrame(both_all_class_occurences)

    ax = df_both_all_class_occurences.plot(
        kind="bar", figsize=(7, 6), rot=0, color=["mediumaquamarine", "lightskyblue", "dodgerblue"]
    )
    plt.ylabel("Occurence", fontsize=18)
    plt.ylabel("Label", fontsize=18)
    plt.legend(
        ["Biodegradation tests", "BIOWIN5\u2122", "BIOWIN6\u2122"],
        fontsize=16,
    )
    plt.xticks(ticks=[0, 1], labels=["NRB", "RB"], fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f"biodegradation/figures/analysis_biodegradation_classes_biowin_{mode}.png")
    plt.close()


def run_classifier_on_removed_data(
    include_speciation: bool,
    run_regression_with_readded: bool,
    run_classification_with_readded: bool,
    fixed_test: bool,
) -> None:
    reg_improved_env_biowin_both = pd.read_csv(
        "biodegradation/dataframes/improved_data/reg_improved_env_biowin_both.csv", index_col=0
    )
    class_improved_env_biowin_both = pd.read_csv(
        "biodegradation/dataframes/improved_data/class_improved_env_biowin_both.csv", index_col=0
    )
    reg_improved_env_biowin_both_readded = pd.read_csv(
        "biodegradation/dataframes/improved_data/reg_improved_env_biowin_both_readded.csv", index_col=0
    )
    class_improved_env_biowin_both_readded = pd.read_csv(
        "biodegradation/dataframes/improved_data/class_improved_env_biowin_both_readded.csv", index_col=0
    )
    dfs_readded = {
        "reg_improved_env_biowin_both_readded": reg_improved_env_biowin_both_readded,
        "class_improved_env_biowin_both_readded": class_improved_env_biowin_both_readded,
    }
    for df_name, df_readded in dfs_readded.items():
        # Analyse performance of BIOWIN
        log.info(
            f"Entries in {df_name} that have no prediction from BIOWIN",
            no_biowin=len(df_readded[df_readded["miti_linear_label"].isna()]),
        )
        log.info(
            f"In {df_name} BIOWIN5 predicted this many NRB entries as in readded dataframe",
            nrb_biowin5="{:.1f}".format(
                (len(df_readded[(df_readded["label"] == df_readded["miti_linear_label"]) & (df_readded["label"] == 0)]))
                / len(df_readded[(df_readded["label"] == 0) & (df_readded["miti_linear_label"].notna())])
                * 100
            ),
        )
        log.info(
            f"In {df_name} BIOWIN5 predicted this many RB entries as in readded dataframe",
            rb_biowin5="{:.1f}".format(
                (len(df_readded[(df_readded["label"] == df_readded["miti_linear_label"]) & (df_readded["label"] == 1)]))
                / len(df_readded[(df_readded["label"] == 1) & (df_readded["miti_linear_label"].notna())])
                * 100
            ),
        )
        log.info(
            f"In {df_name} BIOWIN6 predicted this many NRB entries as in readded dataframe",
            nrb_biowin6="{:.1f}".format(
                (
                    len(
                        df_readded[
                            (df_readded["label"] == df_readded["miti_non_linear_label"]) & (df_readded["label"] == 0)
                        ]
                    )
                )
                / len(df_readded[(df_readded["label"] == 0) & (df_readded["miti_linear_label"].notna())])
                * 100
            ),
        )
        log.info(
            f"In {df_name} BIOWIN6 predicted this many RB entries as in readded dataframe",
            rb_biowin6="{:.1f}".format(
                (
                    len(
                        df_readded[
                            (df_readded["label"] == df_readded["miti_non_linear_label"]) & (df_readded["label"] == 1)
                        ]
                    )
                )
                / len(df_readded[(df_readded["label"] == 1) & (df_readded["miti_linear_label"].notna())])
                * 100
            ),
        )
        log.info(
            f"In {df_name} BIOWIN5&6 predicted this many NRB entries as in readded dataframe",
            nrb_biowin56="{:.1f}".format(
                (
                    len(
                        df_readded[
                            (df_readded["label"] == df_readded["miti_linear_label"])
                            & (df_readded["label"] == df_readded["miti_non_linear_label"])
                            & (df_readded["label"] == 0)
                        ]
                    )
                )
                / (len(df_readded[(df_readded["label"] == 0) & (df_readded["miti_linear_label"].notna())]))
                * 100
            ),
        )
        log.info(
            f"In {df_name} BIOWIN5&6 predicted this many RB entries as in readded dataframe",
            rb_biowin56="{:.1f}".format(
                (
                    len(
                        df_readded[
                            (df_readded["label"] == df_readded["miti_linear_label"])
                            & (df_readded["label"] == df_readded["miti_non_linear_label"])
                            & (df_readded["label"] == 1)
                        ]
                    )
                )
                / (len(df_readded[(df_readded["label"] == 1) & (df_readded["miti_linear_label"].notna())]))
                * 100
            ),
        )
        if "reg" in df_name:
            if run_regression_with_readded:
                log.info("Entries in reg_improved_env_biowin_both", entries=len(reg_improved_env_biowin_both))
                log.info("Entries in df_reg_readded", entries=len(df_readded))
                reg_improved_env_biowin_both = convert_regression_df_to_input(df=reg_improved_env_biowin_both)
                df_readded_input = convert_regression_df_to_input(df=df_readded)
                _, _, _, _ = run_XGBRegressor_Huang_Zhang(
                    df=df_readded_input,
                    column_for_grouping="smiles",
                    random_seed=args.random_seed,
                    nsplits=args.nsplits,
                    include_speciation=include_speciation,
                    fixed_testset=fixed_test,
                    df_smallest=reg_improved_env_biowin_both,
                    dataset_name="reg_improved_env_biowin_both_readded",
                )
        if "class" in df_name:
            if run_classification_with_readded:
                log.info("Entries in class_improved_env_biowin_both", entries=len(class_improved_env_biowin_both))
                log.info(
                    "Entries in class_improved_env_biowin_both that are 0",
                    nrb=len(class_improved_env_biowin_both[class_improved_env_biowin_both["y_true"] == 0]),
                )
                log.info(
                    "Entries in class_improved_env_biowin_both that are 1",
                    rb=len(class_improved_env_biowin_both[class_improved_env_biowin_both["y_true"] == 1]),
                )
                log.info("Entries in df_class_readded", entries=len(df_readded))
                log.info("Entries in df_class_readded that are 0", nrb=len(df_readded[df_readded["y_true"] == 0]))
                log.info("Entries in df_class_readded that are 1", rb=len(df_readded[df_readded["y_true"] == 1]))
                _, _, _, _ = run_XGBClassifier_Huang_Zhang(
                    df=df_readded,
                    random_seed=args.random_seed,
                    nsplits=args.nsplits,
                    use_adasyn=True,
                    include_speciation=include_speciation,
                    fixed_testset=fixed_test,
                    df_smallest=class_improved_env_biowin_both,
                    dataset_name="class_improved_env_biowin_both_readded",
                )


def analyse_biodeg_distribution(
    dfnames_to_dfs: pd.DataFrame, mode: str, bins: List[float], plot_percent: bool, percentage_per_bin: bool
) -> None:
    if mode == "reg":
        analyse_distribution(
            dfnames_to_dfs=dfnames_to_dfs,
            bins=bins,
            df_description="Original df",
            col_name="biodegradation_percent",
            xlabel="Biodegradation (%)",
            ylabel="Number of studies",
            title="Biodegradation percentage distribution",
            saving_name="distribution_of_biodegradation",
            plot_percent=plot_percent,
            percentage_per_bin=percentage_per_bin,
        )
        dfnames_to_dfs_28: Dict[str, pd.DataFrame] = {}
        for df_name, df in dfnames_to_dfs.items():
            dfnames_to_dfs_28[df_name] = df[df["time_day"] == 28.0]
        analyse_distribution(
            dfnames_to_dfs=dfnames_to_dfs,
            bins=bins,
            df_description="Only 28 days",
            col_name="biodegradation_percent",
            xlabel="Biodegradation (%)",
            ylabel="Number of studies",
            title="Biodegradation percentage distribution (only 28 days)",
            saving_name="distribution_of_biodegradation_only_28days",
            plot_percent=plot_percent,
            percentage_per_bin=percentage_per_bin,
        )

    if mode == "class":
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


def analyse_dataset_composition():
    (
        reg_improved_biowin_both_removed,
        class_improved_biowin_both_removed,
        reg_improved_env,
        reg_improved_env_biowin,
        reg_improved_env_biowin_both,
        reg_improved_env_biowin_both_readded,
        class_improved_env,
        class_improved_env_biowin,
        class_improved_env_biowin_both,
        class_improved_env_biowin_both_readded,
    ) = create_dfs_for_improved_data_analysis()

    dfnames_to_dfs_reg = {
        "improved_env": reg_improved_env,
        "improved_env_biowin": reg_improved_env_biowin,
        "improved_env_biowin_both": reg_improved_env_biowin_both,
        "improved_env_biowin_both_readded": reg_improved_env_biowin_both_readded,
    }
    dfnames_to_dfs_class = {
        "improved_env": class_improved_env,
        "improved_env_biowin": class_improved_env_biowin,
        "improved_env_biowin_both": class_improved_env_biowin_both,
        "improved_env_biowin_both_readded": class_improved_env_biowin_both_readded,
    }
    mode_to_df_dict = {"reg": dfnames_to_dfs_reg, "class": dfnames_to_dfs_class}

    analyse_distribution_of_substances(dfnames_to_dfs=dfnames_to_dfs_reg)

    bins = [0, 250, 500, 750, 1000, 2000]
    for mode, dfnames_to_dfs in mode_to_df_dict.items():
        ylabel = "Number of substances" if mode == "class" else "Number of studies"
        analyse_distribution(
            dfnames_to_dfs=dfnames_to_dfs,
            bins=bins,
            df_description=f"\n Analyse distribution of molecular weight {mode}",
            col_name="molecular_weight",
            xlabel="Molecular weight (Da)",
            ylabel=ylabel,
            title="Molecular weight",
            saving_name=f"distribution_of_molecular_weight_{mode}",
            plot_percent=False,
        )

    analyse_distribution_of_days(dfnames_to_dfs_reg)

    for mode, dfnames_to_dfs in mode_to_df_dict.items():
        log.info(f"\n Analyse distribution of halogens {mode}")
        analyse_occurence_of_halogens(dfnames_to_dfs=dfnames_to_dfs, mode=mode)

    # Analyse biowin models
    log.info("\n Analyse BIOWIN models")
    log.info("Regression")
    analyse_biowin_models(
        df_improved_biowin_both=reg_improved_env_biowin_both,
        mode="reg",
        df_both_removed=reg_improved_biowin_both_removed,
    )
    log.info("Classification")
    analyse_biowin_models(
        df_improved_biowin_both=class_improved_env_biowin_both,
        mode="class",
        df_both_removed=class_improved_biowin_both_removed,
    )

    # Run our model on removed dataset
    log.info("\n Run our model on removed data and readd matches")
    run_classifier_on_removed_data(
        include_speciation=False,
        run_regression_with_readded=True,
        run_classification_with_readded=True,
        fixed_test=True,
    )

    # Analyse distribution of degradation percentage or class
    log.info("\n Analyse distribution of degradation percentage or class")
    log.info("Regression")
    for df_name, df in dfnames_to_dfs_reg.items():
        df["biodegradation_percent"] = df["biodegradation_percent"] * 100
        dfnames_to_dfs_reg[df_name] = df * 100
    analyse_biodeg_distribution(
        dfnames_to_dfs=dfnames_to_dfs_reg,
        mode="reg",
        bins=[0, 20, 40, 60, 80, 100],
        plot_percent=True,
        percentage_per_bin=True,
    )
    analyse_biodeg_distribution(
        dfnames_to_dfs=dfnames_to_dfs_reg,
        mode="reg",
        bins=[0, 20, 40, 60, 80, 100],
        plot_percent=False,
        percentage_per_bin=False,
    )

    log.info("Classification")
    analyse_biodeg_distribution(
        dfnames_to_dfs=dfnames_to_dfs_class,
        mode="class",
        bins=[0.0, 0.5, 1.0],
        plot_percent=True,
        percentage_per_bin=True,
    )
    analyse_biodeg_distribution(
        dfnames_to_dfs=dfnames_to_dfs_class,
        mode="class",
        bins=[0.0, 0.5, 1.0],
        plot_percent=False,
        percentage_per_bin=False,
    )


if __name__ == "__main__":
    analyse_dataset_composition()
