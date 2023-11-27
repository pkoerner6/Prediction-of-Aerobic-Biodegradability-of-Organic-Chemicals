import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set
import structlog
import os
from zipfile import ZipFile
import tqdm
import argparse
import xml.dom.minidom as minidom
from io import StringIO
from collections import defaultdict
import math

log = structlog.get_logger()
parser = argparse.ArgumentParser()

parser.add_argument(
    "--unzip",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Unzip the files or use already unzipped files",
)
parser.add_argument(
    "--save",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Save resulting dataframe",
)
parser.add_argument(
    "--path_to_save",
    type=str,
    default="datasets/iuclid_echa.csv",
    help="Give path where to save results",
)
parser.add_argument(
    "--add_info",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Add information at the end",
)
parser.add_argument(
    "--subtype",
    type=str,
    choices=[
        "BiodegradationInWaterScreeningTests",
        "AcuteToxicityDermal",
        "AcuteToxicityOral",
        "AcuteToxicityInhalation",
        "AdditionalEcotoxicologicalInformation",
        "AdditionalPhysicoChemical",
        "AdditionalToxicologicalInformation",
        "AdsorptionDesorption",
        "AutoFlammability",
        "BasicToxicokinetics",
        "BioaccumulationAquaticSediment",
        "BiodegradationInWaterScreeningTests",
        "BiodegradationInSoil",
        "BiodegradationInWaterAndSedimentSimulationTests",
        "BoilingPoint",
        "Carcinogenicity",
        "Density",
        "DissociationConstant",
        "DistributionModelling",
        "Explosiveness",
        "EpidemiologicalData",
        "Explosiveness",
        "ExposureRelatedObservationsOther",
        "EyeIrritation",
        "Flammability",
        "FlashPoint",
        "GeneralInformation",
        "GeneticToxicityVitro",
        "GeneticToxicityVivo",
        "Granulometry",
        "HenrysLawConstant",
        "Hydrolysis",
        "LongTermToxicityToAquaInv",
        "LongTermToxToFish",
        "Melting",
        "ModeOfDegradationInActualUse",
        "OxidisingProperties",
        "Partition",
        "Phototransformation",
        "PhototransformationInAir",
        "PhotoTransformationInSoil",
        "RepeatedDoseToxicityInhalation",
        "RepeatedDoseToxicityOral",
        "SedimentToxicity",
        "SensitisationData",
        "ShortTermToxicityToAquaInv",
        "ShortTermToxicityToFish",
        "SkinIrritationCorrosion",
        "SkinSensitisation",
        "SolubilityOrganic",
        "SpecificInvestigations",
        "StabilityOrganic",
        "SurfaceTension",
        "ToxicityToAquaticAlgae",
        "ToxicityToMicroorganisms",
        "ToxicityToBirds",
        "ToxicityToMicroorganisms",
        "ToxicityToSoilMacroorganismsExceptArthropods",
        "ToxicityToSoilMicroorganisms",
        "ToxicityToTerrestrialArthropods",
        "ToxicityToTerrestrialPlants",
        "Vapour",
        "Viscosity",
        "WaterSolubility",
        "SubstanceComposition",
    ],
    default="BiodegradationInWaterScreeningTests",
    help="Choose the subtype for which to get the information",
)
parser.add_argument(
    "--pickle_excel",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Turn organics excel into pickle after filtering",
)
parser.add_argument(
    "--path_dossiers",
    type=str,
    default="reach_study_results/reach_study_results_dossiers_09_08_2022",
    help="Give the path to the downloaded REACH study results dossiers",
)
args = parser.parse_args()


def unzip_i6z_files(path: str):
    dir_list: List[str] = os.listdir(path)
    for z in tqdm.tqdm(dir_list):
        if ".i6z" not in z:
            log.warn("Not a zip file: ", file=z)
            continue
        joined_path = os.path.join(path, z)
        with ZipFile(joined_path, "r") as zObject:
            folder_name = z.replace(".i6z", "")
            zObject.extractall(path=os.path.join(path + "_unzipped", folder_name))


def check_if_exists(elements: List) -> str:
    if elements:
        return str(elements[0].text)
    return ""


def find_info_in_manifest_per_doc(document: ET.Element) -> List[str]:
    """Find name, IUPAC-name, CAS-number, inventory-number in the given document"""
    infos_names = ["name", "IUPAC-name", "CAS-number", "inventory-number"]
    infos: List[str] = ["", "", "", ""]
    representation = document.findall("manifest:representation", name_space)[0]
    parent = representation.findall("manifest:parent", name_space)
    if not parent:
        return infos
    infos[0] = check_if_exists(parent[0].findall("manifest:name", name_space))
    ref_substance_lst = representation.findall("manifest:reference-substance", name_space)
    if not ref_substance_lst:
        return infos
    ref_substance = ref_substance_lst[0]
    for i, info_name in enumerate(infos_names):
        if i == 0:
            continue
        infos[i] = check_if_exists(ref_substance.findall(f"manifest:{info_name}", name_space))
    return infos


def find_info_in_manifest_per_doc_reference(document: ET.Element) -> List[str]:
    """Find name, IUPAC-name, CAS-number, inventory-number in the given document"""
    infos_names = ["IUPAC-name", "CAS-number", "inventory-number"]
    infos: List[str] = ["", "", ""]
    representation = document.findall("manifest:representation", name_space)[0]
    if not representation:
        return infos
    for i, info_name in enumerate(infos_names):
        infos[i] = check_if_exists(representation.findall(f"manifest:{info_name}", name_space))
    return infos


def create_manifest_df(dir_name: str, name_space: Dict[str, str], subtype: str) -> pd.DataFrame:
    manifest_col_to_values: Dict[str, List] = defaultdict(list)
    tree = ET.parse(dir_name)
    contained_docs = tree.findall("manifest:contained-documents", name_space)[0]
    for doc in contained_docs:
        subtypes = doc.findall("manifest:subtype", name_space)
        for st in subtypes:
            if st.text == subtype:
                manifest_col_to_values["subtype"].append(str(st.text))
                long_id = doc.findall("manifest:name", name_space)
                manifest_col_to_values["long_id"].append(str(long_id[0].get("{http://www.w3.org/1999/xlink}href")))
                infos = find_info_in_manifest_per_doc(document=doc)
                col_names = ["chemical_name", "iupac_name", "cas", "inventory_num"]
                for i, col_name in enumerate(col_names):
                    manifest_col_to_values[col_name].append(infos[i])
                links = doc.findall("manifest:links", name_space)
                sublink = links[0].findall("manifest:link", name_space)
                refs: List[str] = []
                for elem in sublink:
                    reftype = elem.findall("manifest:ref-type", name_space)[0].text
                    if reftype == "REFERENCE":
                        reference = elem.findall("manifest:ref-uuid", name_space)[0].text
                        refs.append(str(reference))
                manifest_col_to_values["reference"].append(refs)
    # Get reference information
    for references in manifest_col_to_values["reference"]:
        uuids: List[str] = []
        for ref in references:
            for doc in contained_docs:
                if (
                    doc.get("id") == ref
                    and doc.findall("manifest:type", name_space)[0].text == "TEST_MATERIAL_INFORMATION"
                ):
                    links = doc.findall("manifest:links", name_space)[0].findall("manifest:link", name_space)
                    for link in links:
                        reftype = link.findall("manifest:ref-type", name_space)[0].text
                        if reftype == "REFERENCE":
                            uuid = str(link.findall("manifest:ref-uuid", name_space)[0].text)
                            uuids.append(uuid)
        manifest_col_to_values["ref_uuids"].append(
            uuids
        )  # ref_uuids are the references to documents containing information on test_material_information, references are references to all related documents
    for uuids in manifest_col_to_values["ref_uuids"]:
        ref_infos_lsts: List[List[str]] = [[], [], []]
        if len(uuids) > 0:
            for uuid in uuids:
                for doc in contained_docs:
                    if (
                        doc.get("id") == uuid
                        and doc.findall("manifest:type", name_space)[0].text == "REFERENCE_SUBSTANCE"
                    ):
                        ref_infos = find_info_in_manifest_per_doc_reference(document=doc)
                        for i, ref_infos_lst in enumerate(ref_infos_lsts):
                            ref_infos_lst.append(ref_infos[i])
        col_names = ["ref_iupac_name", "ref_cas", "ref_inventory_num"]
        for i, col_name in enumerate(col_names):
            result = ", ".join(ref_infos_lsts[i])
            manifest_col_to_values[col_name].append(result)

    manifest_df = pd.DataFrame.from_dict(manifest_col_to_values)
    return manifest_df


def check_tag_of_children(input: ET.Element) -> List[str]:
    return [child.tag.split("}")[1] for child in input]


def get_topics(results: Dict[str, List[str]]) -> List[str]:
    return list({key.split("_")[0] for key in list(results.keys())})


def get_values(
    previous_element: List[ET.Element],
    nspace: str,
    tags: List[str],
    code_to_decoded: Dict[str, str],
    sub_col_name_decode: List[str],
    prev_element_str: str,
    results: Dict[str, List[str]],
    subtype: str,
) -> Dict[str, List[str]]:
    prev_element_str = prev_element_str + "_" + str(previous_element).split("}")[1].split("'")[0]
    if len(tags) == 0:
        return results
    for tag in set(tags):
        element = previous_element[0].findall(f"{nspace}:{tag}", name_space)
        if len(element) == 0:  # if the element is empty, return the given results
            return results
        elif len(element) > 1:  # if we have multiple children we need to iterate over them
            for elem in element:
                tags = check_tag_of_children(elem)
                results = get_values(
                    previous_element=[elem],
                    nspace=nspace,
                    tags=tags,
                    code_to_decoded=code_to_decoded,
                    sub_col_name_decode=sub_col_name_decode,
                    prev_element_str=prev_element_str,
                    results=results,
                    subtype=subtype,
                )
        elif element[0]:
            tags = check_tag_of_children(element[0])
            results = get_values(
                previous_element=element,
                nspace=nspace,
                tags=tags,
                code_to_decoded=code_to_decoded,
                sub_col_name_decode=sub_col_name_decode,
                prev_element_str=prev_element_str,
                results=results,
                subtype=subtype,
            )
        else:  # What happens when we reach the last entry without any children
            replace = [f"_ENDPOINT_STUDY_RECORD.{subtype}_", "_entry", "_value"]
            for item in replace:
                prev_element_str = prev_element_str.replace(item, "")
            key = (prev_element_str + "_" + tag).lower()
            value = [str(element[0].text)]
            # Decode: Only decode if need
            for i, val in enumerate(value):
                if val in list(code_to_decoded.keys()):
                    if val.isnumeric() & (
                        int(val) > 100
                    ):  # Values under 100 are problematic because they might be accidentally be decoded
                        value[i] = code_to_decoded[val]  # Decoding
                    else:
                        dont_decode = True
                        for sub_col_decode in sub_col_name_decode:
                            if sub_col_decode in key:
                                dont_decode = False
                        if not dont_decode:
                            value[i] = code_to_decoded[val]
            # If we have multiple results for a given key of a topic, make sure that the lists of values are all of equal length
            if key not in list(results.keys()):
                results[key] = value
            else:
                for val in value:
                    results[key].append(val)
    return results


def get_code_to_decode(dir_list: List[str], subtype: str) -> Tuple[Dict[str, str], List[str]]:
    for d in dir_list:
        code_to_decoded: Dict[str, str] = {}
        current_dir = os.path.join(directory_to_folders, d, f"ENDPOINT_STUDY_RECORD-{subtype}.xsl")
        if os.path.exists(current_dir):
            tree = ET.parse(current_dir)
        else:
            continue
        root = tree.getroot()
        # Convert to text file
        rough_string = ET.tostring(root, "utf-8")
        reparsed = minidom.parseString(rough_string)
        pretty = reparsed.toprettyxml(indent="\t")
        text_file = StringIO()
        text_file.write(pretty)
        text_file.seek(0)
        data = text_file.read()
        # Process data to get dictionary
        data_into_list = data.replace("\t", "").split("\n")
        data_lst = [s.strip() for s in data_into_list]
        data_lst = list(filter(None, data_lst))
        data_lst_when = [item for item in data_lst if any(sub in item for sub in ["when"])]
        data_lst_when = [sub.replace("\xa0", " ") for sub in data_lst_when]
        to_remove = [
            '<ns0:when test="',
            "name(.) = '",
            "</ns0:when>",
            ". = '",
            "./i6:value = '",
            "'",
            "./unitCode =",
            "./i6:unitCode =",
        ]
        for item_to_remove in to_remove:
            data_lst_when = [sub.replace(item_to_remove, "") for sub in data_lst_when]
        for item in data_lst_when:
            split = item.split('">')
            if len(split) == 1:
                continue
            code_to_decoded[split[0].lstrip()] = split[1].lstrip()
        nones = ["None", "NaN", "nan"]
        for n in nones:
            code_to_decoded[n] = ""
        # Find colum names for which to decode problematic codes - for the other columns we shouldn't decode values below 100
        number_keys = [key for key in list(code_to_decoded.keys()) if key.isdigit()]
        problematic_keys = [key for key in number_keys if int(key) < 100]
        data_template_match_split = data.split("template match=")
        indices: Set[int] = set()
        for problematic_key in problematic_keys:
            search = "'" + '">' + code_to_decoded[problematic_key] + "<"
            item_with_key = [s for s in data_template_match_split if search in s]
            index_with_key = data_template_match_split.index(item_with_key[0])
            indices.add(index_with_key)
        sub_col_name_decode: List[str] = []
        for index in indices:
            data_match = data_template_match_split[index].split("<tr>")
            data_match_str = data_match[0]
            to_replace_match = ["//xt:", '"', ">", "\t", "\n"]
            for string in to_replace_match:
                data_match_str = data_match_str.replace(string, "")
            data_match_list = data_match_str.split(" | ")
            data_match_list = [s.strip() for s in data_match_list]
            data_match_list = [s.lower() for s in data_match_list]
            sub_col_name_decode = sub_col_name_decode + data_match_list
        return code_to_decoded, sub_col_name_decode
    return code_to_decoded, sub_col_name_decode


def get_values_for_dir(
    manifest_df: pd.DataFrame,
    name_space: Dict[str, str],
    dir: str,
    directory_to_folders: str,
    subtype: str,
    code_to_decoded: Dict[str, str],
    sub_col_name_decode: List[str],
) -> pd.DataFrame:
    df = pd.DataFrame()
    if subtype not in str(manifest_df["subtype"].values):
        return df
    parameter_indices = manifest_df.index[manifest_df["subtype"] == subtype].tolist()
    for indx in parameter_indices:
        manifest_info = manifest_df[manifest_df.index == indx].to_dict("records")[0]
        id = manifest_df.at[indx, "long_id"]
        dir_name2 = os.path.join(directory_to_folders, dir, id)
        if os.path.exists(dir_name2):
            tree = ET.parse(dir_name2)
        else:
            continue
        content = tree.findall("parameter:Content", name_space)
        results_discussion = content[0].findall(f"study_record:ENDPOINT_STUDY_RECORD.{subtype}", name_space)
        tags = check_tag_of_children(results_discussion[0])
        if results_discussion:
            nspace = "study_record"
            results = get_values(
                previous_element=results_discussion,
                nspace=nspace,
                tags=tags,
                code_to_decoded=code_to_decoded,
                sub_col_name_decode=sub_col_name_decode,
                prev_element_str="",
                results={},
                subtype=subtype,
            )
            topics = get_topics(results)
            if "resultsanddiscussion" not in topics:
                continue
            keys_in_rad = [key for key in results.keys() if "resultsanddiscussion" in key]
            keys_other = [key for key in results.keys() if "resultsanddiscussion" not in key]
            topics.remove("resultsanddiscussion")
            length_dict = {key: len(value) for key, value in results.items()}
            results_rad = dict((k, results[k]) for k in keys_in_rad if k in results)
            length_dict_rad = {key: len(value) for key, value in results_rad.items()}
            max_len_rad = max(length_dict_rad.values())
            keys_with_max_len = [key for key in keys_in_rad if length_dict[key] == max_len_rad]
            keys_with_other_len = [
                key
                for key in list(results.keys())
                if (((length_dict[key] != max_len_rad) & (key in keys_in_rad)) | (key not in keys_in_rad))
                & (length_dict[key] > 1)
            ]
            for k in keys_with_other_len:
                results[k] = [
                    ", ".join([str(elem) for elem in results[k] if elem != "other: "])
                ]  # Sometimes the value is just "other: " so we can remove it
            keys_in_rad_with_len_1 = [key for key in keys_in_rad if length_dict[key] == 1]
            keys_other = keys_other + keys_in_rad_with_len_1
            if max_len_rad > 1:
                results_lst: List[Dict] = []
                for i in range(max_len_rad):
                    current_results = {key: results[key] for key in keys_other}
                    for key in keys_with_max_len:
                        current_results[key] = [results[key][i]]
                    results_lst.append(current_results)
                df_lst: List[pd.DataFrame] = []
                for res in results_lst:
                    df_res = pd.DataFrame(res)
                    df_lst.append(df_res)
                df_index = pd.concat(df_lst, ignore_index=True)
            else:
                df_index = pd.DataFrame(results)
            for col in ["reference", "ref_uuids"]:
                del manifest_info[col]
            manifest_info_df = pd.DataFrame([manifest_info])
            if len(manifest_info_df) != len(df_index):
                manifest_info_df = pd.concat([manifest_info_df] * len(df_index), ignore_index=True)
            df_index = pd.concat([manifest_info_df, df_index], axis=1, join="inner")
        if len(df) == 0:
            df = df_index
        else:
            df = pd.concat([df, df_index])
    return df


def create_manifest(directory_to_folders: str, name_space: Dict[str, str], dir: str, subtype: str) -> pd.DataFrame:
    dir_name = os.path.join(directory_to_folders, dir, "manifest.xml")
    manifest_df = create_manifest_df(dir_name=dir_name, name_space=name_space, subtype=subtype)
    return manifest_df


def filter_and_turn_excel_to_pickel(excel_dir: str):
    excel_file = pd.read_excel(f"{excel_dir}.xlsx")
    excel_file = excel_file[excel_file["Composition"] != "multi-constituent substance"]
    entries_echa_evaluation_to_delete = ["no", "maybe", "polymer", "no  "]
    for entry in entries_echa_evaluation_to_delete:
        excel_file = excel_file[excel_file["Entry in ECHA DB correct?"] != entry]
    excel_file = excel_file[excel_file["SMILES corresponding to CAS RN"] != "unclear"]
    pd.to_pickle(excel_file, f"{excel_dir}.pkl")


def load_data_to_add_info() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if args.pickle_excel:
        filter_and_turn_excel_to_pickel("reach_study_results/RegisteredSubstances_organic6")
    reg_substances = pd.read_pickle("reach_study_results/RegisteredSubstances_organic6.pkl")
    mixtures = pd.read_csv(
        "reach_study_results/Mixtures_SubsID.csv",
        engine="python",
        sep=";",
    )
    return reg_substances, mixtures


def add_information(df: pd.DataFrame, reg_substances: pd.DataFrame, mixtures: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    new_col_names_to_col_names = {
        "Substance_ID": "substance_id",
        "Substance Type manually": "substance_type",
        "Composition": "composition",
        "Entry in ECHA DB correct?": "entry_echa_evaluation",
        "Substance_Physical_State": "substance_physical_state",
        "Tonnage_Band_Min": "tonnage",
        "Registration_Type": "registration_type",
        "SMILES corresponding to CAS RN": "smiles",
        "SMILES at pH 7.4": "smiles_ph",
        "SMILES ECHA database": "smiles_echa",
        "Molecular_Formula": "molecular_formula",
        "Molecular_Weight": "molecular_weight",
        "EC_No": "ec_num",
        "Substance_Name": "substance_name",
    }
    indices_to_drop = df[
        (df["inventory_num"] == "None") & (df["chemical_name"] == "[No public or meaningful name is available]")
    ].index
    df.drop(indices_to_drop, inplace=True)

    def get_ec_no_from_reg(row) -> str:
        if row["inventory_num"] != "None":
            return row["inventory_num"]
        else:
            chemical_name = row["chemical_name"]
            return reg_substances[reg_substances["Substance_Name"] == chemical_name]["EC_No"].iloc[0]

    df["inventory_num"] = df.apply(get_ec_no_from_reg, axis=1)

    with_inventory_num_df = df[df["inventory_num"].notnull()]
    without_inventory_num_df = df[df["inventory_num"].isnull()]

    result_with_inventory_num_df = pd.merge(
        left=with_inventory_num_df,
        right=reg_substances[list(new_col_names_to_col_names.keys())],
        left_on="inventory_num",
        right_on="EC_No",
    )
    result_without_inventory_num_df = pd.merge(
        left=without_inventory_num_df,
        right=reg_substances[list(new_col_names_to_col_names.keys())],
        left_on="chemical_name",
        right_on="Substance_Name",
    )
    result = pd.concat(
        [result_with_inventory_num_df, result_without_inventory_num_df],
        ignore_index=True,
    )
    result.rename(columns=new_col_names_to_col_names, inplace=True)

    def add_mixture_info(row) -> str:
        if row["substance_id"] not in list(mixtures["Substance_ID"]):
            return row["composition"]
        else:
            return "mixture"

    result["composition"] = result.apply(add_mixture_info, axis=1)

    return result


def get_values_for_dir_list(
    name_space: Dict[str, str],
    dir_list: List[str],
    directory_to_folders: str,
    subtype: str,
) -> pd.DataFrame:
    list_of_dfs: List[pd.DataFrame] = []
    code_to_decoded, sub_col_name_decode = get_code_to_decode(dir_list=dir_list, subtype=subtype)
    reg_substances, mixtures = load_data_to_add_info()
    for dir in tqdm.tqdm(dir_list):
        manifest_df = create_manifest(
            dir=dir,
            name_space=name_space,
            directory_to_folders=directory_to_folders,
            subtype=subtype,
        )
        if len(manifest_df) != 0:
            manifest_df = manifest_df[
                (manifest_df["chemical_name"].isin(reg_substances["Substance_Name"]))
                | (manifest_df["inventory_num"].isin(reg_substances["EC_No"]))
            ]
            df = get_values_for_dir(
                name_space=name_space,
                dir=dir,
                directory_to_folders=directory_to_folders,
                manifest_df=manifest_df,
                subtype=subtype,
                code_to_decoded=code_to_decoded,
                sub_col_name_decode=sub_col_name_decode,
            )
            list_of_dfs.append(df)
    df = pd.concat(list_of_dfs, ignore_index=True)
    if args.add_info:
        df = add_information(df, reg_substances, mixtures)
    df.reset_index(inplace=True, drop=True)

    return df


# Process and format echa_data
cols_to_delete = [
    "long_id",
    "substance_name",
    "ec_num",
    "materialsandmethods_testmaterials_testmaterialinformation",
    "materialsandmethods_guideline_deviation_value",
    "materialsandmethods_glpcompliancestatement_value",
    "materialsandmethods_studydesign_initialtestsubstanceconcentration_basedon_value",
    "materialsandmethods_studydesign_inoculumortestsystem_value",
    "administrativedata_crossreference_reasonpurpose_value",
    "resultsanddiscussion_bod5codresults_bod5cod_keyresult",
    "resultsanddiscussion_bod5codresults_bod5cod_parameter_value",
    "resultsanddiscussion_bod5codresults_bod5cod_value_lowervalue",
    "resultsanddiscussion_bod5codresults_bod5cod_value_unitcode",
    "resultsanddiscussion_bod5codresults_bod5cod_value_upperqualifier",
    "resultsanddiscussion_bod5codresults_bod5cod_value_uppervalue",
    "resultsanddiscussion_bod5codresults_bod5cod_value_lowerqualifier",
    "materialsandmethods_studydesign_initialtestsubstanceconcentration_initialconc_upperqualifier",
    "materialsandmethods_studydesign_initialtestsubstanceconcentration_initialconc_lowerqualifier",
    "materialsandmethods_studydesign_initialtestsubstanceconcentration_initialconc_uppervalue",
    "administrativedata_crossreference_relatedinformation",
    "materialsandmethods_studydesign_durationoftestcontacttime_lowervalue",
    "materialsandmethods_studydesign_durationoftestcontacttime_unitcode",
    "materialsandmethods_studydesign_durationoftestcontacttime_lowerqualifier",
    "materialsandmethods_studydesign_initialtestsubstanceconcentration_initialconc_lowervalue",
    "materialsandmethods_studydesign_initialtestsubstanceconcentration_initialconc_unitcode",
]
oldcolumns_to_new = {
    "subtype": "subtype",
    "substance_id": "substance_id",
    "inventory_num": "inventory_num",
    "chemical_name": "chemical_name",
    "iupac_name": "iupac_name",
    "cas": "cas",
    "substance_type": "substance_type",
    "composition": "composition",
    "ref_inventory_num": "ref_inventory_num",
    "ref_iupac_name": "ref_iupac_name",
    "ref_cas": "ref_cas",
    "entry_echa_evaluation": "entry_echa_evaluation",
    "substance_physical_state": "substance_physical_state",
    "tonnage": "tonnage",
    "registration_type": "registration_type",
    "smiles": "smiles",
    "smiles_ph": "smiles_ph",
    "smiles_echa": "smiles_echa",
    "molecular_formula": "molecular_formula",
    "molecular_weight": "molecular_weight",
    "administrativedata_reliability_value": "reliability",
    "administrativedata_studyresulttype_value": "study_result_type",
    "resultsanddiscussion_degradation_degr_lowervalue": "biodegradation_lowervalue",
    "resultsanddiscussion_degradation_degr_lowerqualifier": "biodegradation_lowerqualifier",
    "resultsanddiscussion_degradation_degr_uppervalue": "biodegradation_uppervalue",
    "resultsanddiscussion_degradation_degr_upperqualifier": "biodegradation_upperqualifier",
    "resultsanddiscussion_degradation_keyresult": "biodegradation_keyresult",
    "resultsanddiscussion_degradation_samplingtime_value": "biodegradation_samplingtime",
    "resultsanddiscussion_degradation_samplingtime_unitcode": "biodegradation_samplingtime_unit",
    "resultsanddiscussion_degradation_stdev": "biodegradation_stdev",
    "administrativedata_purposeflag_value": "purposeflag",
    "administrativedata_endpoint_value": "endpoint",
    "materialsandmethods_guideline_guideline_value": "guideline",
    "materialsandmethods_guideline_qualifier_value": "guideline_qualifier",
    "applicantsummaryandconclusion_validitycriteriafulfilled_value": "validity_criteria_fulfilled",
    "applicantsummaryandconclusion_interpretationofresults_value": "interpretation_of_results",
    "resultsanddiscussion_degradation_parameter_value": "degradation_parameter",
    "materialsandmethods_studydesign_oxygenconditions_value": "oxygen_conditions",
}


def drop_and_rename(df: pd.DataFrame, cols_to_delete: List[str], oldcolumns_to_new: Dict[str, str]) -> pd.DataFrame:
    df = df.drop(labels=cols_to_delete, axis=1)
    df.rename(columns=oldcolumns_to_new, inplace=True)
    df = df[list(oldcolumns_to_new.values())]
    return df


def select_one_guideline(df: pd.DataFrame) -> pd.DataFrame:
    def find_best_guideline(row):
        guidelines = row["guideline"].split(", ")
        guideline_qualifiers = row["guideline_qualifier"].split(", ")
        guideline_order = ["OECD Guideline", "EU Method", "EPA OPPTS", "EPA OTS", "ISO"]
        for best_guideline in guideline_order:
            for i in range(len(guidelines)):
                if best_guideline in guidelines[i]:
                    return pd.Series([guidelines[i], guideline_qualifiers[i]])
        return pd.Series([guidelines[0], guideline_qualifiers[0]])

    df[["guideline", "guideline_qualifier"]] = df.apply(find_best_guideline, axis=1)
    return df


def check_and_process_columns(df: pd.DataFrame, oldcolumns_to_new: Dict[str, str]) -> pd.DataFrame:
    df = df[df["reliability"].notna()]
    reliabilities_to_remove = ["other: ", "d", "None"]
    for reliability in reliabilities_to_remove:
        df = df[df["reliability"] != reliability]
        df = df[df["purposeflag"] != reliability]
    df = df[df["reliability"].apply(lambda x: "3 (not reliable)" not in x)]
    df = df[df["purposeflag"].apply(lambda x: "disregarded due to major methodological deficiencies" not in str(x))]
    df = df[
        (df["biodegradation_lowervalue"].notna()) | (df["biodegradation_uppervalue"].notna())
    ]  # check that results exist and remove otherwise
    qualifiers_to_remove = [
        "ca.",
        ">",
        ">=",
        "<",
        "<=",
    ]  # remove any rows with qualifiers
    for qualifier in qualifiers_to_remove:
        df = df[df["biodegradation_lowerqualifier"] != qualifier]
        df = df[df["biodegradation_upperqualifier"] != qualifier]
    df = df.drop(["biodegradation_lowerqualifier", "biodegradation_upperqualifier"], axis=1)
    oldcolumns_to_new.pop("resultsanddiscussion_degradation_degr_lowerqualifier")
    oldcolumns_to_new.pop("resultsanddiscussion_degradation_degr_upperqualifier")
    nones_to_delete = [", None", "None, ", "None"]
    for col in list(oldcolumns_to_new.values()):  # Remove None strings in all columns
        if (df[col].dtypes) != "object":
            continue
        for n in nones_to_delete:
            df[col] = df[col].str.replace(n, "")

    # Check each column and remove rows with incorrect entries (done with index because couldn't handle nonetypes)
    cols_to_check_if_contain_digits = [
        "subtype",
        "entry_echa_evaluation",
        "substance_physical_state",
        "registration_type",
        "biodegradation_keyresult",
        "biodegradation_samplingtime_unit",
    ]
    df["biodegradation_lowervalue"] = df["biodegradation_lowervalue"].astype(str).replace("nan", np.nan)
    df["biodegradation_uppervalue"] = df["biodegradation_uppervalue"].astype(str).replace("nan", np.nan)

    cols_to_check_if_contain_letters = [
        "substance_id",
        "inventory_num",
        "biodegradation_lowervalue",
        "biodegradation_uppervalue",
    ]
    for col in cols_to_check_if_contain_digits:
        df = df[~df[col].str.match(r"\d", na=False)]
    for col in cols_to_check_if_contain_letters:
        df = df[(~df[col].str.match(r"[a-zA-Z]", na=False))]
    df = df[(df["cas"].str.match(r"\d+\-\d\d\-\d", na=True))]

    df = df[df["biodegradation_lowervalue"].notna() | df["biodegradation_uppervalue"].notna()]
    df["biodegradation_lowervalue"] = df["biodegradation_lowervalue"].astype(float)
    df["biodegradation_uppervalue"] = df["biodegradation_uppervalue"].astype(float)
    df = df[
        (df["biodegradation_lowervalue"] > 0) | (df["biodegradation_uppervalue"] > 0)
    ]  # Remove negative biodegradation values
    df[df["biodegradation_lowervalue"] < 0] = np.nan
    df[df["biodegradation_uppervalue"] < 0] = np.nan

    df["guideline"] = df["guideline"].str.replace("other: ", "")
    guidelines_to_remove = ["", "None", None]
    for guideline_to_remove in guidelines_to_remove:
        df = df[df["guideline"] != guideline_to_remove]
    df = df[df["guideline"].notna()]
    guideline_qualifiers_to_remove = [
        "no guideline followed",
        "no guideline available",
        "no guideline available, no guideline available",
    ]
    for guideline_qualifier in guideline_qualifiers_to_remove:
        df = df[df["guideline_qualifier"] != guideline_qualifier]
    df = df[df["guideline_qualifier"].notna()]
    df = select_one_guideline(df)

    def handle_range_of_values(row) -> float:
        lower = row["biodegradation_lowervalue"]
        upper = row["biodegradation_uppervalue"]
        guideline = row["guideline"]
        difference = abs(lower - upper)
        if (~math.isnan(lower)) and (math.isnan(upper)):
            return lower
        elif (math.isnan(lower)) and (~math.isnan(upper)):
            return upper
        elif difference > 15:
            return -1.0
        elif "DOC Die Away Test" in guideline:
            if (lower < 70) and (upper > 70):
                return -1.0
            else:
                return (lower + upper) / 2
        elif "DOC Die Away Test" not in guideline:
            if (lower < 60) and (upper > 60):
                return -1.0
            else:
                return (lower + upper) / 2
        else:
            log.warn("When handling range of values did not match any condition, therefore deleted")
            return -1.0

    df["biodegradation_lowervalue"] = df.apply(handle_range_of_values, axis=1)
    len_before = len(df)
    df = df[df["biodegradation_lowervalue"] != -1.0]
    len_after = len(df)
    log.info(
        "Entries deleted because range too large or too close to threshold: ",
        deleted_entries=(len_before - len_after),
    )
    df.rename(columns={"biodegradation_lowervalue": "biodegradation_percent"}, inplace=True)
    df.drop(["biodegradation_uppervalue"], axis=1, inplace=True)

    df = df[df["substance_type"] == "organic"]
    df = df[df["composition"] != "mixture"]
    df = df[(df["composition"] == "mono-constituent substance") | (df["composition"].isna())]
    df = df[df["biodegradation_samplingtime"].notna()]
    df = df[df["biodegradation_samplingtime_unit"].notna()]
    df = df[
        (df["endpoint"] == "biodegradation in water: ready biodegradability")
        | (df["endpoint"] == "biodegradation in water: inherent biodegradability")
    ]
    df["endpoint"] = df["endpoint"].str.replace("biodegradation in water: ready biodegradability", "ready")
    df["endpoint"] = df["endpoint"].str.replace("biodegradation in water: inherent biodegradability", "inherent")
    df["interpretation_of_results"] = df["interpretation_of_results"].str.replace("other: ", "")
    df["degradation_parameter"] = df["degradation_parameter"].str.replace("other: ", "")
    df["degradation_parameter"] = df["degradation_parameter"].replace("not specified ", np.NaN)
    df["oxygen_conditions"] = df["oxygen_conditions"].replace("other: ", np.NaN)
    df["oxygen_conditions"] = df["oxygen_conditions"].replace("not specified ", np.NaN)
    df = df[df["study_result_type"] == "experimental study"]
    df = df[(df["smiles"].notna()) | (df["smiles_ph"].notna())]  # Remove any rows for which no smiles

    def split_guideline_adjust_reliability(row):
        reliability = row["reliability"]
        if reliability == "1 (reliable without restriction)":
            reliability = 1
        elif reliability == "2 (reliable with restrictions)":
            reliability = 2
        elif reliability == "4 (not assignable)":
            reliability = 4
        else:
            reliability = -1

        guideline = row["guideline"]
        if ("ISO" in guideline) and ("(" not in guideline):
            guideline_list_iso = guideline.split("-")
            guideline = guideline_list_iso[0].rstrip()
            principle = guideline_list_iso[1]
            for i in range(2, len(guideline_list_iso)):
                principle += guideline_list_iso[i]  # TODO check
            return pd.Series([guideline, "", reliability])
        guideline_list = guideline.split("(")
        guideline = guideline_list[0].rstrip()
        guideline = guideline.replace("Draft", "")
        if len(guideline_list) < 2:
            return pd.Series([guideline, "", reliability])
        principle = guideline_list[1]
        items_to_replace = [
            "Determination of the &quot;Ready&quot; Biodegradability - ",
            "Ready Biodegradability: ",
            "Inherent Biodegradability: ",
            "Inherent Biodegradability - ",
            "Ready biodegradability: ",
            "Ready Biodegradability - ",
            "Inherent biodegradability: ",
        ]
        if len(guideline_list) > 2:
            for i in range(2, len(guideline_list)):
                principle = principle + "(" + guideline_list[i]
            for item in items_to_replace:
                principle = principle.replace(item, "")
            return pd.Series([guideline, principle, reliability])
        principle = principle.replace(")", "")
        for item in items_to_replace:
            principle = principle.replace(item, "")
        if "EU" in guideline:
            if principle == "Carbon Dioxide Evolution Test":
                principle = "CO2 Evolution"
            elif principle == "Closed Bottle Test":
                principle = "Closed Bottle Test"
            elif principle == "Manometric Respirometry Test":
                principle = "Closed Respirometer"
            elif principle == "Dissolved Organic Carbon ":
                principle = "DOC Die Away"
        elif "OECD" in guideline:
            if principle == "CO2 Evolution Test":
                principle = "CO2 Evolution"
            elif principle == "DOC Die Away Test":
                principle = "DOC Die Away"
            elif principle == "Closed Bottle Test":
                principle = "Closed Bottle Test"
            elif principle == "Manometric Respirometry Test":
                principle = "Closed Respirometer"
        return pd.Series([guideline, principle, reliability])

    df[["guideline", "principle", "reliability"]] = df.apply(split_guideline_adjust_reliability, axis=1)
    df["biodegradation_percent"] = df["biodegradation_percent"].div(100)
    df = df[(df["biodegradation_percent"] < 1) & (df["biodegradation_percent"] > 0)]
    return df


def convert_reaction_time_to_days(df: pd.DataFrame) -> pd.DataFrame:
    def convert_samplingtime(row):
        unit = row["biodegradation_samplingtime_unit"]
        sampling_time = float(row["biodegradation_samplingtime"])
        if unit == "d":
            return sampling_time
        elif unit == "mo":
            return sampling_time * 30
        elif unit == "wk":
            return sampling_time * 7
        elif unit == "h":
            return sampling_time / 24
        elif unit == "min":
            return sampling_time / 60 / 24

    df["biodegradation_samplingtime"] = df.apply(convert_samplingtime, axis=1)
    df["biodegradation_samplingtime_unit"] = "d"
    return df


def additionality_to_paper(df_echa: pd.DataFrame) -> None:

    def load_regression_df() -> pd.DataFrame:
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
        return df_regression

    df_reg = load_regression_df()

    log.info("Data points in df_reg: ", echem_entries=len(df_reg))
    log.info("Unique cas in df_reg: ", echem_cas=df_reg["cas"].nunique())
    df_echa_not_in_regression = df_echa[~df_echa["cas"].isin(df_reg["cas"])]
    log.info("Echa data not in regression dataset from Huang et al.: ", new_data=len(df_echa_not_in_regression))
    log.info(
        "Unique new cas in echa data and not in regression dataset: ",
        new_cas=df_echa_not_in_regression["cas"].nunique(),
    )
    df_echa_not_in_regression.reset_index(inplace=True, drop=True)
    df_echa_not_in_regression.to_csv("datasets/biodegradation_echa_data_not_in_regression.csv")


def process_echa_data(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Biodegradation data points before processing", entries=len(df))
    log.info("Number of unique chemicals before processing: ", substance_ids=df["substance_id"].nunique())
    df = drop_and_rename(df, cols_to_delete, oldcolumns_to_new)
    df = check_and_process_columns(df, oldcolumns_to_new)
    df = convert_reaction_time_to_days(df)
    log.info("Biodegradation data points after processing", entries=len(df))
    log.info("Number of unique chemicals after processing: ", substance_ids=df["substance_id"].nunique())
    df.reset_index(inplace=True, drop=True)
    additionality_to_paper(df)
    return df


if __name__ == "__main__":
    if args.unzip:
        log.info("Unzipping")
        unzip_i6z_files(path=args.path_dossiers)
        
    directory_to_folders = args.path_dossiers + "_unzipped"
    dir_list = os.listdir(directory_to_folders)
    dir_list = [x for x in dir_list if "." not in x]  # to eliminate files called ".DS_Store"

    name_space = {
        "manifest": "http://iuclid6.echa.europa.eu/namespaces/manifest/v1",
        "parameter": "http://iuclid6.echa.europa.eu/namespaces/platform-container/v1",
        "study_record": f"http://iuclid6.echa.europa.eu/namespaces/ENDPOINT_STUDY_RECORD-{args.subtype}/7.0",
        "xsl": "http://www.w3.org/1999/XSL/Transform",
    }

    log.info("Getting information for all files")
    df = get_values_for_dir_list(
        name_space=name_space,
        dir_list=dir_list,
        directory_to_folders=directory_to_folders,
        subtype=args.subtype,
    )
    if args.subtype == "BiodegradationInWaterScreeningTests":
        df = process_echa_data(df=df)

    if args.save:
        df.to_csv(args.path_to_save)
