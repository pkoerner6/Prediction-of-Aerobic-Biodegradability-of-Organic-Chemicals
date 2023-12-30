import pytest
import pandas as pd
import numpy as np


d = {"A": [1, 2, 3], "B": [4, 5, 6]}

echem_df_paper_dict = {
    "name": {0: "-", 1: "-", 2: "-", 3: "-", 4: "-", 5: "-"},
    "name_type": {
        0: "IUPAC Name",
        1: "IUPAC Name",
        2: "IUPAC Name",
        3: "IUPAC Name",
        4: "IUPAC Name",
        5: "IUPAC Name",
    },
    "number": {
        0: "7027-11-4",
        1: "unknown",
        2: "5510-99-6",
        3: "417-790-1",
        4: "unknown",
        5: "unknown",
    },
    "number_type": {
        0: "CAS Number",
        1: np.nan,
        2: "CAS Number",
        3: "EC Number",
        4: np.nan,
        5: np.nan,
    },
    "member_of_category": {0: False, 1: False, 2: False, 3: False, 4: False, 5: False},
    "participant": {
        0: "ECHA CHEM",
        1: "ECHA CHEM",
        2: "ECHA CHEM",
        3: "ECHA CHEM",
        4: "ECHA CHEM",
        5: "ECHA CHEM",
    },
    "section": {
        0: "Biodegradation in water: screening tests",
        1: "Biodegradation in water: screening tests",
        2: "Biodegradation in water: screening tests",
        3: "Biodegradation in water: screening tests",
        4: "Biodegradation in water: screening tests",
        5: "Biodegradation in water: screening tests",
    },
    "values": {
        0: "Type of information:experimental study\nReliability:2 (reliable with restrictions)\nEndpoint:inherent biodegradability\n\n\nTest guideline, Guideline:OECD Guideline 302 B (Inherent biodegradability: Zahn-Wellens/EMPA Test)\n\n\nOxygen conditions:aerobic\n\n\n% Degradation, Value:19\n% Degradation, Sampling time:28 d\n\n\nInterpretation of results:other: poorly eliminated in water\n\n\n",
        1: "Type of information:experimental study\nReliability:1 (reliable without restriction)\nEndpoint:inherent biodegradability\n\n\nTest guideline, Guideline:OECD Guideline 302 C (Inherent Biodegradability: Modified MITI Test (II))\n\n\nOxygen conditions:aerobic\n\n\n% Degradation, Value:0\n% Degradation, Sampling time:28 d\n\n\nInterpretation of results:not inherently biodegradable\n\n\n",
        2: "Type of information:experimental study\nReliability:1 (reliable without restriction)\nEndpoint:inherent biodegradability\n\n\nTest guideline, Guideline:OECD Guideline 302 C (Inherent Biodegradability: Modified MITI Test (II))\n\n\nOxygen conditions:aerobic\n\n\n% Degradation, Value:8\n% Degradation, Sampling time:28 d\n\n\nInterpretation of results:not inherently biodegradable\n\n\n",
        3: "Type of information:experimental study\nReliability:2 (reliable with restrictions)\nEndpoint:inherent biodegradability\n\n\nTest guideline, Guideline:EU Method C.9 (Biodegradation: Zahn-Wellens Test)\n\nTest guideline, Guideline:OECD Guideline 302 B (Inherent biodegradability: Zahn-Wellens/EMPA Test)\n\n\nOxygen conditions:aerobic\n\n\n% Degradation, Value:90\n% Degradation, Sampling time:28 d\n\n\nInterpretation of results:inherently biodegradable\n\n\n",
        4: "Type of information:experimental study\nReliability:1 (reliable without restriction)\nEndpoint:inherent biodegradability\n\n\nTest guideline, Guideline:EU Method C.9 (Biodegradation: Zahn-Wellens Test)\n\n\nOxygen conditions:aerobic\n\n\n% Degradation, Value:5.19—5.84\n% Degradation, Sampling time:28 d\n\n\nInterpretation of results:under test conditions no biodegradation observed\n\n\n",
        5: "Type of information:experimental study\nReliability:1 (reliable without restriction)\nEndpoint:inherent biodegradability\n\n\nTest guideline, Guideline:OECD Guideline 302 C (Inherent Biodegradability: Modified MITI Test (II))\n\n\nOxygen conditions:aerobic\n\n\n% Degradation, Value:0\n% Degradation, Sampling time:28 d\n\n\nInterpretation of results:under test conditions no biodegradation observed\n\n\n",
    },
}

new_echem_dict = {
    "substance_name": {
        0: "2-methylaniline",
        1: "2-methylaniline",
        2: "1,2-bis(2-ethylhexyl) benzene-1,2-dicarboxylate",
    },
    "name_type": {0: "IUPAC Name", 1: "IUPAC Name", 2: "IUPAC Name"},
    "number": {0: "95-53-4", 1: "95-53-4", 2: "117-81-7"},
    "number_type": {0: "CAS Number", 1: "CAS Number", 2: "CAS Number"},
    "member_of_category": {0: "False", 1: "False", 2: "False"},
    "substance_link": {
        0: "https://echa.europa.eu/registration-dossier/-/registered-dossier/14441//?documentUUID=ef0de8a5-7c48-41b3-87f0-a6de593f8ba4",
        1: "https://echa.europa.eu/registration-dossier/-/registered-dossier/14441//?documentUUID=ef0de8a5-7c48-41b3-87f0-a6de593f8ba4",
        2: "https://echa.europa.eu/registration-dossier/-/registered-dossier/15358//?documentUUID=94aad6bf-11f2-48d2-b724-d25c1f753e30",
    },
    "participant": {0: "ECHA REACH", 1: "ECHA REACH", 2: "ECHA REACH"},
    "Participant Link": {
        0: "https://echa.europa.eu/information-on-chemicals/registered-substances",
        1: "https://echa.europa.eu/information-on-chemicals/registered-substances",
        2: "https://echa.europa.eu/information-on-chemicals/registered-substances",
    },
    "section": {
        0: "Biodegradation in water: screening tests",
        1: "Biodegradation in water: screening tests",
        2: "Biodegradation in water: screening tests",
    },
    "endpoint_link": {
        0: "https://echa.europa.eu/registration-dossier/-/registered-dossier/14441/5/3/2/?documentUUID=d5e751d8-8854-48a5-a620-7574ab96ffcd",
        1: "https://echa.europa.eu/registration-dossier/-/registered-dossier/14441/5/3/2/?documentUUID=cf8f16b2-5585-4af8-83f9-8297f511e227",
        2: "https://echa.europa.eu/registration-dossier/-/registered-dossier/15358/5/3/2/?documentUUID=ecb102da-6930-44e1-9dcd-42e13074273d",
    },
    "type_of_information": {
        0: "experimental study",
        1: "experimental study",
        2: "experimental study",
    },
    "reliability": {
        0: "2 (reliable with restrictions)",
        1: "2 (reliable with restrictions)",
        2: "2 (reliable with restrictions)",
    },
    "endpoint": {
        0: "ready biodegradability",
        1: "ready biodegradability",
        2: "ready biodegradability",
    },
    "reference_year": {0: "2005", 1: "2005", 2: "2005"},
    "test_guideline": {
        0: "OECD Guideline 301 E (Ready biodegradability: Modified OECD Screening Test)",
        1: "OECD Guideline 301 A (old version) (Ready Biodegradability: Modified AFNOR Test)",
        2: "OECD Guideline 311 (Anaerobic Biodegradability of Organic Compounds in Digested Sludge: Measurement of Gas Production)",
    },
    "degradation_value": {0: "> 90 %", 1: "88—90 %", 2: "100 %"},
    "sampling_time": {0: "28 d", 1: "28 d", 2: "28 d"},
    "interpretation_of_results": {
        0: "readily biodegradable",
        1: "readily biodegradable",
        2: "readily biodegradable",
    },
}


regression_dict_full = {
    "name": {
        0: "(2H)chloroform",
        1: "diquat",
        2: "calcium hydrogen borate",
        3: "3,5-diaminobenzoic acid",
        4: "octane-1,8-diamine",
        5: "N,N'-diphenylguanidine hydrochloride",
        6: "N,N'-diphenylguanidine hydrochloride",
    },
    "name_type": {
        0: np.nan,
        1: "IUPAC Name",
        2: "IUPAC Name",
        3: "IUPAC Name",
        4: "IUPAC Name",
        5: "IUPAC Name",
        6: "IUPAC Name",
    },
    "cas": {
        0: "865-49-6",
        1: "85-00-7",
        2: "12040-58-3",
        3: "535-87-5",
        4: "373-44-4",
        5: "24245-27-0",
        6: "24245-27-0",
    },
    "smiles": {
        0: "[2H]C(Cl)(Cl)Cl",
        1: "[Br-].[Br-].c1cc[n+]2c(c1)-c1cccc[n+]1CC2",
        2: "[Ca+2].[Ca+2].[Ca+2].[O-]B([O-])[O-].[O-]B([O-])[O-]",
        3: "[Cl-].[Cl-].[NH3+]c1cc([NH3+])cc(C(=O)O)c1",
        4: "[Cl-].[Cl-].[NH3+]CCCCCCCC[NH3+]",
        5: "[Cl-].[NH2+]=C(Nc1ccccc1)Nc1ccccc1",
        6: "[Cl-].[NH2+]=C(Nc1ccccc1)Nc1ccccc1",
    },
    "reliability": {0: 2, 1: 1, 2: 1, 3: 2, 4: 2, 5: 1, 6: 1},
    "endpoint": {
        0: "Ready",
        1: "Ready",
        2: "Ready",
        3: "Ready",
        4: "Ready",
        5: "Ready",
        6: "Ready",
    },
    "guideline": {
        0: "OECD Guideline 301 C",
        1: "OECD Guideline 301 C",
        2: "OECD Guideline 301 B",
        3: "OECD Guideline 301 D",
        4: "OECD Guideline 301 E",
        5: "OECD Guideline 301 D",
        6: "OECD Guideline 301 D",
    },
    "principle": {
        0: "Closed Respirometer",
        1: "Closed Respirometer",
        2: "CO2 Evolution",
        3: "Closed Bottle Test",
        4: "DOC Die Away",
        5: "Closed Bottle Test",
        6: "Closed Bottle Test",
    },
    "time_day": {0: 14.0, 1: 28.0, 2: 28.0, 3: 30.0, 4: 29.0, 5: 28.0, 6: 14.0},
    "biodegradation_percent": {
        0: 0.0,
        1: 0.0,
        2: 0.11,
        3: 0.0,
        4: 1.0,
        5: 0.85,
        6: 0.86,
    },
    "pka_acid_1": {0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 9.1, 6: 9.1},
    "pka_acid_2": {0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 10.0, 6: 10.0},
    "pka_acid_3": {0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 10.0, 6: 10.0},
    "pka_acid_4": {0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 10.0, 6: 10.0},
    "pka_base_1": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0},
    "pka_base_2": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0},
    "pka_base_3": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 10.7, 5: 0.0, 6: 0.0},
    "pka_base_4": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 10.7, 5: 9.8, 6: 9.8},
    "α_acid_0": {
        0: 0.9974881135685901,
        1: 0.9974881135685901,
        2: 0.9974881135685901,
        3: 0.9974881135685901,
        4: 0.9974881135685901,
        5: 0.9803894001597887,
        6: 0.9803894001597887,
    },
    "α_acid_1": {
        0: 0.0025055768580650385,
        1: 0.0025055768580650385,
        2: 0.0025055768580650385,
        3: 0.0025055768580650385,
        4: 0.0025055768580650385,
        5: 0.019561340241337752,
        6: 0.019561340241337752,
    },
    "α_acid_2": {
        0: 6.293724512877999e-06,
        1: 6.293724512877999e-06,
        2: 6.293724512877999e-06,
        3: 6.293724512877999e-06,
        4: 6.293724512877999e-06,
        5: 4.913586513435883e-05,
        6: 4.913586513435883e-05,
    },
    "α_acid_3": {
        0: 1.580912120755755e-08,
        1: 1.580912120755755e-08,
        2: 1.580912120755755e-08,
        3: 1.580912120755755e-08,
        4: 1.580912120755755e-08,
        5: 1.234237129314811e-07,
        6: 1.234237129314811e-07,
    },
    "α_acid_4": {
        0: 3.971071705535432e-11,
        1: 3.971071705535432e-11,
        2: 3.971071705535432e-11,
        3: 3.971071705535432e-11,
        4: 3.971071705535432e-11,
        5: 3.1002634983912204e-10,
        6: 3.1002634983912204e-10,
    },
    "α_base_0": {
        0: 2.5118863315095412e-30,
        1: 2.5118863315095412e-30,
        2: 2.5118863315095412e-30,
        3: 2.5118863315095412e-30,
        4: 1.5840988013932986e-15,
        5: 6.28455393475768e-23,
        6: 6.28455393475768e-23,
    },
    "α_base_1": {
        0: 6.309573193613216e-23,
        1: 6.309573193613216e-23,
        2: 6.309573193613216e-23,
        3: 6.309573193613216e-23,
        4: 3.979076285390431e-08,
        5: 1.5786085756808018e-15,
        6: 1.5786085756808018e-15,
    },
    "α_base_2": {
        0: 1.5848931293653668e-15,
        1: 1.5848931293653668e-15,
        2: 1.5848931293653668e-15,
        3: 1.5848931293653668e-15,
        4: 0.9994987731213804,
        5: 3.965285461917285e-08,
        6: 3.965285461917285e-08,
    },
    "α_base_3": {
        0: 3.9810715470456376e-08,
        1: 3.9810715470456376e-08,
        2: 3.9810715470456376e-08,
        3: 3.9810715470456376e-08,
        4: 0.0005009360251145588,
        5: 0.9960346748852266,
        6: 0.9960346748852266,
    },
    "α_base_4": {
        0: 0.9999999601892829,
        1: 0.9999999601892829,
        2: 0.9999999601892829,
        3: 0.9999999601892829,
        4: 2.510627406514083e-07,
        5: 0.003965285461917307,
        6: 0.003965285461917307,
    },
}


regression_dict_with_star = {
    "name": {
        0: "(2H)chloroform",
        1: "diquat",
        2: "calcium hydrogen borate",
        3: "3,5-diaminobenzoic acid",
        4: "octane-1,8-diamine",
        5: "N,N'-diphenylguanidine hydrochloride",
        6: "N,N'-diphenylguanidine hydrochloride",
        7: "Ichthammol",
        8: "trisodium [29H,31H-phthalocyaninetrisulphonato(5-)-N29,N30,N31,N32]cuprate(3-)",
    },
    "name_type": {
        0: np.nan,
        1: "IUPAC Name",
        2: "IUPAC Name",
        3: "IUPAC Name",
        4: "IUPAC Name",
        5: "IUPAC Name",
        6: "IUPAC Name",
        7: "",
        8: "",
    },
    "cas": {
        0: "865-49-6",
        1: "85-00-7",
        2: "12040-58-3",
        3: "535-87-5",
        4: "373-44-4",
        5: "24245-27-0",
        6: "24245-27-0",
        7: "8029-68-3",
        8: "1330-39-8",
    },
    "smiles": {
        0: "[2H]C(Cl)(Cl)Cl",
        1: "[Br-].[Br-].c1cc[n+]2c(c1)-c1cccc[n+]1CC2",
        2: "[Ca+2].[Ca+2].[Ca+2].[O-]B([O-])[O-].[O-]B([O-])[O-]",
        3: "[Cl-].[Cl-].[NH3+]c1cc([NH3+])cc(C(=O)O)c1",
        4: "[Cl-].[Cl-].[NH3+]CCCCCCCC[NH3+]",
        5: "[Cl-].[NH2+]=C(Nc1ccccc1)Nc1ccccc1",
        6: "[Cl-].[NH2+]=C(Nc1ccccc1)Nc1ccccc1",
        7: "*C(*)(*)S(=O)(=O)[O-].[NH4+]",
        8: "*S(=O)(=O)[O-].*S(=O)(=O)[O-].*S(=O)(=O)[O-].[Cu+2].[Na+].[Na+].[Na+].c1ccc2c(c1)-c1nc-2nc2[n-]c(nc3nc(nc4[n-]c(n1)c1ccccc41)-c1ccccc1-3)c1ccccc21",
    },
    "reliability": {0: 2, 1: 1, 2: 1, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1},
    "endpoint": {
        0: "Ready",
        1: "Ready",
        2: "Ready",
        3: "Ready",
        4: "Ready",
        5: "Ready",
        6: "Ready",
        7: "Ready",
        8: "Inherent",
    },
    "guideline": {
        0: "OECD Guideline 301 C",
        1: "OECD Guideline 301 C",
        2: "OECD Guideline 301 B",
        3: "OECD Guideline 301 D",
        4: "OECD Guideline 301 E",
        5: "OECD Guideline 301 D",
        6: "OECD Guideline 301 D",
        7: "EU Method C.4-E",
        8: "OECD Guideline 302 B",
    },
    "principle": {
        0: "Closed Respirometer",
        1: "Closed Respirometer",
        2: "CO2 Evolution",
        3: "Closed Bottle Test",
        4: "DOC Die Away",
        5: "Closed Bottle Test",
        6: "Closed Bottle Test",
        7: "Closed Bottle Test",
        8: "DOC Die Away",
    },
    "time_day": {
        0: 14.0,
        1: 28.0,
        2: 28.0,
        3: 30.0,
        4: 29.0,
        5: 28.0,
        6: 14.0,
        7: 28.0,
        8: 28.0,
    },
    "biodegradation_percent": {
        0: 0.0,
        1: 0.0,
        2: 0.11,
        3: 0.0,
        4: 1.0,
        5: 0.85,
        6: 0.86,
        7: 0.079,
        8: 0.0,
    },
}

regression_dict = {
    "name": {
        0: "(2H)chloroform",
        1: "diquat",
        2: "calcium hydrogen borate",
        3: "3,5-diaminobenzoic acid",
        4: "octane-1,8-diamine",
        5: "N,N'-diphenylguanidine hydrochloride",
        6: "N,N'-diphenylguanidine hydrochloride",
    },
    "name_type": {
        0: np.nan,
        1: "IUPAC Name",
        2: "IUPAC Name",
        3: "IUPAC Name",
        4: "IUPAC Name",
        5: "IUPAC Name",
        6: "IUPAC Name",
    },
    "cas": {
        0: "865-49-6",
        1: "85-00-7",
        2: "12040-58-3",
        3: "535-87-5",
        4: "373-44-4",
        5: "24245-27-0",
        6: "24245-27-0",
    },
    "smiles": {
        0: "[2H]C(Cl)(Cl)Cl",
        1: "[Br-].[Br-].c1cc[n+]2c(c1)-c1cccc[n+]1CC2",
        2: "[Ca+2].[Ca+2].[Ca+2].[O-]B([O-])[O-].[O-]B([O-])[O-]",
        3: "[Cl-].[Cl-].[NH3+]c1cc([NH3+])cc(C(=O)O)c1",
        4: "[Cl-].[Cl-].[NH3+]CCCCCCCC[NH3+]",
        5: "[Cl-].[NH2+]=C(Nc1ccccc1)Nc1ccccc1",
        6: "[Cl-].[NH2+]=C(Nc1ccccc1)Nc1ccccc1",
    },
    "reliability": {0: 2, 1: 1, 2: 1, 3: 2, 4: 2, 5: 1, 6: 1},
    "endpoint": {
        0: "Ready",
        1: "Ready",
        2: "Ready",
        3: "Ready",
        4: "Ready",
        5: "Ready",
        6: "Ready",
    },
    "guideline": {
        0: "OECD Guideline 301 C",
        1: "OECD Guideline 301 C",
        2: "OECD Guideline 301 B",
        3: "OECD Guideline 301 D",
        4: "OECD Guideline 301 E",
        5: "OECD Guideline 301 D",
        6: "OECD Guideline 301 D",
    },
    "principle": {
        0: "Closed Respirometer",
        1: "Closed Respirometer",
        2: "CO2 Evolution",
        3: "Closed Bottle Test",
        4: "DOC Die Away",
        5: "Closed Bottle Test",
        6: "Closed Bottle Test",
    },
    "time_day": {0: 14.0, 1: 28.0, 2: 28.0, 3: 30.0, 4: 29.0, 5: 28.0, 6: 14.0},
    "biodegradation_percent": {
        0: 0.0,
        1: 0.0,
        2: 0.11,
        3: 0.0,
        4: 1.0,
        5: 0.85,
        6: 0.86,
    },
    "inchi": {
        0: "InChI=1S/CHCl3/c2-1(3)4/h1H/i1D",
        1: "InChI=1S/C12H12N2.2BrH/c1-3-7-13-9-10-14-8-4-2-6-12(14)11(13)5-1;;/h1-8H,9-10H2;2*1H/q+2;;/p-2",
        2: "InChI=1S/2BO3.3Ca/c2*2-1(3)4;;;/q2*-3;3*+2",
        3: "InChI=1S/C7H8N2O2.2ClH/c8-5-1-4(7(10)11)2-6(9)3-5;;/h1-3H,8-9H2,(H,10,11);2*1H",
        4: "InChI=1S/C8H20N2.2ClH/c9-7-5-3-1-2-4-6-8-10;;/h1-10H2;2*1H",
        5: "InChI=1S/C13H13N3.ClH/c14-13(15-11-7-3-1-4-8-11)16-12-9-5-2-6-10-12;/h1-10H,(H3,14,15,16);1H",
        6: "InChI=1S/C13H13N3.ClH/c14-13(15-11-7-3-1-4-8-11)16-12-9-5-2-6-10-12;/h1-10H,(H3,14,15,16);1H",
    },
}


class_original_dict = {
    "name": {
        0: "Cyclopenta[c]cyclopropa[g][1,6]diazacyclotetradecine-12a(1H)-carboxylic acid, 2,3,3a,4,5,6,7,8,9,11a,12,13,14,14a-tetradecahydro-2-[[7-methoxy-8-methyl-2-[4-(1-methylethyl)-2-thiazolyl]-4-quinolinyl]oxy]-5-methyl-4,14-dioxo-, (2R,3aR,10Z,11aS,12aR,14aR)-",
        1: "2-methylprop-2-enoic acid",
        2: "3-(Diisobutoxy-thiophosphorylsulfanyl)-2-methyl-propionic acid",
        3: "Phenol, 2-methyl-4,6-dinitro-",
        4: "2-[4-(1,3-dihydro-1,3-dioxo-2H-isoindol-2-yl)phenyl]butyric acid",
        5: "Benzyl (S)-N-(1-oxopentyl)-N-((2'-(1H-tetrazole-5-yl)-1,1'-biphenyl-4-yl)methyl)-L-valinate",
    },
    "name_type": {
        0: np.nan,
        1: "IUPAC Name",
        2: "IUPAC Name",
        3: "IUPAC Name",
        4: np.nan,
        5: "IUPAC Name",
    },
    "cas": {
        0: "923604-58-4",
        1: "79-41-4",
        2: "268567-32-4",
        3: "534-52-1",
        4: "94232-67-4",
        5: "137863-20-8",
    },
    "source": {
        0: "ClassDataset_original",
        1: "ClassDataset_original",
        2: "ClassDataset_original",
        3: "ClassDataset_original",
        4: "ClassDataset_original",
        5: "ClassDataset_original",
    },
    "smiles": {
        0: "COc1ccc2c(O[C@@H]3C[C@H]4C(=O)N[C@]5(C(=O)O)C[C@H]5C=CCCCCN(C)C(=O)[C@@H]4C3)cc(-c3nc(C(C)C)cs3)nc2c1C",
        1: "C=C(C)C(=O)O",
        2: "CC(C)COP(=S)(OCC(C)C)SCC(C)C(=O)O",
        3: "Cc1cc([N+](=O)[O-])cc([N+](=O)[O-])c1O",
        4: "CCC(C(=O)O)c1ccc(N2C(=O)c3ccccc3C2=O)cc1",
        5: "CCCCC(=O)N(Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1)[C@H](C(=O)OCc1ccccc1)C(C)C",
    },
    "y_true": {0: 0, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0},
}

class_external_dict = {
    "smiles": {
        0: "Br/C=C/c1ccccc1",
        1: "Br/C=C\\Br",
        2: "Br/C=C\\c1ccccc1",
        3: "BrC(Br)C(Br)Br",
        4: "BrC(Br)c1ccccc1OCC1CO1",
        5: "Brc1c(Br)c(Br)c(Br)c(Br)c1Br",
    },
    "y_true": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
}

class_all_dict = {
    "name": {
        0: "Cyclopenta[c]cyclopropa[g][1,6]diazacyclotetradecine-12a(1H)-carboxylic acid, 2,3,3a,4,5,6,7,8,9,11a,12,13,14,14a-tetradecahydro-2-[[7-methoxy-8-methyl-2-[4-(1-methylethyl)-2-thiazolyl]-4-quinolinyl]oxy]-5-methyl-4,14-dioxo-, (2R,3aR,10Z,11aS,12aR,14aR)-",
        1: "2-methylprop-2-enoic acid",
        2: "3-(Diisobutoxy-thiophosphorylsulfanyl)-2-methyl-propionic acid",
        3: "Phenol, 2-methyl-4,6-dinitro-",
        4: "2-[4-(1,3-dihydro-1,3-dioxo-2H-isoindol-2-yl)phenyl]butyric acid",
        5: "Benzyl (S)-N-(1-oxopentyl)-N-((2'-(1H-tetrazole-5-yl)-1,1'-biphenyl-4-yl)methyl)-L-valinate",
    },
    "name_type": {
        0: np.nan,
        1: "IUPAC Name",
        2: "IUPAC Name",
        3: "IUPAC Name",
        4: np.nan,
        5: "IUPAC Name",
    },
    "cas": {
        0: "923604-58-4",
        1: "79-41-4",
        2: "268567-32-4",
        3: "534-52-1",
        4: "94232-67-4",
        5: "137863-20-8",
    },
    "source": {
        0: "ClassDataset_original",
        1: "ClassDataset_original",
        2: "ClassDataset_original",
        3: "ClassDataset_original",
        4: "ClassDataset_original",
        5: "ClassDataset_original",
    },
    "smiles": {
        0: "COc1ccc2c(O[C@@H]3C[C@H]4C(=O)N[C@]5(C(=O)O)C[C@H]5C=CCCCCN(C)C(=O)[C@@H]4C3)cc(-c3nc(C(C)C)cs3)nc2c1C",
        1: "C=C(C)C(=O)O",
        2: "CC(C)COP(=S)(OCC(C)C)SCC(C)C(=O)O",
        3: "Cc1cc([N+](=O)[O-])cc([N+](=O)[O-])c1O",
        4: "CCC(C(=O)O)c1ccc(N2C(=O)c3ccccc3C2=O)cc1",
        5: "CCCCC(=O)N(Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1)[C@H](C(=O)OCc1ccccc1)C(C)C",
    },
    "y_true": {0: 0, 1: 1, 2: 0, 3: 0, 4: 1, 5: 0},
}

class_all_no_processing_dict = {
    "Substance Name": {
        0: "Cyclopenta[c]cyclopropa[g][1,6]diazacyclotetradecine-12a(1H)-carboxylic acid, 2,3,3a,4,5,6,7,8,9,11a,12,13,14,14a-tetradecahydro-2-[[7-methoxy-8-methyl-2-[4-(1-methylethyl)-2-thiazolyl]-4-quinolinyl]oxy]-5-methyl-4,14-dioxo-, (2R,3aR,10Z,11aS,12aR,14aR)-",
        1: "2-methylprop-2-enoic acid",
        2: "3-(Diisobutoxy-thiophosphorylsulfanyl)-2-methyl-propionic acid",
        3: "Phenol, 2-methyl-4,6-dinitro-",
        4: "2-[4-(1,3-dihydro-1,3-dioxo-2H-isoindol-2-yl)phenyl]butyric acid",
        5: "Benzyl (S)-N-(1-oxopentyl)-N-((2'-(1H-tetrazole-5-yl)-1,1'-biphenyl-4-yl)methyl)-L-valinate",
    },
    "Name type": {
        0: np.nan,
        1: "IUPAC Name",
        2: "IUPAC Name",
        3: "IUPAC Name",
        4: np.nan,
        5: "IUPAC Name",
    },
    "CAS Number": {
        0: "923604-58-4",
        1: "79-41-4",
        2: "268567-32-4",
        3: "534-52-1",
        4: "94232-67-4",
        5: "137863-20-8",
    },
    "Source": {
        0: "ClassDataset_original",
        1: "ClassDataset_original",
        2: "ClassDataset_original",
        3: "ClassDataset_original",
        4: "ClassDataset_original",
        5: "ClassDataset_original",
    },
    "Smiles": {
        0: "COc1ccc2c(O[C@@H]3C[C@H]4C(=O)N[C@]5(C(=O)O)C[C@H]5C=CCCCCN(C)C(=O)[C@@H]4C3)cc(-c3nc(C(C)C)cs3)nc2c1C",
        1: "C=C(C)C(=O)O",
        2: "CC(C)COP(=S)(OCC(C)C)SCC(C)C(=O)O",
        3: "Cc1cc([N+](=O)[O-])cc([N+](=O)[O-])c1O",
        4: "CCC(C(=O)O)c1ccc(N2C(=O)c3ccccc3C2=O)cc1",
        5: "CCCCC(=O)N(Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1)[C@H](C(=O)OCc1ccccc1)C(C)C",
    },
    "Class": {0: 0, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0},
    "pka_acid_1": {0: 4.1, 1: 4.2, 2: 4.2, 3: 4.2, 4: 4.2, 5: 4.2},
    "pka_acid_2": {0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 10.0},
    "pka_acid_3": {0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 10.0},
    "pka_acid_4": {0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 10.0},
    "pka_base_1": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
    "pka_base_2": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
    "pka_base_3": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
    "pka_base_4": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
    "α_acid_0": {
        0: 0.0004996785048058,
        1: 0.000628976590774,
        2: 0.000628976590774,
        3: 0.000628976590774,
        4: 0.000628976590774,
        5: 0.000628976590774,
    },
    "α_acid_1": {
        0: 0.996989690239032,
        1: 0.996860716935166,
        2: 0.996860716935166,
        3: 0.996860716935166,
        4: 0.996860716935166,
        5: 0.996860716935166,
    },
    "α_acid_2": {
        0: 0.0025043248752663,
        1: 0.0025040009089743,
        2: 0.0025040009089743,
        3: 0.0025040009089743,
        4: 0.0025040009089743,
        5: 0.0025040009089743,
    },
    "α_acid_3": {
        0: 6.29057967427355e-06,
        1: 6.2897659077403885e-06,
        2: 6.2897659077403885e-06,
        3: 6.2897659077403885e-06,
        4: 6.2897659077403885e-06,
        5: 6.2897659077403885e-06,
    },
    "α_acid_4": {
        0: 1.5801221730137746e-08,
        1: 1.5799177641024683e-08,
        2: 1.5799177641024683e-08,
        3: 1.5799177641024683e-08,
        4: 1.5799177641024683e-08,
        5: 1.5799177641024683e-08,
    },
    "α_base_0": {
        0: 2.511886331509541e-30,
        1: 2.511886331509541e-30,
        2: 2.511886331509541e-30,
        3: 2.511886331509541e-30,
        4: 2.511886331509541e-30,
        5: 2.511886331509541e-30,
    },
    "α_base_1": {
        0: 6.309573193613216e-23,
        1: 6.309573193613216e-23,
        2: 6.309573193613216e-23,
        3: 6.309573193613216e-23,
        4: 6.309573193613216e-23,
        5: 6.309573193613216e-23,
    },
    "α_base_2": {
        0: 1.584893129365367e-15,
        1: 1.584893129365367e-15,
        2: 1.584893129365367e-15,
        3: 1.584893129365367e-15,
        4: 1.584893129365367e-15,
        5: 1.584893129365367e-15,
    },
    "α_base_3": {
        0: 3.9810715470456376e-08,
        1: 3.9810715470456376e-08,
        2: 3.9810715470456376e-08,
        3: 3.9810715470456376e-08,
        4: 3.9810715470456376e-08,
        5: 3.9810715470456376e-08,
    },
    "α_base_4": {
        0: 0.9999999601892828,
        1: 0.9999999601892828,
        2: 0.9999999601892828,
        3: 0.9999999601892828,
        4: 0.9999999601892828,
        5: 0.9999999601892828,
    },
}

class_for_labelling_dict = {
    "principle": {
        0: "DOC Die Away",
        1: "Closed Respirometer",
        2: "CO2 Evolution",
        3: "Closed Bottle Test",
        4: "DOC Die Away",
        5: "DOC Die Away",
        6: "Closed Respirometer",
        7: "Closed Bottle Test",
        8: "Closed Bottle Test",
    },
    "biodegradation_percent": {
        0: 0.7,
        1: 0.5,
        2: 0.8,
        3: 0.6,
        4: 0.69,
        5: 0.71,
        6: 0.59,
        7: 0.61,
        8: 0.6,
    },
    "correct_label": {
        0: 1,
        1: 0,
        2: 1,
        3: 1,
        4: 0,
        5: 1,
        6: 0,
        7: 1,
        8: 1,
    },
}

outlier_detection_dict = {
    "cas": {
        0: "865-49-6",
        1: "865-49-6",
        2: "865-49-6",
        3: "865-49-6",
        4: "373-44-4",
        5: "373-44-4",
        6: "373-44-4",
        7: "373-44-4",
        8: "373-44-4",
        9: "24245-27-0",
        10: "24245-27-0",
        11: "24245-27-0",
        12: "24245-27-0",
    },
    "biodegradation_percent": {
        0: 0,
        1: 0.2,
        2: 0.1,
        3: 0.9,
        4: 0.15,
        5: 0,
        6: 0.1,
        7: 0.8,
        8: 0.9,
        9: 0,
        10: 0.1,
        11: 0.9,
        12: 0.15,
    },
}

group_remove_duplicates_dict = {
    "name": {
        0: "Cyclopenta acid",
        1: "2-methylprop-2-enoic acid",
        2: "2-methyl-propionic acid",
        3: "Phenol",
        4: "butyric acid",
        5: "Benzyl",
        6: "Benzyl",
        7: "Benzyl",
        8: "Cyclopenta acid",
    },
    "cas": {
        0: "923604-58-4",
        1: "79-41-4",
        2: "268567-32-4",
        3: "534-52-1",
        4: "94232-67-4",
        5: "137863-20-8",
        6: "137863-20-8",
        7: "137863-20-8",
        8: "923604-58-4",
    },
    "smiles": {
        0: "COc1ccc2c(O[C@@H]3C[C@H]4C(=O)N[C@]5(C(=O)O)C[C@H]5C=CCCCCN(C)C(=O)[C@@H]4C3)cc(-c3nc(C(C)C)cs3)nc2c1C",
        1: "C=C(C)C(=O)O",
        2: "CC(C)COP(=S)(OCC(C)C)SCC(C)C(=O)O",
        3: "Cc1cc([N+](=O)[O-])cc([N+](=O)[O-])c1O",
        4: "CCC(C(=O)O)c1ccc(N2C(=O)c3ccccc3C2=O)cc1",
        5: "CCCCC(=O)N(Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1)[C@H](C(=O)OCc1ccccc1)C(C)C",
        6: "CCCCC(=O)N(Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1)[C@H](C(=O)OCc1ccccc1)C(C)C",
        7: "CCCCC(=O)N(Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1)[C@H](C(=O)OCc1ccccc1)C(C)C",
        8: "COc1ccc2c(O[C@@H]3C[C@H]4C(=O)N[C@]5(C(=O)O)C[C@H]5C=CCCCCN(C)C(=O)[C@@H]4C3)cc(-c3nc(C(C)C)cs3)nc2c1C",
    },
    "y_true": {0: 0, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1},
}


df_a_full_original_dict = {
    "name": {
        0: "(2H)chloroform",
        1: "diquat",
        3: "3,5-diaminobenzoic acid",
        4: "octane-1,8-diamine",
        5: "N,N'-diphenylguanidine hydrochloride",
        6: "N,N'-diphenylguanidine hydrochloride",
        7: "4',5'-dibromo-3',6'-dihydroxy-3H-spiro[2-benzofuran-1,9'-xanthene]-3-one",
    },
    "name_type": {
        0: np.nan,
        1: "IUPAC Name",
        3: "IUPAC Name",
        4: "IUPAC Name",
        5: "IUPAC Name",
        6: "IUPAC Name",
        7: "IUPAC Name",
    },
    "cas": {
        0: "865-49-6",
        1: "85-00-7",
        3: "535-87-5",
        4: "373-44-4",
        5: "24245-27-0",
        6: "24245-27-0",
        7: "596-03-2",
    },
    "smiles": {
        0: "[2H]C(Cl)(Cl)Cl",
        1: "[Br-].[Br-].c1cc[n+]2c(c1)-c1cccc[n+]1CC2",
        3: "[Cl-].[Cl-].[NH3+]c1cc([NH3+])cc(C(=O)O)c1",
        4: "[Cl-].[Cl-].[NH3+]CCCCCCCC[NH3+]",
        5: "[Cl-].[NH2+]=C(Nc1ccccc1)Nc1ccccc1",
        6: "[Cl-].[NH2+]=C(Nc1ccccc1)Nc1ccccc1",
        7: "O=C1OC2(c3ccccc31)c1ccc(O)c(Br)c1Oc1c2ccc(O)c1Br",
    },
    "reliability": {0: 2, 1: 1, 3: 2, 4: 2, 5: 1, 6: 1, 7: 2},
    "endpoint": {
        0: "ready",
        1: "ready",
        3: "ready",
        4: "ready",
        5: "ready",
        6: "ready",
        7: "ready",
    },
    "guideline": {
        0: "OECD Guideline 301 C",
        1: "OECD Guideline 301 C",
        3: "OECD Guideline 301 D",
        4: "OECD Guideline 301 E",
        5: "OECD Guideline 301 D",
        6: "OECD Guideline 301 D",
        7: "OECD Guideline 301 C",
    },
    "principle": {
        0: "Closed Respirometer",
        1: "Closed Respirometer",
        3: "Closed Bottle Test",
        4: "DOC Die Away",
        5: "Closed Bottle Test",
        6: "Closed Bottle Test",
        7: "Closed Respirometer",
    },
    "time_day": {0: 14.0, 1: 28.0, 3: 30.0, 4: 29.0, 5: 28.0, 6: 14.0, 7: 28.0},
    "biodegradation_percent": {
        0: 0.0,
        1: 0.0,
        3: 0.0,
        4: 1.0,
        5: 0.85,
        6: 0.86,
        7: 0.0,
    },
}

df_b_dict = {
    "name": {
        14: "[29H,31H-phthalocyaninato(2-)-N29,N30,N31,N32]iron",
        17: "5-Diazo-2,4,6(1H, 3H, 5H)-pyrimidinetrione",
        21: "sodium chlorate",
        25: "trisodium 12-({4-[2-(sulfonatooxy)ethanesulfonyl]phenyl}sulfamoyl)-9,18,27,36,37,39,40,41-octaaza-38-nickeladecacyclo[17.17.3.1¹⁰,¹⁷.1²⁸,³⁵.0²,⁷.0⁸,³⁷.0¹¹,¹⁶.0²⁰,²⁵.0²⁶,³⁹.0²⁹,³⁴]hentetraconta-1,3,5,7,9,11,13,15,17(41),18,20,22,24,26,28(40),29(34),30,32,35-nonadecaene-4,23-disulfonate",
        26: "amino(phenyl)acetic acid",
        28: "1-[3-(aminomethyl)cyclohexyl]methanamine",
        29: "1,2-Ethanediamine, N1,N2-bis(2-aminoethyl)-",
        32: "(3-aminopropyl)({3-[(3-aminopropyl)amino]propyl})amine",
        36: "zinc(2+) bis(2-sulfanylidene-1,2-dihydropyridin-1-olate)",
        39: "Ichthammol",
    },
    "cas": {
        14: "132-16-1",
        17: "31221-06-4",
        21: "7775-09-9",
        25: "94891-43-7",
        26: "875-74-1",
        28: "2579-20-6",
        29: "112-24-3",
        32: "4605-14-5",
        36: "13463-41-7",
        39: "8029-68-3",
    },
    "smiles_paper": {
        14: "[Fe+2].c1ccc2c(c1)-c1nc-2nc2[n-]c(nc3nc(nc4[n-]c(n1)c1ccccc41)-c1ccccc1-3)c1ccccc21",
        17: "[N-]=[N+]=C1C(=O)[N-]C(=O)[N-]C1=O",
        21: "[Na+].[O-][Cl+2]([O-])[O-]",
        25: "[NH-]c1ccc(S(=O)(=O)CCOS(=O)(=O)[O-])cc1.[Na+]",
        26: "[NH3+][C@@H](C(=O)[O-])c1ccccc1",
        28: "[NH3+]C[C@H]1CCC[C@@H](C[NH3+])C1",
        29: "[NH3+]CC[NH2+]CC[NH2+]CC[NH3+]",
        32: "[NH3+]CCC[NH2+]CCC[NH2+]CCC[NH3+]",
        36: "[O-][n+]1ccccc1[S-][Zn+2][S-]c1cccc[n+]1[O-]",
        39: "*C(*)(*)S(=O)(=O)[O-].[NH4+]",
    },
}

df_a_dict = {
    "name": {
        0: "(2H)chloroform",
        1: "diquat",
        3: "3,5-diaminobenzoic acid",
        4: "octane-1,8-diamine",
        5: "N,N'-diphenylguanidine hydrochloride",
        6: "2-amino-6-chloro-4-nitrophenol",
        8: "1-(diphenylmethyl)piperazine",
    },
    "cas": {
        0: "865-49-6",
        1: "85-00-7",
        3: "535-87-5",
        4: "373-44-4",
        5: "24245-27-0",
        6: "6358-09-4",
        8: "841-77-0",
    },
    "smiles_paper": {
        0: "[2H]C(Cl)(Cl)Cl",
        1: "[Br-].[Br-].c1cc[n+]2c(c1)-c1cccc[n+]1CC2",
        3: "[Cl-].[Cl-].[NH3+]c1cc([NH3+])cc(C(=O)O)c1",
        4: "[Cl-].[Cl-].[NH3+]CCCCCCCC[NH3+]",
        5: "[Cl-].[NH2+]=C(Nc1ccccc1)Nc1ccccc1",
        6: "[Cl-].[NH3+]c1cc([N+](=O)[O-])cc(Cl)c1O",
        8: "[Cl-].c1ccc(C(c2ccccc2)[NH+]2CCNCC2)cc1",
    },
    "smiles": {
        0: "[2H]C(Cl)(Cl)Cl",
        1: "[Br-].[Br-].C1C[N+]2=C(C=CC=C2)C2=[N+]1C=CC=C2",
        3: "NC1=CC(=CC(N)=C1)C([O-])=O",
        4: "[NH3+]CCCCCCCC[NH3+]",
        5: "[NH2+]=C(NC1=CC=CC=C1)NC1=CC=CC=C1",
        6: "NC1=C([O-])C(Cl)=CC(=C1)N(=O)=O",
        8: "C1CN(CC[NH2+]1)C(C1=CC=CC=C1)C1=CC=CC=C1",
    },
    "ec_num": {
        0: "212-742-4",
        1: "201-579-4",
        3: "208-621-0",
        4: "206-764-3",
        5: "246-107-8",
        6: "228-762-1",
        8: "212-667-7",
    },
}

smiles_not_found_dict = {
    "name": {
        0: "calcium hydrogen borate",
        3: "4-chloro-2-[(E)-2-(2-{[(2Z)-6-chloro-10,12-dioxa-2,3-diaza-11-chromatetracyclo[11.8.0.0⁴,⁹.0¹⁶,²¹]henicosa-1(21),2,4,6,8,13,15,17,19-nonaen-11-yl]oxy}naphthalen-1-yl)diazen-1-yl]phenol",
        4: "Iodo(triphenylphosphino)copper",
        5: "Ferrocene",
        8: "trisodium 12-({4-[2-(sulfonatooxy)ethanesulfonyl]phenyl}sulfamoyl)-9,18,27,36,37,39,40,41-octaaza-38-nickeladecacyclo[17.17.3.1¹⁰,¹⁷.1²⁸,³⁵.0²,⁷.0⁸,³⁷.0¹¹,¹⁶.0²⁰,²⁵.0²⁶,³⁹.0²⁹,³⁴]hentetraconta-1,3,5,7,9,11,13,15,17(41),18,20,22,24,26,28(40),29(34),30,32,35-nonadecaene-4,23-disulfonate",
    },
    "cas": {
        0: "12040-58-3",
        3: "31714-55-3",
        4: "47107-74-4",
        5: "102-54-5",
        8: "94891-43-7",
    },
    "smiles_paper": {
        0: "[Ca+2].[Ca+2].[Ca+2].[O-]B([O-])[O-].[O-]B([O-])[O-]",
        3: "[Cr+3].[H+].[O-]c1ccc(Cl)cc1N=Nc1c([O-])ccc2ccccc12.[O-]c1ccc(Cl)cc1N=Nc1c([O-])ccc2ccccc12",
        4: "[Cu+].[I-].c1ccc(P(c2ccccc2)c2ccccc2)cc1",
        5: "[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1",
        8: "[NH-]c1ccc(S(=O)(=O)CCOS(=O)(=O)[O-])cc1.[Na+]",
    },
    "smiles_from_cas_cc": {0: "", 3: "", 4: "", 5: "", 8: ""},
    "inchi_from_cas_cc": {0: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 8: np.nan},
    "isomeric_smiles_pubchem": {
        0: "B([O-])([O-])[O-].B([O-])([O-])[O-].[Ca+2].[Ca+2].[Ca+2]",
        3: "[H+].C1=CC=C2C(=C1)C=CC(=C2N=NC3=C(C=CC(=C3)Cl)[O-])[O-].C1=CC=C2C(=C1)C=CC(=C2N=NC3=C(C=CC(=C3)Cl)[O-])[O-].[Cr+3]",
        4: "C1=CC=C(C=C1)[PH+](C2=CC=CC=C2)C3=CC=CC=C3.[Cu]I",
        5: "",
        8: "",
    },
    "canonical_smiles_pubchem": {
        0: "B([O-])([O-])[O-].B([O-])([O-])[O-].[Ca+2].[Ca+2].[Ca+2]",
        3: "[H+].C1=CC=C2C(=C1)C=CC(=C2N=NC3=C(C=CC(=C3)Cl)[O-])[O-].C1=CC=C2C(=C1)C=CC(=C2N=NC3=C(C=CC(=C3)Cl)[O-])[O-].[Cr+3]",
        4: "C1=CC=C(C=C1)[PH+](C2=CC=CC=C2)C3=CC=CC=C3.[Cu]I",
        5: "",
        8: "",
    },
    "inchi_pubchem": {
        0: "InChI=1S/2BO3.3Ca/c2*2-1(3)4;;;/q2*-3;3*+2",
        3: "InChI=1S/2C16H11ClN2O2.Cr/c2*17-11-6-8-14(20)13(9-11)18-19-16-12-4-2-1-3-10(12)5-7-15(16)21;/h2*1-9,20-21H;/q;;+3/p-3",
        4: "InChI=1S/C18H15P.Cu.HI/c1-4-10-16(11-5-1)19(17-12-6-2-7-13-17)18-14-8-3-9-15-18;;/h1-15H;;1H/q;+1;",
        5: np.nan,
        8: np.nan,
    },
    "cas_found_comptox": {
        0: np.nan,
        3: "31714-55-3",
        4: "47107-74-4",
        5: "102-54-5",
        8: np.nan,
    },
    "dtxcid_comptox": {
        0: np.nan,
        3: "DTXCID401324081",
        4: "DTXCID301391479",
        5: "DTXCID301768066",
        8: np.nan,
    },
    "inchikey_comptox": {
        0: np.nan,
        3: "HOMXPGRNXLQWJA-VFUQPONKSA-K",
        4: "COTHKJMLMQDXCG-UHFFFAOYSA-N",
        5: np.nan,
        8: np.nan,
    },
    "iupac_name_comptox": {
        0: np.nan,
        3: "Hydrogen bis[1-{[5-chloro-2-(hydroxy-kappaO)phenyl]diazenyl-kappaN~1~}naphthalen-2-olato(2-)-kappaO]chromate(1-)",
        4: "Copper--triphenylphosphane--hydrogen iodide (1/1/1)",
        5: "Iron(2+) dicyclopenta-2,4-dien-1-ide",
        8: np.nan,
    },
    "smiles_comptox": {
        0: "",
        3: "[H+].[Cr+3].[O-]C1=CC=C(Cl)C=C1\\N=N\\C1=C2C=CC=CC2=CC=C1[O-].[O-]C1=CC=C(Cl)C=C1\\N=N\\C1=C2C=CC=CC2=CC=C1[O-]",
        4: "[Cu].I.C1=CC=C(C=C1)P(C1=CC=CC=C1)C1=CC=CC=C1",
        5: "[Fe++].c1cccc1.c1cccc1",
        8: "",
    },
    "inchi_comptox": {
        0: np.nan,
        3: "InChI=1S/2C16H11ClN2O2.Cr/c2*17-11-6-8-14(20)13(9-11)18-19-16-12-4-2-1-3-10(12)5-7-15(16)21;/h2*1-9,20-21H;/q;;+3/p-3/b2*19-18+;\n",
        4: "InChI=1S/C18H15P.Cu.HI/c1-4-10-16(11-5-1)19(17-12-6-2-7-13-17)18-14-8-3-9-15-18;;/h1-15H;;1H\n",
        5: np.nan,
        8: np.nan,
    },
    "molecular_formula_comptox": {
        0: np.nan,
        3: "C32H19Cl2CrN4O4",
        4: "C18H16CuIP",
        5: "C10H10Fe",
        8: np.nan,
    },
    "inchi_from_smiles_paper": {
        0: "InChI=1S/2BO3.3Ca/c2*2-1(3)4;;;/q2*-3;3*+2",
        3: "InChI=1S/2C16H11ClN2O2.Cr/c2*17-11-6-8-14(20)13(9-11)18-19-16-12-4-2-1-3-10(12)5-7-15(16)21;/h2*1-9,20-21H;/q;;+3/p-3",
        4: "InChI=1S/C18H15P.Cu.HI/c1-4-10-16(11-5-1)19(17-12-6-2-7-13-17)18-14-8-3-9-15-18;;/h1-15H;;1H/q;+1;/p-1",
        5: "InChI=1S/2C5H5.Fe/c2*1-2-4-5-3-1;/h2*1-5H;/q2*-1;+2",
        8: "InChI=1S/C8H10NO6S2.Na/c9-7-1-3-8(4-2-7)16(10,11)6-5-15-17(12,13)14;/h1-4,9H,5-6H2,(H,12,13,14);/q-1;+1/p-1",
    },
    "inchi_from_smiles_from_cas_cc": {0: "", 3: "", 4: "", 5: "", 8: ""},
    "inchi_from_isomeric_smiles_pubchem": {
        0: "InChI=1S/2BO3.3Ca/c2*2-1(3)4;;;/q2*-3;3*+2",
        3: "InChI=1S/2C16H11ClN2O2.Cr/c2*17-11-6-8-14(20)13(9-11)18-19-16-12-4-2-1-3-10(12)5-7-15(16)21;/h2*1-9,20-21H;/q;;+3/p-3",
        4: "InChI=1S/C18H15P.Cu.HI/c1-4-10-16(11-5-1)19(17-12-6-2-7-13-17)18-14-8-3-9-15-18;;/h1-15H;;1H/q;+1;",
        5: "",
        8: "",
    },
    "inchi_from_canonical_smiles_pubchem": {
        0: "InChI=1S/2BO3.3Ca/c2*2-1(3)4;;;/q2*-3;3*+2",
        3: "InChI=1S/2C16H11ClN2O2.Cr/c2*17-11-6-8-14(20)13(9-11)18-19-16-12-4-2-1-3-10(12)5-7-15(16)21;/h2*1-9,20-21H;/q;;+3/p-3",
        4: "InChI=1S/C18H15P.Cu.HI/c1-4-10-16(11-5-1)19(17-12-6-2-7-13-17)18-14-8-3-9-15-18;;/h1-15H;;1H/q;+1;",
        5: "",
        8: "",
    },
    "inchi_from_smiles_comptox": {
        0: "",
        3: "InChI=1S/2C16H11ClN2O2.Cr/c2*17-11-6-8-14(20)13(9-11)18-19-16-12-4-2-1-3-10(12)5-7-15(16)21;/h2*1-9,20-21H;/q;;+3/p-3/b2*19-18+;",
        4: "InChI=1S/C18H15P.Cu.HI/c1-4-10-16(11-5-1)19(17-12-6-2-7-13-17)18-14-8-3-9-15-18;;/h1-15H;;1H",
        5: "InChI=1S/2C5H5.Fe/c2*1-2-4-5-3-1;/h2*1-5H;/q;;+2",
        8: "",
    },
}


df_a_full_dict = {
    "name": {
        0: "(2H)chloroform",
        1: "diquat",
        3: "3,5-diaminobenzoic acid",
        4: "octane-1,8-diamine",
        7: "2-amino-6-chloro-4-nitrophenol",
    },
    "name_type": {
        0: np.nan,
        1: "IUPAC Name",
        3: "IUPAC Name",
        4: "IUPAC Name",
        7: "IUPAC Name",
    },
    "cas": {0: "865-49-6", 1: "85-00-7", 3: "535-87-5", 4: "373-44-4", 7: "6358-09-4"},
    "smiles": {
        0: "[2H]C(Cl)(Cl)Cl",
        1: "[Br-].[Br-].C1C[N+]2=C(C=CC=C2)C2=[N+]1C=CC=C2",
        3: "NC1=CC(=CC(N)=C1)C([O-])=O",
        4: "[NH3+]CCCCCCCC[NH3+]",
        7: "NC1=C([O-])C(Cl)=CC(=C1)N(=O)=O",
    },
    "reliability": {0: 2, 1: 1, 3: 2, 4: 2, 7: 1},
    "endpoint": {0: "ready", 1: "ready", 3: "ready", 4: "ready", 7: "ready"},
    "guideline": {
        0: "OECD Guideline 301 C",
        1: "OECD Guideline 301 C",
        3: "OECD Guideline 301 D",
        4: "OECD Guideline 301 E",
        7: "OECD Guideline 301 F",
    },
    "principle": {
        0: "Closed Respirometer",
        1: "Closed Respirometer",
        3: "Closed Bottle Test",
        4: "DOC Die Away",
        7: "Closed Respirometer",
    },
    "time_day": {0: 14.0, 1: 28.0, 3: 30.0, 4: 29.0, 7: 28.0},
    "biodegradation_percent": {0: 0.0, 1: 0.0, 3: 0.0, 4: 1.0, 7: 0.0},
    "ec_num": {
        0: "212-742-4",
        1: "201-579-4",
        3: "208-621-0",
        4: "206-764-3",
        7: "228-762-1",
    },
    "to_remove": {0: 0, 1: 0, 3: 0, 4: 0, 7: 0},
}

df_b_found_full_dict = {
    "name": {
        2: "calcium hydrogen borate",
        8: "1,2,3,4-tetrahydronaphthalen-1-amine",
        12: "[29H,31H-phthalocyaninato-N29,N30,N31,N32]cobalt",
        13: "4-chloro-2-[(E)-2-(2-{[(2Z)-6-chloro-10,12-dioxa-2,3-diaza-11-chromatetracyclo[11.8.0.0⁴,⁹.0¹⁶,²¹]henicosa-1(21),2,4,6,8,13,15,17,19-nonaen-11-yl]oxy}naphthalen-1-yl)diazen-1-yl]phenol",
        14: "Iodo(triphenylphosphino)copper",
    },
    "name_type": {
        2: "IUPAC Name",
        8: "IUPAC Name",
        12: np.nan,
        13: "IUPAC Name",
        14: np.nan,
    },
    "cas": {
        2: "12040-58-3",
        8: "2217-40-5",
        12: "3317-67-7",
        13: "31714-55-3",
        14: "47107-74-4",
    },
    "reliability": {2: 1, 8: 1, 12: 1, 13: 1, 14: 1},
    "endpoint": {2: "ready", 8: "ready", 12: "ready", 13: "ready", 14: "ready"},
    "guideline": {
        2: "OECD Guideline 301 B",
        8: "OECD Guideline 301 A",
        12: "OECD Guideline 301 F",
        13: "OECD Guideline 301 D",
        14: "OECD Guideline 301 B",
    },
    "principle": {
        2: "CO2 Evolution",
        8: "DOC Die Away",
        12: "Closed Respirometer",
        13: "Closed Bottle Test",
        14: "CO2 Evolution",
    },
    "time_day": {2: 28.0, 8: 28.0, 12: 28.0, 13: 28.0, 14: 28.0},
    "biodegradation_percent": {2: 0.11, 8: 0.1, 12: 0.0, 13: 0.0412, 14: 0.12},
    "smiles": {
        2: "B([O-])([O-])[O-].B([O-])([O-])[O-].[Ca+2].[Ca+2].[Ca+2]",
        8: "C1CC(C2=CC=CC=C2C1)N",
        12: "C1=CC=C2C(=C1)C3=NC4=NC(=NC5=C6C=CC=CC6=C([N-]5)N=C7C8=CC=CC=C8C(=N7)N=C2[N-]3)C9=CC=CC=C94.[Co+2]",
        13: "[H+].C1=CC=C2C(=C1)C=CC(=C2N=NC3=C(C=CC(=C3)Cl)[O-])[O-].C1=CC=C2C(=C1)C=CC(=C2N=NC3=C(C=CC(=C3)Cl)[O-])[O-].[Cr+3]",
        14: "C1=CC=C(C=C1)[PH+](C2=CC=CC=C2)C3=CC=CC=C3.[Cu]I",
    },
}


df_pubchem_dict = {
    "name": {
        2: "calcium hydrogen borate",
        7: "1,2,3,4-tetrahydronaphthalen-1-amine",
        10: "[29H,31H-phthalocyaninato-N29,N30,N31,N32]cobalt",
        11: "4-chloro-2-[(E)-2-(2-{[(2Z)-6-chloro-10,12-dioxa-2,3-diaza-11-chromatetracyclo[11.8.0.0⁴,⁹.0¹⁶,²¹]henicosa-1(21),2,4,6,8,13,15,17,19-nonaen-11-yl]oxy}naphthalen-1-yl)diazen-1-yl]phenol",
        12: "Iodo(triphenylphosphino)copper",
    },
    "cas": {
        2: "12040-58-3",
        7: "2217-40-5",
        10: "3317-67-7",
        11: "31714-55-3",
        12: "47107-74-4",
    },
    "smiles_paper": {
        2: "[Ca+2].[Ca+2].[Ca+2].[O-]B([O-])[O-].[O-]B([O-])[O-]",
        7: "[Cl-].[NH3+]C1CCCc2ccccc21",
        10: "[Co+2].c1ccc2c(c1)-c1nc-2nc2[n-]c(nc3nc(nc4[n-]c(n1)c1ccccc41)-c1ccccc1-3)c1ccccc21",
        11: "[Cr+3].[H+].[O-]c1ccc(Cl)cc1N=Nc1c([O-])ccc2ccccc12.[O-]c1ccc(Cl)cc1N=Nc1c([O-])ccc2ccccc12",
        12: "[Cu+].[I-].c1ccc(P(c2ccccc2)c2ccccc2)cc1",
    },
    "smiles_from_cas_cc": {2: np.nan, 7: np.nan, 10: np.nan, 11: np.nan, 12: np.nan},
    "inchi_from_cas_cc": {2: np.nan, 7: np.nan, 10: np.nan, 11: np.nan, 12: np.nan},
    "isomeric_smiles_pubchem": {
        2: "B([O-])([O-])[O-].B([O-])([O-])[O-].[Ca+2].[Ca+2].[Ca+2]",
        7: "C1CC(C2=CC=CC=C2C1)N",
        10: "C1=CC=C2C(=C1)C3=NC4=NC(=NC5=C6C=CC=CC6=C([N-]5)N=C7C8=CC=CC=C8C(=N7)N=C2[N-]3)C9=CC=CC=C94.[Co+2]",
        11: "[H+].C1=CC=C2C(=C1)C=CC(=C2N=NC3=C(C=CC(=C3)Cl)[O-])[O-].C1=CC=C2C(=C1)C=CC(=C2N=NC3=C(C=CC(=C3)Cl)[O-])[O-].[Cr+3]",
        12: "C1=CC=C(C=C1)[PH+](C2=CC=CC=C2)C3=CC=CC=C3.[Cu]I",
    },
    "canonical_smiles_pubchem": {
        2: "B([O-])([O-])[O-].B([O-])([O-])[O-].[Ca+2].[Ca+2].[Ca+2]",
        7: "C1CC(C2=CC=CC=C2C1)N",
        10: "C1=CC=C2C(=C1)C3=NC4=NC(=NC5=C6C=CC=CC6=C([N-]5)N=C7C8=CC=CC=C8C(=N7)N=C2[N-]3)C9=CC=CC=C94.[Co+2]",
        11: "[H+].C1=CC=C2C(=C1)C=CC(=C2N=NC3=C(C=CC(=C3)Cl)[O-])[O-].C1=CC=C2C(=C1)C=CC(=C2N=NC3=C(C=CC(=C3)Cl)[O-])[O-].[Cr+3]",
        12: "C1=CC=C(C=C1)[PH+](C2=CC=CC=C2)C3=CC=CC=C3.[Cu]I",
    },
    "inchi_pubchem": {
        2: "InChI=1S/2BO3.3Ca/c2*2-1(3)4;;;/q2*-3;3*+2",
        7: "InChI=1S/C10H13N/c11-10-7-3-5-8-4-1-2-6-9(8)10/h1-2,4,6,10H,3,5,7,11H2",
        10: "InChI=1S/C32H16N8.Co/c1-2-10-18-17(9-1)25-33-26(18)38-28-21-13-5-6-14-22(21)30(35-28)40-32-24-16-8-7-15-23(24)31(36-32)39-29-20-12-4-3-11-19(20)27(34-29)37-25;/h1-16H;/q-2;+2",
        11: "InChI=1S/2C16H11ClN2O2.Cr/c2*17-11-6-8-14(20)13(9-11)18-19-16-12-4-2-1-3-10(12)5-7-15(16)21;/h2*1-9,20-21H;/q;;+3/p-3",
        12: "InChI=1S/C18H15P.Cu.HI/c1-4-10-16(11-5-1)19(17-12-6-2-7-13-17)18-14-8-3-9-15-18;;/h1-15H;;1H/q;+1;",
    },
}

df_lunghini_added_cas_dict = {
    "smiles": {
        0: "Br/C=C/c1ccccc1",
        1: "Br/C=C\\Br",
        2: "Br/C=C\\c1ccccc1",
        3: "BrC(Br)Br",
        4: "BrC(Br)C(Br)Br",
        5: "BrC(Br)c1ccccc1OCC1CO1",
    },
    "cas_external": {
        0: "103-64-0, 000103-64-0",
        1: "540-49-8",
        2: "1335-06-4",
        3: "75-25-2, 000075-25-2",
        4: "79-27-6, 000079-27-6, 1335-06-4",
        5: "NOCAS_865529",
    },
    "y_true": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
    "dataset": {
        0: "MITI, VEGA, EPISUITE",
        1: "OPERA",
        2: "OPERA",
        3: "VEGA, EPISUITE",
        4: "MITI, OPERA, EPISUITE",
        5: "OPERA",
    },
    "inchikey": {
        0: "YMOONIIMQBGTDU-VOTSOKGWSA-N",
        1: "UWTUEMKLYAGTNQ-UPHRSURJSA-N",
        2: "YMOONIIMQBGTDU-SREVYHEPSA-N",
        3: "DIKBFYAXUHHXCS-UHFFFAOYSA-N",
        4: "QXSZNDIIPUOQMB-UHFFFAOYSA-N",
        5: "RUBIQRSGTVJYHR-UHFFFAOYSA-N",
    },
    "inchi": {
        0: "InChI=1S/C8H7Br/c9-7-6-8-4-2-1-3-5-8/h1-7H/b7-6+",
        1: "InChI=1S/C2H2Br2/c3-1-2-4/h1-2H/b2-1-",
        2: "InChI=1S/C8H7Br/c9-7-6-8-4-2-1-3-5-8/h1-7H/b7-6-",
        3: "InChI=1S/CHBr3/c2-1(3)4/h1H",
        4: "InChI=1S/C2H2Br4/c3-1(4)2(5)6/h1-2H",
        5: "InChI=1S/C10H10Br2O2/c11-10(12)8-3-1-2-4-9(8)14-6-7-5-13-7/h1-4,7,10H,5-6H2",
    },
    "cid": {
        0: 5314126.0,
        1: 643776.0,
        2: 5369379.0,
        3: 5558.0,
        4: 6588.0,
        5: 3035339.0,
    },
    "cas_pubchem": {
        0: "588-72-7, 103-64-0, 1335-06-4",
        1: "590-11-4, 540-49-8",
        2: "588-73-8, 1335-06-4, 588-72-7",
        3: "75-25-2, 4471-18-5",
        4: "4471-18-5",
        5: "30171-80-3",
    },
    "cas_ref_pubchem": {
        0: "588-72-7: CAS Common Chemistry, ChemIDplus, FDA Global Substance Registration System (GSRS); 103-64-0: ChemIDplus, DTP/NCI, EPA Chemicals under the TSCA, EPA DSSTox, Hazardous Substances Data Bank (HSDB); 1335-06-4: European Chemicals Agency (ECHA)",
        1: "590-11-4: ChemIDplus, EPA DSSTox, FDA Global Substance Registration System (GSRS); 540-49-8: DTP/NCI",
        2: "588-73-8: ChemIDplus, DTP/NCI, FDA Global Substance Registration System (GSRS); 1335-06-4: ChemIDplus, EPA Chemicals under the TSCA, EPA DSSTox, European Chemicals Agency (ECHA); 588-72-7: DTP/NCI",
        3: "75-25-2: CAMEO Chemicals, CAS Common Chemistry, ChemIDplus, DrugBank, DTP/NCI, EPA Chemicals under the TSCA, EPA DSSTox, European Chemicals Agency (ECHA), FDA Global Substance Registration System (GSRS), Hazardous Substances Data Bank (HSDB), ILO International Chemical Safety Cards (ICSC), Occupational Safety and Health Administration (OSHA), The National Institute for Occupational Safety and Health (NIOSH); 4471-18-5: ChemIDplus",
        4: "79-27-6: CAMEO Chemicals, CAS Common Chemistry, ChemIDplus, DTP/NCI, EPA Chemicals under the TSCA, EPA DSSTox, European Chemicals Agency (ECHA), FDA Global Substance Registration System (GSRS), Hazardous Substances Data Bank (HSDB), ILO International Chemical Safety Cards (ICSC), Occupational Safety and Health Administration (OSHA), The National Institute for Occupational Safety and Health (NIOSH)",
        5: "30171-80-3: ChemIDplus, EPA DSSTox, European Chemicals Agency (ECHA)",
    },
    "deprecated_cas_pubchem": {
        0: "1340-14-3",
        1: np.nan,
        2: "41380-64-7",
        3: np.nan,
        4: np.nan,
        5: np.nan,
    },
    "inchi_from_smiles": {
        0: "InChI=1S/C8H7Br/c9-7-6-8-4-2-1-3-5-8/h1-7H/b7-6+",
        1: "InChI=1S/C2H2Br2/c3-1-2-4/h1-2H/b2-1-",
        2: "InChI=1S/C8H7Br/c9-7-6-8-4-2-1-3-5-8/h1-7H/b7-6-",
        3: "InChI=1S/CHBr3/c2-1(3)4/h1H",
        4: "InChI=1S/C2H2Br4/c3-1(4)2(5)6/h1-2H",
        5: "InChI=1S/C10H10Br2O2/c11-10(12)8-3-1-2-4-9(8)14-6-7-5-13-7/h1-4,7,10H,5-6H2",
    },
}

class_df_dict = {
    "cas": {
        0: "85-00-7",
        1: "24245-27-0",
        2: "1100-88-5",
        3: "3317-67-7",
        4: "31714-55-3",
        5: "75-25-2",
    },
    "smiles": {
        0: "[Br-].[Br-].c1cc[n+]2c(c1)-c1cccc[n+]1CC2",
        1: "[Cl-].[NH2+]=C(Nc1ccccc1)Nc1ccccc1",
        2: "[Cl-].c1ccc(C[P+](c2ccccc2)(c2ccccc2)c2ccccc2)cc1",
        3: "[Co+2].c1ccc2c(c1)-c1nc-2nc2[n-]c(nc3nc(nc4[n-]c(n1)c1ccccc41)-c1ccccc1-3)c1ccccc21",
        4: "[Cr+3].[H+].[O-]c1ccc(Cl)cc1N=Nc1c([O-])ccc2ccccc12.[O-]c1ccc(Cl)cc1N=Nc1c([O-])ccc2ccccc12",
        5: "BrC(Br)Br",
    },
    "principle": {
        0: "Closed Respirometer",
        1: "Closed Bottle Test",
        2: "Closed Respirometer",
        3: "Closed Respirometer",
        4: "Closed Bottle Test",
        5: "Closed Bottle Test",
    },
    "biodegradation_percent": {0: 0.0, 1: 0.85, 2: 0.0025, 3: 0.0, 4: 0.0412, 5: 0.06},
    "y_true": {0: 0, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0},
    "threshold": {0: 0.6, 1: 0.6, 2: 0.6, 3: 0.6, 4: 0.6, 5: 0.6},
}

class_df_long_dict = {
    "name": {
        0: "Cyclopenta[c]cyclopropa[g][1,6]diazacyclotetradecine-12a(1H)-carboxylic acid, 2,3,3a,4,5,6,7,8,9,11a,12,13,14,14a-tetradecahydro-2-[[7-methoxy-8-methyl-2-[4-(1-methylethyl)-2-thiazolyl]-4-quinolinyl]oxy]-5-methyl-4,14-dioxo-, (2R,3aR,10Z,11aS,12aR,14aR)-",
        1: "2-methylprop-2-enoic acid",
        2: "3-(Diisobutoxy-thiophosphorylsulfanyl)-2-methyl-propionic acid",
        3: "Phenol, 2-methyl-4,6-dinitro-",
        4: "2-[4-(1,3-dihydro-1,3-dioxo-2H-isoindol-2-yl)phenyl]butyric acid",
        5: "Benzyl (S)-N-(1-oxopentyl)-N-((2'-(1H-tetrazole-5-yl)-1,1'-biphenyl-4-yl)methyl)-L-valinate",
        6: "4-(decanoyloxy)benzoic acid",
        7: "potassium hydrogen benzene-1,2-dicarboxylate",
        8: "4-bromo-2,2-diphenylbutanoic acid",
        9: "Benzoic acid, 3,4,5-trihydroxy-",
        10: "4-{2-[(2-methylprop-2-enoyl)oxy]ethoxy}-4-oxobutanoic acid",
        11: "prop-2-enoic acid",
        12: "Chromium acetate",
        13: "3-[(2-acetamidoacetyl)amino]propanoic acid",
        14: "2-[(1Z,2S,3aS,3bS,5aS,6S,7R,9aS,9bS,10R,11aR)-2-(acetyloxy)-7,10-dihydroxy-3a,3b,6,9a-tetramethyl-hexadecahydro-1H-cyclopenta[a]phenanthren-1-ylidene]-6-methylhept-5-enoic acid",
        15: "2-(4-{4-[4-(hydroxydiphenylmethyl)piperidin-1-yl]but-1-yn-1-yl}phenyl)-2-methylpropanoic acid",
        16: "3-tert-butylhexanedioic acid",
        17: "Benzoic acid, 4-(1,1-dimethylethyl)-",
        18: "2-methylpentanedioic acid",
        19: "barium(2+) ion bis(3-methylbenzoate)",
    },
    "name_type": {
        0: np.nan,
        1: "IUPAC Name",
        2: "IUPAC Name",
        3: "IUPAC Name",
        4: np.nan,
        5: "IUPAC Name",
        6: "IUPAC Name",
        7: "IUPAC Name",
        8: "IUPAC Name",
        9: "IUPAC Name",
        10: "IUPAC Name",
        11: "IUPAC Name",
        12: np.nan,
        13: np.nan,
        14: "IUPAC Name",
        15: "IUPAC Name",
        16: "IUPAC Name",
        17: "IUPAC Name",
        18: "IUPAC Name",
        19: "IUPAC Name",
    },
    "cas": {
        0: "923604-58-4",
        1: "79-41-4",
        2: "268567-32-4",
        3: "534-52-1",
        4: "94232-67-4",
        5: "137863-20-8",
        6: "86960-46-5",
        7: "877-24-7",
        8: "37742-98-6",
        9: "149-91-7",
        10: "20882-04-6",
        11: "79-10-7",
        12: "17593-70-3",
        13: "1016788-34-3",
        14: "6990-06-3",
        15: "832088-68-3",
        16: "10347-88-3",
        17: "98-73-7",
        18: "617-62-9",
        19: "68092-47-7",
    },
    "source": {
        0: "ClassDataset_original",
        1: "ClassDataset_original",
        2: "ClassDataset_original",
        3: "ClassDataset_original",
        4: "ClassDataset_original",
        5: "ClassDataset_original",
        6: "ClassDataset_original",
        7: "ClassDataset_original",
        8: "ClassDataset_original",
        9: "ClassDataset_original",
        10: "ClassDataset_original",
        11: "ClassDataset_original",
        12: "ClassDataset_original",
        13: "ClassDataset_original",
        14: "ClassDataset_original",
        15: "ClassDataset_original",
        16: "ClassDataset_original",
        17: "ClassDataset_original",
        18: "ClassDataset_original",
        19: "ClassDataset_original",
    },
    "smiles": {
        0: "COc1ccc2c(O[C@@H]3C[C@H]4C(=O)N[C@]5(C(=O)O)C[C@H]5C=CCCCCN(C)C(=O)[C@@H]4C3)cc(-c3nc(C(C)C)cs3)nc2c1C",
        1: "C=C(C)C(=O)O",
        2: "CC(C)COP(=S)(OCC(C)C)SCC(C)C(=O)O",
        3: "Cc1cc([N+](=O)[O-])cc([N+](=O)[O-])c1O",
        4: "CCC(C(=O)O)c1ccc(N2C(=O)c3ccccc3C2=O)cc1",
        5: "CCCCC(=O)N(Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1)[C@H](C(=O)OCc1ccccc1)C(C)C",
        6: "CCCCCCCCCC(=O)Oc1ccc(C(=O)O)cc1",
        7: "O=C([O-])c1ccccc1C(=O)O.[K+]",
        8: "O=C(O)C(CCBr)(c1ccccc1)c1ccccc1",
        9: "O=C(O)c1cc(O)c(O)c(O)c1",
        10: "C=C(C)C(=O)OCCOC(=O)CCC(=O)O",
        11: "C=CC(=O)O",
        12: "CC(=O)[O-].[Cr]",
        13: "CC(=O)NCC(=O)NCCC(=O)O",
        14: "CC(=O)O[C@H]1C[C@@]2(C)[C@@H](C[C@@H](O)[C@H]3[C@@]4(C)CC[C@@H](O)[C@@H](C)[C@@H]4CC[C@@]32C)/C1=C(\\CCC=C(C)C)C(=O)O",
        15: "CC(C)(C(=O)O)c1ccc(C#CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1",
        16: "CC(C)(C)C(CCC(=O)O)CC(=O)O",
        17: "CC(C)(C)c1ccc(C(=O)O)cc1",
        18: "CC(CCC(=O)O)C(=O)O",
        19: "Cc1cccc(C(=O)O)c1",
    },
    "y_true": {
        0: 0,
        1: 1,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 1,
        7: 1,
        8: 0,
        9: 1,
        10: 1,
        11: 1,
        12: 1,
        13: 1,
        14: 0,
        15: 0,
        16: 0,
        17: 0,
        18: 1,
        19: 1,
    },
    "pka_acid_1": {
        0: 4.1,
        1: 4.2,
        2: 4.2,
        3: 4.2,
        4: 4.2,
        5: 4.2,
        6: 4.2,
        7: 4.2,
        8: 4.2,
        9: 4.2,
        10: 4.3,
        11: 4.3,
        12: 4.3,
        13: 4.3,
        14: 4.3,
        15: 4.3,
        16: 4.3,
        17: 4.3,
        18: 4.3,
        19: 4.3,
    },
    "pka_acid_2": {
        0: 10.0,
        1: 10.0,
        2: 10.0,
        3: 10.0,
        4: 10.0,
        5: 10.0,
        6: 10.0,
        7: 10.0,
        8: 10.0,
        9: 8.2,
        10: 10.0,
        11: 10.0,
        12: 10.0,
        13: 10.0,
        14: 10.0,
        15: 10.0,
        16: 4.5,
        17: 10.0,
        18: 4.3,
        19: 10.0,
    },
    "pka_acid_3": {
        0: 10.0,
        1: 10.0,
        2: 10.0,
        3: 10.0,
        4: 10.0,
        5: 10.0,
        6: 10.0,
        7: 10.0,
        8: 10.0,
        9: 8.2,
        10: 10.0,
        11: 10.0,
        12: 10.0,
        13: 10.0,
        14: 10.0,
        15: 10.0,
        16: 10.0,
        17: 10.0,
        18: 10.0,
        19: 10.0,
    },
    "pka_acid_4": {
        0: 10.0,
        1: 10.0,
        2: 10.0,
        3: 10.0,
        4: 10.0,
        5: 10.0,
        6: 10.0,
        7: 10.0,
        8: 10.0,
        9: 8.3,
        10: 10.0,
        11: 10.0,
        12: 10.0,
        13: 10.0,
        14: 10.0,
        15: 10.0,
        16: 10.0,
        17: 10.0,
        18: 10.0,
        19: 10.0,
    },
    "pka_base_1": {
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0,
        6: 0.0,
        7: 0.0,
        8: 0.0,
        9: 0.0,
        10: 0.0,
        11: 0.0,
        12: 0.0,
        13: 0.0,
        14: 0.0,
        15: 0.0,
        16: 0.0,
        17: 0.0,
        18: 0.0,
        19: 0.0,
    },
    "pka_base_2": {
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0,
        6: 0.0,
        7: 0.0,
        8: 0.0,
        9: 0.0,
        10: 0.0,
        11: 0.0,
        12: 0.0,
        13: 0.0,
        14: 0.0,
        15: 0.0,
        16: 0.0,
        17: 0.0,
        18: 0.0,
        19: 0.0,
    },
    "pka_base_3": {
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0,
        6: 0.0,
        7: 0.0,
        8: 0.0,
        9: 0.0,
        10: 0.0,
        11: 0.0,
        12: 0.0,
        13: 0.0,
        14: 0.0,
        15: 0.0,
        16: 0.0,
        17: 0.0,
        18: 0.0,
        19: 0.0,
    },
    "pka_base_4": {
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0,
        6: 0.0,
        7: 0.0,
        8: 0.0,
        9: 0.0,
        10: 0.0,
        11: 0.0,
        12: 0.0,
        13: 0.0,
        14: 0.0,
        15: 8.7,
        16: 0.0,
        17: 0.0,
        18: 0.0,
        19: 0.0,
    },
    "α_acid_0": {
        0: 0.0004996785048058,
        1: 0.000628976590774,
        2: 0.000628976590774,
        3: 0.000628976590774,
        4: 0.000628976590774,
        5: 0.000628976590774,
        6: 0.000628976590774,
        7: 0.000628976590774,
        8: 0.000628976590774,
        9: 0.0005313766133463,
        10: 0.0007917056779277,
        11: 0.0007917056779277,
        12: 0.0007917056779277,
        13: 0.0007917056779277,
        14: 0.0007917056779277,
        15: 0.0007917056779277,
        16: 9.962360990580045e-07,
        17: 0.0007917056779277,
        18: 6.288737880211342e-07,
        19: 0.0007917056779277,
    },
    "α_acid_1": {
        0: 0.996989690239032,
        1: 0.996860716935166,
        2: 0.996860716935166,
        3: 0.996860716935166,
        4: 0.996860716935166,
        5: 0.996860716935166,
        6: 0.996860716935166,
        7: 0.996860716935166,
        8: 0.996860716935166,
        9: 0.8421751771256935,
        10: 0.996698396604992,
        11: 0.996698396604992,
        12: 0.996698396604992,
        13: 0.996698396604992,
        14: 0.996698396604992,
        15: 0.996698396604992,
        16: 0.0012541869412508,
        17: 0.996698396604992,
        18: 0.000791705192551,
        19: 0.996698396604992,
    },
    "α_acid_2": {
        0: 0.0025043248752663,
        1: 0.0025040009089743,
        2: 0.0025040009089743,
        3: 0.0025040009089743,
        4: 0.0025040009089743,
        5: 0.0025040009089743,
        6: 0.0025040009089743,
        7: 0.0025040009089743,
        8: 0.0025040009089743,
        9: 0.1334757705086249,
        10: 0.0025035931787394,
        11: 0.0025035931787394,
        12: 0.0025035931787394,
        13: 0.0025035931787394,
        14: 0.0025035931787394,
        15: 0.0025035931787394,
        16: 0.9962360990580108,
        17: 0.0025035931787394,
        18: 0.9966977855519372,
        19: 0.0025035931787394,
    },
    "α_acid_3": {
        0: 6.29057967427355e-06,
        1: 6.2897659077403885e-06,
        2: 6.2897659077403885e-06,
        3: 6.2897659077403885e-06,
        4: 6.2897659077403885e-06,
        5: 6.2897659077403885e-06,
        6: 6.2897659077403885e-06,
        7: 6.2897659077403885e-06,
        8: 6.2897659077403885e-06,
        9: 0.0211544840037622,
        10: 6.288741735695572e-06,
        11: 6.288741735695572e-06,
        12: 6.288741735695572e-06,
        13: 6.288741735695572e-06,
        14: 6.288741735695572e-06,
        15: 6.288741735695572e-06,
        16: 0.0025024319398038,
        17: 6.288741735695572e-06,
        18: 0.0025035916438435,
        19: 6.288741735695572e-06,
    },
    "α_acid_4": {
        0: 1.5801221730137746e-08,
        1: 1.5799177641024683e-08,
        2: 1.5799177641024683e-08,
        3: 1.5799177641024683e-08,
        4: 1.5799177641024683e-08,
        5: 1.5799177641024683e-08,
        6: 1.5799177641024683e-08,
        7: 1.5799177641024683e-08,
        8: 1.5799177641024683e-08,
        9: 0.0026631917485729,
        10: 1.5796605037161773e-08,
        11: 1.5796605037161773e-08,
        12: 1.5796605037161773e-08,
        13: 1.5796605037161773e-08,
        14: 1.5796605037161773e-08,
        15: 1.5796605037161773e-08,
        16: 6.285824835369541e-06,
        17: 1.5796605037161773e-08,
        18: 6.288737880211442e-06,
        19: 1.5796605037161773e-08,
    },
    "α_base_0": {
        0: 2.5118863315095412e-30,
        1: 2.5118863315095412e-30,
        2: 2.5118863315095412e-30,
        3: 2.5118863315095412e-30,
        4: 2.5118863315095412e-30,
        5: 2.5118863315095412e-30,
        6: 2.5118863315095412e-30,
        7: 2.5118863315095412e-30,
        8: 2.5118863315095412e-30,
        9: 2.5118863315095412e-30,
        10: 2.5118863315095412e-30,
        11: 2.5118863315095412e-30,
        12: 2.5118863315095412e-30,
        13: 2.5118863315095412e-30,
        14: 2.5118863315095412e-30,
        15: 6.008437965372989e-23,
        16: 2.5118863315095412e-30,
        17: 2.5118863315095412e-30,
        18: 2.5118863315095412e-30,
        19: 2.5118863315095412e-30,
    },
    "α_base_1": {
        0: 6.309573193613216e-23,
        1: 6.309573193613216e-23,
        2: 6.309573193613216e-23,
        3: 6.309573193613216e-23,
        4: 6.309573193613216e-23,
        5: 6.309573193613216e-23,
        6: 6.309573193613216e-23,
        7: 6.309573193613216e-23,
        8: 6.309573193613216e-23,
        9: 6.309573193613216e-23,
        10: 6.309573193613216e-23,
        11: 6.309573193613216e-23,
        12: 6.309573193613216e-23,
        13: 6.309573193613216e-23,
        14: 6.309573193613216e-23,
        15: 1.5092513799787496e-15,
        16: 6.309573193613216e-23,
        17: 6.309573193613216e-23,
        18: 6.309573193613216e-23,
        19: 6.309573193613216e-23,
    },
    "α_base_2": {
        0: 1.584893129365367e-15,
        1: 1.584893129365367e-15,
        2: 1.584893129365367e-15,
        3: 1.584893129365367e-15,
        4: 1.584893129365367e-15,
        5: 1.584893129365367e-15,
        6: 1.584893129365367e-15,
        7: 1.584893129365367e-15,
        8: 1.584893129365367e-15,
        9: 1.584893129365367e-15,
        10: 1.584893129365367e-15,
        11: 1.584893129365367e-15,
        12: 1.584893129365367e-15,
        13: 1.584893129365367e-15,
        14: 1.584893129365367e-15,
        15: 3.791068063105745e-08,
        16: 1.584893129365367e-15,
        17: 1.584893129365367e-15,
        18: 1.584893129365367e-15,
        19: 1.584893129365367e-15,
    },
    "α_base_3": {
        0: 3.9810715470456376e-08,
        1: 3.9810715470456376e-08,
        2: 3.9810715470456376e-08,
        3: 3.9810715470456376e-08,
        4: 3.9810715470456376e-08,
        5: 3.9810715470456376e-08,
        6: 3.9810715470456376e-08,
        7: 3.9810715470456376e-08,
        8: 3.9810715470456376e-08,
        9: 3.9810715470456376e-08,
        10: 3.9810715470456376e-08,
        11: 3.9810715470456376e-08,
        12: 3.9810715470456376e-08,
        13: 3.9810715470456376e-08,
        14: 3.9810715470456376e-08,
        15: 0.9522732428644664,
        16: 3.9810715470456376e-08,
        17: 3.9810715470456376e-08,
        18: 3.9810715470456376e-08,
        19: 3.9810715470456376e-08,
    },
    "α_base_4": {
        0: 0.9999999601892828,
        1: 0.9999999601892828,
        2: 0.9999999601892828,
        3: 0.9999999601892828,
        4: 0.9999999601892828,
        5: 0.9999999601892828,
        6: 0.9999999601892828,
        7: 0.9999999601892828,
        8: 0.9999999601892828,
        9: 0.9999999601892828,
        10: 0.9999999601892828,
        11: 0.9999999601892828,
        12: 0.9999999601892828,
        13: 0.9999999601892828,
        14: 0.9999999601892828,
        15: 0.0477267192248515,
        16: 0.9999999601892828,
        17: 0.9999999601892828,
        18: 0.9999999601892828,
        19: 0.9999999601892828,
    },
}

metal_smiles_dict = {
    "smiles": {
        0: "N(CCNC(=S)[S-])C=1[S-][Mn+2][S]1",
        1: "BrN1C(=O)CCC1=O",
        2: "C=CC(=O)OCCCCOC(=O)OC1=CC=C(C=C1)C(=O)C1=CC=CC=C1",
        3: "CCCCCCCC(=O)NO",
        4: "CCOc1ccccc1C(N)=N",
        5: "CCCCCCCCCCCCCCCCCCn1c(=O)c2ccc3sc4ccccc4c4ccc(c2c34)c1=O",
        6: "O=c1[nH]cc(F)c(=O)[nH]1",
        7: "c1ccccc1[Bi](c1ccccc1)c1ccccc1",
        8: "[Ni++].CCCCC(CC)C([O-])=O.CCCCC(CC)C([O-])=O",
        9: "CCB(CC)OC",
    }
}

df_improved_class_dict = {
    "cas": {0: "100-09-4", 1: "100-10-7", 2: "100-14-1", 3: "100-26-5", 4: "100-37-8", 5: "100-40-3", 6: "100-41-4"},
    "smiles": {
        0: "COC1=CC=C(C=C1)C([O-])=O",
        1: "CN(C)C1=CC=C(C=O)C=C1",
        2: "N(=O)(=O)C1=CC=C(CCl)C=C1",
        3: "[O-]C(=O)C1=CC=C(N=C1)C([O-])=O",
        4: "CC[NH+](CC)CCO",
        5: "C(=C)C1CCC=CC1",
        6: "CCC1=CC=CC=C1",
    },
    "reliability": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0},
    "endpoint": {0: "ready", 1: "ready", 2: "ready", 3: "ready", 4: "ready", 5: "ready", 6: "ready"},
    "guideline": {
        0: "OECD Guideline 301 F",
        1: "OECD Guideline 301 F",
        2: "OECD Guideline 301 C",
        3: "OECD Guideline 301 C",
        4: "OECD Guideline 301 C",
        5: "OECD Guideline 301 C",
        6: "OECD Guideline 301 C",
    },
    "time_day": {0: 28.0, 1: 28.0, 2: 28.0, 3: 28.0, 4: 28.0, 5: 28.0, 6: 28.0},
    "y_true": {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
    "inchi_from_smiles": {
        0: "InChI=1S/C8H8O3/c1-11-7-4-2-6(3-5-7)8(9)10/h2-5H,1H3,(H,9,10)/p-1",
        1: "InChI=1S/C9H11NO/c1-10(2)9-5-3-8(7-11)4-6-9/h3-7H,1-2H3",
        2: "InChI=1S/C7H6ClNO2/c8-5-6-1-3-7(4-2-6)9(10)11/h1-4H,5H2",
        3: "InChI=1S/C7H5NO4/c9-6(10)4-1-2-5(7(11)12)8-3-4/h1-3H,(H,9,10)(H,11,12)/p-2",
        4: "InChI=1S/C6H15NO/c1-3-7(4-2)5-6-8/h8H,3-6H2,1-2H3/p+1",
        5: "InChI=1S/C8H12/c1-2-8-6-4-3-5-7-8/h2-4,8H,1,5-7H2",
        6: "InChI=1S/C8H10/c1-2-8-6-4-3-5-7-8/h3-7H,2H2,1H3",
    },
}

df_improved_reg_dict = {
    "cas": {3: "100-00-5", 4: "100-01-6", 5: "100-02-7", 6: "100-09-4", 14: "100-37-8"},
    "smiles": {
        3: "ClC1=CC=C(C=C1)N(=O)=O",
        4: "NC1=CC=C(C=C1)[N+]([O-])=O",
        5: "[O-]C1=CC=C(C=C1)N(=O)=O",
        6: "COC1=CC=C(C=C1)C([O-])=O",
        14: "CC[NH+](CC)CCO",
    },
    "reliability": {3: 2, 4: 1, 5: 1, 6: 1, 14: 1},
    "endpoint": {3: "ready", 4: "ready", 5: "ready", 6: "ready", 14: "inherent"},
    "guideline": {
        3: "OECD Guideline 301 D",
        4: "OECD Guideline 301 C",
        5: "OECD Guideline 301 C",
        6: "OECD Guideline 301 F",
        14: "OECD Guideline 302 B",
    },
    "principle": {
        3: "Closed Bottle Test",
        4: "Closed Respirometer",
        5: "DOC Die Away",
        6: "Closed Respirometer",
        14: "DOC Die Away",
    },
    "time_day": {3: 20.0, 4: 14.0, 5: 14.0, 6: 28.0, 14: 0.125},
    "biodegradation_percent": {3: 0.2565, 4: 0.0, 5: 0.043, 6: 0.89, 14: 0.65},
}

class_improved_dict = {
    "cas": {0: "100-09-4", 1: "100-10-7", 2: "100-14-1", 3: "100-26-5", 4: "100-36-7", 5: "100-37-8"},
    "smiles": {
        0: "COC1=CC=C(C=C1)C(O)=O",
        1: "CN(C)C1=CC=C(C=O)C=C1",
        2: "N(=O)(=O)C1=CC=C(CCl)C=C1",
        3: "C(O)(=O)C=1C=CC(C(O)=O)=NC1",
        4: "N(CCN)(CC)CC",
        5: "CCN(CC)CCO",
    },
    "reliability": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 2.0, 5: 1.0},
    "endpoint": {0: "ready", 1: "ready", 2: "ready", 3: "ready", 4: "inherent", 5: "ready"},
    "guideline": {
        0: "OECD Guideline 301 F",
        1: "OECD Guideline 301 F",
        2: "OECD Guideline 301 C",
        3: "OECD Guideline 301 C",
        4: "OECD Guideline 302 B",
        5: "OECD Guideline 301 C",
    },
    "time_day": {0: 28.0, 1: 28.0, 2: 28.0, 3: 28.0, 4: 28.0, 5: 28.0},
    "y_true": {0: 1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0},
    "inchi_from_smiles": {
        0: "InChI=1S/C8H8O3/c1-11-7-4-2-6(3-5-7)8(9)10/h2-5H,1H3,(H,9,10)",
        1: "InChI=1S/C9H11NO/c1-10(2)9-5-3-8(7-11)4-6-9/h3-7H,1-2H3",
        2: "InChI=1S/C7H6ClNO2/c8-5-6-1-3-7(4-2-6)9(10)11/h1-4H,5H2",
        3: "InChI=1S/C7H5NO4/c9-6(10)4-1-2-5(7(11)12)8-3-4/h1-3H,(H,9,10)(H,11,12)",
        4: "InChI=1S/C6H16N2/c1-3-8(4-2)6-5-7/h3-7H2,1-2H3",
        5: "InChI=1S/C6H15NO/c1-3-7(4-2)5-6-8/h8H,3-6H2,1-2H3",
    },
    "threshold": {0: 0.6, 1: 0.6, 2: 0.6, 3: 0.6, 4: 0.7, 5: 0.6},
}


df_for_aggregate_dict = {
    "cas": {
        0: "865-49-6",
        1: "865-49-6",
        2: "85-00-7",
        3: "85-00-7",
        4: "535-87-5",
        5: "535-87-5",
        6: "535-87-5",
    },
    "smiles": {
        0: "[2H]C(Cl)(Cl)Cl",
        1: "[2H]C(Cl)(Cl)Cl",
        2: "[Br-].[Br-].c1cc[n+]2c(c1)-c1cccc[n+]1CC2",
        3: "[Br-].[Br-].c1cc[n+]2c(c1)-c1cccc[n+]1CC2",
        4: "[Cl-].[Cl-].[NH3+]CCCCCCCC",
        5: "[Cl-].[Cl-].[NH3+]CCCCCCCC[NH3+]",
        6: "[Cl-].[Cl-].[NH3+]CCCCCCCC[NH3+]",
    },
    "reliability": {
        0: 2,
        1: 2,
        2: 1,
        3: 1,
        4: 2,
        5: 2,
        6: 2,
    },
    "endpoint": {0: "Ready", 1: "Ready", 2: "Ready", 3: "Ready", 4: "Ready", 5: "Ready", 6: "Inherent"},
    "guideline": {
        0: "OECD Guideline 301 C",
        1: "OECD Guideline 301 C",
        2: "OECD Guideline 301 B",
        3: "OECD Guideline 301 B",
        4: "OECD Guideline 301 D",
        5: "OECD Guideline 301 D",
        6: "OECD Guideline 301 D",
    },
    "principle": {
        0: "Closed Respirometer",
        1: "Closed Respirometer",
        2: "CO2 Evolution",
        3: "CO2 Evolution",
        4: "DOC Die Away",
        5: "DOC Die Away",
        6: "DOC Die Away",
    },
    "time_day": {0: 14.0, 1: 14.0, 2: 28.0, 3: 28.0, 4: 29.0, 5: 29.0, 6: 29.0},
    "biodegradation_percent": {
        0: 0.0,
        1: 1.0,
        2: 0.0,
        3: 1.0,
        4: 0.0,
        5: 1.0,
        6: 0.5,
    },
    "inchi_from_smiles": {
        0: "InChI=1S/C8H8O3/c1-11-7-4-2-6(3-5-7)8(9)10/h2-5H,1H3,(H,9,10)",
        1: "InChI=1S/C8H8O3/c1-11-7-4-2-6(3-5-7)8(9)10/h2-5H,1H3,(H,9,10)",
        2: "InChI=1S/C8H8O3/c1-11-7-4-2-6(3-5-7)8(9)10/h2-5H,1H3,(H,9,10)",
        3: "InChI=1S/C8H8O3/c1-11-7-4-2-6(3-5-7)8(9)10/h2-5H,1H3,(H,9,10)",
        4: "InChI=1S/C8H8O3/c1-11-7-4-2-6(3-5-7)8(9)10/h2-5H,1H3,(H,9,10)",
        5: "InChI=1S/C8H8O3/c1-11-7-4-2-6(3-5-7)8(9)10/h2-5H,1H3,(H,9,10)",
        6: "InChI=1S/C8H8O3/c1-11-7-4-2-6(3-5-7)8(9)10/h2-5H,1H3,(H,9,10)",
    },
}


@pytest.fixture
def xml_data() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(d)
    return df


@pytest.fixture
def echem_df_paper() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(echem_df_paper_dict)
    return df


@pytest.fixture
def new_echem() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(new_echem_dict)
    return df


@pytest.fixture
def regression_paper_full() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(regression_dict_full)
    return df


@pytest.fixture
def regression_paper() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(regression_dict)
    return df


@pytest.fixture
def regression_paper_with_star() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(regression_dict_with_star)
    return df


@pytest.fixture
def class_original_paper() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(class_original_dict)
    return df


@pytest.fixture
def class_external_paper() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(class_external_dict)
    return df


@pytest.fixture
def class_all_paper() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(class_all_dict)
    return df


@pytest.fixture
def class_all_no_processing() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(class_all_no_processing_dict)
    return df


@pytest.fixture
def class_for_labelling() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(class_for_labelling_dict)
    return df


@pytest.fixture
def outlier_detection() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(outlier_detection_dict)
    return df


@pytest.fixture
def group_remove_duplicates() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(group_remove_duplicates_dict)
    return df


@pytest.fixture
def df_a_full_original() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(df_a_full_original_dict)
    return df


@pytest.fixture
def df_a() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(df_a_dict)
    return df


@pytest.fixture
def df_b() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(df_b_dict)
    return df


@pytest.fixture
def smiles_not_found() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(smiles_not_found_dict)
    return df


@pytest.fixture
def df_a_full() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(df_a_full_dict)
    return df


@pytest.fixture
def df_b_found_full() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(df_b_found_full_dict)
    return df


@pytest.fixture
def df_pubchem() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(df_pubchem_dict)
    return df


@pytest.fixture
def df_lunghini_added_cas() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(df_lunghini_added_cas_dict)
    return df


@pytest.fixture
def class_df() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(class_df_dict)
    return df


@pytest.fixture
def class_df_long() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(class_df_long_dict)
    return df


@pytest.fixture
def metal_smiles() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(metal_smiles_dict)
    return df


@pytest.fixture
def df_improved_class() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(df_improved_class_dict)
    return df


@pytest.fixture
def df_improved_reg() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(df_improved_reg_dict)
    return df


@pytest.fixture
def class_improved() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(class_improved_dict)
    return df


@pytest.fixture
def df_for_aggregate() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(df_for_aggregate_dict)
    return df