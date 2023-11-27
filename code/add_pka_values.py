"""In this file the model MolGpKa is used to predict pKa values.
Most of this code was taken from https://github.com/Xundrug/MolGpKa/tree/master """
"""Requirements are in requirements_molgpka.txt"""

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
from rdkit.Chem.MolStandardize import rdMolStandardize
import os.path as osp
import numpy as np
import pandas as pd
import torch
import os
from typing import List, Tuple
from torch_geometric.nn import GlobalAttention
from torch_geometric.data import Data
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from pHcalc import Acid
import math
import structlog
import argparse
from tqdm import tqdm

tqdm.pandas()
log = structlog.get_logger()

root = os.path.join(os.path.abspath(""), "MolGpKa")
smarts_file = os.path.join(root, "src/utils/smarts_pattern.tsv")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_curated_no_metal",
    default=False,
    type=bool,
    help="Whether to add pka values to reg_curated_s_no_metal & reg_curated_scs_no_metal",
)
parser.add_argument(
    "--run_iuclid_data",
    default=False,
    type=bool,
    help="Whether to add pka values to iuclid_echa",
)
parser.add_argument(
    "--paths_to_dfs_to_run",
    "--names-list",
    nargs="+",
    default=[],
    help="List of paths to dfs to which pka and alpha values should be added",
)
args = parser.parse_args()

# ----------------------------------------------------------------------------------
# Code from MolGpKa


def split_acid_base_pattern(smarts_file):
    df_smarts = pd.read_csv(smarts_file, sep="\t")
    df_smarts_acid = df_smarts[df_smarts.Acid_or_base == "A"]
    df_smarts_base = df_smarts[df_smarts.Acid_or_base == "B"]
    return df_smarts_acid, df_smarts_base


def unique_acid_match(matches):
    single_matches = list(set([m[0] for m in matches if len(m) == 1]))
    double_matches = [m for m in matches if len(m) == 2]
    single_matches = [[j] for j in single_matches]
    double_matches.extend(single_matches)
    return double_matches


def match_acid(df_smarts_acid, mol):
    matches = []
    for idx, name, smarts, index, acid_base in df_smarts_acid.itertuples():
        pattern = Chem.MolFromSmarts(smarts)
        match = mol.GetSubstructMatches(pattern)
        if len(match) == 0:
            continue
        if len(index) > 2:
            index = index.split(",")
            index = [int(i) for i in index]
            for m in match:
                matches.append([m[index[0]], m[index[1]]])
        else:
            index = int(index)
            for m in match:
                matches.append([m[index]])
    matches = unique_acid_match(matches)
    matches_modify = []
    for i in matches:
        for j in i:
            matches_modify.append(j)
    return matches_modify


def match_base(df_smarts_base, mol):
    matches = []
    for idx, name, smarts, indexs, acid_base in df_smarts_base.itertuples():
        pattern = Chem.MolFromSmarts(smarts)
        match = mol.GetSubstructMatches(pattern)
        if len(match) == 0:
            continue
        index_split = indexs.split(",")
        for index in index_split:
            index = int(index)
            for m in match:
                matches.append([m[index]])
    matches = unique_acid_match(matches)
    matches_modify = []
    for i in matches:
        for j in i:
            matches_modify.append(j)
    return matches_modify


def get_ionization_aid(mol, acid_or_base=None):
    df_smarts_acid, df_smarts_base = split_acid_base_pattern(smarts_file)

    if mol == None:
        raise RuntimeError("read mol error: {}".format(mol))
    acid_matches = match_acid(df_smarts_acid, mol)
    base_matches = match_base(df_smarts_base, mol)
    if acid_or_base == None:
        return acid_matches, base_matches
    elif acid_or_base == "acid":
        return acid_matches
    else:
        return base_matches


def one_hot(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_bond_pair(mol):
    bonds = mol.GetBonds()
    res = [[], []]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res


def get_atom_features(mol, aid):
    AllChem.ComputeGasteigerCharges(mol)
    Chem.AssignStereochemistry(mol)

    acceptor_smarts_one = "[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]"
    acceptor_smarts_two = "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]"
    donor_smarts_one = "[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]"
    donor_smarts_two = "[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]"

    hydrogen_donor_one = Chem.MolFromSmarts(donor_smarts_one)
    hydrogen_donor_two = Chem.MolFromSmarts(donor_smarts_two)
    hydrogen_acceptor_one = Chem.MolFromSmarts(acceptor_smarts_one)
    hydrogen_acceptor_two = Chem.MolFromSmarts(acceptor_smarts_two)

    hydrogen_donor_match_one = mol.GetSubstructMatches(hydrogen_donor_one)
    hydrogen_donor_match_two = mol.GetSubstructMatches(hydrogen_donor_two)
    hydrogen_donor_match = []
    hydrogen_donor_match.extend(hydrogen_donor_match_one)
    hydrogen_donor_match.extend(hydrogen_donor_match_two)
    hydrogen_donor_match = list(set(hydrogen_donor_match))

    hydrogen_acceptor_match_one = mol.GetSubstructMatches(hydrogen_acceptor_one)
    hydrogen_acceptor_match_two = mol.GetSubstructMatches(hydrogen_acceptor_two)
    hydrogen_acceptor_match = []
    hydrogen_acceptor_match.extend(hydrogen_acceptor_match_one)
    hydrogen_acceptor_match.extend(hydrogen_acceptor_match_two)
    hydrogen_acceptor_match = list(set(hydrogen_acceptor_match))

    ring = mol.GetRingInfo()

    m = []
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)

        o = []
        o += one_hot(atom.GetSymbol(), ["C", "H", "O", "N", "S", "Cl", "F", "Br", "P", "I"])
        o += [atom.GetDegree()]
        o += one_hot(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ],
        )
        o += [atom.GetImplicitValence()]
        o += [atom.GetIsAromatic()]
        o += [
            ring.IsAtomInRingOfSize(atom_idx, 3),
            ring.IsAtomInRingOfSize(atom_idx, 4),
            ring.IsAtomInRingOfSize(atom_idx, 5),
            ring.IsAtomInRingOfSize(atom_idx, 6),
            ring.IsAtomInRingOfSize(atom_idx, 7),
            ring.IsAtomInRingOfSize(atom_idx, 8),
        ]

        o += [atom_idx in hydrogen_donor_match]
        o += [atom_idx in hydrogen_acceptor_match]
        o += [atom.GetFormalCharge()]
        if atom_idx == aid:
            o += [0]
        else:
            o += [len(Chem.rdmolops.GetShortestPath(mol, atom_idx, aid))]

        if atom_idx == aid:
            o += [True]
        else:
            o += [False]
        m.append(o)
    return m


def mol2vec(mol, atom_idx, evaluation=True, pka=None):
    node_f = get_atom_features(mol, atom_idx)
    edge_index = get_bond_pair(mol)
    if evaluation:
        batch = np.zeros(
            len(node_f),
        )
        data = Data(
            x=torch.tensor(node_f, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            batch=torch.tensor(batch, dtype=torch.long),
        )
    else:
        data = Data(
            x=torch.tensor(node_f, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            pka=torch.tensor([[pka]], dtype=torch.float),
        )
    return data


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, improved=False, cached=False, bias=True, **kwargs):
        super(GCNConv, self).__init__(aggr="add", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    "Cached {} number of edges, but found {}. Please "
                    "disable the caching behavior of this layer by removing "
                    "the `cached=True` argument in its constructor.".format(self.cached_num_edges, edge_index.size(1))
                )

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class GCNNet(torch.nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(
            29, 1024, cached=False  # n_features = 29
        )  # if you defined cache=True, the shape of batch must be same!
        self.bn1 = BatchNorm1d(1024)
        self.conv2 = GCNConv(1024, 512, cached=False)
        self.bn2 = BatchNorm1d(512)
        self.conv3 = GCNConv(512, 256, cached=False)
        self.bn3 = BatchNorm1d(256)
        self.conv4 = GCNConv(256, 512, cached=False)
        self.bn4 = BatchNorm1d(512)
        self.conv5 = GCNConv(512, 1024, cached=False)
        self.bn5 = BatchNorm1d(1024)

        self.att = GlobalAttention(Linear(1024, 1))  # hidden = 1024
        self.fc2 = Linear(1024, 128)
        self.fc3 = Linear(128, 16)
        self.fc4 = Linear(16, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.conv5.reset_parameters()

        self.att.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = self.att(x, batch)

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def load_model(model_file, device="cpu"):
    model = GCNNet().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    return model


def model_pred(m2, aid, model, device="cpu"):
    data = mol2vec(m2, aid)
    with torch.no_grad():
        data = data.to(device)
        pKa = model(data)
        pKa = pKa.cpu().numpy()
        pka = pKa[0][0]
    return pka


def predict_acid(mol):
    model_file = osp.join(root, "models/weight_acid.pth")
    model_acid = load_model(model_file)

    acid_idxs = get_ionization_aid(mol, acid_or_base="acid")
    acid_res = {}
    for aid in acid_idxs:
        apka = model_pred(mol, aid, model_acid)
        acid_res.update({aid: apka})
    return acid_res


def predict_base(mol):
    model_file = osp.join(root, "models/weight_base.pth")
    model_base = load_model(model_file)

    base_idxs = get_ionization_aid(mol, acid_or_base="base")
    base_res = {}
    for aid in base_idxs:
        bpka = model_pred(mol, aid, model_base)
        base_res.update({aid: bpka})
    return base_res


def predict(mol, uncharged=True):
    if uncharged:
        un = rdMolStandardize.Uncharger()
        mol = un.uncharge(mol)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    mol = AllChem.AddHs(mol)
    base_dict = predict_base(mol)
    acid_dict = predict_acid(mol)
    AllChem.EmbedMolecule(mol)
    return base_dict, acid_dict, mol


# ----------------------------------------------------------------------------------
# Own functions


def calculate_alpha_values_based_on_pkas(pkas: List[float]) -> List[float]:
    pH = 7.4  # pH at which biodegradation tests are performed
    phos_acid = Acid(pKa=pkas[:4], charge=0)
    alphas_acid = phos_acid.alpha(pH)
    phos_base = Acid(pKa=pkas[4:], charge=0)
    alphas_base = phos_base.alpha(pH)
    return list(alphas_acid) + list(alphas_base)


def predict_pKa_for_df(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    def get_pKa(row):
        acid_values = [10.0, 10.0, 10.0, 10.0]
        base_values = [0.0, 0.0, 0.0, 0.0]
        smiles = row[smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        try:
            base_dict, acid_dict, _ = predict(mol)
            acid_values_found = sorted(list(acid_dict.values()))
            base_values_found = sorted(list(base_dict.values()))
            acid_values_found = [x for x in acid_values_found if (x < 9.4 and x > 0)]
            base_values_found = [x for x in base_values_found if (x > 5.4 and x < 10)]
        except:
            log.error(
                "Could not predict for this smiles, default values taken: ",
                smiles=smiles,
            )
            base_values_found = base_values
            acid_values_found = acid_values
        if len(acid_values_found) > 4:
            log.warn(
                "Found more than 4 acids!!! Only the 4 first items were taken!",
                acid_values_found=acid_values_found,
            )
            acid_values_found = acid_values_found[:4]
        if len(base_values_found) > 4:
            log.warn(
                "Found more than 4 acids!!! Only the 4 first items were taken!",
                base_values_found=base_values_found,
            )
            base_values_found = base_values_found[:4]
        for i_a in range(len(acid_values_found)):
            acid_values[i_a] = acid_values_found[i_a]
        bases_i = 3
        for i_b in range(len(base_values_found) - 1, -1, -1):
            base_values[bases_i] = base_values_found[i_b]
            bases_i -= 1

        pkas = [
            acid_values[0],
            acid_values[1],
            acid_values[2],
            acid_values[3],
            base_values[0],
            base_values[1],
            base_values[2],
            base_values[3],
        ]

        alphas = calculate_alpha_values_based_on_pkas(pkas)

        return pd.Series(pkas + alphas)

    df[
        [
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
    ] = df.progress_apply(get_pKa, axis=1)
    return df


def load_regression_df_curated_s_no_metal() -> pd.DataFrame:
    df_regression = pd.read_csv("datasets/curated_data/reg_curated_s_no_metal.csv", index_col=0)
    df_regression = df_regression[
        df_regression["cas"] != "1803551-73-6"
    ]  # Remove because cannot be converted to fingerprint (Explicit valence for atom # 0 F, 2, is greater than permitted)
    return df_regression


def load_regression_df_curated_scs_no_metal() -> pd.DataFrame:
    df_regression = pd.read_csv(
        "datasets/curated_data/reg_curated_scs_no_metal.csv", index_col=0
    )
    return df_regression


if __name__ == "__main__":
    if args.run_iuclid_data: 
        df = pd.read_csv("datasets/iuclid_echa.csv", index_col=0)
        df_pred = predict_pKa_for_df(df=df, smiles_col="smiles")
        df_pred.to_csv("datasets/iuclid_echa.csv")
    if args.run_improved_no_metal:
        df = load_regression_df_curated_s_no_metal()
        df_pred = predict_pKa_for_df(df=df, smiles_col="smiles")
        df_pred.to_csv("datasets/curated_data/reg_curated_s_no_metal.csv")
        df = load_regression_df_curated_scs_no_metal()
        df_pred = predict_pKa_for_df(df=df, smiles_col="smiles")
        df_pred.to_csv("datasets/curated_data/reg_curated_scs_no_metal.csv")

    for df_path in args.paths_to_dfs_to_run:
        df = pd.read_csv(df_path, index_col=0)
        df_pred = predict_pKa_for_df(df=df, smiles_col="smiles")
        df_pred.to_csv(df_path)
