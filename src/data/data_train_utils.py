import os
import sys
import dill
import pickle
import random
import warnings
import itertools
from collections import defaultdict

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.data import HeteroData

from scipy.spatial.transform import Rotation

import Bio
# These are just annoying :')
import Bio.PDB
warnings.filterwarnings("ignore",
    category=Bio.PDB.PDBExceptions.PDBConstructionWarning)
from Bio.Data.IUPACData import protein_letters_3to1

from utils import load_csv, printt
from utils import compute_rmsd


# -------- DATA LOADING -------
DATA_CACHE_VERSION = "v2"

class Loader:
    """
        Raw data loader with shared utilities
        @param (bool) data  default=True to load complexes, not poses
    """
    def __init__(self, args, data=True):
        self.args = args
        if data:
            self.root = args.data_path
            self.data_file = args.data_file
        else:
            self.root = args.pose_path
            self.data_file = args.pose_file
        # read in file paths
        self.data_to_load = load_csv(self.data_file)
        self.data_cache = self.data_file.replace(".csv", f"_cache_{DATA_CACHE_VERSION}.pkl")
        # cache for post-processed data, optional
        self.graph_cache = self.data_file.replace(".csv",
                f"_graph_{args.resolution}.pkl")
        self.esm_cache = self.data_file.replace(".csv", f"_esm.pkl")
        #if args.use_orientation_features:
        #    self.data_cache = self.data_cache.replace(".pkl", "_orientation.pkl")
        #    self.graph_cache = self.graph_cache.replace(".pkl", "_orientation.pkl")
        #    self.esm_cache = self.esm_cache.replace(".pkl", "_orientation.pkl")
        if args.debug:
            self.data_cache = self.data_cache.replace(".pkl", "_debug.pkl")
            self.graph_cache = self.graph_cache.replace(".pkl", "_debug.pkl")
            self.esm_cache = self.esm_cache.replace(".pkl", "_debug.pkl")
        # biopython PDB parser
        self.parser = Bio.PDB.PDBParser()

    def read_files(self):
        raise Exception("implement me")

    @staticmethod
    def add_values_to_dict(dictionary, **values):
        for key, value in values.items():
            dictionary[key].append(value)

    def parse_pdb(self, fp, models=None, chains=None):
        """
            Parse PDB file via Biopython
            (instead of my hacky implementation based on the spec)
            @return  (residue sequence, coordinates, atom sequence)
        """
        all_res, all_atom, all_pos = [], [], []
        visualization_values = defaultdict(list)

        structure = self.parser.get_structure(fp, fp)  # name, path
        # (optional) select subset of models
        if models is not None:
            models = [structure[m] for m in models]
        else:
            models = structure
        for model in models:
            # (optional) select subset of chains
            if chains is not None:
                chains = [model[c] for c in chains if c in model]
            else:
                chains = model
            # retrieve all atoms in desired chains
            for chain in chains:
                res = chain.get_residues()
                for res in chain:
                    # ignore HETATM records
                    if res.id[0] != " ":
                        continue
                    for atom in res:
                        all_res.append(res.get_resname())
                        all_pos.append(torch.from_numpy(atom.get_coord()))
                        all_atom.append((atom.get_name(), atom.element))
                        self.add_values_to_dict(visualization_values,
                                                chain=repr(chain).replace("<Chain id=", "").replace(">", ""),
                                                residue=res.get_segid(),
                                                atom_name=atom.get_name(),
                                                element=atom.element)
        all_pos = torch.stack(all_pos, dim=0)
        visualization_values["resname"] = all_res

        return all_res, all_atom, all_pos, visualization_values

    def _to_dict(self, pdb, p1, p2):
        item = {
            "pdb_id": pdb,
            "receptor": p1,
            "ligand": p2
        }
        return item

    def convert_pdb(self, all_res, all_atom, all_pos, all_visualization_values):
        """
            Unify PDB representation across different dataset formats.
            Given all residues, coordinates, and atoms, select subset
            of all atoms to keep.

            @return tuple(list, torch.Tensor) seq, pos
        """
        if self.args.resolution == "atom":
            atoms_to_keep = None
        elif self.args.resolution == "backbone":
            atoms_to_keep = ["N", "CA", "O", "CB"]
        elif self.args.resolution == "residue":
            atoms_to_keep = ["CA"]
        else:
            raise Exception(f"invalid resolution {self.args.resolution}")

        if atoms_to_keep is not None:
            to_keep, atoms = [], []
            for i,a in enumerate(all_atom):
                if a[0] in atoms_to_keep:  # atom name
                    to_keep.append(i)
                    atoms.append(a[1])  # element symbol
            seq = [all_res[i] for i in to_keep]
            visualization_values = {key: [values[i] for i in to_keep] for key, values in all_visualization_values.items()}
            pos = all_pos[torch.tensor(to_keep)]
        else:
            atoms = all_atom
            seq = all_res
            pos = all_pos
            visualization_values = all_visualization_values
        assert pos.shape[0] == len(seq) == len(atoms)
        return atoms, seq, pos, visualization_values

    def extract_coords(self, all_res, all_atom, all_pos):
        """
            Extract coordinates of CA, N, and C atoms that are needed 
            to compute orientation features, as proposed in EquiDock

            @return c_alpha_coords, n_coords, c_coords (all torch.Tensor)
        """
        c_alpha_coords = []
        n_coords = []
        c_coords = []
        for i, a in enumerate(all_atom):
            if a[0] == "CA":
                c_alpha_coords.append(all_pos[i])
            if a[0] == "N":
                n_coords.append(all_pos[i])
            if a[0] == "C":
                c_coords.append(all_pos[i])

        c_alpha_coords = torch.stack(c_alpha_coords)
        n_coords = torch.stack(n_coords)
        c_coords = torch.stack(c_coords)
        
        assert c_alpha_coords.shape == n_coords.shape == c_coords.shape

        return c_alpha_coords, n_coords, c_coords

    def compute_orientation_vectors(self, c_alpha_coords, n_coords, c_coords):
        """
            Extract 3D coordinates and n_i, u_i, v_i vectors of representative residues
        """
        num_residues = c_alpha_coords.shape[0]
        
        n_i_list = []
        u_i_list = []
        v_i_list = []
        for i in range(num_residues):
            n_coord = n_coords[i]
            c_alpha_coord = c_alpha_coords[i]
            c_coord = c_coords[i]
            u_i = (n_coord - c_alpha_coord) / torch.linalg.vector_norm(n_coord - c_alpha_coord)
            t_i = (c_coord - c_alpha_coord) / torch.linalg.vector_norm(c_coord - c_alpha_coord)
            n_i = torch.linalg.cross(u_i, t_i) / torch.linalg.vector_norm(torch.linalg.cross(u_i, t_i))
            v_i = torch.linalg.cross(n_i, u_i)
            assert (torch.abs(torch.linalg.vector_norm(v_i) - 1.) < 1e-5), "v_i norm is not 1"
            n_i_list.append(n_i)
            u_i_list.append(u_i)
            v_i_list.append(v_i)

        n_i_feat = torch.stack(n_i_list)
        u_i_feat = torch.stack(u_i_list)
        v_i_feat = torch.stack(v_i_list)
        
        assert n_i_feat.shape == u_i_feat.shape == v_i_feat.shape
        
        return n_i_feat, u_i_feat, v_i_feat

    def process_data(self, data, args):
        """
            Tokenize, etc.
        """
        # check if cache exists
        if not args.no_graph_cache and os.path.exists(self.graph_cache):
            with open(self.graph_cache, "rb") as f:
                data, data_params = pickle.load(f)
                printt("Loaded processed data from cache")
                return data, data_params

        # non-data parameters
        data_params = {}

        #### select subset of residues that match desired
        # resolution (e.g. residue-level vs. atom-level)
        for item in data.values():
            item["visualization_values"] = {}
            for k in ["receptor", "ligand"]:
                # >>> in what case would this be false?
                if k in item:
                    subset = self.convert_pdb(*item[k])
                    item[f"{k}_atom"] = subset[0]
                    item[f"{k}_seq"] = subset[1]
                    item[f"{k}_xyz"] = subset[2]
                    item["visualization_values"][k] = subset[3]

                    # for orientation features we need coordinates of CA, N, and C atoms
                    # These coordinates define the local coordinate system of each residue
                    if args.use_orientation_features:
                        c_alpha_coords, n_coords, c_coords = self.extract_coords(*item[k])
                        n_i_feat, u_i_feat, v_i_feat = self.compute_orientation_vectors(c_alpha_coords, n_coords, c_coords)
                        item[f"{k}_n_i_feat"] = n_i_feat
                        item[f"{k}_u_i_feat"] = u_i_feat
                        item[f"{k}_v_i_feat"] = v_i_feat

        #### convert to HeteroData graph objects
        for item in data.values():
            item["graph"] = self.to_graph(item)

        if not args.no_graph_cache:
            with open(self.graph_cache, "wb+") as f:
                pickle.dump([data, data_params], f)

        return data, data_params

    def process_embed(self, data, data_params):
        #### tokenize AFTER converting to graph
        if self.args.lm_embed_dim > 0:
            data = self.compute_embeddings(data)
            data_params["num_residues"] = 23  # <cls> <sep> <pad>
            printt("finished tokenizing residues with ESM")
        else:
            # tokenize residues for non-ESM
            tokenizer = tokenize(data.values(), "receptor_seq")
            tokenize(data.values(), "ligand_seq", tokenizer)
            self.esm_model = None
            data_params["num_residues"] = len(tokenizer)
            data_params["tokenizer"] = tokenizer
            printt("finished tokenizing residues")

        #### protein sequence tokenization
        # tokenize atoms
        atom_tokenizer = tokenize(data.values(), "receptor_atom")
        tokenize(data.values(), "ligand_atom", atom_tokenizer)
        data_params["atom_tokenizer"] = atom_tokenizer
        printt("finished tokenizing all inputs")

        return data, data_params

    def to_graph(self, item):
        """
            Convert raw dictionary to PyTorch geometric object
        """
        data = HeteroData()
        data["name"] = item["path"]
        # retrieve position and compute kNN
        for key in ["receptor", "ligand"]:
            data[key].pos = item[f"{key}_xyz"].float()
            data[key].x = item[f"{key}_seq"]  # _seq is residue id
            if self.args.use_orientation_features:
                data[key].n_i_feat = item[f"{key}_n_i_feat"].float()
                data[key].u_i_feat = item[f"{key}_u_i_feat"].float()
                data[key].v_i_feat = item[f"{key}_v_i_feat"].float()
            # kNN graph
            edge_index = knn_graph(data[key].pos, self.args.knn_size)
            data[key, "contact", key].edge_index = edge_index
        # center receptor at origin
        center = data["receptor"].pos.mean(dim=0, keepdim=True)
        for key in ["receptor", "ligand"]:
            data[key].pos = data[key].pos - center
        data.center = center  # save old center
        return data

    def compute_embeddings(self, data):
        """
            Pre-compute ESM2 embeddings.
        """
        # check if we already computed embeddings
        if os.path.exists(self.esm_cache):
            with open(self.esm_cache, "rb") as f:
                path_to_rep = pickle.load(f)
            self._save_esm_rep(data, path_to_rep)
            printt("Loaded cached ESM embeddings")
            return data

        printt("Computing ESM embeddings")
        # load pretrained model
        esm_model, alphabet = torch.hub.load(
                "facebookresearch/esm:main",
                "esm2_t33_650M_UR50D")
        self.esm_model = esm_model.cuda().eval()
        tokenizer = alphabet.get_batch_converter()
        # convert to 3 letter codes
        aa_code = defaultdict(lambda: "<unk>")
        aa_code.update(
            {k.upper():v for k,v in protein_letters_3to1.items()})
        # fix ordering
        all_pdbs = sorted(data)
        all_graphs = [data[pdb]["graph"] for pdb in all_pdbs]
        rec_seqs = [g["receptor"].x for g in all_graphs]
        lig_seqs = [g["ligand"].x for g in all_graphs]
        rec_seqs = ["".join(aa_code[s] for s in seq) for seq in rec_seqs]
        lig_seqs = ["".join(aa_code[s] for s in seq) for seq in lig_seqs]
        # batchify sequences
        rec_batches = self._esm_batchify(rec_seqs, tokenizer)
        lig_batches = self._esm_batchify(lig_seqs, tokenizer)
        with torch.no_grad():
            pad_idx = alphabet.padding_idx
            rec_reps = self._run_esm(rec_batches, pad_idx)
            lig_reps = self._run_esm(lig_batches, pad_idx)

        # dump to cache
        path_to_rep = {}
        for idx, pdb in enumerate(all_pdbs):
            # cat one-hot representation and ESM embedding
            rec_graph_x = torch.cat([rec_reps[idx][0],
                rec_reps[idx][1]], dim=1)
            lig_graph_x = torch.cat([lig_reps[idx][0],
                lig_reps[idx][1]], dim=1)
            path_to_rep[pdb] = rec_graph_x, lig_graph_x
        with open(self.esm_cache, "wb+") as f:
            pickle.dump(path_to_rep, f)

        # overwrite graph.x for each element in batch
        self._save_esm_rep(data, path_to_rep)

        return data

    def _esm_batchify(self, seqs, tokenizer):
        batch_size = find_largest_diviser_smaller_than(len(seqs), 32)
        #self.args.batch_size if len(seqs)%self.args.batch_size == 0 else find_smallest_diviser(len(seqs))

        iterator = range(0, len(seqs), batch_size)
        # group up sequences
        batches = [seqs[i:i + batch_size] for i in iterator]
        batches = [[("", seq) for seq in batch] for batch in batches]
        # tokenize
        batch_tokens = [tokenizer(batch)[2] for batch in batches]
        return batch_tokens

    def _run_esm(self, batches, padding_idx):
        """
            Wrapper around ESM specifics
            @param (list)  batch
            @return (list)  same order as batch
        """
        # run ESM model
        all_reps = []
        for batch in tqdm(batches, desc="ESM", ncols=50):
            reps = self.esm_model(batch.cuda(), repr_layers=[33])
            reps = reps["representations"][33].cpu().squeeze()[:,1:]
            all_reps.append(reps)
        # crop to length
        # exclude <cls> <sep>
        cropped = []
        for i, batch in enumerate(batches):
            batch_lens = (batch != padding_idx).sum(1) - 2
            for j, length in enumerate(batch_lens):
                rep_crop = all_reps[i][j,:length]
                token_crop = batch[j,1:length+1,None]
                cropped.append((rep_crop, token_crop))
        return cropped

    def _save_esm_rep(self, data, path_to_rep):
        """
            Assign new ESM representation to graph.x
        """
        for pdb, (rec_rep, lig_rep) in path_to_rep.items():
            rec_graph = data[pdb]["graph"]["receptor"]
            lig_graph = data[pdb]["graph"]["ligand"]
            rec_graph.x = rec_rep
            lig_graph.x = lig_rep
            assert len(rec_graph.pos) == len(rec_graph.x)
            # print(f'len(rec_graph.pos): {len(rec_graph.pos)}')
            # print(f'len(rec_graph.x): {len(rec_graph.x)}')
            # print('***')
            # print(f'len(lig_graph.pos): {len(lig_graph.pos)}')
            # print(f'len(lig_graph.x): {len(lig_graph.x)}')
            # print('-----')
            assert len(lig_graph.pos) == len(lig_graph.x)
        return data

    def split_data(self, raw_data, args):
        # separate out train/test
        data = {
            "train": {},
            "val":   {},
            "test":  {}
        }
        ## if test split is pre-specified, split into train/test
        # otherwise, allocate all data to train for cross-validation
        for k, item in raw_data.items():
            if "split" in item:
                data[item["split"]][k] = item
            else:
                data["train"] = raw_data
                break
        # split train into separate folds
        # sometimes we only want to load val data. Then train data is empty
        if len(data["train"]) > 0:
            data["train"] = split_into_folds(data["train"], args)
        return data

    def crossval_split(self, fold_num):
        """
            number of folds-way cross validation
            @return  dict: split -> [pdb_ids]
        """
        splits = { "train": [] }
        # split into train/val/test
        folds = [list(sorted(f)) for f in self.splits["train"]]
        val_data = list(sorted(self.splits["val"]))
        test_data = list(sorted(self.splits["test"]))
        # if val split is pre-specified, do not take fold
        if len(val_data) == 0:
            val_fold = fold_num
            splits["val"] = folds[val_fold]
        else:
            val_fold = -1  # all remaining folds go to train
            splits["val"] = val_data
        # if test split is pre-specified, do not take fold
        if len(test_data) > 0:
            test_fold = val_fold  # must specify to allocate train folds
            splits["test"] = test_data
        # otherwise both val/test labels depend on fold_num
        else:
            test_fold = (fold_num+1) % self.args.num_folds
            splits["test"] = folds[test_fold]
        # add remaining to train
        for idx in range(self.args.num_folds):
            if idx in [val_fold, test_fold]:
                continue
            splits["train"].extend(folds[idx])
        return splits


class DIPSLoader(Loader):
    def __init__(self, args, split=None, batch=None, verbose=True):
        super(DIPSLoader, self).__init__(args)
        # load and process files
        data = self.read_files()
        data = self.assign_receptor(data)
        data, data_params = self.process_data(data, args)
        data, data_params = self.process_embed(data, data_params)

        #### pre-compute ESM embeddings if needed
        self.data = data
        self.data_params = data_params
        printt(len(self.data), "entries loaded")
        # split into folds
        self.splits = self.split_data(data, args)

    def read_files(self):
        data = {}
        # check if loaded previously
        if not self.args.recache and os.path.exists(self.data_cache):
            with open(self.data_cache, "rb") as f:
                path_to_data = pickle.load(f)
        else:
            path_to_data = {}
        for line in tqdm(self.data_to_load,
                         desc="data loading", ncols=50):
            path = line["path"]
            item = path_to_data.get(path)
            if item is None:
                item = self.parse_dill(os.path.join(self.root, path), path)
                path_to_data[path] = item
            item.update(line)  # add meta-data
            data[path] = item
            if self.args.debug:
                if len(data) >= 2:
                    break
        # write to cache
        if self.args.recache or not os.path.exists(self.data_cache):
            with open(self.data_cache, "wb+") as f:
                pickle.dump(path_to_data, f)
        return data

    def assign_receptor(self, data):
        """
            For docking, we assigned smaller protein as ligand
            for this dataset (since no canonical receptor/ligand
            assignments)
        """
        for item in data.values():
            rec = item["receptor"]
            lig = item["ligand"]
            if len(rec[0]) < len(lig[0]):
                item["receptor"] = lig
                item["ligand"] = rec
        return data

    def parse_dill(self, fp, pdb_id):
        with open(fp, "rb") as f:
            data = dill.load(f)
        p1, p2 = data[1], data[2]
        p1, p2 = self.parse_df(p1), self.parse_df(p2)
        return self._to_dict(pdb_id, p1, p2)

    def parse_df(self, df):
        """
            Parse PDB DataFrame
        """
        # extract dataframe values
        all_res = df["resname"]
        all_pos = torch.tensor([df["x"], df["y"], df["z"]]).t()
        all_atom = list(zip(df["atom_name"], df["element"]))
        visualization_values = {"chain": df["chain"],
                                "resname": all_res,
                                "residue": df["residue"],
                                "atom_name": df["atom_name"],
                                "element": df["element"]}
        # convert to seq, pos
        return all_res, all_atom, all_pos, visualization_values


class DB5Loader(Loader):
    """
        Docking benchmark 5.5
    """
    def __init__(self, args):
        super(DB5Loader, self).__init__(args)
        # load and process files
        self.root = os.path.join(self.root, "structures")
        # cache file dependent on use_unbound
        if args.use_unbound:
            printt('Using Unbound structures')
            self.data_cache = self.data_cache.replace(".pkl", "_u.pkl")
            self.esm_cache = self.esm_cache.replace(".pkl", "_u.pkl")
        else:
            printt('Using Bound structures')
            self.data_cache = self.data_cache.replace(".pkl", "_b.pkl")
            self.esm_cache = self.esm_cache.replace(".pkl", "_b.pkl")
        data = self.read_files()
        data, data_params = self.process_data(data, args)
        data, data_params = self.process_embed(data, data_params)
        self.data = data
        self.data_params = data_params
        printt(len(self.data), "entries loaded")
        # split into folds
        self.splits = self.split_data(data, args)

    def read_files(self):
        data = {}
        # check if loaded previously
        if not self.args.recache and os.path.exists(self.data_cache):
            with open(self.data_cache, "rb") as f:
                path_to_data = pickle.load(f)
        else:
            path_to_data = {}
        for line in tqdm(self.data_to_load,
                         desc="data loading", ncols=50):
            pdb = line["path"]
            item = path_to_data.get(pdb)
            if item is None:
                item = self.parse_path(pdb, os.path.join(self.root, pdb))
                path_to_data[pdb] = item
            item.update(line)  # add meta-data
            data[pdb] = item
            if self.args.debug:
                if len(data) >= 2:
                    break
        # write to cache
        if self.args.recache or not os.path.exists(self.data_cache):
            with open(self.data_cache, "wb+") as f:
                pickle.dump(path_to_data, f)
        return data

    def parse_path(self, pdb, path):
        if self.args.use_unbound:
            fp_rec = f"{path}_r_u.pdb"
            fp_lig = f"{path}_l_u.pdb"
        else:
            fp_rec = f"{path}_r_b.pdb"
            fp_lig = f"{path}_l_b.pdb"
        p1, p2 = self.parse_pdb(fp_rec), self.parse_pdb(fp_lig)
        return self._to_dict(pdb, p1, p2)


class SabDabLoader(Loader):
    """
        Structure Antibody Database
        Downloaded May 2, 2022.
    """
    def __init__(self, args):
        super(SabDabLoader, self).__init__(args)
        # standard workflow
        data = self.read_files()
        data, data_params = self.process_data(data, args)
        data, data_params = self.process_embed(data, data_params)
        self.data = data
        self.data_params = data_params
        printt(len(self.data), "entries loaded")
        # split into folds
        self.splits = self.split_data(data, args)

    def read_files(self):
        data = {}
        # check if loaded previously
        if os.path.exists(self.data_cache):
            with open(self.data_cache, "rb") as f:
                path_to_data = pickle.load(f)
        else:
            path_to_data = {}

        for line in tqdm(self.data_to_load,
                         desc="data loading", ncols=50):
            pdb = line["pdb"]
            item = path_to_data.get(pdb)
            if item is None:
                # parse sabdab entry
                pdb = line["pdb"]
                rec_chains = [
                    c.strip() for c in line["antigen_chain"].split("|")]
                lig_chains = set([c.upper() for c in [
                    line["Hchain"], line["Lchain"]] if c is not None])
                # parse pdb object
                path = os.path.join(self.root, f"{pdb}.pdb")
                rec = self.parse_pdb(path, line["model"], rec_chains)
                lig = self.parse_pdb(path, line["model"], lig_chains)
                # TODO SabDab pdb_id needs to be de-duped
                item = self._to_dict(pdb, rec, lig)
                path_to_data[pdb] = item
            item.update(line)  # add meta-data
            data[pdb] = item
        # write to cache
        if not os.path.exists(self.data_cache):
            with open(self.data_cache, "wb+") as f:
                pickle.dump(path_to_data, f)
        return data


# ------ DATA PROCESSING ------

def tokenize(data, key, tokenizer=None):
    """
        Tokenize every item[key] in data.
        Modifies item[key] and copies original value to item[key_raw].
        @param (list) data
        @param (str) key  item[key] is iterable
    """
    if len(data) == 0:  # sometimes no val split, etc.
        return
    # if tokenizer is not provided, create index
    all_values = [item[key] for item in data]
    all_values = set(itertools.chain(*all_values))
    if tokenizer is None:
        tokenizer = {}  # never used
    if type(tokenizer) is dict:
        for item in sorted(all_values):
            if item not in tokenizer:
                tokenizer[item] = len(tokenizer) + 1  # 1-index
        f_token = lambda seq: [tokenizer[x] for x in seq]
    else:
        aa_code = defaultdict(lambda: "<unk>")
        aa_code.update(
            {k.upper():v for k,v in protein_letters_3to1.items()})
        def f_token(seq):
            seq = "".join([aa_code[s] for s in seq])
            seq = tokenizer([("", seq)])[2][0]
            return seq
    # tokenize items and modify data in place
    raw_key = f"{key}_raw"
    for item in data:
        raw_item = item[key]
        item[raw_key] = raw_item  # save old
        # tokenize and convert to tensor if applicable
        item[key] = f_token(raw_item)
        if not torch.is_tensor(item[key]):
            item[key] = torch.tensor(item[key])
    return tokenizer


def split_into_folds(data, args):
    """
        Split into train/val/test folds
        @param (list) data
    """
    # split by cluster
    for item in data.values():
        by_cluster = ("cluster" in item)
        break
    if by_cluster:
        clst_to_pdb = defaultdict(list)
        for k in data:
            clst_to_pdb[data[k]["cluster"]].append(k)
    else:
        clst_to_pdb = {k:[k] for k in data}
    keys = list(sorted(clst_to_pdb))
    random.shuffle(keys)
    # split into folds
    cur_clst = 0
    folds = [{} for _ in range(args.num_folds)]
    fold_ratio = 1. / args.num_folds
    max_fold_size = int(len(data) * fold_ratio)
    for fold_num in range(args.num_folds):
        fold = []
        while (len(fold) < max_fold_size) and (cur_clst < len(clst_to_pdb)):
            fold.extend(clst_to_pdb[keys[cur_clst]])
            cur_clst += 1
        for k in fold:
            folds[fold_num][k] = data[k]
    return folds

# ------ DATA COLLATION -------

def get_mask(lens):
    """ torch.MHA style mask (False = attend, True = mask) """
    mask = torch.arange(max(lens))[None, :] >= lens[:, None]
    return mask

def find_smallest_diviser(n):
    """
    Returns smallest diviser > 1
    """
    i = 2
    while n%i != 0:
        i += 1
    return i

def find_largest_diviser_smaller_than(n, upper):
    """
    Returns largest diviser of n smaller than upper
    """
    i = upper
    while n%i !=0:
        i = i-1
    return i
