import os
import sys
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader, DataListLoader

from . import data_train_utils as data_utils
from utils import compute_rmsd
from geom_utils import NoiseTransform, set_time, sample_vec, axis_angle_to_matrix
from scipy.spatial.transform import Rotation as R


def load_data(args, split=None, batch=None, verbose=True):
    """
        Load and minimally process data

        Args:
            split: If none, all splits are loaded. Otherwise, only the specified splut.
            batch: If none, all batches are loaded. Otherwise, only the specified batch.
    """
    # add others like db5 or sabdab if we run on those
    if args.dataset == "dips":
        data = data_utils.DIPSLoader(args, split=split, batch=batch, verbose=verbose)
    elif args.dataset == "db5":
        data = data_utils.DB5Loader(args)
    elif args.dataset == "toy":
        data = data_utils.ToyLoader(args, split=split, batch=batch, verbose=verbose)
    else:
        raise Exception("invalid --dataset", args.dataset)
    return data


def get_data(dataset, fold_num, args, for_reverse_diffusion=False,num_samples=1):
    """
        Convert raw data into DataLoaders for training.
    """
    if args.use_randomized_confidence_data:
        dataset_class = RandomizedConfidenceDataset
    else:
        dataset_class = BindingDataset

    if args.debug: #and False: TODO
        splits = {}
        splits["train"] = dataset_class(args, dataset.data, apply_transform=not for_reverse_diffusion)
        splits["train"].data = [splits["train"].data[1]] * args.multiplicity # take first element and repeat it
        splits["train"].length = len(splits["train"].data)
        splits["val"] = dataset_class(args, dataset.data, apply_transform=not for_reverse_diffusion)
        splits["val"].data = [splits["val"].data[1]] # take first element
        splits["val"].length = len(splits["val"].data)
        splits["test"] = copy.deepcopy(splits["train"]) # TODO
        print("test",len(splits["test"]),len(splits["train"]))
        if for_reverse_diffusion:
            return splits
        return _get_loader(splits, args)
    use_pose = type(dataset) is tuple
    if use_pose:
        dataset, poses = dataset
    # smush folds and convert to Dataset object
    # or extract val and rest are train
    # for training, without crossval_split, weird stuff happens
    splits = dataset.crossval_split(fold_num) if args.mode == "train" else dataset.splits # TODO
    # splits = dataset.splits

    for split, pdb_ids in splits.items():
        # debug mode: only load small dataset
        if args.debug and False:
            pdb_ids = pdb_ids[:320] # 10 batches
        splits[split] = dataset_class(args, dataset.data, pdb_ids, apply_transform=not for_reverse_diffusion)
    # current reverse diffusion does NOT use DataLoader
    if for_reverse_diffusion:
        return splits
    # convert to DataLoader
    data_loaders = _get_loader(splits, args)
    return data_loaders


def _get_loader(splits, args):
    """
        Convert lists into DataLoader
    """
    # current reverse diffusion does NOT use DataLoader
    if args.mode == "test":
        return splits
    # convert to DataLoader
    loaders = {}
    for split, data in splits.items():
        # account for test-only datasets
        if len(data) == 0:
            loaders[split] = []
            continue
        # do not shuffle val/test
        shuffle = (split == "train")
        # set proper DataLoader object (PyG)
        if torch.cuda.is_available() and args.num_gpu > 1:
            loader = DataListLoader
        else:
            loader = DataLoader
        loaders[split] = loader(data,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=False,
                                pin_memory=True,
                                shuffle=shuffle)
    return loaders

# ------ DATASET -------


class BindingDataset(Dataset):
    """
        Protein-protein binding dataset
    """
    def __init__(self, args, data, pdb_ids=None, apply_transform=True):
        super(BindingDataset, self).__init__(
            transform = NoiseTransform(args) if apply_transform else None
        )
        self.args = args
        # select subset for given split
        if pdb_ids is not None:
            data = {k:v for k,v in data.items() if k in pdb_ids}
            self.pdb_ids = [k for k in data if k in pdb_ids]
        else:
            self.pdb_ids = list(data)
        # convert to PyTorch geometric objects upon GET not INIT
        if isinstance(data, list):
            self.data = data
        else:
            self.data = list(data.values())
            
        self.length = len(self.data)

    def len(self):
        return self.length

    def __delitem__(self, idx):
        """
            Easier deletion interface. MUST update length.
        """
        del self.data[idx]
        self.len = len(self.data)

    def get(self, idx):
        """
            Create graph object to keep original object intact,
            so we can modify pos, etc.
        """
        item = self.data[idx]["graph"]
        # >>> fix this later no need to copy tensors only references
        data = copy.deepcopy(item)
        return data

    def get_visualization_values(self, pdb_name=None, index=None):
        if index is not None:
            pass
        elif pdb_name is not None:
            index = self.pdb_ids.index(pdb_name)
        else:
            raise Exception("Either pdb_name or index should be given!")

        return self.data[index]["visualization_values"]

    def set_graph(self, idx, new_graph):
        self.data[idx]["graph"] = new_graph

class RandomizedConfidenceDataset(Dataset):
    """
        Protein-protein dataset of randomly perturbed ligand poses used to train a confidence model. (experimental)
    """
    def __init__(self, args, data, pdb_ids=None, apply_transform=False):
        super(RandomizedConfidenceDataset, self).__init__()
        self.args = args
        # select subset for given split
        if pdb_ids is not None:
            data = {k:v for k,v in data.items() if k in pdb_ids}
            self.pdb_ids = [k for k in data if k in pdb_ids]
        else:
            self.pdb_ids = list(data)
        # convert to PyTorch geometric objects upon GET not INIT
        self.data = list(data.values())
        self.length = len(self.data)

    def len(self):
        return self.length

    def __delitem__(self, idx):
        """
            Easier deletion interface. MUST update length.
        """
        del self.data[idx]
        self.len = len(self.data)

    def get(self, idx):
        """
            Create graph object to keep original object intact,
            so we can modify pos, etc.
        """
        item = self.data[idx]["graph"]
        # >>> fix this later no need to copy tensors only references
        data = copy.deepcopy(item)
        set_time(data, 0, 0, 0, 1)
        if np.random.rand() < 0.05:
            tr_s_max=5
            rot_s_max=0.2
        else:
            tr_s_max=2
            rot_s_max=0.1
        return self.randomize_position_and_compute_rmsd(data, tr_s_max=tr_s_max, rot_s_max=rot_s_max) # 2 and 0.1 yields almost 50% -> good. 2 and 0.2 yields 13%

    def set_graph(self, idx, new_graph):
        self.data[idx]["graph"] = new_graph

    def randomize_position_and_compute_rmsd(self, complex_graph, tr_s_max, rot_s_max):
        # randomize rotation
        original_pos = complex_graph["ligand"].pos
        center = torch.mean(original_pos, dim=0, keepdim=True)
        # one way to generate random rotation matrix
        # random_rotation = torch.from_numpy(R.random().as_matrix())
        
        # Another way
        rot_update = sample_vec(eps=rot_s_max)# * rot_s_max
        rot_update = torch.from_numpy(rot_update).float()
        random_rotation = axis_angle_to_matrix(rot_update.squeeze())
        
        # yet another way
        #x = np.random.randn(3)
        #x /= np.linalg.norm(x)
        #x *= rot_s_max
        #random_rotation = R.from_euler('zyx', x, degrees=True)
        #random_rotation = torch.from_numpy(random_rotation.as_matrix())
        pos = (original_pos - center) @ random_rotation.T.float()

        # random translation
        tr_update = torch.normal(0, tr_s_max, size=(1, 3))
        pos = pos + tr_update + center
        complex_graph["ligand"].pos = pos

        # compute rmsd
        rmsd = compute_rmsd(original_pos, pos)
        return complex_graph, rmsd
