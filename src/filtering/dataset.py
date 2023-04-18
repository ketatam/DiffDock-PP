import os

import torch
import copy

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader, DataListLoader
from tqdm import tqdm

import random

from geom_utils import set_time



def deserialize_batch(batch_directory, duplicate_rec_and_esm=True, use_complex_rmsd=False, use_interface_rmsd=False):
    first_iteration = torch.load(f"{batch_directory}/first_iteration.pkl")
    ligand_positions = torch.load(f"{batch_directory}/ligand_positions.pkl")
    if use_complex_rmsd:
        rmsd_multiple_iterations = torch.load(f"{batch_directory}/crmsds.pkl")
    elif use_interface_rmsd:
        rmsd_multiple_iterations = torch.load(f"{batch_directory}/irmsds.pkl")
    else:
        rmsd_multiple_iterations = torch.load(f"{batch_directory}/rmsds.pkl")
    if not duplicate_rec_and_esm:
        ligand_positions = list(map(list, zip(*ligand_positions))) # transpose list
        rmsd_multiple_iterations = list(map(list, zip(*rmsd_multiple_iterations))) # transpose list
        return first_iteration, ligand_positions, rmsd_multiple_iterations

    next_iters_ligand_positions = ligand_positions[1:]
    next_iters = [copy.deepcopy(first_iteration) for i in range(len(next_iters_ligand_positions))]
    for i, iteration in enumerate(next_iters):
        for j, graph in enumerate(iteration):
            next_iters[i][j]["ligand"].pos = next_iters_ligand_positions[i][j]
    return [first_iteration] + next_iters, rmsd_multiple_iterations


def deserialize_batch_into_single_list(batch_directory, use_complex_rmsd=False, use_interface_rmsd=False):
    complexes, ligand_positions, rmsds = deserialize_batch(batch_directory, duplicate_rec_and_esm=False, use_complex_rmsd=use_complex_rmsd, use_interface_rmsd=use_interface_rmsd)

    return complexes, sum(ligand_positions, []), sum(rmsds, [])  # Concat samples

class CondfidenceBindingDataset(Dataset):
    """
        Protein-protein binding dataset
    """

    def __init__(self, split, samples_directory, batch=None, use_complex_rmsd=False, use_interface_rmsd=False):
        super(CondfidenceBindingDataset, self).__init__()

        self.complexes = []
        self.ligand_positions = []
        self.rmsds = []

        split_directory = f"{samples_directory}/{split}"

        if batch is None:
            existing_batch_dirs = [batch_dir for batch_dir in os.listdir(split_directory) if "batch" in batch_dir]

            for batch_dir in tqdm(existing_batch_dirs):
                batch_directory = f"{split_directory}/{batch_dir}"
                try:
                    complexes, ligand_positions, rmsds = deserialize_batch_into_single_list(batch_directory, use_complex_rmsd=use_complex_rmsd, use_interface_rmsd=use_interface_rmsd)
                except RuntimeError as e:
                    print(f"Skipping {batch_directory} because of error: {str(e)}")
                    continue

                assert (len(ligand_positions)/len(complexes)).is_integer()

                self.complexes += complexes
                self.ligand_positions += ligand_positions
                self.rmsds += rmsds

            self.n_samples_per_complex = len(self.ligand_positions) / len(self.complexes)
            assert self.n_samples_per_complex.is_integer()
            self.n_samples_per_complex = int(self.n_samples_per_complex)

            self.length = len(self.ligand_positions)

        else:
            batch_directory = f"{split_directory}/batch-{batch}"
            self.samples, self.rmsds = deserialize_batch_into_single_list(batch_directory)
            self.length = len(self.samples)

    def len(self):
        return self.length

    def __delitem__(self, idx):
        """
            Easier deletion interface. MUST update length.
        """
        del self.samples[idx], self.rmsds[idx]
        self.length = len(self.samples)

    def get(self, idx):
        """
            Create graph object to keep original object intact,
            so we can modify pos, etc.
        """
        complex = self.complexes[idx//self.n_samples_per_complex]
        ligand_position = self.ligand_positions[idx]
        rmsd = self.rmsds[idx]

        # >>> fix this later no need to copy tensors only references
        sample = copy.deepcopy(complex)
        set_time(sample, 0, 0, 0, 1)
        assert sample["ligand"].pos.shape == ligand_position.shape
        sample["ligand"].pos = ligand_position
        sample.rmsd = rmsd

        return sample#, rmsd


def is_pkl_empty(batch_path):
    file_name = f"{batch_path}/first_iteration.pkl"
    return not os.path.exists(file_name) or os.path.getsize(file_name) < 1000000

class CondfidenceBindingDiskLoader(Dataset):
    """
        Protein-protein binding dataset
    """

    def __init__(self, split, samples_directory, shuffle=True, drop_last_batch_when_shuffling=True):
        super(CondfidenceBindingDiskLoader, self).__init__()

        self.one_batch_size = 128 * 4  # 128 - number of samples per chunk, 4 - number of predictions per sample
        self.split_directory = f"{samples_directory}/{split}"
        self.shuffle = shuffle

        self.existing_batch_dirs = [batch_dir for batch_dir in os.listdir(self.split_directory)
                                    if "batch" in batch_dir
                                    and not is_pkl_empty(f"{self.split_directory}/{batch_dir}")]

        # Remove the last batch so that all batches have the same size
        last_batch_index = max([int(dir_name.split('-')[-1]) for dir_name in self.existing_batch_dirs])
        last_batch = f"batch-{last_batch_index}"
        self.existing_batch_dirs.remove(last_batch)

        # Compute size
        self.length = len(self.existing_batch_dirs) * self.one_batch_size

        if self.shuffle:
            random.shuffle(self.existing_batch_dirs)
        
        # If you are not shuffling, you can still use the last batch, just put it at the end
        # Or if you explicitly specify it
        if not shuffle or not drop_last_batch_when_shuffling:
            self.existing_batch_dirs.append(last_batch)
            # Compute new length
            batch = deserialize_batch_into_single_list(f"{self.split_directory}/{last_batch}")
            self.length += len(batch)
            del batch

        self.current_batch_index = 0
        self.load_new_batch(self.current_batch_index)

    def load_new_batch(self, batch_index):
        self.current_batch_index = batch_index

        batch_directory_name = self.existing_batch_dirs[batch_index]
        batch_directory_path = f"{self.split_directory}/{batch_directory_name}"
        self.samples, self.rmsds = deserialize_batch_into_single_list(batch_directory_path)

        # Shuffle the rmsds and samples
        if self.shuffle:
            random_permutation = list(range(len(self.samples)))
            random.shuffle(random_permutation)
            self.samples = [self.samples[i] for i in random_permutation]
            self.rmsds = [self.rmsds[i] for i in random_permutation]


    def get_batch_index(self, idx):
        return idx // self.one_batch_size

    def get_index_inside_the_batch(self, idx):
        return idx % self.one_batch_size

    def get(self, idx):
        """
            Create graph object to keep original object intact,
            so we can modify pos, etc.
        """
        # Check if we have to load a new batch
        batch_index = self.get_batch_index(idx)
        if batch_index != self.current_batch_index:
            self.load_new_batch(batch_index)

        index_inside_of_current_batch = self.get_index_inside_the_batch(idx)

        sample = self.samples[index_inside_of_current_batch]
        rmsd = self.rmsds[index_inside_of_current_batch]

        # >>> fix this later no need to copy tensors only references
        sample = copy.deepcopy(sample)

        return sample, rmsd

    def len(self):
        return self.length


def get_confidence_loader(split, args, shuffle, batch=None,samples=None):

    if samples==None:
        samples = CondfidenceBindingDataset(split, args.samples_directory, batch=batch, use_complex_rmsd=args.use_complex_rmsd, use_interface_rmsd=args.use_interface_rmsd)

    # set proper DataLoader object (PyG)
    if torch.cuda.is_available() and args.num_gpu > 1:
        loader = DataListLoader
    else:
        loader = DataLoader

    return loader(samples, batch_size=args.batch_size,
                  num_workers=args.num_workers,
                  drop_last=False,
                  pin_memory=True,
                  shuffle=shuffle)



def get_confidence_disc_loader(split, args, shuffle):
    samples = CondfidenceBindingDiskLoader(split, args.samples_directory, shuffle=shuffle)

    # set proper DataLoader object (PyG)
    if torch.cuda.is_available() and args.num_gpu > 1:
        loader = DataListLoader
    else:
        loader = DataLoader

    return loader(samples, batch_size=args.batch_size,
                  num_workers=args.num_workers,
                  drop_last=False,
                  pin_memory=True,
                  shuffle=False)