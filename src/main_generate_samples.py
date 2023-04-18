import os
import shutil
import sys
import yaml
import random
import resource
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


from args import parse_args
from data import load_data, get_data
from model import load_model, to_cuda
from utils import printt
from train import evaluate_pose
from sample import sample
from pathlib import Path
from tqdm import tqdm

from data.utils import DIPSLoader

DATA_CACHE_VERSION = "v1"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(args=None):
    # he he he
    with open("data/goodluck.txt") as f:
        for line in f:
            print(line, end="")
    if args is None:
        args = parse_args()

    torch.cuda.set_device(args.gpu)
    torch.hub.set_dir(args.torchhub_path)

    # needs to be set if DataLoader does heavy lifting
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # needs to be set if sharing resources
    if args.num_workers >= 1:
        torch.multiprocessing.set_sharing_strategy("file_system")

    fold = 0

    #### set up fold experiment
    # set_seed(args.seed)

    # load data params
    data_params = load_data(args, split="test", batch=0).data_params
    print("Loaded first batch")

    # get model and load checkpoint, if relevant
    model = load_model(args, data_params, fold, load_best=True) # load last_model to continue training
    model = to_cuda(model, args)
    printt("finished loading model")

    # Log number of parameters
    numel = sum([p.numel() for p in model.parameters()])
    printt('Model with', numel, 'parameters')
    # Get cache directory

    for split in ("test", "val", "train"):
        print(f"Inference for {split} split!")

        n_batches = DIPSLoader.get_n_batches()[split]
        batch_indexes = list(range(n_batches))
        # get_random_indexes_ignore_seed(batch_indexes, args.seed)
        random.shuffle(batch_indexes)

        for batch_index in tqdm(batch_indexes):
            # Get current directory
            directory = f"{args.samples_directory}/{split}/batch-{batch_index}"

            # If directory exists and is not empty, continue
            if os.path.exists(directory):
                print(f"batch {batch_index} is already generated. Continue!")
                continue

            # Load batch
            data = load_data(args, split=split, batch=batch_index, verbose=False)
            batch = get_data(data, fold, args, for_reverse_diffusion=True)[split]
            if len(batch) == 0:
                print("Zero batch!")
                continue

            # If not, create directory and continue
            os.makedirs(directory, exist_ok=True)

            # run reverse diffusion process
            samples_multiple_iterations = []
            rmsd_multiple_iterations = []
            try:
                for i in range(args.generate_n_predictions):
                    iteration = sample(batch, model, args)
                    rmsd = evaluate_pose(batch, iteration)["rmsd"]

                    samples_multiple_iterations.append(iteration)
                    rmsd_multiple_iterations.append(rmsd)

            except Exception as e:
                printt("RuntimeError. ")
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                shutil.rmtree(directory)
                continue

            # For some reason, there were some samples without complex_t
            if "complex_t" not in samples_multiple_iterations[0][0]:
                print(f"No complex_t! Batch: {batch_index}.")

            serialize(samples_multiple_iterations, rmsd_multiple_iterations, directory=directory)


def get_random_indexes_ignore_seed(batches, current_seed):
    batch_indexes = list(range(len(batches)))

    # Randomize batch indexes
    random.seed(None)
    random.shuffle(batch_indexes)
    random.seed(current_seed)
    return batch_indexes


def split_list_into_batches(data, batch_size):
    number_of_batches = len(data) // batch_size + int(len(data) % batch_size > 0)
    batches = [data[i: i * batch_size] for i in range(number_of_batches)]
    return batches


def serialize(samples_multiple_iterations, rmsd_multiple_iterations, directory="."):
    torch.save(samples_multiple_iterations[0], f"{directory}/first_iteration.pkl")
    ligand_positions = [[graph["ligand"].pos for graph in graphs] for graphs in samples_multiple_iterations]
    torch.save(ligand_positions, f"{directory}/ligand_positions.pkl")
    torch.save(rmsd_multiple_iterations, f"{directory}/rmsds.pkl")


if __name__ == "__main__":
    main()

