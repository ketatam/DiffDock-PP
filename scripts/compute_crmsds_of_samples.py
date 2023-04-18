import yaml
import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import random

sys.path.append("/data/rsg/nlp/sdobers/ruslan/diffdock-protein/src")

from data import load_data, get_data
from data.utils import DIPSLoader
from evaluation.compute_rmsd import evaluate_all_rmsds
from filtering.dataset import deserialize_batch

class Dict2Class:
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])
            
with open("/data/rsg/nlp/sdobers/ruslan/diffdock-protein/config/dips_esm_batchwise_loading.yaml", "r") as f:
    args_loaded = yaml.safe_load(f)

args_dict = {} 
for small_dict in args_loaded.values():
    args_dict = {**args_dict, **small_dict}
    
args_dict = {
    **args_dict, 
    "samples_directory": "/data/rsg/nlp/sdobers/DIPS/confidence_full_20_poses",
    "debug": False,
    "recache": False,
    "use_randomized_confidence_data": False,
    "mode": ""
}

args = Dict2Class(args_dict)

fold = 0

# load data params
data_params = load_data(args, split="test", batch=0).data_params
print("Loaded first batch")

for split in ("train", "test", "val"):
    print(f"Inference for {split} split!")

    n_batches = DIPSLoader.get_n_batches()[split]
    batch_indexes = list(range(n_batches))
    random.shuffle(batch_indexes)

    for batch_index in tqdm(batch_indexes):
        # Get current directory
        directory = f"{args.samples_directory}/{split}/batch-{batch_index}"
        
        if os.path.exists(f"{directory}/crmsds.pkl"):
            continue
        
        data = load_data(args, split=split, batch=batch_index, verbose=False)
        gt = get_data(data, fold, args, for_reverse_diffusion=True)[split]
        
        iterations, _ = deserialize_batch(directory)
        
        gt_names = [graph.name for graph in gt]
        iterations_names = [graph.name for graph in iterations[0]]
        assert all(np.array(gt_names) == np.array(iterations_names))
        
        rmsds = [evaluate_all_rmsds(gt, iteration).complex_rmsd_list for iteration in iterations]
        
        torch.save(rmsds, f"{directory}/crmsds.pkl")