import pandas as pd
import numpy as np
import random

import os
import subprocess
import shutil

from tqdm import tqdm
from biopandas.pdb import PandasPdb

from scipy.spatial.transform import Rotation

DATA_PATH = "/data/rsg/nlp/sdobers/data/db5/structures"
DATASET = "db5_unbound"
EQUIDOCK_PUBLIC = "/data/rsg/nlp/sdobers/ruslan/equidock_public"
GT_DIRECTORY =  f"{EQUIDOCK_PUBLIC}/test_sets_pdb/{DATASET}_test_random_transformed/complexes"
INPUT_DIRECTORY = f"{EQUIDOCK_PUBLIC}/test_sets_pdb/{DATASET}_test_random_transformed/random_transformed"

def rotate_translate(df, seed):
    # get random rotation
    rot = Rotation.random(random_state=seed)
    rot = rot.as_matrix().squeeze()
    # get random translation
    t = np.random.randn(3, 1)
    t = t / np.sqrt(np.sum(t * t))
    length = np.random.uniform(low=0, high=20)
    t = (t * length)
    # subtract mean
    xyz = np.stack([df['x_coord'], df['y_coord'], df['z_coord']])
    xyz = xyz - np.mean(xyz, axis=1, keepdims=True)
    # apply rotation
    xyz = rot @ xyz + t
    return xyz, rot, t

def rotate_translate_pdb_file(input_file, output_file, seed):
    ppdb_model = PandasPdb().read_pdb(input_file)
    atoms = ppdb_model.df['ATOM']
    atoms['x_coord'], atoms['y_coord'], atoms['z_coord'] = rotate_translate(atoms, seed)[0]
    
    assert ppdb_model.df["ATOM"]['x_coord'].values[0] == atoms['x_coord'].values[0]
    
    ppdb_model.to_pdb(path=output_file, records=None)


# Read file list
df = pd.read_csv("/data/rsg/nlp/sdobers/data/db5/splits.csv")
df_test = df[df["split"] == "test"]
paths = df_test.path.values

print("Prepare data")
os.makedirs(GT_DIRECTORY, exist_ok=True)
os.makedirs(INPUT_DIRECTORY, exist_ok=True)

for path in tqdm(paths):
    gt_receptor = f"{GT_DIRECTORY}/{path.upper()}_r_u_COMPLEX.pdb"
    gt_ligand = f"{GT_DIRECTORY}/{path.upper()}_l_u_COMPLEX.pdb"

    input_receptor = f"{INPUT_DIRECTORY}/{path.upper()}_r_u.pdb"
    input_ligand = f"{INPUT_DIRECTORY}/{path.upper()}_l_u.pdb"

    shutil.copyfile(f"{DATA_PATH}/{path}_r_u.pdb", gt_receptor)
    shutil.copyfile(f"{DATA_PATH}/{path}_l_u.pdb", gt_ligand)
    
    shutil.copyfile(f"{DATA_PATH}/{path}_r_u.pdb", input_receptor)

    rotate_translate_pdb_file(f"{DATA_PATH}/{path}_l_u.pdb", input_ligand,  seed=hash(path) % 2**16)

