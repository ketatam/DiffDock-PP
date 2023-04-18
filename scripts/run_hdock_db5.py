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
HDOCK_DATA_PATH = "/data/rsg/nlp/sdobers/data/db5/hdock"
HDOCK_EXECUTABLE = "/data/rsg/chemistry/rmwu/bin/hdock"
HDOCK_OUTPUT_INTERPETER= "/data/rsg/chemistry/rmwu/bin/createpl"

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
random.shuffle(paths)


# print("Prepare data")
# for suffix in ("unbound", "bound"):
#     for path in tqdm(paths):
#         target_dir = f"{DATA_PATH}/{suffix}/{path.lower()}"
#         os.makedirs(target_dir, exist_ok=True)
        
#         shutil.copyfile(f"{DATA_PATH}/{path}_r_{suffix[0]}.pdb", f"{target_dir}/receptor.pdb")
#         shutil.copyfile(f"{DATA_PATH}/{path}_l_{suffix[0]}.pdb", f"{target_dir}/ligand_gt.pdb")
        
#         rotate_translate_pdb_file(f"{DATA_PATH}/{path}_l_{suffix[0]}.pdb", f"{target_dir}/ligand.pdb",  seed=hash(path) % 2**16)
        
print("Run HDock")
with open(f"{HDOCK_DATA_PATH}/log.txt", "a") as file:
    for suffix in ("unbound", "bound"):
        for path in tqdm(paths):
            hdock_path = f"{DATA_PATH}/{suffix}/{path.lower()}"
            os.chdir(hdock_path)
            if os.path.exists("output.txt"):
                continue
            cmd = f"{HDOCK_EXECUTABLE} receptor.pdb ligand.pdb -out output.txt"
            file.write(f"{cmd}\n")
            subprocess.call(cmd, shell=True, stdout=file)

print("Parse output.txt to pdb")
for suffix in ("unbound", "bound"):
    for path in tqdm(paths):
        hdock_path = f"{DATA_PATH}/{suffix}/{path.lower()}"
        os.chdir(hdock_path)
        if os.path.exists("output.pdb"):
            continue
        if not os.path.exists("output.txt"):
            print(f"No output for {path}")
            continue
        cmd = f"{HDOCK_OUTPUT_INTERPETER} output.txt output.pdb"
        subprocess.call(cmd, shell=True)