import pandas as pd
import numpy as np
import random

import dill
import os
import subprocess

from tqdm import tqdm
from scipy.spatial.transform import Rotation

DATA_PATH = "/data/rsg/nlp/sdobers/data/DIPS/pairs_pruned/"
HDOCK_DATA_PATH = "/data/rsg/nlp/sdobers/data/DIPS/hdock"
HDOCK_EXECUTABLE = "/data/rsg/chemistry/rmwu/bin/hdock"
HDOCK_OUTPUT_INTERPETER= "/data/rsg/chemistry/rmwu/bin/createpl"


def main():
    df = pd.read_csv("/data/rsg/nlp/sdobers/data/DIPS/data_file.csv")
    df_test = df[df["split"] == "test"]

    paths = df_test.path.values
    random.shuffle(paths)

    print("Convert test dill files to pdb.")
    for path in tqdm(paths):
        dill_to_pdb(path)

    print("Run HDock")
    with open(f"{HDOCK_DATA_PATH}/log.txt", "a") as file:
        for path in tqdm(paths):
            hdock_path = f"{HDOCK_DATA_PATH}/{path}".replace(".dill", '')
            os.chdir(hdock_path)
            if os.path.exists("output.txt"):
                continue
            cmd = f"{HDOCK_EXECUTABLE} receptor.pdb ligand.pdb -out output.txt"
            file.write(f"{cmd}\n")
            subprocess.call(cmd, shell=True, stdout=file)

    print("Parse output.txt to pdb")
    for path in tqdm(paths):
        hdock_path = f"{HDOCK_DATA_PATH}/{path}".replace(".dill", '')
        os.chdir(hdock_path)
        if os.path.exists("output.pdb"):
            continue
        if not os.path.exists("output.txt"):
            print(f"No output for {path}")
            continue
        cmd = f"{HDOCK_OUTPUT_INTERPETER} output.txt output.pdb"
        subprocess.call(cmd, shell=True)


def parse_dill(fp):
    with open(f"{DATA_PATH}/{fp}", "rb") as f:
        data = dill.load(f)
    p1, p2 = data[1], data[2]
    return p1, p2


def dill_to_pdb(path):
    p1, p2 = parse_dill(path)
    path = path.replace('.dill', '')
    out_path = f'{HDOCK_DATA_PATH}/{path}'
    os.makedirs(out_path, exist_ok=True)

    # set p1 as receptor
    if len(p1.index) < len(p2.index):
        p1, p2 = p2, p1

    p1_lines = to_pdb_lines(p1)
    p2_lines = to_pdb_lines(p2)

    fp_p1 = f'{out_path}/receptor.pdb'
    fp_p2 = f'{out_path}/ligand_gt.pdb'

    with open(fp_p1, 'w') as f:
        f.writelines(p1_lines)
    with open(fp_p2, 'w') as f:
        f.writelines(p2_lines)

    # randomly rotate ligand
    p2_xyz, rot, t = rotate_translate(p2, seed=hash(path) % 2**16)
    p2.x, p2.y, p2.z = p2_xyz[0], p2_xyz[1], p2_xyz[2]

    p2_lines = to_pdb_lines(p2)
    fp_p2 = f'{out_path}/ligand.pdb'
    with open(fp_p2, 'w') as f:
        f.writelines(p2_lines)


def to_pdb_lines(df):
    df = df.to_dict(orient='records')
    lines = []
    for i, item in enumerate(df):
        item = {k: v.strip() if type(v) is str else v for k, v in item.items()}
        line = f'ATOM  {i + 1:>5} {item["atom_name"]:>4} '
        line = line + f'{item["resname"]} {item["chain"]}{item["residue"]:>4}    '
        line = line + f'{item["x"]:>8.3f}{item["y"]:>8.3f}{item["z"]:>8.3f}'
        line = line + '  1.00  0.00          '
        line = line + f'{item["element"]:>2} 0\n'
        lines.append(line)
    return lines


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
    xyz = np.stack([df['x'], df['y'], df['z']])
    xyz = xyz - np.mean(xyz, axis=1, keepdims=True)
    # apply rotation
    xyz = rot @ xyz + t
    return xyz, rot, t


if __name__ == "__main__":
    main()