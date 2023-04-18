import math
import pandas as pd
import os

from biopandas.pdb import PandasPdb
import numpy as np

import torch

from matplotlib import pyplot as plt
import scipy.spatial as spa
from tqdm import tqdm


# Input: expects 3xN matrix of points
# Returns such R, t so that rmsd(R @ A + t, B) is min
# Uses Kabsch algorithm (https://en.wikipedia.org/wiki/Kabsch_algorithm)
# R = 3x3 rotation matrix
# t = 3x1 column vector
# This already takes residue identity into account.
def rigid_transform_Kabsch_3D(A, B):
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")


    # find mean column wise: 3 x 1
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = np.diag([1.,1.,-1.])
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(np.linalg.det(R) - 1) < 1e-5

    t = -R @ centroid_A + centroid_B
    return R, t


def compute_rmsd(pred, true):
    return np.sqrt(np.mean(np.sum((pred - true) ** 2, axis=1)))

def get_rmsd_summary(rmsds):
    rmsds_np = np.array(rmsds)
    return {
        'mean': np.mean(rmsds_np),
        'median': np.median(rmsds_np),
        'std': np.std(rmsds_np),
        'lt1': 100 * (rmsds_np < 1.0).sum() / len(rmsds_np),
        'lt2': 100 * (rmsds_np < 2.0).sum() / len(rmsds_np),
        'lt5': 100 * (rmsds_np < 5.0).sum() / len(rmsds_np),
        'lt10': 100 * (rmsds_np < 10.0).sum() / len(rmsds_np)
    }


class RMSDComputer():
    def __init__(self):
        self.complex_rmsd_list = []
        self.ligand_rmsd_list = []
        self.interface_rmsd_list = []
    
    def update_all_rmsd(self, ligand_coors_pred, ligand_coors_true, receptor_coors):
        complex_rmsd = self.update_complex_rmsd(ligand_coors_pred, ligand_coors_true, receptor_coors)
        ligand_rmsd = self.update_ligand_rmsd(ligand_coors_pred, ligand_coors_true)
        interface_rmsd = self.update_interface_rmsd(ligand_coors_pred, ligand_coors_true, receptor_coors)
        return complex_rmsd, ligand_rmsd, interface_rmsd

    def update_complex_rmsd(self, ligand_coors_pred, ligand_coors_true, receptor_coors):
        complex_coors_pred = np.concatenate((ligand_coors_pred, receptor_coors), axis=0)
        complex_coors_true = np.concatenate((ligand_coors_true, receptor_coors), axis=0)

        R,t = rigid_transform_Kabsch_3D(complex_coors_pred.T, complex_coors_true.T)
        complex_coors_pred_aligned = (R @ complex_coors_pred.T + t).T

        complex_rmsd = compute_rmsd(complex_coors_pred_aligned, complex_coors_true)
        self.complex_rmsd_list.append(complex_rmsd)

        return complex_rmsd

    def update_ligand_rmsd(self, ligand_coors_pred, ligand_coors_true):
        ligand_rmsd = compute_rmsd(ligand_coors_pred, ligand_coors_true)
        self.ligand_rmsd_list.append(ligand_rmsd)
        
        return ligand_rmsd


    def update_interface_rmsd(self, ligand_coors_pred, ligand_coors_true, receptor_coors):
        ligand_receptor_distance = spa.distance.cdist(ligand_coors_true, receptor_coors)
        positive_tuple = np.where(ligand_receptor_distance < 8.)
        
        active_ligand = positive_tuple[0]
        active_receptor = positive_tuple[1]
        
        ligand_coors_pred = ligand_coors_pred[active_ligand, :]
        ligand_coors_true = ligand_coors_true[active_ligand, :]
        receptor_coors = receptor_coors[active_receptor, :]

        complex_coors_pred = np.concatenate((ligand_coors_pred, receptor_coors), axis=0)
        complex_coors_true = np.concatenate((ligand_coors_true, receptor_coors), axis=0)

        R,t = rigid_transform_Kabsch_3D(complex_coors_pred.T, complex_coors_true.T)
        complex_coors_pred_aligned = (R @ complex_coors_pred.T + t).T

        interface_rmsd = compute_rmsd(complex_coors_pred_aligned, complex_coors_true)
        self.interface_rmsd_list.append(interface_rmsd)

        return interface_rmsd

    
    def summarize(self, verbose=True):
        ligand_rmsd_summarized = get_rmsd_summary(self.ligand_rmsd_list) if self.ligand_rmsd_list else None
        complex_rmsd_summarized = get_rmsd_summary(self.complex_rmsd_list)
        interface_rmsd_summarized = get_rmsd_summary(self.interface_rmsd_list)

        if verbose:
            print(f'ligand_rmsd_summarized: {ligand_rmsd_summarized}')
            print(f'complex_rmsd_summarized: {complex_rmsd_summarized}')
            print(f'interface_rmsd_summarized: {interface_rmsd_summarized}')

        return ligand_rmsd_summarized, complex_rmsd_summarized, interface_rmsd_summarized

    def pretty_print(self, lrmsd, crmsd, irmsd):
        num_test_files = len(self.complex_rmsd_list)
        print(f'Number of samples:\t\t{num_test_files}')
        print()
        print(f"Ligand RMSD median/mean:\t{lrmsd['median']:.3}/{lrmsd['mean']:.3} ± {lrmsd['std']:.3}")
        print(f"Complex RMSD median/mean:\t{crmsd['median']:.3}/{crmsd['mean']:.3} ± {crmsd['std']:.3}")
        print(f"Interface RMSD median/mean:\t{irmsd['median']:.3}/{irmsd['mean']:.3} ± {irmsd['std']:.3}")
        print()
        print(f"Ligand lt1/lt2/lt5/lt10:\t{lrmsd['lt1']:.3}%/{lrmsd['lt2']:.3}%/{lrmsd['lt5']:.3}%/{lrmsd['lt10']:.3}%")
        print(f"Complex lt1/lt2/lt5/lt10:\t{crmsd['lt1']:.3}%/{crmsd['lt2']:.3}%/{crmsd['lt5']:.3}%/{crmsd['lt10']:.3}%")
        print(f"Interface lt1/lt2/lt5/lt10:\t{irmsd['lt1']:.3}%/{irmsd['lt2']:.3}%/{irmsd['lt5']:.3}%/{irmsd['lt10']:.3}%")


def evaluate_all_rmsds(data_list, samples_list):
    """
        Evaluate sampled pose vs. ground truth
    """
    meter = RMSDComputer()
    assert len(data_list) == len(samples_list)
    for true_graph, pred_graph in zip(data_list, samples_list):
        true_xyz = true_graph["ligand"].pos
        rec_xyz = true_graph["receptor"].pos

        pred_xyz = pred_graph["ligand"].pos

        if pred_graph["mirrored"]:
            true_xyz, rec_xyz = rec_xyz, true_xyz
            assert rec_xyz.shape == pred_graph["receptor"].pos.shape

        assert true_xyz.shape == pred_xyz.shape

        if not pred_graph["mirrored"]:
            assert (rec_xyz == pred_graph["receptor"].pos).all()

        complex_rmsd, ligand_rmsd, interface_rmsd = meter.update_all_rmsd(pred_xyz.numpy(), true_xyz.numpy(), rec_xyz.numpy())

    return meter
