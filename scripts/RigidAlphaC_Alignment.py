##################
#Author: Jessica Freeze
#Last Major Update: Oct 16, 2023
#Goal:The outputted files of DiffDock - PP contain alpha carbon traces (or backbones) with no other residue atoms. This program adds back the residues to the alpha carbon trace in the docked position. 
#     Note, no additional minimization is done here.
##################

import numpy as np
import mdtraj as md
import argparse

#This function courtesy of this tutorial: https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/
def kabsch_umeyama(A, B):
    assert A.shape == B.shape
    n, m = A.shape

    #stationary mean
    EA = np.mean(A, axis=0)
    #moving mean
    EB = np.mean(B, axis=0)
    #Normalization transformation
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    #Covariance matrix computation
    H = ((A - EA).T @ (B - EB)) / n
    #Singular value decomposition
    U, D, VT = np.linalg.svd(H)
    #Handling relfections
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])
    #Calculating the rotation matrix
    R = U @ S @ VT
    #Calculating the scaling factor
    c = VarA / np.trace(np.diag(D) @ S)
    #Calculating the translation vector
    t = EA - c * R @ EB

    return R, c, t

def load_and_extract_alpha(fname):
    # Load the PDB file structure
    all_coords = md.load(fname)

    # Find the positions of the alpha carbons from the structure
    alpha_carbon_pos = all_coords.topology.select("name CA")

    return all_coords,alpha_carbon_pos


#########Main Program###########
def main():
    #Set up parsing options
    parser = argparse.ArgumentParser(
                        prog='align_AlphaC_PDB',
                        description='This program takes alpha carbon backbones from rigid transformations from a full protein starting structure and transforms the original protein structure to the new alpha backbone coordinates.')
    parser.add_argument('-m', '--moving', help='The PDB file you wish to have aligned, usually the original file input into DiffDock-PP')
    parser.add_argument('-s', '--stationary', help='The PDB file you wish to use as reference for alignment, usually the output file input into DiffDock-PP')
    parser.add_argument('-o', '--output', help='Name for output PDB file containing aligned coordinates.')
    args = parser.parse_args()

    # Load the moving PDB file
    moving_all,moving_alpha_pos = load_and_extract_alpha(args.moving)
    # Load the stationary PDB file
    stationary_all,stationary_alpha_pos = load_and_extract_alpha(args.stationary)


    # Extract the coordinates of the alpha carbons from the moving structure
    moving_alpha = moving_all.xyz[0][moving_alpha_pos]

    # Extract the coordinates of the alpha carbons from the stationary structure
    stationary_alpha = stationary_all.xyz[0][stationary_alpha_pos]

    #Determine the transformation
    R, scaling, t = kabsch_umeyama(stationary_alpha, moving_alpha)

    #Perform the transformation
    moving_all.xyz = t+scaling*np.dot(moving_all.xyz, R.T)

    #Save the transformed pdb coordinates
    moving_all.save(args.output)


if __name__ == "__main__":
    main()