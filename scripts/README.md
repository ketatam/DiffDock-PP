# DiffDock - PP (Protein-Protein) Post-Processing Script

## Post Processing

The outputted files of DiffDock - PP contain alpha carbon traces (or backbones) with no other residue atoms. Since the diffusion model treats both bodies as rigid structures, we may use the Kabsch-Umeyama algorithm for applying the transformation from starting structure to docked structure. This algorithm is useful for rotation, translation, and scaling transformations. For the purposes of our discussion we shall henceforth refer to the "starting" structure as the "moving" structure, and "docked" as "stationary". It is also important to note that even the larger protein that the smaller protein is being docked into has it's coordinates shifted and therefore will need its original coordinates shifted according to a new alpha carbon backbone.

To implement this algorithm we wrote the utility program: RigidAlphaC_Alignment.py

### Installing Dependencies: Numpy, MdTraj, argparse

       pip install numpy
       pip install mdtraj
       pip install argparse


### Running RigidAlphaC_Alignment.py

       python3 RigidAlphaC_Alignment.py -m moving.pdb -s stationary.pdb -o align_output.pdb


### Arguments

       parser.add_argument('-m', '--moving', help='The PDB file you wish to have aligned, usually the original file input into DiffDock-PP')
       parser.add_argument('-s', '--stationary', help='The PDB file you wish to use as reference for alignment, usually the output file input into DiffDock-PP')
       parser.add_argument('-o', '--output', help='Name for output PDB file containing aligned coordinates.')

