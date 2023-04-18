"""
This file is a nice quality-of-life script.
Given that you formatted your checkpoints in the way my code does,
this script will parse the filenames and delete all but
the last five checkpoints.

This allows you to keep multiple checkpoints just in case,
to verify/debug predictions over time, but still delete them easily
once you're done.

You need to add your checkpoint directory to "roots"
(and if you run this you need to delete mine because
you don't have permissions lol.)
"""

import os
import glob

roots = ["/data/scratch/rmwu/tmp-runs",
         "/data/scratch/rmwu/tmp-runs/ml-energy"]

for root in roots:
    for exps in os.listdir(root):
        exps = os.path.join(root, exps)
        if not os.path.isdir(exps):
            continue
        for run in os.listdir(exps):
            run = os.path.join(exps, run)

            for fold in range(5):
                fold_dir = f"{run}/fold_{fold}"
                if not os.path.exists(fold_dir):
                    break

                try:
                    models = sorted(glob.glob(f"{fold_dir}/*.pth"),
                                    key=lambda s:(int(s.split("/")[-1].split("_")[3]),
                                                  int(s.split("/")[-1].split("_")[2])))
                except:
                    print(glob.glob(f"{fold_dir}/*.pth"))
                    continue
                if len(models) == 0:
                    print(f"no models found at {fold_dir}: {os.listdir(fold_dir)}")
                    break

                checkpoint = models[-1]
                rest = models[:-5]

                for fp in rest:
                    cmd = f"rm {fp}"
                    os.system(cmd)
                    print("cleaned", fold_dir)

