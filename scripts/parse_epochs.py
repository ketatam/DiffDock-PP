"""
This script is a lazy way to parse the best epoch for each fold
from a given experiment folder.

Usage:
    python parse_epochs.py dips_esm

You need to modify "root" to point to your checkpoint directory
"""

import sys
import glob
from collections import defaultdict


def main():
    root = "/data/scratch/rmwu/tmp-runs/glue"
    base = sys.argv[1]
    path = f"{root}/{base}/fold_*/*best*pth"
    save = glob.glob(path)
    best = defaultdict(int)
    for fp in save:
        fold, epoch = parse_path(fp)
        best[fold] = max(epoch, best[fold])
    # print stats
    print("best epochs",
          " ".join([str(epoch) for _, epoch in sorted(best.items())]))


def parse_path(fp):
    fp = fp.split("/")
    fold, path = fp[-2], fp[-1]
    fold = int(fold[5:])
    epoch = int(path.split("_")[3])
    return fold, epoch


if __name__ == "__main__":
    main()

