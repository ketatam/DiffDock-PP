import sys

sys.path.append("src")


from args import parse_args
from data import load_data
from tqdm import tqdm

from data.utils import DIPSLoader


def main(args=None):
    # he he he
    with open("data/goodluck.txt") as f:
        for line in f:
            print(line, end="")
    if args is None:
        args = parse_args()
    args.debug = False

    for split in ("test", "val", "train"):
        print(f"Generate {split} cache")

        n_batches = DIPSLoader.get_n_batches()[split]
        batch_indexes = list(range(n_batches))

        for batch_index in tqdm(batch_indexes):
            load_data(args, split=split, batch=batch_index)


if __name__ == "__main__":
    main()

